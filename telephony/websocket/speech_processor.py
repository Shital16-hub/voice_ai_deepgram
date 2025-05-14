# telephony/websocket/speech_processor.py

"""
Enhanced speech processor with temporal-based echo detection and proper v2 integration.
"""
import logging
import re
import asyncio
import time
import os
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np

# Import the updated Google Cloud STT v2
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Enhanced speech processor with advanced echo detection using temporal analysis.
    Optimized for telephony with Google Cloud STT v2.
    """
    
    def __init__(self, pipeline):
        """Initialize enhanced speech processor for telephony."""
        self.pipeline = pipeline
        
        logger.info("Initializing Enhanced SpeechProcessor with v2 API and echo detection")
        
        # Get project ID with automatic extraction
        project_id = self._get_project_id()
        
        # Use Google Cloud STT v2 with optimal telephony settings
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,  # Match Twilio's 8kHz
            encoding="MULAW",   # Direct mulaw support
            channels=1,
            interim_results=False,  # Disable for better accuracy
            project_id=project_id,
            enhanced_model=True,    # Use enhanced telephony model
            location="global"       # Can be changed to specific region
        )
        
        # Minimal transcription cleaning - let v2 API handle most processing
        self.cleanup_patterns = [
            # Only remove obvious artifacts
            (re.compile(r'\[.*?\]'), ''),  # [inaudible]
            (re.compile(r'\<.*?\>'), ''),  # <music>
        ]
        
        # Advanced echo detection with temporal tracking
        self.echo_history = []
        self.max_echo_history = 10
        self.tts_output_timestamps = []  # Track when TTS outputs speech
        self.echo_detection_window = 3.0  # 3 second window for echo detection
        
        # Performance statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        self.echo_detections = 0
        
        logger.info("Enhanced SpeechProcessor initialized with temporal echo detection")
    
    def _get_project_id(self) -> str:
        """Auto-extract project ID from environment or credentials."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"SpeechProcessor: Auto-extracted project ID: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        raise ValueError(
            "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
            "or ensure your credentials file contains a project_id field."
        )
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with advanced echo detection and v2 optimization.
        """
        self.audio_chunks_received += 1
        current_time = time.time()
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Process through Google Cloud STT v2
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            
            if result and result.text and result.is_final:
                self.successful_transcriptions += 1
                
                # Clean up transcription minimally
                cleaned_text = self.cleanup_transcription(result.text)
                
                if cleaned_text:
                    # Advanced echo detection using temporal analysis
                    if self._is_echo_temporal(cleaned_text, current_time):
                        self.echo_detections += 1
                        logger.info(f"ECHO DETECTED (temporal): '{cleaned_text}'")
                        return None
                    
                    # Traditional echo detection as backup
                    if self.is_echo_of_system_speech(cleaned_text):
                        self.echo_detections += 1
                        logger.info(f"ECHO DETECTED (content): '{cleaned_text}'")
                        return None
                    
                    logger.info(f"Valid transcription: '{cleaned_text}' (confidence: {result.confidence:.2f})")
                    return cleaned_text
                else:
                    logger.debug("Transcription cleaned to empty string")
            else:
                logger.debug("No final transcription result")
                self.failed_transcriptions += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Minimal cleanup - let v2 API handle most processing.
        """
        if not text:
            return ""
        
        original_text = text
        cleaned = text
        
        # Apply minimal cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Ensure proper capitalization
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        if original_text != cleaned:
            logger.debug(f"Cleaned: '{original_text}' -> '{cleaned}'")
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Validate transcription with minimal requirements.
        """
        cleaned = self.cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Must have at least one word
        words = cleaned.split()
        if len(words) < 1:
            return False
        
        # Must be longer than 1 character
        if len(cleaned) < 2:
            return False
        
        return True
    
    def register_tts_output(self, output_text: str) -> None:
        """
        Register when TTS outputs speech for temporal echo detection.
        
        Args:
            output_text: The text that was converted to speech
        """
        current_time = time.time()
        
        # Add to TTS output history with timestamp
        self.tts_output_timestamps.append({
            'text': output_text.lower().strip(),
            'timestamp': current_time,
            'words': set(output_text.lower().split())
        })
        
        # Clean old entries (older than echo detection window)
        cutoff_time = current_time - self.echo_detection_window
        self.tts_output_timestamps = [
            entry for entry in self.tts_output_timestamps 
            if entry['timestamp'] > cutoff_time
        ]
        
        logger.debug(f"Registered TTS output for echo detection: '{output_text[:30]}...'")
    
    def _is_echo_temporal(self, transcription: str, transcription_time: float) -> bool:
        """
        Advanced echo detection using temporal analysis.
        
        Args:
            transcription: The transcribed text
            transcription_time: When the transcription was received
            
        Returns:
            True if this is likely an echo
        """
        if not self.tts_output_timestamps:
            return False
        
        normalized_transcription = transcription.lower().strip()
        trans_words = set(normalized_transcription.split())
        
        # Check against recent TTS outputs
        for tts_entry in self.tts_output_timestamps:
            time_diff = transcription_time - tts_entry['timestamp']
            
            # Echo typically occurs within 1-2 seconds of TTS output
            if 0.5 <= time_diff <= 2.5:
                # Calculate word overlap
                word_overlap = len(trans_words & tts_entry['words'])
                overlap_ratio = word_overlap / max(len(trans_words), len(tts_entry['words']))
                
                # High overlap within the time window indicates echo
                if overlap_ratio > 0.6:  # 60% word overlap
                    logger.debug(f"Echo detected: {overlap_ratio:.1%} overlap, "
                               f"{time_diff:.1f}s after TTS")
                    return True
                
                # For exact matches, use lower threshold
                if normalized_transcription == tts_entry['text']:
                    logger.debug(f"Exact echo detected: {time_diff:.1f}s after TTS")
                    return True
        
        return False
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Traditional echo detection using string matching.
        """
        if not transcription or not self.echo_history:
            return False
        
        normalized_trans = transcription.lower().strip()
        
        # Check against recent responses
        for recent_response in self.echo_history:
            normalized_response = recent_response.lower().strip()
            
            # Check for various echo patterns
            if self._is_echo_match(normalized_trans, normalized_response):
                return True
        
        return False
    
    def _is_echo_match(self, transcription: str, response: str) -> bool:
        """Check if transcription matches response (echo detection)."""
        # Exact match
        if transcription == response:
            return True
        
        # Substring match for longer texts
        if len(response) > 15:
            if transcription in response or response in transcription:
                return True
        
        # Word overlap detection
        trans_words = set(transcription.split())
        response_words = set(response.split())
        
        if trans_words and response_words:
            overlap_ratio = len(trans_words & response_words) / max(len(trans_words), len(response_words))
            return overlap_ratio > 0.7  # 70% word overlap
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to traditional echo history."""
        if response and len(response) > 5:
            self.echo_history.append(response)
            if len(self.echo_history) > self.max_echo_history:
                self.echo_history.pop(0)
            
            # Also register for temporal detection
            self.register_tts_output(response)
            
            logger.debug(f"Added to echo history: '{response[:30]}...'")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session and log statistics."""
        await self.speech_client.stop_streaming()
        
        # Log comprehensive statistics
        total_attempts = self.audio_chunks_received
        success_rate = (self.successful_transcriptions / max(total_attempts, 1)) * 100
        echo_rate = (self.echo_detections / max(self.successful_transcriptions + self.echo_detections, 1)) * 100
        
        logger.info("Speech session ended - Enhanced Statistics:")
        logger.info(f"Total chunks: {total_attempts}")
        logger.info(f"Successful transcriptions: {self.successful_transcriptions}")
        logger.info(f"Failed transcriptions: {self.failed_transcriptions}")
        logger.info(f"Echo detections: {self.echo_detections}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Echo detection rate: {echo_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including echo detection."""
        total_attempts = max(self.audio_chunks_received, 1)
        total_processed = self.successful_transcriptions + self.echo_detections
        
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "echo_detections": self.echo_detections,
            "echo_history_size": len(self.echo_history),
            "tts_tracking_entries": len(self.tts_output_timestamps),
            "success_rate": round((self.successful_transcriptions / total_attempts) * 100, 2),
            "echo_detection_rate": round((self.echo_detections / max(total_processed, 1)) * 100, 2),
            "stt_stats": self.speech_client.get_stats()
        }