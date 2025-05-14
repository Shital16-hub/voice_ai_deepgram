# telephony/websocket/speech_processor.py

"""
Optimized speech processor using Google Cloud STT v2.32.0+ for telephony.
"""
import logging
import re
import asyncio
import time
import os
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np

# Import the updated Google Cloud STT
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Speech processor optimized for telephony using Google Cloud STT v2.32.0+.
    Minimal preprocessing, optimal configuration.
    """
    
    def __init__(self, pipeline):
        """Initialize optimized speech processor for telephony."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor for telephony with v2 API")
        
        # Get project ID with automatic extraction
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not in environment, try to extract from credentials file
        if not project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id')
                        logger.info(f"SpeechProcessor: Auto-extracted project ID from credentials: {project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
        
        if not project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
        
        # Use Google Cloud STT v2 optimized for telephony
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,  # Match Twilio's 8kHz
            encoding="MULAW",   # Match Twilio's mulaw encoding
            channels=1,
            interim_results=False,  # Disable for better accuracy
            project_id=project_id,
            enhanced_model=True,    # Use enhanced telephony model
            location="global"       # Can be changed to specific region if needed
        )
        
        # Minimal transcription cleaning patterns
        self.cleanup_patterns = [
            # Remove only obvious technical artifacts
            (re.compile(r'\[.*?\]'), ''),  # [inaudible], [music]
            (re.compile(r'\<.*?\>'), ''),  # <noise>
            # Don't remove filler words - they're natural speech
        ]
        
        # Echo detection
        self.echo_history = []
        self.max_echo_history = 5
        
        # Statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        
        logger.info("SpeechProcessor initialized with telephony optimization and v2 API")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with minimal intervention for best accuracy.
        """
        self.audio_chunks_received += 1
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Pass audio directly to Google Cloud STT v2
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            
            if result and result.text:
                self.successful_transcriptions += 1
                
                # Apply minimal cleanup
                cleaned_text = self.cleanup_transcription(result.text)
                
                if cleaned_text:
                    logger.info(f"Transcription successful: '{cleaned_text}' (confidence: {result.confidence:.2f})")
                    return cleaned_text
                else:
                    logger.debug("Transcription cleaned to empty string")
            else:
                logger.debug("No transcription result")
                self.failed_transcriptions += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Apply minimal cleanup to preserve natural speech.
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
        
        # Ensure first letter is capitalized
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        if original_text != cleaned:
            logger.debug(f"Cleaned transcription: '{original_text}' -> '{cleaned}'")
        
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
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Detect echo using simple string matching.
        """
        if not transcription or not self.echo_history:
            return False
        
        normalized_trans = transcription.lower().strip()
        
        # Check against recent responses
        for recent_response in self.echo_history:
            normalized_response = recent_response.lower().strip()
            
            # Check for various echo patterns
            if self._is_echo_match(normalized_trans, normalized_response):
                logger.info(f"Detected echo: '{transcription}'")
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
        """Add response to echo history."""
        if response and len(response) > 5:
            self.echo_history.append(response)
            if len(self.echo_history) > self.max_echo_history:
                self.echo_history.pop(0)
            logger.debug(f"Added to echo history: '{response[:30]}...'")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session."""
        await self.speech_client.stop_streaming()
        
        # Log statistics
        logger.info("Speech session ended")
        logger.info(f"Total chunks: {self.audio_chunks_received}")
        logger.info(f"Successful: {self.successful_transcriptions}")
        logger.info(f"Failed: {self.failed_transcriptions}")
        
        success_rate = (self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "echo_history_size": len(self.echo_history),
            "success_rate": round((self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100, 2),
            "stt_stats": self.speech_client.get_stats()
        }