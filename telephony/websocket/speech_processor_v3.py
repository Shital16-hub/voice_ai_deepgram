# telephony/websocket/speech_processor_v3.py

"""
Advanced speech processor using Google Cloud STT v2.32.0 automatic features.
Removes hardcoding and leverages API's built-in intelligence.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
import re

# Import the fixed STT implementation
from speech_to_text.google_cloud_stt_v2 import GoogleCloudStreamingSTT_V2

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Advanced speech processor that leverages Google Cloud STT's automatic features.
    No hardcoded keywords or patterns - lets the API do its job!
    """
    
    def __init__(self, pipeline):
        """Initialize speech processor with minimal intervention."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor with Google Cloud STT automatic features")
        
        # Use the fixed Google Cloud STT with automatic features enabled
        self.speech_client = GoogleCloudStreamingSTT_V2(
            language="en-US",
            sample_rate=8000,  # Twilio's sample rate
            encoding="MULAW",   # Twilio's encoding
            channels=1,
            interim_results=True,  # Enable for better responsiveness
            enhanced_model=True,  # Use premium model for best accuracy
            timeout=60.0  # Increased timeout for better connection stability
        )
        
        # Minimal intervention - only basic cleanup
        self.basic_cleanup_patterns = [
            # Remove only clear technical artifacts
            (re.compile(r'\[INAUDIBLE\]', re.IGNORECASE), ''),
            (re.compile(r'\[MUSIC\]', re.IGNORECASE), ''),
            (re.compile(r'\[NOISE\]', re.IGNORECASE), ''),
            
            # Fix spacing around punctuation (API sometimes adds extra spaces)
            (re.compile(r'\s+([.!?])'), r'\1'),
            (re.compile(r'([.!?])\s+([.!?])'), r'\1\2'),
            
            # Normalize multiple spaces
            (re.compile(r'\s{2,}'), ' '),
        ]
        
        # Echo detection (simplified)
        self.recent_system_outputs = []
        self.max_echo_history = 3
        
        # Statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.interim_results_count = 0
        self.partial_transcriptions = []  # Track partial results
        
        # Utterance management
        self.current_utterance = ""
        self.utterance_timeout = 2.0  # 2 seconds of silence before considering utterance complete
        self.last_audio_time = 0
        
        logger.info("SpeechProcessor initialized - relying on API's automatic features")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with minimal intervention, letting the API handle detection.
        
        Args:
            audio_data: Audio data as bytes
            callback: Optional callback function
            
        Returns:
            Final transcription text or None
        """
        self.audio_chunks_received += 1
        self.last_audio_time = time.time()
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Define callback to handle results
            async def handle_result(result):
                if result.is_final:
                    self.successful_transcriptions += 1
                    
                    # Apply minimal cleanup
                    cleaned_text = self._minimal_cleanup(result.text)
                    
                    # Log with confidence
                    logger.info(f"Final transcription: '{cleaned_text}' "
                              f"(confidence: {result.confidence:.2f})")
                    
                    # Call user callback
                    if callback:
                        await callback(result)
                    
                    return cleaned_text
                else:
                    # Track interim results
                    self.interim_results_count += 1
                    logger.debug(f"Interim result: '{result.text}' (confidence: {result.confidence:.2f})")
                    
                    # Call user callback for interim results too
                    if callback:
                        await callback(result)
            
            # Process through Google Cloud STT
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=handle_result
            )
            
            # Return final result if available
            if result and result.is_final:
                return self._minimal_cleanup(result.text)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return None
    
    def _minimal_cleanup(self, text: str) -> str:
        """
        Apply minimal cleanup to preserve API's intelligent processing.
        
        Args:
            text: Original transcription from API
            
        Returns:
            Minimally cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Apply only essential cleanup patterns
        for pattern, replacement in self.basic_cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Let the API handle capitalization - only fix obvious issues
        if cleaned and not cleaned[0].isupper() and cleaned[0].isalpha():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Simple validation - let the API's confidence scores do most of the work.
        
        Args:
            text: Transcription text
            
        Returns:
            True if valid (very permissive)
        """
        if not text:
            return False
        
        # Apply minimal cleanup
        cleaned = self._minimal_cleanup(text)
        
        # Very basic checks
        if not cleaned or len(cleaned) < 1:
            return False
        
        # Check if it's mostly non-alphabetic characters (likely noise)
        alpha_ratio = sum(1 for c in cleaned if c.isalpha()) / len(cleaned)
        if alpha_ratio < 0.3:  # Less than 30% alphabetic
            return False
        
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Simple echo detection using string similarity.
        
        Args:
            transcription: User transcription
            
        Returns:
            True if it's likely an echo
        """
        if not transcription or not self.recent_system_outputs:
            return False
        
        cleaned_transcription = transcription.lower().strip()
        
        # Check against recent system outputs
        for output in self.recent_system_outputs:
            cleaned_output = output.lower().strip()
            
            # Simple check: if transcription is contained in output or vice versa
            if (len(cleaned_transcription) > 10 and len(cleaned_output) > 10):
                if (cleaned_transcription in cleaned_output or 
                    cleaned_output in cleaned_transcription):
                    
                    # Additional check for substantial overlap
                    shorter = min(len(cleaned_transcription), len(cleaned_output))
                    longer = max(len(cleaned_transcription), len(cleaned_output))
                    
                    if longer / shorter < 2.0:  # Less than 2x difference in length
                        logger.info(f"Detected potential echo: '{transcription}'")
                        return True
        
        return False
    
    def add_to_echo_history(self, text: str) -> None:
        """
        Add system output to echo detection history.
        
        Args:
            text: System output text
        """
        if text and len(text) > 5:
            self.recent_system_outputs.append(text)
            
            # Maintain limited history
            if len(self.recent_system_outputs) > self.max_echo_history:
                self.recent_system_outputs.pop(0)
    
    async def handle_utterance_timeout(self) -> Optional[str]:
        """
        Handle timeout of current utterance - return accumulated partial results.
        
        Returns:
            Accumulated partial transcription if available
        """
        current_time = time.time()
        if (self.current_utterance and 
            current_time - self.last_audio_time > self.utterance_timeout):
            
            logger.info(f"Utterance timeout - returning partial: '{self.current_utterance}'")
            result = self.current_utterance
            self.current_utterance = ""
            return result
        
        return None
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session and get final results."""
        # Get any final results
        final_text, duration = await self.speech_client.stop_streaming()
        
        # Calculate statistics
        total_chunks = max(self.audio_chunks_received, 1)
        success_rate = (self.successful_transcriptions / total_chunks) * 100
        
        # Log session summary
        logger.info("Speech session ended")
        logger.info(f"Final text: '{final_text}'")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Chunks processed: {self.audio_chunks_received}")
        logger.info(f"Successful transcriptions: {self.successful_transcriptions}")
        logger.info(f"Interim results: {self.interim_results_count}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Reset for next session
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.interim_results_count = 0
        self.current_utterance = ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_chunks = max(self.audio_chunks_received, 1)
        
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "interim_results_count": self.interim_results_count,
            "echo_history_size": len(self.recent_system_outputs),
            "success_rate": round((self.successful_transcriptions / total_chunks) * 100, 2),
            "stt_stats": self.speech_client.get_stats(),
            "current_utterance_length": len(self.current_utterance),
            "last_audio_time": self.last_audio_time
        }