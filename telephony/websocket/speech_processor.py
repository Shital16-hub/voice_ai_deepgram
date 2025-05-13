# telephony/websocket/speech_processor.py

"""
Updated speech processor using Google Cloud STT v2.25.0+.
Removes deprecated fields and simplifies the implementation.
"""
import logging
import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np

# Import the updated Google Cloud STT
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Updated speech processor for Google Cloud STT v2.25.0+.
    Removes all deprecated field usage and complex abstractions.
    """
    
    def __init__(self, pipeline):
        """Initialize updated speech processor."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor")
        
        # Use the updated Google Cloud STT (keeps original class name for compatibility)
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,  # Twilio uses 8kHz
            encoding="MULAW",   # Twilio uses MULAW (NOT ALAW)
            channels=1,
            interim_results=False,  # Disable for lower latency
            enhanced_model=True
        )
        
        # Simple transcription cleaning patterns
        self.cleanup_patterns = [
            # Remove technical annotations
            (re.compile(r'\[.*?\]'), ''),
            (re.compile(r'\(.*?\)'), ''),
            (re.compile(r'<.*?>'), ''),
            
            # Common filler words
            (re.compile(r'\b(um|uh|er|ah|hmm|mmm|like|you know)\b', re.IGNORECASE), ''),
            
            # Multiple spaces
            (re.compile(r'\s+'), ' '),
        ]
        
        # Echo detection
        self.echo_history = []
        self.max_echo_history = 5
        
        # Statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        
        logger.info("SpeechProcessor initialized successfully")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with the updated Google Cloud STT.
        
        Args:
            audio_data: Audio data as bytes
            callback: Optional callback function
            
        Returns:
            Transcription text or None
        """
        self.audio_chunks_received += 1
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Process through updated STT
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            
            if result and result.text:
                self.successful_transcriptions += 1
                
                # Clean up transcription
                cleaned_text = self.cleanup_transcription(result.text)
                
                if cleaned_text:
                    logger.info(f"Transcription successful: '{cleaned_text}'")
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
        Clean up transcription using simple patterns.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
        
        original_text = text
        cleaned = text
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Remove very short results
        if len(cleaned) <= 1:
            return ""
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        if original_text != cleaned:
            logger.debug(f"Cleaned transcription: '{original_text}' -> '{cleaned}'")
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Validate transcription using simple criteria.
        
        Args:
            text: Transcription text
            
        Returns:
            True if valid
        """
        cleaned = self.cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Must have at least one word
        words = cleaned.split()
        if len(words) < 1:
            return False
        
        # Must be longer than 2 characters
        if len(cleaned) < 2:
            return False
        
        # Check for reasonable word lengths
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 1 or avg_word_length > 20:
            return False
        
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Detect echo using simple string matching.
        
        Args:
            transcription: User transcription
            
        Returns:
            True if it's an echo
        """
        if not transcription or not self.echo_history:
            return False
        
        normalized_trans = transcription.lower().strip()
        
        # Check against recent responses
        for recent_response in self.echo_history:
            normalized_response = recent_response.lower().strip()
            
            # Exact match
            if normalized_trans == normalized_response:
                logger.info(f"Detected exact echo: '{transcription}'")
                return True
            
            # Substring match for longer texts
            if len(normalized_response) > 20:
                if normalized_trans in normalized_response or normalized_response in normalized_trans:
                    logger.info(f"Detected substring echo: '{transcription}'")
                    return True
            
            # Word overlap for shorter texts
            trans_words = set(normalized_trans.split())
            response_words = set(normalized_response.split())
            
            if trans_words and response_words:
                overlap_ratio = len(trans_words & response_words) / max(len(trans_words), len(response_words))
                if overlap_ratio > 0.7:  # 70% overlap
                    logger.info(f"Detected word-overlap echo: {overlap_ratio:.1%}")
                    return True
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo history."""
        if response and len(response) > 10:
            self.echo_history.append(response)
            if len(self.echo_history) > self.max_echo_history:
                self.echo_history.pop(0)
            logger.debug(f"Added to echo history: '{response[:30]}...'")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session."""
        await self.speech_client.stop_streaming()
        
        # Log statistics
        logger.info("Stopped speech session")
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