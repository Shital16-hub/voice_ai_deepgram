"""
Enhanced speech processor with extensive debugging and error handling.
"""
import logging
import re
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np

from speech_to_text.simple_google_stt import SimpleGoogleSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Speech processor with comprehensive debugging."""
    
    def __init__(self, pipeline):
        """Initialize speech processor."""
        self.pipeline = pipeline
        
        # Initialize with debug logging
        logger.info("Initializing SpeechProcessor")
        
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=8000,
            enable_automatic_punctuation=True,
            enhanced_telephony=True
        )
        self.google_speech_active = False
        
        # Simple patterns
        self.non_speech_patterns = re.compile(
            r'\[.*?\]|\(.*?\)|music playing|background noise|static',
            re.IGNORECASE
        )
        
        # Echo detection
        self.echo_detection_history = []
        self.max_echo_history = 3
        self.min_words_for_valid_query = 1
        
        # Debugging counters
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
        Process audio with comprehensive debugging.
        """
        try:
            self.audio_chunks_received += 1
            logger.info(f"Processing audio chunk #{self.audio_chunks_received}, size: {len(audio_data)} bytes")
            
            # Initialize speech session if needed
            if not self.google_speech_active:
                logger.info("Starting Google Speech session...")
                await self.speech_client.start_streaming()
                self.google_speech_active = True
                logger.info("Google Speech session started successfully")
            
            # Results collection
            transcription_results = []
            
            async def transcription_callback(result):
                logger.info(f"Received transcription callback: is_final={result.is_final}, text='{result.text}'")
                
                if result.is_final:
                    transcription_results.append(result)
                    logger.info(f"Added final result: '{result.text}' (confidence: {result.confidence})")
                    self.successful_transcriptions += 1
                
                # Call original callback if provided
                if callback:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.error(f"Error in original callback: {e}")
            
            logger.info("Sending audio to Google Cloud STT...")
            
            # Process the audio
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=transcription_callback
            )
            
            logger.info(f"Google Cloud STT returned: {result}")
            
            # Return best result
            if transcription_results:
                best_result = max(transcription_results, key=lambda r: r.confidence)
                logger.info(f"Returning best result: '{best_result.text}'")
                return best_result.text
            elif result and result.text:
                logger.info(f"Returning direct result: '{result.text}'")
                return result.text
            else:
                logger.warning("No transcription results obtained")
                self.failed_transcriptions += 1
                return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription text with logging."""
        if not text:
            return ""
        
        original_text = text
        
        # Remove non-speech annotations
        cleaned_text = self.non_speech_patterns.sub('', text)
        
        # Basic cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if original_text != cleaned_text:
            logger.info(f"Cleaned transcription: '{original_text}' -> '{cleaned_text}'")
        
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """Check if transcription is valid with detailed logging."""
        logger.info(f"Validating transcription: '{text}'")
        
        cleaned_text = self.cleanup_transcription(text)
        
        if not cleaned_text:
            logger.info("Rejected: Empty after cleanup")
            return False
        
        # Always accept questions
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'can', 'could', 'would', 'is', 'are']
        if any(word in cleaned_text.lower() for word in question_words):
            logger.info(f"Accepted: Contains question word - '{cleaned_text}'")
            return True
        
        # Accept if it contains relevant keywords
        relevant_words = ['feature', 'price', 'cost', 'plan', 'service', 'support', 'voiceassist']
        if any(word in cleaned_text.lower() for word in relevant_words):
            logger.info(f"Accepted: Contains relevant keyword - '{cleaned_text}'")
            return True
        
        # Minimum word count
        word_count = len(cleaned_text.split())
        if word_count >= self.min_words_for_valid_query:
            logger.info(f"Accepted: Meets word count ({word_count}) - '{cleaned_text}'")
            return True
        
        logger.info(f"Rejected: Doesn't meet criteria - '{cleaned_text}'")
        return False
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Check for echo with detailed logging."""
        if not transcription or not self.echo_detection_history:
            return False
        
        logger.debug(f"Checking for echo: '{transcription}'")
        logger.debug(f"Recent responses: {self.echo_detection_history}")
        
        # Simple echo detection
        for i, recent_phrase in enumerate(self.echo_detection_history):
            if len(recent_phrase) > 15:
                # Check for word overlap
                transcription_words = set(transcription.lower().split())
                phrase_words = set(recent_phrase.lower().split())
                overlap = len(transcription_words & phrase_words)
                
                if overlap > 2 and overlap > len(phrase_words) * 0.5:
                    logger.info(f"Detected echo: {overlap} words overlap with response #{i}")
                    logger.info(f"  Transcription: '{transcription}'")
                    logger.info(f"  Recent phrase: '{recent_phrase}'")
                    return True
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo history with logging."""
        if len(response) > 20:
            self.echo_detection_history.append(response)
            if len(self.echo_detection_history) > self.max_echo_history:
                removed = self.echo_detection_history.pop(0)
                logger.debug(f"Removed old echo history: '{removed}'")
            logger.debug(f"Added to echo history: '{response}' (total: {len(self.echo_detection_history)})")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session with stats."""
        if self.google_speech_active:
            try:
                await self.speech_client.stop_streaming()
                self.google_speech_active = False
                logger.info("Stopped Google Speech session")
                logger.info(f"Session stats: {self.audio_chunks_received} chunks received, "
                           f"{self.successful_transcriptions} successful, "
                           f"{self.failed_transcriptions} failed")
            except Exception as e:
                logger.error(f"Error stopping speech session: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "google_speech_active": self.google_speech_active,
            "echo_history_size": len(self.echo_detection_history)
        }