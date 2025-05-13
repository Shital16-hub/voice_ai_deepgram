# telephony/websocket/speech_processor.py

"""
Enhanced speech processor with better error handling and optimized Google Cloud STT.
"""
import logging
import re
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np
import time

from speech_to_text.simple_google_stt import SimpleGoogleSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Enhanced speech processor with better recognition and error handling.
    """
    
    def __init__(self, pipeline):
        """Initialize enhanced speech processor."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor")
        
        # Initialize with enhanced settings focused on quality, not domain-specific phrases
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=8000,
            enable_automatic_punctuation=True,
            enhanced_telephony=True
        )
        self.google_speech_active = False
        
        # General-purpose pattern matching (no business-specific terms)
        self.non_speech_patterns = re.compile(
            r'\[.*?\]|\(.*?\)|<.*?>|'  # Technical annotations
            r'music playing|background noise|static|'  # Common annotations
            r'\b(um|uh|er|ah|hmm|mmm|like|you know)\b',  # Filler words
            re.IGNORECASE
        )
        
        # Echo detection (content-agnostic)
        self.echo_detection_history = []
        self.max_echo_history = 5
        self.min_words_for_valid_query = 1
        
        # Processing statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        
        logger.info("SpeechProcessor initialized with enhanced recognition")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with enhanced reliability and error handling.
        """
        try:
            self.audio_chunks_received += 1
            logger.info(f"Processing audio chunk #{self.audio_chunks_received}, "
                       f"size: {len(audio_data)} bytes")
            
            # Initialize speech session if needed
            if not self.google_speech_active:
                logger.info("Starting Google Speech session...")
                await self.speech_client.start_streaming()
                self.google_speech_active = True
                logger.info("Google Speech session started successfully")
            
            # Results collection
            transcription_results = []
            
            async def transcription_callback(result):
                logger.info(f"Received transcription: is_final={result.is_final}, "
                           f"text='{result.text}' (confidence: {result.confidence})")
                
                if result.is_final and result.text:
                    transcription_results.append(result)
                    logger.info(f"Added final result: '{result.text}'")
                    self.successful_transcriptions += 1
                
                # Call original callback if provided
                if callback:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.error(f"Error in original callback: {e}")
            
            logger.info("Sending audio to enhanced Google Cloud STT...")
            
            # Process the audio with enhanced client
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=transcription_callback
            )
            
            logger.info(f"Enhanced Google Cloud STT returned: {result}")
            
            # Return best result
            if transcription_results:
                # Get the most recent final result
                best_result = transcription_results[-1]
                logger.info(f"Returning transcription: '{best_result.text}'")
                return best_result.text
            elif result and result.text:
                logger.info(f"Returning direct result: '{result.text}'")
                return result.text
            else:
                logger.info("No transcription results obtained")
                self.failed_transcriptions += 1
                return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """Enhanced transcription cleanup with general rules."""
        if not text:
            return ""
        
        original_text = text
        
        # Remove non-speech annotations
        cleaned_text = self.non_speech_patterns.sub('', text)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)(\s+\1\b){2,}', r'\1', cleaned_text)
        
        # Clean up whitespace and punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Capitalize sentences
        sentences = cleaned_text.split('. ')
        capitalized_sentences = [s.capitalize() for s in sentences if s]
        cleaned_text = '. '.join(capitalized_sentences)
        
        if original_text != cleaned_text:
            logger.info(f"Cleaned transcription: '{original_text}' -> '{cleaned_text}'")
        
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """Enhanced transcription validation with general criteria."""
        logger.info(f"Validating transcription: '{text}'")
        
        cleaned_text = self.cleanup_transcription(text)
        
        if not cleaned_text:
            logger.info("Rejected: Empty after cleanup")
            return False
        
        # Basic quality checks
        words = cleaned_text.split()
        
        # Check for minimum word count
        if len(words) < self.min_words_for_valid_query:
            logger.info(f"Rejected: Too few words ({len(words)})")
            return False
        
        # Check for reasonable character distribution
        if len(cleaned_text) < 2:
            logger.info("Rejected: Too short")
            return False
        
        # Check for question patterns (generally useful regardless of domain)
        question_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which',
                              'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
                              'is', 'are', 'was', 'were', 'have', 'has', 'had']
        
        text_lower = cleaned_text.lower()
        if any(word in text_lower.split() for word in question_indicators):
            logger.info(f"Accepted: Contains question indicator - '{cleaned_text}'")
            return True
        
        # Length-based acceptance (more lenient)
        if len(words) >= 2:
            # Check if it forms a meaningful phrase
            if not all(len(word) <= 2 for word in words):  # Not all short words
                logger.info(f"Accepted: Substantial content - '{cleaned_text}'")
                return True
        
        logger.info(f"Rejected: Doesn't meet validation criteria - '{cleaned_text}'")
        return False
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Enhanced echo detection using general algorithms."""
        if not transcription or not self.echo_detection_history:
            return False
        
        logger.debug(f"Checking for echo: '{transcription}'")
        
        # Normalize transcription for comparison
        normalized_transcription = transcription.lower().strip()
        words = set(normalized_transcription.split())
        
        # Check against recent responses
        for i, recent_phrase in enumerate(self.echo_detection_history):
            normalized_phrase = recent_phrase.lower().strip()
            phrase_words = set(normalized_phrase.split())
            
            # Multiple echo detection strategies
            
            # 1. Exact match (case-insensitive)
            if normalized_transcription == normalized_phrase:
                logger.info(f"Detected exact echo: '{transcription}' matches recent response")
                return True
            
            # 2. Substring match for longer phrases
            if len(normalized_phrase) > 20:
                if normalized_transcription in normalized_phrase or normalized_phrase in normalized_transcription:
                    logger.info(f"Detected substring echo: '{transcription}' overlaps with recent response")
                    return True
            
            # 3. Word overlap analysis
            if len(phrase_words) > 3:  # Only for substantial phrases
                overlap = len(words & phrase_words)
                max_words = max(len(words), len(phrase_words))
                
                # High overlap indicates possible echo
                if overlap > 3 and overlap / max_words > 0.7:
                    logger.info(f"Detected word overlap echo: {overlap}/{max_words} words match")
                    return True
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo history with intelligent filtering."""
        if len(response) > 10:  # Only track substantial responses
            self.echo_detection_history.append(response)
            if len(self.echo_detection_history) > self.max_echo_history:
                removed = self.echo_detection_history.pop(0)
                logger.debug(f"Removed old echo history: '{removed[:30]}...'")
            logger.debug(f"Added to echo history: '{response[:30]}...' "
                        f"(total: {len(self.echo_detection_history)})")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session with comprehensive cleanup."""
        if self.google_speech_active:
            try:
                await self.speech_client.stop_streaming()
                self.google_speech_active = False
                
                # Log session statistics
                logger.info("Stopped Google Speech session")
                logger.info(f"Session stats: {self.audio_chunks_received} chunks received")
                logger.info(f"  Successful: {self.successful_transcriptions}")
                logger.info(f"  Failed: {self.failed_transcriptions}")
                
                success_rate = (self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100
                logger.info(f"  Success rate: {success_rate:.1f}%")
                
            except Exception as e:
                logger.error(f"Error stopping speech session: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        total_chunks = max(self.audio_chunks_received, 1)
        
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "google_speech_active": self.google_speech_active,
            "echo_history_size": len(self.echo_detection_history),
            "success_rate": round((self.successful_transcriptions / total_chunks) * 100, 2),
        }