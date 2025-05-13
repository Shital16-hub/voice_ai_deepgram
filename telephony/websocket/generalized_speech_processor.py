# telephony/websocket/generalized_speech_processor.py

"""
Generalized speech processor that works across different domains
without hardcoded business logic or domain-specific context.
"""
import logging
import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

from speech_to_text.optimized_google_stt import OptimizedGoogleSTT

logger = logging.getLogger(__name__)

class GeneralizedSpeechProcessor:
    """
    Generalized speech processor that focuses on audio quality and recognition
    without domain-specific hardcoding. Suitable for any application domain.
    """
    
    def __init__(self, pipeline, language_code: str = "en-US"):
        """Initialize generalized speech processor."""
        self.pipeline = pipeline
        self.language_code = language_code
        
        logger.info("Initializing GeneralizedSpeechProcessor")
        
        # Use optimized STT client with no domain-specific configuration
        self.speech_client = OptimizedGoogleSTT(
            language_code=language_code,
            sample_rate=8000,
            enable_automatic_punctuation=True,
            model="phone_call",  # Best for telephony, domain-agnostic
            use_enhanced=True     # Premium model for better accuracy
        )
        
        # General-purpose text cleaning patterns
        self.cleanup_patterns = [
            # Remove technical annotations
            (re.compile(r'\[.*?\]'), ''),
            (re.compile(r'\(.*?\)'), ''),
            (re.compile(r'<.*?>'), ''),
            
            # Common transcription artifacts
            (re.compile(r'\b(um|uh|er|ah|hmm|mmm)\b', re.IGNORECASE), ''),
            (re.compile(r'\b(like|you know)\b', re.IGNORECASE), ''),
            
            # Multiple spaces and leading/trailing whitespace
            (re.compile(r'\s+'), ' '),
        ]
        
        # Echo detection (content-agnostic)
        self.recent_outputs = []
        self.max_echo_history = 5
        
        # Processing statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        self.empty_results = 0
        
        # Timing optimization
        self.last_transcription_time = 0
        self.min_transcription_gap = 0.2  # 200ms minimum between attempts
        
        logger.info("GeneralizedSpeechProcessor initialized with domain-agnostic configuration")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with quality-focused approach, no domain assumptions.
        """
        try:
            self.audio_chunks_received += 1
            current_time = time.time()
            
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Prevent too frequent processing
            if current_time - self.last_transcription_time < self.min_transcription_gap:
                logger.debug("Throttling: too soon after last transcription")
                return None
            
            # Process through optimized STT
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=callback
            )
            
            if result and result.text:
                self.last_transcription_time = current_time
                self.successful_transcriptions += 1
                
                # Clean up transcription (domain-agnostic)
                cleaned_text = self._cleanup_transcription(result.text)
                
                if cleaned_text:
                    logger.info(f"Transcription successful: '{cleaned_text}'")
                    return cleaned_text
                else:
                    logger.debug("Transcription cleaned to empty string")
                    self.empty_results += 1
            else:
                logger.debug("No transcription result")
                self.failed_transcriptions += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def _cleanup_transcription(self, text: str) -> str:
        """
        Clean up transcription using general-purpose rules.
        No domain-specific logic.
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Apply general cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Remove very short or single-character results
        if len(cleaned) <= 1:
            return ""
        
        # Capitalize first letter of sentences
        sentences = cleaned.split('. ')
        capitalized = [s.capitalize() for s in sentences if s]
        cleaned = '. '.join(capitalized)
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Validate transcription using general criteria.
        No business or domain-specific validation.
        """
        if not text:
            return False
        
        # Clean first
        cleaned = self._cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Basic quality checks
        words = cleaned.split()
        
        # Minimum word count (adjustable)
        if len(words) < 1:
            return False
        
        # Check for reasonable character distribution
        if len(cleaned) < 2:
            return False
        
        # Reject if all the same character repeated
        if len(set(cleaned.lower())) < 2:
            return False
        
        # Check for reasonable word lengths
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 1 or avg_word_length > 20:
            return False
        
        logger.debug(f"Transcription validated: '{cleaned}'")
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Detect echo using general string matching.
        No domain-specific context required.
        """
        if not transcription or not self.recent_outputs:
            return False
        
        # Normalize for comparison
        normalized_transcription = transcription.lower().strip()
        
        # Check against recent outputs
        for recent_output in self.recent_outputs:
            normalized_output = recent_output.lower().strip()
            
            # Exact match
            if normalized_transcription == normalized_output:
                logger.info(f"Detected exact echo: '{transcription}'")
                return True
            
            # Substring match for longer texts
            if len(normalized_output) > 20:
                if (normalized_transcription in normalized_output or 
                    normalized_output in normalized_transcription):
                    logger.info(f"Detected substring echo: '{transcription}'")
                    return True
            
            # Word overlap for shorter texts
            trans_words = set(normalized_transcription.split())
            output_words = set(normalized_output.split())
            
            if trans_words and output_words:
                overlap_ratio = len(trans_words & output_words) / max(len(trans_words), len(output_words))
                if overlap_ratio > 0.7:  # 70% overlap
                    logger.info(f"Detected word-overlap echo: {overlap_ratio:.1%}")
                    return True
        
        return False
    
    def add_to_echo_history(self, output: str) -> None:
        """Add output to echo detection history (general purpose)."""
        if output and len(output) > 3:  # Only track substantial outputs
            self.recent_outputs.append(output)
            
            # Maintain limited history
            if len(self.recent_outputs) > self.max_echo_history:
                self.recent_outputs.pop(0)
            
            logger.debug(f"Added to echo history: '{output[:30]}...' "
                        f"(total: {len(self.recent_outputs)})")
    
    async def stop_session(self) -> None:
        """Stop processing session and log statistics."""
        # Flush any remaining audio
        if hasattr(self.speech_client, 'flush_buffer'):
            await self.speech_client.flush_buffer()
        
        # Log session statistics
        total_attempts = self.audio_chunks_received
        success_rate = (self.successful_transcriptions / max(total_attempts, 1)) * 100
        
        logger.info("Speech processing session ended")
        logger.info(f"Total chunks: {total_attempts}")
        logger.info(f"Successful: {self.successful_transcriptions}")
        logger.info(f"Failed: {self.failed_transcriptions}")
        logger.info(f"Empty results: {self.empty_results}")
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        total_attempts = max(self.audio_chunks_received, 1)
        
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "empty_results": self.empty_results,
            "echo_history_size": len(self.recent_outputs),
            "success_rate": round((self.successful_transcriptions / total_attempts) * 100, 2),
            "failure_rate": round((self.failed_transcriptions / total_attempts) * 100, 2),
            "empty_rate": round((self.empty_results / total_attempts) * 100, 2),
            "language_code": self.language_code,
            "stt_stats": self.speech_client.get_stats()
        }