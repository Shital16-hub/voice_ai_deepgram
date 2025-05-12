"""
Speech recognition management for telephony integration.
"""
import logging
import asyncio
import numpy as np
from typing import Optional

from speech_to_text.simple_google_stt import SimpleGoogleSTT
from telephony.speech.transcription_cleaner import TranscriptionCleaner

logger = logging.getLogger(__name__)

class SpeechRecognitionManager:
    """Manages Google Cloud Speech recognition for telephony."""
    
    def __init__(self):
        self.speech_client = None
        self.cleaner = TranscriptionCleaner()
        self.session_active = False
        self.min_words_for_valid_query = 1
    
    async def start_session(self) -> None:
        """Start speech recognition session."""
        try:
            self.speech_client = SimpleGoogleSTT(
                language_code="en-US",
                sample_rate=16000,
                enable_automatic_punctuation=True
            )
            
            await self.speech_client.start_streaming()
            self.session_active = True
            logger.info("Started Google Cloud Speech streaming session")
        except Exception as e:
            logger.error(f"Error starting speech session: {e}")
            self.session_active = False
    
    async def stop_session(self) -> None:
        """Stop speech recognition session."""
        if self.session_active and self.speech_client:
            try:
                await self.speech_client.stop_streaming()
                logger.info("Stopped Google Cloud Speech streaming session")
            except Exception as e:
                logger.error(f"Error stopping speech session: {e}")
        
        self.session_active = False
    
    async def process_audio(self, audio_data: np.ndarray) -> str:
        """Process audio through speech recognition."""
        if not self.session_active:
            logger.warning("Speech session not active")
            return ""
        
        try:
            transcription_results = []
            
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.debug(f"Received final result: {result.text}")
            
            # Convert to bytes format
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Process chunk
            await self.speech_client.process_audio_chunk(
                audio_chunk=audio_bytes,
                callback=transcription_callback
            )
            
            # Wait for results
            await asyncio.sleep(0.5)
            
            # Get best transcription
            if transcription_results:
                best_result = max(transcription_results, 
                                key=lambda r: getattr(r, 'confidence', 0))
                transcription = best_result.text
            else:
                # Try to get final result by restarting session
                final_transcription, _ = await self.speech_client.stop_streaming()
                await self.speech_client.start_streaming()
                transcription = final_transcription
            
            # Clean transcription
            logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
            cleaned = self.cleaner.cleanup_transcription(transcription)
            logger.info(f"CLEANED TRANSCRIPTION: '{cleaned}'")
            
            return cleaned
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            # Reset session on error
            await self._reset_session()
            return ""
    
    def is_valid_transcription(self, text: str) -> bool:
        """Check if transcription is valid."""
        cleaned_text = self.cleaner.cleanup_transcription(text)
        
        if not cleaned_text:
            return False
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset recognition state."""
        self.session_active = False
    
    async def _reset_session(self) -> None:
        """Reset speech recognition session after error."""
        if self.session_active:
            try:
                await self.stop_session()
                await self.start_session()
                logger.info("Reset speech recognition session")
            except Exception as e:
                logger.error(f"Error resetting session: {e}")