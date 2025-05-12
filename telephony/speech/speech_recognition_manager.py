"""
Speech recognition management for telephony integration.
"""
import logging
import asyncio
import numpy as np
from typing import Optional

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from integration.stt_integration import STTIntegration
from telephony.speech.transcription_cleaner import TranscriptionCleaner

logger = logging.getLogger(__name__)

class SpeechRecognitionManager:
    """Manages Google Cloud Speech recognition for telephony with proper integration."""
    
    def __init__(self, stt_integration: Optional[STTIntegration] = None):
        self.stt_integration = stt_integration
        self.cleaner = TranscriptionCleaner()
        self.session_active = False
        self.min_words_for_valid_query = 1
    
    async def start_session(self) -> None:
        """Start speech recognition session."""
        try:
            if not self.stt_integration:
                # Create new STT integration if not provided
                self.stt_integration = STTIntegration(language="en-US")
                await self.stt_integration.init()
            
            if not self.session_active:
                await self.stt_integration.start_streaming()
                self.session_active = True
                logger.info("Started Google Cloud Speech streaming session")
        except Exception as e:
            logger.error(f"Error starting speech session: {e}")
            self.session_active = False
    
    async def stop_session(self) -> None:
        """Stop speech recognition session."""
        if self.session_active and self.stt_integration:
            try:
                await self.stt_integration.end_streaming()
                self.session_active = False
                logger.info("Stopped Google Cloud Speech streaming session")
            except Exception as e:
                logger.error(f"Error stopping speech session: {e}")
    
    async def process_audio(self, audio_data: np.ndarray) -> str:
        """Process audio through speech recognition."""
        if not self.session_active or not self.stt_integration:
            logger.warning("Speech session not active")
            return ""
        
        try:
            # Process audio through STT integration
            result = await self.stt_integration.transcribe_audio_data(
                audio_data=audio_data,
                is_short_audio=False
            )
            
            # Extract transcription
            transcription = result.get("transcription", "")
            original_transcription = result.get("original_transcription", "")
            
            # Log the processing
            logger.info(f"RAW TRANSCRIPTION: '{original_transcription}'")
            logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
            
            return transcription
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            # Reset session on error
            await self._reset_session()
            return ""
    
    def is_valid_transcription(self, text: str) -> bool:
        """Check if transcription is valid."""
        if not self.stt_integration:
            return False
            
        return self.stt_integration.is_valid_transcription(text, self.min_words_for_valid_query)
    
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