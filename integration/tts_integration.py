"""
Simplified TTS integration using Google Cloud TTS.
"""
import logging
import asyncio
from typing import Optional, Dict, Any

from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Simplified Text-to-Speech integration using Google Cloud TTS.
    """
    
    def __init__(
        self,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration with Google Cloud TTS.
        
        Args:
            voice_name: Voice name to use
            voice_gender: Voice gender (MALE, FEMALE, NEUTRAL)
            language_code: Language code (defaults to en-US)
            enable_caching: Whether to enable TTS caching
        """
        # Set default voice name if not provided
        if not voice_name:
            voice_name = "en-US-Neural2-C"  # Default Neural2 voice
            
        self.voice_name = voice_name
        self.voice_gender = voice_gender or "NEUTRAL"
        self.language_code = language_code or "en-US"
        self.enable_caching = enable_caching
        self.tts_client = None
        self.initialized = False
        
        logger.info(f"TTSIntegration initialized with voice: {self.voice_name}")
    
    async def init(self) -> None:
        """Initialize the TTS client."""
        if self.initialized:
            return
            
        try:
            # Initialize Google Cloud TTS with telephony optimization
            self.tts_client = GoogleCloudTTS(
                voice_name=self.voice_name,
                voice_gender=self.voice_gender,
                language_code=self.language_code,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,          # For Twilio compatibility
                enable_caching=self.enable_caching,
                voice_type="NEURAL2"
            )
            
            self.initialized = True
            logger.info(f"Initialized TTS with Google Cloud TTS, voice: {self.voice_name}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            raise

    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Google Cloud TTS.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (mulaw format for Twilio)
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Use Google Cloud TTS (returns mulaw format)
            return await self.tts_client.synthesize(text)
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the TTS configuration."""
        return {
            "provider": "Google Cloud TTS",
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "sample_rate": 8000,
            "format": "mulaw",
            "initialized": self.initialized
        }