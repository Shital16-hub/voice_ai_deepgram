"""
Updated TTS integration using the fixed Google Cloud TTS.
"""
import logging
import asyncio
from typing import Optional, Dict, Any

from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Updated Text-to-Speech integration using the fixed Google Cloud TTS.
    """
    
    def __init__(
        self,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        enable_caching: bool = True,
        credentials_file: Optional[str] = None
    ):
        """
        Initialize the TTS integration with fixed Google Cloud TTS.
        
        Args:
            voice_name: Voice name to use (e.g., "en-US-Neural2-C")
            voice_gender: Voice gender (MALE, FEMALE, or None for Neural2 voices)
            language_code: Language code (defaults to en-US)
            enable_caching: Whether to enable TTS caching
            credentials_file: Path to Google Cloud credentials file
        """
        # Set default voice name if not provided
        if not voice_name:
            voice_name = "en-US-Neural2-C"  # Default Neural2 voice
        
        # Don't set gender for Neural2 voices
        if voice_name and "Neural2" in voice_name:
            voice_gender = None
            
        self.voice_name = voice_name
        self.voice_gender = voice_gender
        self.language_code = language_code or "en-US"
        self.enable_caching = enable_caching
        self.credentials_file = credentials_file
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
                credentials_file=self.credentials_file,
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
        info = {
            "provider": "Google Cloud TTS",
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "sample_rate": 8000,
            "format": "mulaw",
            "initialized": self.initialized
        }
        
        # Add stats from TTS client if available
        if self.tts_client and hasattr(self.tts_client, 'get_stats'):
            info["stats"] = self.tts_client.get_stats()
        
        return info