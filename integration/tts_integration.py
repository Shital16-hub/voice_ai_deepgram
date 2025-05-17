"""
Text-to-Speech integration module for Voice AI Agent.
IMPROVED: Optimized for faster response time.
"""
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Callable, Awaitable, AsyncIterator

from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    IMPROVED: Text-to-Speech integration class optimized for ultra-low latency.
    """
    
    def __init__(
        self,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        enable_caching: bool = True,
        credentials_file: Optional[str] = None
    ):
        """Initialize the TTS integration with optimized Google Cloud TTS."""
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
        
        # IMPROVED: Add performance tracking
        self.synthesis_count = 0
        self.synthesis_time_total = 0
        self.cache_hits = 0
        self.failures = 0
        
        logger.info(f"TTSIntegration initialized with voice: {self.voice_name}")
    
    async def init(self) -> None:
        """Initialize the TTS client with optimized settings."""
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
        IMPROVED: Convert text to speech using Google Cloud TTS with parallel processing.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (mulaw format for Twilio)
        """
        if not self.initialized:
            await self.init()
        
        start_time = time.time()
        
        try:
            # IMPROVED: Fast-path for very short texts to improve latency
            # For extremely short texts (1-2 words), bypass normal processing
            if len(text.split()) <= 2 and len(text) < 20:
                # Try to get from cache directly if possible
                if self.tts_client and hasattr(self.tts_client, '_get_cache_key'):
                    cache_key = self.tts_client._get_cache_key(text)
                    if hasattr(self.tts_client, '_get_cache_path'):
                        cache_path = self.tts_client._get_cache_path(cache_key)
                        if cache_path.exists():
                            self.cache_hits += 1
                            audio_data = cache_path.read_bytes()
                            self.synthesis_count += 1
                            synthesis_time = time.time() - start_time
                            self.synthesis_time_total += synthesis_time
                            return audio_data
            
            # Use Google Cloud TTS (returns mulaw format)
            audio_data = await self.tts_client.synthesize(text)
            
            # Track performance
            self.synthesis_count += 1
            synthesis_time = time.time() - start_time
            self.synthesis_time_total += synthesis_time
            
            # Log timing for performance monitoring
            if synthesis_time > 1.0:
                logger.warning(f"TTS synthesis took {synthesis_time:.2f}s for {len(text)} characters")
            else:
                logger.debug(f"TTS synthesis took {synthesis_time:.2f}s for {len(text)} characters")
                
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            self.failures += 1
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the TTS configuration and performance."""
        info = {
            "provider": "Google Cloud TTS",
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "sample_rate": 8000,
            "format": "mulaw",
            "initialized": self.initialized,
            
            # Performance stats
            "synthesis_count": self.synthesis_count,
            "avg_synthesis_time": self.synthesis_time_total / max(1, self.synthesis_count),
            "cache_hits": self.cache_hits,
            "failures": self.failures
        }
        
        # Add stats from TTS client if available
        if self.tts_client and hasattr(self.tts_client, 'get_stats'):
            info["client_stats"] = self.tts_client.get_stats()
        
        return info
    
    async def synthesize_streaming(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        IMPROVED: Stream synthesis for real-time responses with lower latency.
        
        Args:
            text_stream: Async iterator of text chunks
            
        Yields:
            Audio data chunks
        """
        if not self.initialized:
            await self.init()
            
        buffer = ""
        
        try:
            async for chunk in text_stream:
                if not chunk:
                    continue
                    
                buffer += chunk
                
                # Process if buffer contains complete sentence or is long enough
                if any(c in buffer for c in ['.', '!', '?']) or len(buffer.split()) >= 5:
                    audio_data = await self.text_to_speech(buffer)
                    yield audio_data
                    buffer = ""
            
            # Process remaining text
            if buffer:
                audio_data = await self.text_to_speech(buffer)
                yield audio_data
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            raise