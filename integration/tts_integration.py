"""
Optimized TTS integration with MULAW output for Twilio.
"""
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List, Union

from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Optimized TTS integration with MULAW support for Twilio.
    """
    
    def __init__(
        self,
        voice_id: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize TTS integration with optimizations.
        
        Args:
            voice_id: ElevenLabs voice ID
            enable_caching: Whether to cache TTS results
        """
        self.voice_id = voice_id
        self.enable_caching = enable_caching
        self.tts_client = None
        self.initialized = False
        
        # Performance optimizations for telephony
        self.add_pause_after_speech = True
        self.pause_duration_ms = 300  # Reduced for faster response
        self.optimize_for_latency = True
        
        # Performance tracking
        self.total_syntheses = 0
        self.total_cache_hits = 0
        self.total_errors = 0
        self.synthesis_times = []
        self.max_time_history = 50
    
    async def init(self) -> None:
        """Initialize TTS with MULAW optimization."""
        if self.initialized:
            return
            
        try:
            # Get configuration from environment
            import os
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")
            
            # Initialize ElevenLabs with MULAW configuration
            self.tts_client = ElevenLabsTTS(
                api_key=api_key,
                voice_id=self.voice_id,
                enable_caching=self.enable_caching,
                container_format="mulaw",  # Direct MULAW output
                sample_rate=8000,  # 8kHz for Twilio
                model_id=model_id,
                optimize_streaming_latency=3  # High optimization for telephony
            )
            
            self.initialized = True
            logger.info(f"TTS initialized with MULAW support, voice: {self.voice_id}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    def _split_text_optimally(self, text: str) -> List[str]:
        """Split text for optimal speech generation."""
        if len(text) <= 100:
            return [text]
        
        # Split by sentences for better prosody
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Combine short sentences for efficiency
            if len(current_chunk) > 0 and len(current_chunk + " " + sentence) <= 150:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks if chunks else [text]
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with MULAW output.
        
        Args:
            text: Text to convert
            
        Returns:
            MULAW audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        start_time = time.time()
        
        try:
            # Track synthesis
            self.total_syntheses += 1
            
            # Generate speech with enhanced settings
            audio_data = await self._generate_speech_optimized(text)
            
            # Track timing
            synthesis_time = time.time() - start_time
            self.synthesis_times.append(synthesis_time)
            if len(self.synthesis_times) > self.max_time_history:
                self.synthesis_times.pop(0)
            
            return audio_data
                
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            self.total_errors += 1
            raise
    
    async def _generate_speech_optimized(self, text: str) -> bytes:
        """Generate speech with telephony optimizations."""
        # Split text for better processing
        chunks = self._split_text_optimally(text)
        
        # For short text, process directly
        if len(chunks) == 1 and len(text) < 100:
            # Single chunk processing with optimal settings
            return await self._synthesize_chunk_with_settings(text)
        
        # Process multiple chunks with optimized settings
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_audio = await self._synthesize_chunk_with_settings(chunk)
            
            # Ensure even byte length
            if len(chunk_audio) % 2 != 0:
                chunk_audio = chunk_audio + b'\x00'
            
            audio_chunks.append(chunk_audio)
            
            # Add inter-chunk pause for better prosody (shorter for telephony)
            if i < len(chunks) - 1 and self.add_pause_after_speech:
                pause_size = int(8000 * (150 / 1000))  # 150ms pause
                silence_data = b'\x00' * pause_size
                audio_chunks.append(silence_data)
        
        # Combine chunks
        combined_audio = b''.join(audio_chunks)
        
        # Add final pause
        if self.add_pause_after_speech:
            final_pause_size = int(8000 * (self.pause_duration_ms / 1000))
            final_silence = b'\x00' * final_pause_size
            combined_audio = combined_audio + final_silence
        
        return combined_audio
    
    async def _synthesize_chunk_with_settings(self, text: str) -> bytes:
        """Synthesize chunk with optimized settings for telephony."""
        # Check cache first
        if self.enable_caching and hasattr(self.tts_client, '_get_cache_path'):
            cache_path = self.tts_client._get_cache_path(text, self.voice_id, self.tts_client.model_id)
            if cache_path.exists():
                self.total_cache_hits += 1
                logger.debug(f"Cache hit for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Synthesize with optimized parameters
        audio_data = await self.tts_client.synthesize(
            text,
            voice_id=self.voice_id,
            # Optimized voice settings for telephony
            stability=0.4,  # Slightly lower for clarity
            clarity=0.7,    # High clarity for phone calls
            style=0.1,      # Minimal style for consistency
            optimize_streaming_latency=3  # High optimization
        )
        
        return audio_data
    
    async def text_to_speech_streaming(self, text: str) -> bytes:
        """
        Stream text to speech for real-time applications.
        
        Args:
            text: Text to convert
            
        Returns:
            MULAW audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # For telephony, we prioritize speed over streaming
            # Use regular synthesis with optimizations
            return await self.text_to_speech(text)
            
        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            self.total_errors += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS performance statistics."""
        cache_hit_rate = (self.total_cache_hits / max(1, self.total_syntheses)) * 100
        avg_synthesis_time = sum(self.synthesis_times) / max(1, len(self.synthesis_times))
        
        return {
            "total_syntheses": self.total_syntheses,
            "total_cache_hits": self.total_cache_hits,
            "total_errors": self.total_errors,
            "cache_hit_rate": cache_hit_rate,
            "average_synthesis_time": avg_synthesis_time,
            "recent_synthesis_times": self.synthesis_times[-10:] if self.synthesis_times else [],
            "initialized": self.initialized,
            "voice_id": self.voice_id,
            "enable_caching": self.enable_caching
        }
    
    async def close(self):
        """Close TTS client connections."""
        if self.tts_client and hasattr(self.tts_client, 'close'):
            await self.tts_client.close()
            logger.info("TTS client connections closed")