"""
Optimized ElevenLabs TTS client with MULAW support for Twilio.
"""
import os
import logging
import asyncio
import aiohttp
import hashlib
import json
import audioop
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union, List

from .config import config
from .exceptions import TTSError

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """
    Optimized ElevenLabs TTS client with direct MULAW output for Twilio.
    """

    BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    MAX_CHUNK_SIZE = 1200  # Reduced for faster processing
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        container_format: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        optimize_streaming_latency: int = 3
    ):
        """
        Initialize the ElevenLabs TTS client with MULAW optimization.
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID for synthesis  
            model_id: Model ID to use
            sample_rate: Audio sample rate (8000 for Twilio)
            container_format: Audio format (mulaw for Twilio)
            enable_caching: Whether to cache results
            optimize_streaming_latency: Latency optimization level (1-3)
        """
        self.api_key = api_key or config.elevenlabs_api_key
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        self.voice_id = voice_id or config.voice_id
        if not self.voice_id:
            raise ValueError("ElevenLabs voice ID is required")
            
        self.model_id = model_id or config.model_id
        self.sample_rate = sample_rate or config.sample_rate
        self.container_format = container_format or config.container_format
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        self.optimize_streaming_latency = max(1, min(3, optimize_streaming_latency))
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"TTS caching enabled. Cache directory: {self.cache_dir}")
        
        # Connection pool for better performance
        self._connector = None
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            if self._connector:
                self._connector.close()
            
            # Create connector with connection pooling
            self._connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(
                    total=15,  # Total timeout
                    connect=3,  # Connection timeout
                    sock_read=5  # Socket read timeout
                )
            )
        
        return self._session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/*"
        }
    
    def _get_output_format(self) -> str:
        """Get output format for Twilio optimization."""
        # For Twilio, we need µ-law at 8kHz
        if self.container_format == "mulaw" or self.container_format == "ulaw":
            return "ulaw_8000"
        
        # For MP3 with quality optimization
        if self.container_format == "mp3":
            return f"mp3_{self.sample_rate}_128"
            
        # For PCM (will convert to MULAW later)
        return f"pcm_{self.sample_rate}"
    
    def _get_cache_path(self, text: str, voice_id: str, model_id: str) -> Path:
        """Generate cache file path."""
        params = {
            "voice_id": voice_id,
            "model_id": model_id,
            "output_format": self._get_output_format()
        }
        cache_key = hashlib.md5(f"{text}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        # Determine file extension
        ext = "ulaw" if self.container_format in ["mulaw", "ulaw"] else "wav"
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _split_text_for_optimization(self, text: str) -> List[str]:
        """Split text into optimized chunks."""
        if len(text) <= self.MAX_CHUNK_SIZE:
            return [text]
        
        chunks = []
        
        # Split by sentences first
        sentences = []
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > self.MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If single sentence is too long, split by commas
                if len(sentence) > self.MAX_CHUNK_SIZE:
                    parts = sentence.split(', ')
                    temp_chunk = ""
                    for part in parts:
                        if len(temp_chunk) + len(part) > self.MAX_CHUNK_SIZE:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = part
                        else:
                            temp_chunk += ", " + part if temp_chunk else part
                    current_chunk = temp_chunk
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.debug(f"Split text of {len(text)} chars into {len(chunks)} chunks")
        return chunks
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize text to speech with MULAW optimization.
        
        Args:
            text: Text to synthesize
            **kwargs: Additional parameters
            
        Returns:
            Audio data as bytes (MULAW format)
        """
        if not text:
            logger.warning("Empty text provided")
            return b''
        
        # Use provided parameters or defaults
        voice_id = kwargs.get('voice_id', self.voice_id)
        model_id = kwargs.get('model_id', self.model_id)
        
        # Check cache first
        if self.enable_caching:
            cache_path = self._get_cache_path(text, voice_id, model_id)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Handle long text
        if len(text) > 2000:
            logger.info(f"Text too long ({len(text)} chars), splitting into chunks")
            chunks = self._split_text_for_optimization(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_audio = await self._synthesize_chunk(chunk, voice_id, model_id)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with other chunks
            
            if not audio_chunks:
                raise TTSError("Failed to synthesize any chunks")
            
            # Combine audio chunks
            return self._combine_audio_chunks(audio_chunks)
        else:
            return await self._synthesize_chunk(text, voice_id, model_id)
    
    async def _synthesize_chunk(self, text: str, voice_id: str, model_id: str) -> bytes:
        """Synthesize a single text chunk."""
        # Prepare optimized request data
        request_data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.4,  # Optimized for clarity
                "similarity_boost": 0.6,  # Better similarity
                "style": 0.1  # Minimal style for consistency
            }
        }
        
        # Construct URL with optimizations
        output_format = self._get_output_format()
        url = f"{self.BASE_URL}/{voice_id}"
        
        # Add parameters
        params = {
            "output_format": output_format,
            "optimize_streaming_latency": self.optimize_streaming_latency
        }
        
        try:
            session = await self._get_session()
            
            async with session.post(
                url,
                headers=self._get_headers(),
                json=request_data,
                params=params
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise TTSError(f"ElevenLabs API error: {response.status} - {error_msg}")
                
                audio_data = await response.read()
            
            # Convert to MULAW if needed
            if self.container_format in ["mulaw", "ulaw"] and not output_format.startswith("ulaw"):
                audio_data = await self._convert_to_mulaw(audio_data)
            
            # Cache result
            if self.enable_caching and audio_data:
                cache_path = self._get_cache_path(text, voice_id, model_id)
                cache_path.write_bytes(audio_data)
            
            return audio_data
            
        except aiohttp.ClientError as e:
            raise TTSError(f"Network error: {str(e)}")
        except Exception as e:
            raise TTSError(f"TTS error: {str(e)}")
    
    async def _convert_to_mulaw(self, audio_data: bytes) -> bytes:
        """Convert audio to MULAW format."""
        try:
            # If it's already MULAW, return as-is
            if self._is_mulaw_format(audio_data):
                return audio_data
            
            # Convert PCM to MULAW
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                # Extract PCM from WAV
                import wave
                import io
                
                with io.BytesIO(audio_data) as f:
                    with wave.open(f, 'rb') as wav:
                        pcm_data = wav.readframes(wav.getnframes())
                        sample_rate = wav.getframerate()
                        channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        
                        # Resample to 8kHz if needed
                        if sample_rate != 8000:
                            pcm_data, _ = audioop.ratecv(
                                pcm_data, sample_width, channels,
                                sample_rate, 8000, None
                            )
                        
                        # Convert to mono if needed
                        if channels > 1:
                            pcm_data = audioop.tomono(pcm_data, sample_width, 1, 0)
                        
                        # Convert to 16-bit if needed
                        if sample_width != 2:
                            pcm_data = audioop.lin2lin(pcm_data, sample_width, 2)
                        
                        # Convert to MULAW
                        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
                        return mulaw_data
            else:
                # For MP3 or other formats, use subprocess if available
                # This is a simplified conversion - production might need ffmpeg
                try:
                    # Assume it's raw PCM and convert directly
                    mulaw_data = audioop.lin2ulaw(audio_data, 2)
                    return mulaw_data
                except:
                    logger.warning("Cannot convert to MULAW, returning original")
                    return audio_data
            
        except Exception as e:
            logger.error(f"Error converting to MULAW: {e}")
            return audio_data
    
    def _is_mulaw_format(self, audio_data: bytes) -> bool:
        """Check if audio is already in MULAW format."""
        # Simple heuristic - MULAW uses full 8-bit range
        if len(audio_data) < 100:
            return False
        
        sample = audio_data[:100]
        
        # Check for uniform distribution
        ranges = [0] * 8
        for byte_val in sample:
            ranges[byte_val // 32] += 1
        
        # MULAW should have relatively uniform distribution
        non_zero_ranges = sum(1 for r in ranges if r > 0)
        return non_zero_ranges >= 6
    
    def _combine_audio_chunks(self, chunks: List[bytes]) -> bytes:
        """Combine multiple audio chunks."""
        if not chunks:
            return b''
        
        # For MULAW, simple concatenation works
        if self.container_format in ["mulaw", "ulaw"]:
            return b''.join(chunks)
        
        # For other formats, return the first chunk for simplicity
        # In production, you'd properly combine the audio files
        return chunks[0]
    
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream speech synthesis for real-time applications.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio data chunks as they are generated
        """
        voice_id = self.voice_id
        model_id = self.model_id
        output_format = self._get_output_format()
        
        # Optimized request data
        request_data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.6,
                "style": 0.1
            }
        }
        
        # Streaming endpoint
        url = f"{self.BASE_URL}/{voice_id}/stream"
        params = {
            "output_format": output_format,
            "optimize_streaming_latency": self.optimize_streaming_latency
        }
        
        try:
            session = await self._get_session()
            
            async with session.post(
                url,
                headers=self._get_headers(),
                json=request_data,
                params=params
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise TTSError(f"ElevenLabs streaming error: {response.status} - {error_msg}")
                
                # Stream chunks with optimized size
                all_chunks = []
                async for chunk in response.content.iter_chunked(2048):  # Larger chunks
                    all_chunks.append(chunk)
                    yield chunk
                
                # Convert to MULAW if needed
                if self.container_format in ["mulaw", "ulaw"] and not output_format.startswith("ulaw"):
                    combined_audio = b''.join(all_chunks)
                    mulaw_audio = await self._convert_to_mulaw(combined_audio)
                    
                    # Yield the converted audio in chunks
                    chunk_size = 2048
                    for i in range(0, len(mulaw_audio), chunk_size):
                        yield mulaw_audio[i:i+chunk_size]
                
        except aiohttp.ClientError as e:
            raise TTSError(f"Streaming network error: {str(e)}")
        except Exception as e:
            raise TTSError(f"Streaming TTS error: {str(e)}")
    
    # Alias for compatibility
    text_to_speech = synthesize
    
    async def close(self):
        """Close the session and connector."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector:
            await self._connector.close()