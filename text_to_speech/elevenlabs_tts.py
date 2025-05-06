# text_to_speech/elevenlabs_tts.py

"""
ElevenLabs Text-to-Speech client for the Voice AI Agent.
"""
import os
import logging
import asyncio
import aiohttp
import hashlib
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union, List

from .config import config
from .exceptions import TTSError

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """
    Client for the ElevenLabs Text-to-Speech API with support for streaming.
    
    This class handles both batch and streaming TTS operations using ElevenLabs' API,
    optimized for low-latency voice AI applications.
    """

    BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    MAX_CHUNK_SIZE = 1500  # Maximum safe character chunk size for ElevenLabs API
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        container_format: Optional[str] = None,
        enable_caching: Optional[bool] = None
    ):
        """
        Initialize the ElevenLabs TTS client.
        
        Args:
            api_key: ElevenLabs API key (defaults to environment variable)
            voice_id: Voice ID for synthesis (defaults to config)
            model_id: Model ID to use (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            container_format: Audio format (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
        """
        self.api_key = api_key or config.elevenlabs_api_key
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required. Set it in .env file or pass directly.")
        
        self.voice_id = voice_id or config.voice_id
        if not self.voice_id:
            raise ValueError("ElevenLabs voice ID is required. Set it in config or pass directly.")
            
        self.model_id = model_id or config.model_id
        self.sample_rate = sample_rate or config.sample_rate
        self.container_format = container_format or config.container_format
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for ElevenLabs API requests."""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/*"
        }
    
    def _get_output_format(self) -> str:
        """Get the output format parameter for ElevenLabs API."""
        # For Twilio, we need µ-law encoded audio at 8kHz
        if self.container_format == "mulaw" or self.container_format == "ulaw":
            return "ulaw_8000"
        
        # For MP3
        if self.container_format == "mp3":
            return f"mp3_{self.sample_rate}_128"
            
        # For WAV (PCM)
        return f"pcm_{self.sample_rate}"
    
    def _get_cache_path(self, text: str, voice_id: str, model_id: str) -> Path:
        """
        Generate a cache file path based on text and parameters.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID
            model_id: Model ID
            
        Returns:
            Path to the cache file
        """
        # Create a unique hash based on text and params
        params = {
            "voice_id": voice_id,
            "model_id": model_id,
            "output_format": self._get_output_format()
        }
        cache_key = hashlib.md5(f"{text}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        # Determine file extension based on format
        ext = "wav"
        if self.container_format == "mp3":
            ext = "mp3"
        elif self.container_format in ["mulaw", "ulaw"]:
            ext = "ulaw"
            
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text into smaller chunks that won't exceed API limits.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.MAX_CHUNK_SIZE:
            return [text]
            
        chunks = []
        sentences = []
        
        # First try to split by sentences (periods, exclamation marks, question marks)
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > self.MAX_CHUNK_SIZE:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If a single sentence is longer than the limit, split it by words
                if len(sentence) > self.MAX_CHUNK_SIZE:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > self.MAX_CHUNK_SIZE:
                            chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += " " + word
                    
                    if word_chunk.strip():
                        current_chunk = word_chunk.strip()
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Split text of length {len(text)} into {len(chunks)} chunks")
        return chunks
    
    async def synthesize(
        self, 
        text: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech in a single request.
        
        Args:
            text: Text to synthesize
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        if not text:
            logger.warning("Empty text provided to synthesize")
            return b''
            
        # Use provided voice_id or default
        voice_id = kwargs.get('voice_id', self.voice_id)
        
        # Use provided model_id or default
        model_id = kwargs.get('model_id', self.model_id)
        
        # Check if text exceeds ElevenLabs' limit
        if len(text) > 2000:
            logger.info(f"Text length ({len(text)}) exceeds ElevenLabs' 2000 character limit. Splitting into chunks.")
            chunks = self._split_long_text(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
                    chunk_audio = await self._synthesize_chunk(chunk, voice_id, model_id)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with next chunk instead of failing completely
            
            # Combine audio chunks
            if not audio_chunks:
                raise TTSError("Failed to synthesize any chunks of the text")
                
            # For simplicity, just concatenate audio data
            # In a production implementation, you'd properly combine the audio files
            return b''.join(audio_chunks)
        else:
            # Standard case - text is within limits
            return await self._synthesize_chunk(text, voice_id, model_id)
    
    async def _synthesize_chunk(
        self, 
        text: str,
        voice_id: str,
        model_id: str
    ) -> bytes:
        """
        Synthesize a single chunk of text.
        
        Args:
            text: Text chunk to synthesize
            voice_id: Voice ID to use
            model_id: Model ID to use
            
        Returns:
            Audio data as bytes
        """
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(text, voice_id, model_id)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Prepare request data
        request_data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        # Add output_format parameter for Twilio compatibility if needed
        output_format = self._get_output_format()
        url = f"{self.BASE_URL}/{voice_id}"
        if self.container_format:
            url += f"?output_format={output_format}"
            
            # Add streaming optimization for lower latency
            url += "&optimize_streaming_latency=3"
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=request_data
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise TTSError(f"ElevenLabs API error: {response.status} - {error_msg}")
                    
                    audio_data = await response.read()
            
            # Cache result if enabled
            if self.enable_caching and audio_data:
                cache_path = self._get_cache_path(text, voice_id, model_id)
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except aiohttp.ClientError as e:
            raise TTSError(f"Network error when connecting to ElevenLabs: {str(e)}")
        except Exception as e:
            raise TTSError(f"Error during TTS synthesis: {str(e)}")
    
    async def synthesize_with_streaming(
        self, 
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech synthesis for real-time applications.
        
        Takes a streaming text input and returns streaming audio output,
        optimized for low-latency voice applications.
        
        Args:
            text_stream: Async generator producing text chunks
            
        Yields:
            Audio data chunks as they are generated
        """
        buffer = ""
        max_chunk_size = min(config.max_text_chunk_size, self.MAX_CHUNK_SIZE)
        
        try:
            async for text_chunk in text_stream:
                if not text_chunk:
                    continue
                    
                # Add to buffer
                buffer += text_chunk
                
                # Process buffer if it's large enough or contains sentence-ending punctuation
                if len(buffer) >= max_chunk_size or any(c in buffer for c in ['.', '!', '?', '\n']):
                    # Process the buffered text
                    audio_data = await self.synthesize(buffer)
                    yield audio_data
                    buffer = ""
            
            # Process any remaining text in the buffer
            if buffer:
                audio_data = await self.synthesize(buffer)
                yield audio_data
                
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            raise TTSError(f"Streaming TTS error: {str(e)}")
    
    async def synthesize_streaming(
        self, 
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream speech synthesis for a complete text.
        
        This method uses the streaming endpoint to receive chunks as they are generated.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio data chunks as they are generated
        """
        voice_id = self.voice_id
        model_id = self.model_id
        output_format = self._get_output_format()
        
        # Prepare request data
        request_data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        # Add streaming optimization for lower latency
        url = f"{self.BASE_URL}/{voice_id}/stream?output_format={output_format}&optimize_streaming_latency=3"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=request_data
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise TTSError(f"ElevenLabs API error: {response.status} - {error_msg}")
                    
                    # Stream the response chunks
                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk
                        
        except aiohttp.ClientError as e:
            raise TTSError(f"Network error when connecting to ElevenLabs: {str(e)}")
        except Exception as e:
            raise TTSError(f"Error during streaming TTS synthesis: {str(e)}")
            
    text_to_speech = synthesize  # Alias for compatibility with existing code