"""
ElevenLabs Text-to-Speech client for Voice AI Agent.

This module provides the ElevenLabs TTS implementation optimized for 
telephony applications.
"""
import os
import logging
import asyncio
import aiohttp
import hashlib
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union, List

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """
    Client for the ElevenLabs Text-to-Speech API with support for streaming.
    
    This class handles both batch and streaming TTS operations using ElevenLabs' API,
    optimized for low-latency telephony applications.
    """

    BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    # Telephony-optimized model and voice from the recommendations
    DEFAULT_MODEL = "eleven_flash_v2_5"  # Optimized for low-latency applications
    DEFAULT_VOICE = "CwhRBWXzGAHq8TQ4Fs17"  # Roger voice
    MAX_CHUNK_SIZE = 1500  # Maximum safe character chunk size for ElevenLabs API
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        output_format: Optional[str] = None,
        enable_caching: Optional[bool] = True
    ):
        """
        Initialize the ElevenLabs TTS client.
        
        Args:
            api_key: ElevenLabs API key (defaults to environment variable)
            model_id: TTS model ID to use (defaults to class default)
            voice_id: Voice ID for synthesis (defaults to class default)
            output_format: Audio format (defaults to mp3)
            enable_caching: Whether to cache results (defaults to True)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required. Set in env var ELEVENLABS_API_KEY or pass directly.")
        
        self.model_id = model_id or self.DEFAULT_MODEL
        self.voice_id = voice_id or self.DEFAULT_VOICE
        self.output_format = output_format or "mp3_44100_128"
        self.enable_caching = enable_caching
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path('./cache/tts_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for ElevenLabs API requests."""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def _get_voice_settings(self) -> Dict[str, Any]:
        """Get voice settings optimized for telephony."""
        return {
            "stability": 0.75,  # Higher stability for consistent telephony output
            "similarity_boost": 0.75,  # Balanced voice characteristic preservation
            "style": 0.0,  # Neutral style
            "use_speaker_boost": True  # Enhance clarity for telephony
        }
    
    def _get_params(self, **kwargs) -> Dict[str, Any]:
        """Get the parameters for a TTS request, with overrides from kwargs."""
        params = {
            "output_format": self.output_format,
            "optimize_streaming_latency": 3  # Range 0-4, higher values optimize latency at some quality cost
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
                
        return params
    
    def _get_cache_path(self, text: str, voice_id: str, model_id: str, params: Dict[str, Any]) -> Path:
        """
        Generate a cache file path based on text and parameters.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            model_id: Model ID to use
            params: TTS parameters
            
        Returns:
            Path to the cache file
        """
        # Create a unique hash based on text and params
        cache_key = hashlib.md5(
            f"{text}:{voice_id}:{model_id}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()
        
        # Determine file extension based on format
        if "pcm" in self.output_format or "linear" in self.output_format:
            ext = "wav"
        elif "mp3" in self.output_format:
            ext = "mp3"
        else:
            ext = "audio"
            
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
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech in a single request.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID (defaults to configured ID)
            model_id: Model ID (defaults to configured ID)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        if not text:
            logger.warning("Empty text provided to synthesize")
            return b''
        
        # Use configured IDs if not specified
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
            
        # Check if text exceeds length limit
        if len(text) > 2000:
            logger.info(f"Text length ({len(text)}) exceeds limit. Splitting into chunks.")
            chunks = self._split_long_text(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
                    chunk_audio = await self._synthesize_chunk(chunk, voice_id, model_id, **kwargs)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with next chunk instead of failing completely
            
            # Combine audio chunks
            if not audio_chunks:
                raise RuntimeError("Failed to synthesize any chunks of the text")
                
            return b''.join(audio_chunks)
        else:
            # Standard case - text is within limits
            return await self._synthesize_chunk(text, voice_id, model_id, **kwargs)
    
    async def _synthesize_chunk(
        self, 
        text: str,
        voice_id: str,
        model_id: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize a single chunk of text.
        
        Args:
            text: Text chunk to synthesize
            voice_id: Voice ID to use
            model_id: Model ID to use
            **kwargs: Additional parameters
            
        Returns:
            Audio data as bytes
        """
        params = self._get_params(**kwargs)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(text, voice_id, model_id, params)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Prepare request
        url = f"{self.BASE_URL}/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": self._get_voice_settings()
        }
        
        # Make API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    params=params
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise RuntimeError(f"ElevenLabs API error: {response.status} - {error_msg}")
                    
                    audio_data = await response.read()
            
            # Cache result if enabled
            if self.enable_caching and audio_data:
                cache_path = self._get_cache_path(text, voice_id, model_id, params)
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error when connecting to ElevenLabs: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error during TTS synthesis: {str(e)}")
    
    async def synthesize_streaming(
        self, 
        text_stream: AsyncGenerator[str, None],
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech synthesis for real-time applications.
        
        Takes a streaming text input and returns streaming audio output,
        optimized for low-latency voice applications.
        
        Args:
            text_stream: Async generator producing text chunks
            voice_id: Voice ID (defaults to configured ID)
            model_id: Model ID (defaults to configured ID)
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            Audio data chunks as they are generated
        """
        # Use configured IDs if not specified
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        
        buffer = ""
        max_chunk_size = min(250, self.MAX_CHUNK_SIZE)  # Smaller chunks for faster streaming
        
        try:
            async for text_chunk in text_stream:
                if not text_chunk:
                    continue
                    
                # Add to buffer
                buffer += text_chunk
                
                # Process buffer if it's large enough or contains sentence-ending punctuation
                if len(buffer) >= max_chunk_size or any(c in buffer for c in ['.', '!', '?', '\n']):
                    # Process the buffered text
                    audio_data = await self.synthesize(buffer, voice_id, model_id, **kwargs)
                    yield audio_data
                    buffer = ""
            
            # Process any remaining text in the buffer
            if buffer:
                audio_data = await self.synthesize(buffer, voice_id, model_id, **kwargs)
                yield audio_data
                
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            raise
    
    async def synthesize_with_ssml(
        self, 
        ssml: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize speech using SSML markup for advanced control.
        
        Args:
            ssml: SSML-formatted text
            voice_id: Voice ID (defaults to configured ID)
            model_id: Model ID (defaults to configured ID)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        # Note: ElevenLabs doesn't directly support SSML. 
        # This function strips SSML tags and sends the plain text.
        # In a full implementation, you might want to parse the SSML and 
        # translate it to ElevenLabs voice settings.
        
        # Use configured IDs if not specified
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        
        # Simple SSML tag removal (replace with proper SSML parser in production)
        import re
        plain_text = re.sub(r'<[^>]+>', '', ssml)
        
        logger.warning("ElevenLabs doesn't directly support SSML. Converting to plain text.")
        
        return await self.synthesize(plain_text, voice_id, model_id, **kwargs)