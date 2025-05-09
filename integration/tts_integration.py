"""
TTS integration with optimized streaming and caching for low-latency telephony applications.
"""
import logging
import asyncio
import time
import hashlib
import json
import os
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

# Import ElevenLabs TTS
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Optimized Text-to-Speech integration.
    """
    
    def __init__(
        self,
        voice_id: Optional[str] = None,
        enable_caching: bool = True,
        optimize_streaming_latency: int = 3  # Reduced from 4 for better quality/latency balance
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice_id: Voice ID to use for ElevenLabs TTS
            enable_caching: Whether to enable TTS caching
            optimize_streaming_latency: Optimization level for streaming (1-3)
        """
        self.voice_id = voice_id
        self.enable_caching = enable_caching
        self.optimize_streaming_latency = optimize_streaming_latency
        self.tts_client = None
        self.initialized = False
        
        # Enhanced caching
        self.cache_dir = Path('./cache/tts_cache')
        self.memory_cache = {}  # Fast in-memory cache
        self.cache_stats = {"hits": 0, "misses": 0, "entries": 0}
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 300  # Reduced from 500ms for better responsiveness
        
        # Prepare chunking for streaming
        self.max_chunk_size = 100  # Maximum text chunk size in characters
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Get API key from environment
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY in environment variables.")
                
            model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")  # Use the latest model
            
            # Create cache directory if needed
            if self.enable_caching:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"TTS cache directory: {self.cache_dir}")
            
            # Initialize the ElevenLabs TTS client with settings optimized for Twilio
            self.tts_client = ElevenLabsTTS(
                api_key=api_key,
                voice_id=self.voice_id, 
                enable_caching=self.enable_caching,
                container_format="mulaw",  # Use mulaw for Twilio compatibility
                sample_rate=8000,  # Set sample rate for telephony
                model_id=model_id,
                optimize_streaming_latency=self.optimize_streaming_latency
            )
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs, voice: {self.voice_id or 'default'}, model: {model_id}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text string.
        
        Args:
            text: Text to generate a key for
            
        Returns:
            Cache key string
        """
        # Simple hashing function for cache keys
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Include voice and model in the key
        if self.tts_client:
            voice_id = self.tts_client.voice_id
            model_id = self.tts_client.model_id
            return f"{voice_id}_{model_id}_{text_hash}"
        
        return text_hash
    
    def _split_text_at_sentence_boundaries(self, text: str) -> List[str]:
        """
        Split text at sentence boundaries for improved speech pacing.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence chunks
        """
        # Split on sentence endings, keeping the punctuation with the sentence
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences and combine very short ones
        result = []
        current = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If current sentence is very short, combine with next
            if len(current) > 0 and len(current) < 30:
                current += " " + sentence
            elif len(current) > 0:
                result.append(current)
                current = sentence
            else:
                current = sentence
        
        # Add the last sentence if there is one
        if current:
            result.append(current)
            
        return result
    
    async def _check_cache(self, text: str) -> Optional[bytes]:
        """
        Check if speech for this text is in cache.
        
        Args:
            text: Text to check in cache
            
        Returns:
            Cached audio or None if not found
        """
        if not self.enable_caching:
            return None
            
        cache_key = self._get_cache_key(text)
        
        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            logger.debug(f"Memory cache hit for '{text[:20]}...'")
            return self.memory_cache[cache_key]
            
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.mulaw"
        if cache_file.exists():
            try:
                audio_data = cache_file.read_bytes()
                
                # Add to memory cache for faster future access
                self.memory_cache[cache_key] = audio_data
                
                self.cache_stats["hits"] += 1
                logger.debug(f"Disk cache hit for '{text[:20]}...'")
                return audio_data
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    async def _save_to_cache(self, text: str, audio_data: bytes) -> None:
        """
        Save synthesized speech to cache.
        
        Args:
            text: Original text
            audio_data: Synthesized audio data
        """
        if not self.enable_caching or not audio_data:
            return
            
        try:
            cache_key = self._get_cache_key(text)
            
            # Save to memory cache
            self.memory_cache[cache_key] = audio_data
            self.cache_stats["entries"] = len(self.memory_cache)
            
            # Save to disk cache
            cache_file = self.cache_dir / f"{cache_key}.mulaw"
            cache_file.write_bytes(audio_data)
            
            logger.debug(f"Cached speech for '{text[:20]}...'")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with optimized latency and caching.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        # Skip processing for empty text
        if not text or text.strip() == "":
            logger.warning("Empty text provided to text_to_speech")
            return b''
        
        try:
            # Check cache first for fastest response
            cached_audio = await self._check_cache(text)
            if cached_audio:
                logger.info(f"Using cached speech for '{text[:30]}...'")
                return cached_audio
            
            # For short text (under 100 chars), use direct synthesis
            if len(text) < 100:
                # Generate speech with balanced quality/latency parameters
                params = {
                    "stability": 0.5,       # Balanced stability 
                    "clarity": 0.75,        # Good clarity
                    "style": 0.15,          # Light speaking style for natural sound
                }
                
                # Generate speech
                audio_data = await self.tts_client.synthesize(text, **params)
                
                # Cache the result for future use
                await self._save_to_cache(text, audio_data)
                
                # Add a pause after speech if configured
                if self.add_pause_after_speech:
                    # Generate silence based on pause_duration_ms
                    silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                    silence_data = b'\x00' * silence_size
                    
                    # Append silence to audio data
                    audio_data = audio_data + silence_data
                
                logger.info(f"Generated speech for short text: {len(audio_data)} bytes")
                return audio_data
            
            # For longer text, use sentence splitting for better pacing
            sentences = self._split_text_at_sentence_boundaries(text)
            
            # Generate speech for all sentences
            all_audio = []
            
            for i, sentence in enumerate(sentences):
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # Check cache for this sentence
                cached_sentence_audio = await self._check_cache(sentence)
                if cached_sentence_audio:
                    logger.debug(f"Using cached speech for sentence {i+1}/{len(sentences)}")
                    all_audio.append(cached_sentence_audio)
                    continue
                    
                # Generate speech for this sentence with quality parameters
                params = {
                    "stability": 0.5,
                    "clarity": 0.75,
                    "style": 0.15,
                }
                
                sentence_audio = await self.tts_client.synthesize(sentence, **params)
                
                # Save to cache
                await self._save_to_cache(sentence, sentence_audio)
                
                all_audio.append(sentence_audio)
                
                # Add shorter inter-sentence pauses for all but the last sentence
                if i < len(sentences) - 1:
                    # Add a shorter pause between sentences (100ms)
                    sentence_pause_ms = 100
                    inter_sentence_silence = b'\x00' * int(8000 * (sentence_pause_ms / 1000))
                    all_audio.append(inter_sentence_silence)
            
            # Combine all audio chunks
            combined_audio = b''.join(all_audio)
            
            # Add the final pause after the complete speech
            if self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                
                # Append silence to audio data
                combined_audio = combined_audio + silence_data
            
            logger.info(f"Generated speech for {len(sentences)} sentences: {len(combined_audio)} bytes")
            return combined_audio
            
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}", exc_info=True)
            raise
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = {
            "enabled": self.enable_caching,
            "entries": len(self.memory_cache) if self.enable_caching else 0,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_ratio": self.cache_stats["hits"] / max(1, (self.cache_stats["hits"] + self.cache_stats["misses"]))
        }
        
        return cache_stats
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up old cache entries.
        
        Args:
            max_age_hours: Maximum age in hours for cache entries
            
        Returns:
            Number of entries removed
        """
        if not self.enable_caching:
            return 0
            
        try:
            # Clear memory cache (simple approach)
            previous_count = len(self.memory_cache)
            self.memory_cache.clear()
            
            # Clear disk cache based on age
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            files_removed = 0
            for cache_file in self.cache_dir.glob("*.mulaw"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    files_removed += 1
            
            logger.info(f"Cleaned up {previous_count} memory cache entries and {files_removed} disk cache files")
            return previous_count + files_removed
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0