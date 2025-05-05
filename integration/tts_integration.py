"""
TTS Integration module for Voice AI Agent with ElevenLabs.
"""
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent.
    
    Provides an abstraction layer for TTS functionality, handling initialization,
    single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice_id: Voice ID to use for ElevenLabs TTS
            model_id: Model ID to use for ElevenLabs TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice_id = voice_id  # Will use default if None
        self.model_id = model_id  # Will use default if None
        self.enable_caching = enable_caching
        self.tts_client = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # 500ms pause after speech
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the ElevenLabs TTS client
            self.tts_client = ElevenLabsTTS(
                voice_id=self.voice_id, 
                model_id=self.model_id,
                enable_caching=self.enable_caching
            )
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs - Voice: {self.voice_id or 'default'}, Model: {self.model_id or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Get audio data from TTS client
            audio_data = await self.tts_client.synthesize(text)
            return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            raise
    
    async def text_to_speech_streaming(
        self, 
        text_generator: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech conversion.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Yields:
            Audio data chunks
        """
        if not self.initialized:
            await self.init()
        
        try:
            async for audio_chunk in self.tts_client.synthesize_streaming(text_generator):
                yield audio_chunk
                
        except Exception as e:
            logger.error(f"Error in streaming text to speech: {e}")
            raise
    
    async def process_realtime_text(
        self,
        text_chunks: AsyncIterator[str],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process text chunks in real-time and generate speech.
        
        Args:
            text_chunks: Async iterator of text chunks
            audio_callback: Callback to handle audio data
            
        Returns:
            Statistics about the processing
        """
        if not self.initialized:
            await self.init()
        
        # Start measuring time
        start_time = time.time()
        
        # Process each text chunk
        total_chunks = 0
        total_audio_bytes = 0
        
        try:
            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                
                # Process the text chunk
                audio_data = await self.text_to_speech(chunk)
                
                # Track statistics
                total_chunks += 1
                total_audio_bytes += len(audio_data)
                
                # Send audio to callback
                await audio_callback(audio_data)
                
                # Log progress periodically
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} text chunks")
        
        except Exception as e:
            logger.error(f"Error processing realtime text: {e}")
            return {
                "error": str(e),
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "elapsed_time": time.time() - start_time
            }
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        
        return {
            "total_chunks": total_chunks,
            "total_audio_bytes": total_audio_bytes,
            "elapsed_time": elapsed_time,
            "avg_chunk_size": total_audio_bytes / total_chunks if total_chunks > 0 else 0
        }
    
    async def process_ssml(self, ssml: str) -> bytes:
        """
        Process SSML text and convert to speech.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Note: ElevenLabs doesn't directly support SSML
            audio_data = await self.tts_client.synthesize_with_ssml(ssml)
            return audio_data
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # No specific cleanup needed for ElevenLabs TTS client
        pass