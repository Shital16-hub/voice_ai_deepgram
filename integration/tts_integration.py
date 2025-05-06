"""
Text-to-Speech integration module for Voice AI Agent.

This module provides classes and functions for integrating text-to-speech
capabilities with the Voice AI Agent system using ElevenLabs.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union, AsyncIterator

from text_to_speech import ElevenLabsTTS, RealTimeResponseHandler, AudioProcessor

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
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice_id: Voice ID to use for ElevenLabs TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice_id = voice_id
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # 500ms pause after speech
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the ElevenLabs TTS client with settings optimized for Twilio
            self.tts_client = ElevenLabsTTS(
                voice_id=self.voice_id, 
                enable_caching=self.enable_caching,
                container_format="mulaw",  # Use mulaw for Twilio compatibility
                sample_rate=8000,  # Set sample rate for telephony
                model_id=os.getenv("TTS_MODEL_ID", "eleven_turbo_v2"),  # Use the latest model
                optimize_streaming_latency=4  # Maximum optimization for real-time performance
            )
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs, voice: {self.voice_id or 'default'}, model: {self.tts_client.model_id}, format: mulaw for Twilio")
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
            
            # Ensure the audio data has an even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
                logger.debug("Padded audio data to make even length")
            
            # Add a short pause after speech for better conversation flow
            if self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                
                # Append silence to audio data
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after speech")
            
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
            text_generator: Async generator producing text chunks
            
        Yields:
            Audio data chunks as they are generated
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Track if we need to add the final pause
            needs_final_pause = False
            
            async for text_chunk in text_generator:
                if not text_chunk:
                    continue
                    
                # Convert text chunk to speech
                audio_data = await self.tts_client.synthesize(text_chunk)
                
                # Ensure each chunk has an even number of bytes
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data + b'\x00'
                
                # Only the last chunk should get the pause
                needs_final_pause = True
                yield audio_data
            
            # Add a pause at the end of the complete audio stream
            if needs_final_pause and self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                yield silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause at end of streaming audio")
                
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
        
        # Reset the TTS handler for this new session
        if self.tts_handler:
            await self.tts_handler.stop()
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
        
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
        
        Note: ElevenLabs does not support SSML directly, so we strip SSML tags.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # ElevenLabs doesn't support SSML directly, so we need to strip SSML tags
            # This is a simple approach and might not handle all SSML features
            import re
            text = re.sub(r'<[^>]*>', '', ssml)
            
            # Generate speech from the cleaned text
            audio_data = await self.tts_client.synthesize(text)
            
            # Ensure even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            # Add a pause at the end if needed
            if self.add_pause_after_speech:
                silence_size = int(8000 * (self.pause_duration_ms / 1000))
                silence_data = b'\x00' * silence_size
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after speech")
                
            return audio_data
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_handler:
            try:
                await self.tts_handler.stop()
            except Exception as e:
                logger.error(f"Error during TTS cleanup: {e}")