"""
TTS Integration module with ElevenLabs support.
"""
import logging
import time
import os
import sys
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

from telephony.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration with ElevenLabs support.
    
    Provides an abstraction layer for TTS functionality, handling initialization,
    single-text processing, and streaming capabilities.
    """
    
    def __init__(
        self,
        voice: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice: Voice ID to use for ElevenLabs
            enable_caching: Whether to enable TTS caching
        """
        self.voice = voice or "CwhRBWXzGAHq8TQ4Fs17"  # Default to Roger voice
        self.enable_caching = enable_caching
        self.tts_client = None
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # 500ms pause after speech
    
    async def init(self) -> None:
        """Initialize the TTS components with ElevenLabs."""
        if self.initialized:
            return
            
        try:
            # Add the directory containing elevenlabs_tts.py to the path
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            
            # Import the ElevenLabs TTS client
            from elevenlabs_tts import ElevenLabsTTS
            
            # Initialize the ElevenLabs TTS client
            self.tts_client = ElevenLabsTTS(
                voice_id=self.voice,
                model_id="eleven_flash_v2_5",  # Fast model for telephony
                enable_caching=self.enable_caching
            )
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs - Voice: {self.voice}, Model: eleven_flash_v2_5")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with proper telephony format conversion.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Get audio data from ElevenLabs TTS client (MP3 format)
            audio_data = await self.tts_client.synthesize(text)
            
            # Get audio info before conversion
            logger.debug(f"Raw ElevenLabs audio: {len(audio_data)} bytes")
            
            # Create audio processor
            audio_processor = AudioProcessor()
            
            # Convert to Twilio-compatible format
            try:
                processed_audio = audio_processor.prepare_audio_for_telephony(
                    audio_data,
                    format="mp3",
                    target_sample_rate=8000,  # Twilio uses 8kHz
                    target_channels=1         # Mono for telephony
                )
            except Exception as conv_error:
                logger.error(f"Error preparing audio for telephony: {conv_error}")
                # Return empty data if conversion fails
                return b''
            
            # Add a short pause after speech for better conversation flow
            if self.add_pause_after_speech and processed_audio:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz sample rate
                silence_data = b'\x7f' * silence_size  # μ-law silence
                
                # Append silence to audio data
                processed_audio = processed_audio + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after speech")
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            # Return empty data rather than raising
            return b''
    
    async def text_to_speech_streaming(
        self, 
        text_generator: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech conversion with telephony optimization.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Yields:
            Audio data chunks ready for Twilio
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Create audio processor
            audio_processor = AudioProcessor()
            
            # Track if we need to add the final pause
            needs_final_pause = False
            
            async for audio_chunk in self.tts_client.synthesize_streaming(text_generator):
                # Process each chunk for telephony compatibility
                try:
                    processed_chunk = audio_processor.prepare_audio_for_telephony(
                        audio_chunk,
                        format="mp3",
                        target_sample_rate=8000,
                        target_channels=1
                    )
                    
                    # Only the last chunk should get the pause
                    needs_final_pause = True
                    yield processed_chunk
                    
                except Exception as chunk_error:
                    logger.error(f"Error processing streaming chunk: {chunk_error}")
                    # Skip problematic chunks
                    continue
            
            # Add a pause at the end of the complete audio stream
            if needs_final_pause and self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz sample rate
                silence_data = b'\x7f' * silence_size  # μ-law silence
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
        Process text chunks in real-time and generate speech with telephony optimization.
        
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
        
        # Create audio processor
        audio_processor = AudioProcessor()
        
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
        Process SSML text and convert to speech with telephony optimization.
        
        Args:
            ssml: SSML-formatted text
            
        Returns:
            Audio data as bytes (Twilio-compatible)
        """
        if not self.initialized:
            await self.init()
        
        try:
            # ElevenLabs doesn't directly support SSML, but we'll pass it through
            audio_data = await self.tts_client.synthesize_with_ssml(ssml)
            
            # Create audio processor
            audio_processor = AudioProcessor()
            
            # Process for telephony
            processed_audio = audio_processor.prepare_audio_for_telephony(
                audio_data,
                format="mp3",
                target_sample_rate=8000,
                target_channels=1
            )
            
            # Add a pause at the end if needed
            if self.add_pause_after_speech:
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz sample rate
                silence_data = b'\x7f' * silence_size  # μ-law silence
                processed_audio = processed_audio + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after SSML speech")
                
            return processed_audio
        except Exception as e:
            logger.error(f"Error in SSML processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass