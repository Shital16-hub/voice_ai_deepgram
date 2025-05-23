"""
Streaming functionality for text-to-speech processing with Google Cloud TTS.

This module provides utilities for handling streaming text input
and audio output for real-time voice applications.
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Awaitable
import queue
import threading

from .google_cloud_tts import GoogleCloudTTS
from .config import config
from .exceptions import TTSStreamingError

logger = logging.getLogger(__name__)

class TTSStreamer:
    """
    Manages real-time streaming of text to speech using Google Cloud TTS.
    
    Optimized for low-latency voice applications, this class manages the streaming
    of text from a knowledge base to speech output in real-time.
    """

    def __init__(
        self,
        tts_client: Optional[GoogleCloudTTS] = None,
        **tts_kwargs
    ):
        """
        Initialize the TTS streamer.
        
        Args:
            tts_client: Existing GoogleCloudTTS client, or one will be created
            **tts_kwargs: Arguments to pass to GoogleCloudTTS if creating a new client
        """
        # Set defaults for Twilio compatibility if not provided
        tts_kwargs.setdefault('container_format', 'mulaw')
        tts_kwargs.setdefault('sample_rate', 8000)
        tts_kwargs.setdefault('voice_type', 'NEURAL2')
        
        self.tts_client = tts_client or GoogleCloudTTS(**tts_kwargs)
        self.text_queue = asyncio.Queue()
        self.running = False
    
    async def _text_generator(self) -> AsyncGenerator[str, None]:
        """
        Generate text chunks from the internal queue.
        
        Yields:
            Text chunks as they become available
        """
        while self.running:
            try:
                # Get text from queue with timeout
                text = await asyncio.wait_for(
                    self.text_queue.get(), 
                    timeout=config.stream_timeout
                )
                yield text
                self.text_queue.task_done()
            except asyncio.TimeoutError:
                # No new text received within timeout, yield empty string to check if streaming should continue
                if self.running:
                    continue
                else:
                    break
            except Exception as e:
                logger.error(f"Error in text generator: {e}")
                if self.running:  # Only raise if we're still supposed to be running
                    raise TTSStreamingError(f"Text streaming error: {str(e)}")
    
    async def add_text(self, text: str) -> None:
        """
        Add text to the streaming queue.
        
        Args:
            text: Text chunk to add to the queue
        """
        if not self.running:
            raise TTSStreamingError("Cannot add text: streamer is not running")
        await self.text_queue.put(text)

    async def start_streaming(self) -> AsyncGenerator[bytes, None]:
        """
        Start the TTS streaming process.
        
        Returns:
            An async generator yielding audio chunks
        """
        self.running = True
        text_stream = self._text_generator()
        
        try:
            # Use the Google Cloud TTS client to synthesize streaming audio
            async for audio_chunk in self.tts_client.synthesize_streaming(text_stream):
                yield audio_chunk
        finally:
            self.running = False
    
    async def stop_streaming(self) -> None:
        """Stop the TTS streaming process."""
        self.running = False
        # Clear any pending items in the queue
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except asyncio.QueueEmpty:
                break


class RealTimeResponseHandler:
    """
    Handles real-time response text from the knowledge base to speech output using Google Cloud TTS.
    
    This class is designed to receive word-by-word output from the knowledge base
    and efficiently stream it to the TTS system.
    """
    
    def __init__(
        self,
        tts_client: Optional[GoogleCloudTTS] = None,
        tts_streamer: Optional[TTSStreamer] = None,
        **tts_kwargs
    ):
        """
        Initialize the real-time response handler.
        
        Args:
            tts_client: Existing GoogleCloudTTS client
            tts_streamer: Existing TTSStreamer or one will be created
            **tts_kwargs: Arguments to pass to TTSStreamer/GoogleCloudTTS if creating new ones
        """
        # Set defaults for Twilio compatibility
        tts_kwargs.setdefault('container_format', 'mulaw')
        tts_kwargs.setdefault('sample_rate', 8000)
        tts_kwargs.setdefault('voice_type', 'NEURAL2')
        
        # Initialize TTS client if provided
        self.tts_client = tts_client
        
        # Initialize TTS streamer, using the provided TTS client if available
        if tts_streamer:
            self.tts_streamer = tts_streamer
        elif self.tts_client:
            self.tts_streamer = TTSStreamer(tts_client=self.tts_client)
        else:
            self.tts_streamer = TTSStreamer(**tts_kwargs)
            
        self.buffer = ""
        self.buffer_lock = asyncio.Lock()
        self.audio_queue = asyncio.Queue()
        self.stream_task = None
        
    async def start(self) -> AsyncGenerator[bytes, None]:
        """
        Start the real-time response handler.
        
        Returns:
            An async generator yielding audio data
        """
        # Start the TTS streaming process
        self.stream_task = asyncio.create_task(self._stream_processor())
        
        # Return generator for audio data
        while True:
            try:
                audio_chunk = await self.audio_queue.get()
                if audio_chunk is None:  # End signal
                    break
                yield audio_chunk
                self.audio_queue.task_done()
            except Exception as e:
                logger.error(f"Error in audio generator: {str(e)}")
                await self.stop()
                raise
    
    async def _stream_processor(self) -> None:
        """Process the TTS stream and put audio chunks into the queue."""
        try:
            async for audio_chunk in self.tts_streamer.start_streaming():
                await self.audio_queue.put(audio_chunk)
        except Exception as e:
            logger.error(f"Error in TTS stream processor: {str(e)}")
        finally:
            # Signal the end of streaming
            await self.audio_queue.put(None)
    
    async def add_word(self, word: str) -> None:
        """
        Add a single word from the knowledge base to the TTS stream.
        
        This method is optimized for word-by-word output from the knowledge base.
        
        Args:
            word: Single word to add to the stream
        """
        async with self.buffer_lock:
            # Add space before word if needed
            if self.buffer and not self.buffer.endswith((' ', '\n', '\t')):
                self.buffer += ' '
            self.buffer += word
            
            # Process the buffer if we have a complete sentence or enough words
            if (any(c in self.buffer for c in ['.', '!', '?']) or 
                self.buffer.count(' ') >= 5):
                await self.tts_streamer.add_text(self.buffer)
                self.buffer = ""
    
    async def add_text(self, text: str) -> None:
        """
        Add text to the TTS stream.
        
        This handles larger chunks of text from the knowledge base.
        
        Args:
            text: Text to add to the stream
        """
        async with self.buffer_lock:
            # Add space before text if needed
            if self.buffer and not self.buffer.endswith((' ', '\n', '\t')) and not text.startswith((' ', '\n', '\t')):
                self.buffer += ' '
            self.buffer += text
            
            # Process the buffer
            await self.tts_streamer.add_text(self.buffer)
            self.buffer = ""
    
    async def flush(self) -> None:
        """Flush any remaining text in the buffer to the TTS stream."""
        async with self.buffer_lock:
            if self.buffer:
                await self.tts_streamer.add_text(self.buffer)
                self.buffer = ""
    
    async def stop(self) -> None:
        """Stop the real-time response handler."""
        # Flush any remaining text
        await self.flush()
        
        # Stop the TTS streamer
        await self.tts_streamer.stop_streaming()
        
        # Cancel the stream task if it exists
        if self.stream_task:
            try:
                self.stream_task.cancel()
                await asyncio.gather(self.stream_task, return_exceptions=True)
            except:
                pass