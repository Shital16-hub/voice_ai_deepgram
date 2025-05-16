"""
Google Cloud Speech-to-Text streaming implementation for real-time transcription.
"""
import os
import logging
import asyncio
import queue
import threading
import re
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass

from google.cloud import speech
import numpy as np

from ..config import config
from .models import TranscriptionConfig
from .exceptions import STTError, STTStreamingError

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None

class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text streaming client.
    
    This class provides real-time streaming speech recognition
    optimized for telephony applications.
    """
    
    def __init__(
        self, 
        credentials_file: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        channels: int = 1,
        interim_results: Optional[bool] = None
    ):
        """
        Initialize the Google Cloud streaming STT client.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            model_name: STT model to use (defaults to config)
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results (default True)
        """
        self.credentials_file = credentials_file or config.credentials_file
        
        # Set environment variable for credentials if provided
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        
        self.model_name = model_name or config.model_name
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results if interim_results is not None else config.interim_results
        
        # Create Google Cloud Speech client
        try:
            self.client = speech.SpeechClient()
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Speech client: {e}")
            raise STTStreamingError(f"Failed to initialize Google Cloud Speech client: {e}")
        
        # State management
        self.is_streaming = False
        self.streaming_thread = None
        self.audio_queue = None
        self.streaming_responses = None
        self.stop_stream = None
        self.stream_future = None
        self.result_queue = None
        self.utterance_id = 0
        
        # Define the patterns for non-speech annotations
        self.non_speech_pattern = re.compile('|'.join([
            r'\[.*?\]',           # Anything in square brackets
            r'\(.*?\)',           # Anything in parentheses
            r'\<.*?\>',           # Anything in angle brackets
            r'music playing',     # Common transcription
            r'background noise',  # Common transcription
            r'static',            # Common transcription
            r'\b(um|uh|hmm|mmm)\b',  # Common filler words
        ]))
    
    def _get_streaming_config(
        self, 
        config_obj: Optional[TranscriptionConfig] = None, 
        **kwargs
    ) -> speech.StreamingRecognitionConfig:
        """
        Get the streaming recognition configuration.
        
        Args:
            config_obj: Optional configuration object
            **kwargs: Additional parameters to override defaults
            
        Returns:
            StreamingRecognitionConfig for Google Cloud Speech-to-Text
        """
        # Determine encoding type
        if self.encoding.upper() == "LINEAR16":
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif self.encoding.upper() == "MULAW":
            encoding = speech.RecognitionConfig.AudioEncoding.MULAW
        else:
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Create base recognition config
        recognition_config = {
            "encoding": encoding,
            "sample_rate_hertz": self.sample_rate,
            "language_code": self.language,
            "model": self.model_name,
            "enable_automatic_punctuation": config.enable_automatic_punctuation,
            "enable_word_time_offsets": config.enable_word_time_offsets,
            "profanity_filter": config.profanity_filter,
            "use_enhanced": config.use_enhanced_model,
        }
        
        # Add speech contexts (keywords) for better recognition
        if config.speech_contexts:
            recognition_config["speech_contexts"] = [
                {"phrases": config.speech_contexts, "boost": 15.0}
            ]
        
        # Add telephony-specific optimizations
        if config.use_enhanced_telephony:
            recognition_config["use_enhanced"] = True
            recognition_config["model"] = "phone_call"
        
        # Override with any config object settings
        if config_obj:
            config_dict = config_obj.dict(exclude_none=True, exclude_unset=True)
            for key, value in config_dict.items():
                recognition_config[key] = value
        
        # Override with any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                recognition_config[key] = value
        
        # Create RecognitionConfig and StreamingRecognitionConfig
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(**recognition_config),
            interim_results=self.interim_results
        )
        
        return streaming_config
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Clean up transcription by removing non-speech annotations.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        logger.debug("Starting new Google Cloud Speech streaming session")
        
        # Initialize state
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
        self.stop_stream = threading.Event()
        self.is_streaming = True
        self.utterance_id = 0
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(
            target=self._run_streaming,
            daemon=True
        )
        self.streaming_thread.start()
        
        # Create future to track when streaming is complete
        self.stream_future = asyncio.get_event_loop().create_future()
    
    def _run_streaming(self):
        """Run the streaming recognition in a background thread."""
        try:
            # Create a streaming config
            streaming_config = self._get_streaming_config()
            
            # Define a generator for the audio stream
            def audio_generator():
                while not self.stop_stream.is_set():
                    try:
                        # Get the next chunk with a timeout
                        chunk = self.audio_queue.get(block=True, timeout=1.0)
                        
                        # Create a streaming request with the audio content
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                        
                        # Mark task as done
                        self.audio_queue.task_done()
                    except queue.Empty:
                        continue
                
                logger.debug("Audio generator completed")
            
            # Create streaming recognize request
            request_generator = self._generate_streaming_request(streaming_config, audio_generator())
            
            # Start streaming recognize and process responses
            self.streaming_responses = self.client.streaming_recognize(request_generator)
            
            current_chunk_id = 0
            
            # Process streaming responses
            for response in self.streaming_responses:
                if self.stop_stream.is_set():
                    break
                
                # Handle streaming response
                for result in response.results:
                    current_chunk_id += 1
                    
                    # Get the first alternative (most likely)
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create a result object
                        streaming_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence if result.is_final else 0.0,
                            start_time=0.0,  # Not provided in streaming responses
                            end_time=0.0,    # Not provided in streaming responses
                            chunk_id=current_chunk_id,
                            words=None       # Words not provided in streaming responses
                        )
                        
                        # Add to result queue
                        asyncio.run_coroutine_threadsafe(
                            self.result_queue.put(streaming_result),
                            asyncio.get_event_loop()
                        )
        
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")
            # Put an error in the result queue
            error_result = StreamingTranscriptionResult(
                text=f"Error: {str(e)}",
                is_final=True,
                confidence=0.0,
                start_time=0.0,
                end_time=0.0,
                chunk_id=0,
                words=None
            )
            asyncio.run_coroutine_threadsafe(
                self.result_queue.put(error_result),
                asyncio.get_event_loop()
            )
        finally:
            # Signal that streaming is complete
            if self.stream_future and not self.stream_future.done():
                asyncio.run_coroutine_threadsafe(
                    self._complete_streaming(),
                    asyncio.get_event_loop()
                )
    
    async def _complete_streaming(self):
        """Complete the streaming future."""
        if not self.stream_future.done():
            self.stream_future.set_result(True)
    
    def _generate_streaming_request(self, streaming_config, audio_generator):
        """Generate streaming recognition requests."""
        # First request contains only the config
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        
        # Subsequent requests contain audio data
        for audio_content in audio_generator:
            yield audio_content
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray, memoryview],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process an audio chunk for streaming recognition.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional async callback for results
            
        Returns:
            Optional transcription result
        """
        if not self.is_streaming:
            raise STTStreamingError("Not streaming - call start_streaming() first")
        
        # Add audio chunk to queue
        self.audio_queue.put(audio_chunk)
        
        # Check for results
        final_result = None
        
        # Process any results in the queue
        while not self.result_queue.empty():
            result = await self.result_queue.get()
            
            # Call callback if provided
            if callback:
                await callback(result)
            
            # Keep track of final results
            if result.is_final:
                final_result = result
            
            self.result_queue.task_done()
        
        # Return final result if available
        return final_result
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        logger.debug("Stopping Google Cloud Speech streaming session")
        
        # Set stop flag
        self.stop_stream.set()
        
        # Wait for streaming to complete
        if self.stream_future:
            try:
                await asyncio.wait_for(self.stream_future, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for streaming to complete")
        
        # Get any final results from the queue
        final_text = ""
        duration = 0.0
        
        try:
            # Process any remaining results
            while not self.result_queue.empty():
                result = await self.result_queue.get()
                
                # Keep track of final results
                if result.is_final and result.text:
                    final_text = result.text
                    duration = result.end_time - result.start_time if result.end_time > 0 else 0.0
                
                self.result_queue.task_done()
        except Exception as e:
            logger.error(f"Error processing final results: {e}")
        
        # Reset state
        self.is_streaming = False
        self.streaming_thread = None
        self.audio_queue = None
        self.streaming_responses = None
        self.stop_stream = None
        self.stream_future = None
        self.result_queue = None
        
        # Clean up the transcription
        cleaned_text = self.cleanup_transcription(final_text)
        
        return cleaned_text, duration
    
    async def stream_audio_file(
        self,
        file_path: str,
        chunk_size: int = 4096,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        simulate_realtime: bool = False
    ) -> List[StreamingTranscriptionResult]:
        """
        Stream an audio file to the Google Cloud Speech-to-Text API.
        
        Args:
            file_path: Path to audio file
            chunk_size: Size of audio chunks to send
            callback: Optional async callback for results
            simulate_realtime: Whether to simulate real-time streaming
            
        Returns:
            List of final transcription results
        """
        try:
            # Start streaming
            await self.start_streaming()
            
            # Open the audio file
            with open(file_path, 'rb') as f:
                # Collect final results
                final_results = []
                
                # Stream chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Process chunk
                    result = await self.process_audio_chunk(chunk, callback)
                    
                    # Add final result if available
                    if result and result.is_final:
                        final_results.append(result)
                    
                    # Simulate real-time streaming if requested
                    if simulate_realtime:
                        await asyncio.sleep(chunk_size / self.sample_rate / 2)  # Half real-time speed
            
            # Close the stream
            await self.stop_streaming()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error streaming audio file: {e}")
            await self.stop_streaming()
            raise STTStreamingError(f"Error streaming audio file: {str(e)}")