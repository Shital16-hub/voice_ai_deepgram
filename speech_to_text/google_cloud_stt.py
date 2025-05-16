"""
Fixed Ultra Low Latency Google Cloud STT Configuration
Corrected async callback handling for multi-threaded environment.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
import uuid
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Import Speech-to-Text v2 API
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account
from google.protobuf.duration_pb2 import Duration

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Simplified result for ultra low latency."""
    text: str
    is_final: bool
    confidence: float = 0.0
    session_id: str = ""

class GoogleCloudStreamingSTT:
    """
    Ultra Low Latency Google Cloud Speech-to-Text v2 client.
    Fixed async callback handling for threading environment.
    """
    
    # Ultra low latency constants
    STREAMING_LIMIT = 60000   # 1 minute
    CHUNK_TIMEOUT = 1.0       # Ultra fast timeout
    RECONNECT_DELAY = 0.1     # Immediate reconnection
    MAX_SILENCE_TIME = 2.0    # Reduced silence detection
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_file: Optional[str] = None,
        enable_vad: bool = True,
        enable_echo_suppression: bool = False,
        **kwargs
    ):
        """Initialize with ultra low latency settings."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        self.enable_vad = enable_vad
        self.enable_echo_suppression = enable_echo_suppression
        
        # Get project ID
        self.project_id = self._get_project_id(project_id)
        
        # Initialize client
        self._initialize_client()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Setup ultra low latency configuration
        self._setup_config()
        
        # State tracking (minimal)
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=50)  # Smaller queue
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # FIXED: Store the main event loop and callback
        self._main_loop = None
        self._current_callback = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Minimal metrics
        self.total_chunks = 0
        self.successful_transcriptions = 0
        
        logger.info(f"Ultra Low Latency STT initialized - Project: {self.project_id}")
    
    def _get_project_id(self, project_id: Optional[str]) -> str:
        """Get project ID with minimal overhead."""
        if project_id:
            return project_id
            
        # Try environment variable
        env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if env_project_id:
            return env_project_id
        
        # Try credentials file
        credentials_file_to_check = self.credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file_to_check and os.path.exists(credentials_file_to_check):
            try:
                with open(credentials_file_to_check, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        raise ValueError("Google Cloud project ID is required")
    
    def _initialize_client(self):
        """Initialize client with minimal error handling."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.client = SpeechClient(credentials=credentials)
            else:
                self.client = SpeechClient()
                
        except Exception as e:
            logger.error(f"Error initializing Speech client: {e}")
            raise
    
    def _setup_config(self):
        """Setup configuration optimized for ultra low latency."""
        # Audio encoding
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Ultra fast recognition config
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony_short",  # Optimized for short utterances and speed
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=False,  # Disabled for speed
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                profanity_filter=False,
                enable_word_confidence=False,        # Disabled for speed
                max_alternatives=1,                  # Only best result for speed
                enable_word_time_offsets=False,      # Disabled for speed
            ),
        )
        
        # Aggressive streaming config that meets Google Cloud requirements
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    # Meet Google Cloud's minimum requirements
                    speech_start_timeout=Duration(seconds=1),      # Wait only 1s for speech
                    speech_end_timeout=Duration(nanos=500000000)   # 500ms minimum
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Ultra fast request generator with minimal processing."""
        # Send initial config
        yield self.config_request
        
        # Send audio chunks immediately
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with minimal timeout
                chunk = self.audio_queue.get(timeout=0.01)
                if chunk is None:
                    break
                
                # Send immediately
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                break
    
    def _handle_callback_sync(self, streaming_result: StreamingTranscriptionResult):
        """FIXED: Synchronous callback handler that schedules async callback."""
        if self._current_callback and self._main_loop:
            try:
                # Schedule the async callback in the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    self._current_callback(streaming_result),
                    self._main_loop
                )
                # Don't wait for the result to avoid blocking
            except Exception as e:
                logger.error(f"Error scheduling callback: {e}")
    
    def _run_streaming(self):
        """FIXED: Ultra fast streaming with proper callback handling."""
        try:
            # Create streaming call
            self.current_stream = self.client.streaming_recognize(
                requests=self._request_generator(),
                timeout=30  # Reduced timeout
            )
            
            # Process responses immediately
            for response in self.current_stream:
                if self.stop_event.is_set():
                    break
                
                # Handle results immediately
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result
                        streaming_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence if result.is_final else 0.0,
                            session_id=self.session_id
                        )
                        
                        # FIXED: Handle callback synchronously to avoid event loop issues
                        if self._current_callback:
                            self._handle_callback_sync(streaming_result)
                        
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Transcription: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
    
    async def start_streaming(self) -> None:
        """FIXED: Start streaming with proper event loop capture."""
        if self.is_streaming:
            return
        
        # FIXED: Capture the current event loop
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no running loop, create one
            self._main_loop = asyncio.get_event_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Start thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started ultra fast streaming: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming with minimal cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end
        try:
            self.audio_queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        
        # Wait briefly for thread
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
        
        return "", 0.0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """FIXED: Process audio with proper callback storage."""
        # Store callback and ensure we have the main loop
        self._current_callback = callback
        if not self._main_loop:
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._main_loop = asyncio.get_event_loop()
        
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        # Add to queue immediately
        try:
            audio_bytes = bytes(audio_chunk)
            self.audio_queue.put(audio_bytes, block=False)
        except queue.Full:
            # Drop oldest chunk if queue full
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                self.audio_queue.put(audio_bytes, block=False)
            except (queue.Empty, queue.Full):
                pass
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get minimal stats."""
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "queue_size": self.audio_queue.qsize(),
            "project_id": self.project_id
        }
    
    async def cleanup(self):
        """FIXED: Cleanup with proper resource management."""
        await self.stop_streaming()
        
        # Cleanup executor
        if self._executor:
            self._executor.shutdown(wait=False)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break