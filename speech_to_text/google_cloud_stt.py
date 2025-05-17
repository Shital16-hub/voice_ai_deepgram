# speech_to_text/google_cloud_stt.py - CRITICAL FIXES FOR TELEPHONY

"""
Google Cloud Speech-to-Text v2 client FIXED for Twilio integration.
IMPROVED: Ultra-low latency speech detection with better session management.
"""
import os
import asyncio
import time
import json
import queue
import threading
import uuid
import logging
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
    IMPROVED: Google Cloud Speech-to-Text v2 client for Twilio MULAW audio.
    Ultra-optimized for telephony speech detection with sub-2s latency.
    """
    
    # IMPROVED: Optimized constants for faster response
    STREAMING_LIMIT = 120000      # 2 minutes (reduced from 4 min for better refresh)
    CHUNK_TIMEOUT = 0.5           # Faster timeout (reduced from 1.0s)
    RECONNECT_DELAY = 0.1         # Shorter delay (reduced from 0.2s)
    MAX_SILENCE_TIME = 1.0        # More responsive silence detection (reduced from 2.0s)
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = True,  # IMPROVED: Enable for better debugging
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_file: Optional[str] = None,
        enable_vad: bool = True,
        enable_echo_suppression: bool = False,
        **kwargs
    ):
        """Initialize with IMPROVED telephony settings."""
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
        
        # IMPROVED: Setup ultra-optimized configuration
        self._setup_config()
        
        # State tracking (minimal)
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=200)  # Increased queue size (from 100)
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # IMPROVED: Callback handling with better error recovery
        self._main_loop = None
        self._current_callback = None
        self._executor = ThreadPoolExecutor(max_workers=3)  # Increased from 2
        
        # IMPROVED: Result tracking for persistent state
        self._last_final_result = None
        self._result_lock = threading.Lock()
        
        # IMPROVED: Stream reset tracking with faster rotation
        self._restart_counter = 0
        self._max_restarts = 5  # Increased from 3 to allow more restarts
        self._last_streaming_start = None
        self._streaming_time_limit = 110  # Seconds (reduced from 240)
        
        # IMPROVED: Metrics for debugging
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.interim_transcriptions = 0
        self.errors = 0
        
        logger.info(f"FIXED STT initialized - Project: {self.project_id}, Model: telephony_short")
    
    def _get_project_id(self, project_id: Optional[str]) -> str:
        """IMPROVED: Get project ID with faster fallbacks."""
        if project_id:
            return project_id
            
        # Try environment variable
        env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if env_project_id:
            return env_project_id
        
        # Try credentials file with minimal overhead
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
        """Initialize client with better error handling."""
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
        """IMPROVED: Ultra-optimized configuration for telephony."""
        # Audio encoding - FIXED for MULAW
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # IMPROVED: Recognition config super-optimized for telephony
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony_short",  # BEST model for telephony
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                profanity_filter=False,
                enable_word_confidence=True,
                max_alternatives=1,
                enable_word_time_offsets=False,  # Disabled for speed
            ),
        )
        
        # IMPROVED: Streaming config with ULTRA-responsive voice activity timeouts
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    # IMPROVED: Ultra-responsive timeouts
                    speech_start_timeout=Duration(seconds=3),  # Reduced from 5s to 3s
                    speech_end_timeout=Duration(seconds=0, nanos=600000000)  # 600ms (reduced from 1s)
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """IMPROVED: Faster request generator with better queue management."""
        # Send initial config
        yield self.config_request
        
        # IMPROVED: Better audio chunk processing
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with reduced timeout for faster processing
                chunk = self.audio_queue.get(timeout=0.05)  # Reduced from 0.1
                if chunk is None:
                    break
                
                # IMPROVED: Faster validation
                if len(chunk) > 0:
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                    self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                # Continue with error recovery instead of breaking
                continue
    
    def _handle_callback_sync(self, streaming_result: StreamingTranscriptionResult):
        """IMPROVED: Synchronous callback with better error handling."""
        # Track final results to solve second call issue
        if streaming_result.is_final:
            with self._result_lock:
                self._last_final_result = streaming_result
        
        if self._current_callback and self._main_loop:
            try:
                # Schedule the async callback in the main event loop with timeout handling
                future = asyncio.run_coroutine_threadsafe(
                    self._current_callback(streaming_result),
                    self._main_loop
                )
                # Add timeout for callback completion
                future.result(timeout=1.0)  # 1s timeout
            except Exception as e:
                logger.error(f"Error in callback: {e}")
                # Continue with error recovery - don't abort
    
    def _run_streaming(self):
        """IMPROVED: Enhanced streaming with better error recovery."""
        try:
            # Create streaming call with proper timeout
            self.current_stream = self.client.streaming_recognize(
                requests=self._request_generator(),
                timeout=self.STREAMING_LIMIT
            )
            
            # Process responses immediately
            for response in self.current_stream:
                if self.stop_event.is_set():
                    break
                
                # IMPROVED: Better result processing with debugging
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
                        
                        # IMPROVED: Handle callback with better error recovery
                        try:
                            if self._current_callback:
                                self._handle_callback_sync(streaming_result)
                        except Exception as callback_error:
                            logger.error(f"Callback error: {callback_error}")
                            # Continue processing instead of breaking
                        
                        # IMPROVED: Track metrics with enhanced logging
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"FINAL: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
                        else:
                            self.interim_transcriptions += 1
                            logger.info(f"INTERIM: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            self.errors += 1
            # IMPROVED: Better error recovery - restart if needed after a short delay
            if not self.stop_event.is_set():
                logger.info("Auto-recovery after streaming error")
                time.sleep(0.2)  # Reduced delay (from 0.5s)
                self.is_streaming = False  # Reset state to trigger restart
    
    async def start_streaming(self) -> None:
        """IMPROVED: Start streaming with better session management."""
        # Check if we need to restart an existing session
        if self.is_streaming:
            if self._last_streaming_start:
                streaming_duration = time.time() - self._last_streaming_start
                if streaming_duration > self._streaming_time_limit:
                    logger.info(f"Streaming session active for {streaming_duration}s, forcing restart")
                    await self.stop_streaming()
                elif self._restart_counter >= self._max_restarts:
                    logger.info(f"Reached {self._restart_counter} restarts, forcing new streaming session")
                    await self.stop_streaming()
                else:
                    logger.info("Streaming already active, incrementing restart counter")
                    self._restart_counter += 1
                    return
            else:
                # No timestamp, just stop the existing session to be safe
                await self.stop_streaming()
        
        # Capture the current event loop
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = asyncio.get_event_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        
        # Reset state tracking values
        self._last_streaming_start = time.time()
        self._restart_counter = 0
        
        # IMPROVED: Clear queue more efficiently
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Reset result tracking
        with self._result_lock:
            self._last_final_result = None
        
        # Start thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started FIXED streaming: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """IMPROVED: Stop streaming with minimal delay and reliable cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end faster
        try:
            self.audio_queue.put(None, block=False)  # Changed from timeout to non-blocking
        except queue.Full:
            # Clear one item to make room
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                self.audio_queue.put(None, block=False)
            except Exception:
                pass
        
        # Wait for thread with shorter timeout
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=0.5)  # Reduced from 1.0s
        
        # IMPROVED: Return last final result if available
        final_text = ""
        duration = 0.0
        
        with self._result_lock:
            if self._last_final_result:
                final_text = self._last_final_result.text
                duration = 0.0  # We don't track duration in this implementation
        
        logger.info(f"Stopped streaming session: {self.session_id}")
        return final_text, duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """IMPROVED: Faster audio processing with better session health checks."""
        # Store callback and ensure we have the main loop
        self._current_callback = callback
        if not self._main_loop:
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._main_loop = asyncio.get_event_loop()
        
        if not self.is_streaming:
            logger.warning("Received audio chunk but streaming is not active - starting streaming")
            await self.start_streaming()
        
        self.total_chunks += 1
        
        # IMPROVED: Validate audio chunk more efficiently
        if not audio_chunk or len(audio_chunk) == 0:
            logger.warning("Received empty audio chunk")
            return None
        
        # IMPROVED: Add to queue with better error handling and queue management
        try:
            # Check if queue is getting full (>80% capacity)
            if self.audio_queue.qsize() > self.audio_queue.maxsize * 0.8:
                # Clear oldest chunks to make room
                try:
                    for _ in range(min(5, self.audio_queue.qsize() // 10)):
                        self.audio_queue.get_nowait()
                        self.audio_queue.task_done()
                except queue.Empty:
                    pass
                    
            # Add new chunk with no waiting (non-blocking)
            self.audio_queue.put(bytes(audio_chunk), block=False)
            
        except queue.Full:
            # If still full, clear more aggressively
            logger.warning("Audio queue full, dropping several oldest chunks")
            try:
                # Clear up to 20% of queue
                for _ in range(min(10, self.audio_queue.qsize() // 5)):
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                # Try again
                self.audio_queue.put(bytes(audio_chunk), block=False)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
        
        return None
    
    async def check_health(self) -> bool:
        """IMPROVED: More thorough health check with proactive session rotation."""
        # Check streaming session age
        if self.is_streaming and self._last_streaming_start:
            streaming_duration = time.time() - self._last_streaming_start
            # Rotate session more frequently
            if streaming_duration > 60:  # Reduced from 120s to 60s
                logger.info(f"Stream health check: session active for {streaming_duration}s, restarting")
                await self.stop_streaming()
                await asyncio.sleep(0.1)  # Brief pause
                await self.start_streaming()
                return True
                
        # Check if streaming is inactive but should be active
        if not self.is_streaming:
            logger.info("Stream health check: streaming inactive, restarting")
            await self.start_streaming()
            return True
            
        # Check queue health
        if self.audio_queue.qsize() > self.audio_queue.maxsize * 0.9:
            logger.warning("Queue nearing capacity, clearing old items")
            try:
                # Clear up to 30% of queue
                for _ in range(min(20, self.audio_queue.qsize() // 3)):
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
            except queue.Empty:
                pass
                
        return True  # Healthy or repaired
    
    def get_stats(self) -> Dict[str, Any]:
        """IMPROVED: More detailed stats for debugging."""
        with self._result_lock:
            last_result_text = self._last_final_result.text if self._last_final_result else None
        
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "interim_transcriptions": self.interim_transcriptions,
            "queue_size": self.audio_queue.qsize(),
            "queue_capacity": self.audio_queue.maxsize,
            "queue_usage_pct": (self.audio_queue.qsize() / self.audio_queue.maxsize) * 100 if self.audio_queue.maxsize > 0 else 0,
            "project_id": self.project_id,
            "model": "telephony_short",
            "interim_results": self.interim_results,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "last_final_result": last_result_text,
            "restart_counter": self._restart_counter,
            "max_restarts": self._max_restarts,
            "errors": self.errors,
            "session_age": time.time() - self._last_streaming_start if self._last_streaming_start else 0,
            "session_limit": self._streaming_time_limit,
            "voice_activity_timeout": {
                "speech_start": "3s",  # Updated values
                "speech_end": "0.6s"
            }
        }
    
    async def cleanup(self):
        """IMPROVED: Better cleanup with resource management."""
        await self.stop_streaming()
        
        # Cleanup executor
        if self._executor:
            self._executor.shutdown(wait=False)  # Changed from wait=True for faster shutdown
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break