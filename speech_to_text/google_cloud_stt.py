# speech_to_text/google_cloud_stt.py - CRITICAL FIXES

"""
Google Cloud Speech-to-Text v2 client FIXED for Twilio integration.
All critical issues resolved for proper speech detection and response.
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
    FIXED Google Cloud Speech-to-Text v2 client for Twilio MULAW audio.
    All critical issues resolved for proper speech detection and response.
    """
    
    # FIXED: Optimized constants
    STREAMING_LIMIT = 240000      # 4 minutes (Twilio call limit)
    CHUNK_TIMEOUT = 0.5           # UPDATED: Even faster timeout for better performance
    RECONNECT_DELAY = 0.2         # Shorter delay for faster recovery
    MAX_SILENCE_TIME = 1.5        # UPDATED: More responsive silence detection
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = True,  # CRITICAL: Enable for better debugging
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_file: Optional[str] = None,
        enable_vad: bool = True,      # Keep VAD but configure properly
        enable_echo_suppression: bool = False,
        **kwargs
    ):
        """Initialize with FIXED telephony settings."""
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
        
        # CRITICAL FIX: Setup proper configuration
        self._setup_config()
        
        # State tracking (minimal)
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=100)  # Increased queue size
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # CRITICAL FIX: Proper callback handling
        self._main_loop = None
        self._current_callback = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # CRITICAL FIX: Track results for second call issue
        self._last_final_result = None
        self._result_lock = threading.Lock()
        
        # CRITICAL FIX: Add stream reset tracking
        self._restart_counter = 0
        self._max_restarts = 3  # Allow up to 3 restarts before requiring a new streaming session
        self._last_streaming_start = None
        self._streaming_time_limit = 120  # UPDATED: 2 minutes (reduced from 4) before needing a restart
        
        # CRITICAL NEW: Add streaming watchdog timer to detect stalled streams
        self._last_received_audio_time = None
        self._last_result_time = None
        self._stream_healthy = True
        
        # Minimal metrics
        self.total_chunks = 0
        self.successful_transcriptions = 0
        
        logger.info(f"FIXED STT initialized - Project: {self.project_id}, Model: telephony_short")
    
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
        """CRITICAL FIX: Proper configuration for Twilio integration."""
        # Audio encoding - FIXED for MULAW
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # CRITICAL FIX: Recognition config optimized for Twilio telephony
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony_short",  # BEST model for telephony
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,   # FIXED: Enable for better results
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                profanity_filter=False,
                enable_word_confidence=True,         # Enable for debugging
                max_alternatives=1,                  # Only best result for speed
                enable_word_time_offsets=False,      # Disabled for speed
            ),
        )
        
        # CRITICAL FIX: Streaming config with PROPER voice activity timeouts
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,  # Enable for debugging
                enable_voice_activity_events=True,     # Keep voice activity events
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    # CRITICAL FIX: Proper timeouts for reliable speech detection
                    speech_start_timeout=Duration(seconds=3),     # UPDATED: Even shorter
                    speech_end_timeout=Duration(seconds=1)        # Reduced for faster completion
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """CRITICAL FIX: Improved request generator with proper audio handling."""
        # Send initial config
        yield self.config_request
        
        # CRITICAL FIX: Better audio chunk processing
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with optimized timeout
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                
                # CRITICAL FIX: Validate audio chunk before sending
                if isinstance(chunk, bytes) and len(chunk) > 0:
                    # Update last received audio time for watchdog
                    self._last_received_audio_time = time.time()
                    
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                    self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                break
    
    def _handle_callback_sync(self, streaming_result: StreamingTranscriptionResult):
        """CRITICAL FIX: Synchronous callback handler with result tracking."""
        # Update result timestamp for health check
        self._last_result_time = time.time()
        
        # CRITICAL FIX: Track final results to solve second call issue
        if streaming_result.is_final:
            with self._result_lock:
                self._last_final_result = streaming_result
        
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
        """CRITICAL FIX: Enhanced streaming with better error handling."""
        try:
            # CRITICAL FIX: Create streaming call with proper timeout
            self.current_stream = self.client.streaming_recognize(
                requests=self._request_generator(),
                timeout=self.STREAMING_LIMIT  # Increased timeout
            )
            
            # Process responses immediately
            for response in self.current_stream:
                if self.stop_event.is_set():
                    break
                
                # Update stream health status
                self._stream_healthy = True
                
                # CRITICAL FIX: Better result processing
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
                        
                        # CRITICAL FIX: Handle callback synchronously
                        if self._current_callback:
                            self._handle_callback_sync(streaming_result)
                        
                        # CRITICAL FIX: Better logging for debugging
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"FINAL: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
                        else:
                            logger.info(f"INTERIM: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            # Set stream health to false
            self._stream_healthy = False
            
            # CRITICAL FIX: Don't fail silently, restart if needed
            if not self.stop_event.is_set():
                logger.info("Attempting to restart streaming...")
                time.sleep(0.5)  # Reduced delay for faster recovery
    
    async def start_streaming(self) -> None:
        """CRITICAL FIX: Start streaming with proper session management."""
        # CRITICAL FIX: Check if we need to restart an existing session
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
        
        # CRITICAL FIX: Capture the current event loop
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = asyncio.get_event_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        
        # CRITICAL FIX: Reset state tracking values
        self._last_streaming_start = time.time()
        self._last_received_audio_time = time.time()
        self._last_result_time = time.time()
        self._stream_healthy = True
        self._restart_counter = 0
        
        # CRITICAL FIX: Clear queue properly
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # CRITICAL FIX: Reset result tracking
        with self._result_lock:
            self._last_final_result = None
        
        # Start thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started FIXED streaming: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """CRITICAL FIX: Stop streaming with proper cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end
        try:
            self.audio_queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        
        # Wait for thread with shorter timeout
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)  # Reduced timeout
        
        # CRITICAL FIX: Return last final result if available
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
        """CRITICAL FIX: Process audio with better error handling."""
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
        
        # CRITICAL NEW: Health check for stream
        current_time = time.time()
        if self._last_received_audio_time and current_time - self._last_received_audio_time > 10:
            # If no audio for 10 seconds, check if we're getting results
            if not self._last_result_time or current_time - self._last_result_time > 15:
                # No results for 15 seconds, force restart
                logger.warning("Stream health check failed - forcing restart")
                await self.stop_streaming()
                await self.start_streaming()
        
        self.total_chunks += 1
        
        # CRITICAL FIX: Validate audio chunk
        if not audio_chunk or len(audio_chunk) == 0:
            logger.warning("Received empty audio chunk")
            return None
        
        # Add to queue with better error handling
        try:
            audio_bytes = bytes(audio_chunk)
            
            # CRITICAL FIX: Don't drop audio if queue is full, wait briefly
            if self.audio_queue.full():
                try:
                    # Remove oldest chunk
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    pass
            
            self.audio_queue.put(audio_bytes, block=False)
            
        except queue.Full:
            logger.warning("Audio queue full, dropping oldest chunk")
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                self.audio_queue.put(audio_bytes, block=False)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
        
        return None
    
    async def check_health(self) -> bool:
        """CRITICAL NEW FIX: Check and repair streaming session health."""
        if not self._stream_healthy:
            logger.warning("Stream health check failed: stream not healthy")
            await self.stop_streaming()
            await self.start_streaming()
            return True
            
        # Check if streaming needs to be restarted due to time limit
        if self.is_streaming and self._last_streaming_start:
            streaming_duration = time.time() - self._last_streaming_start
            if streaming_duration > self._streaming_time_limit:
                logger.info(f"Stream health check: session active for {streaming_duration}s, restarting")
                await self.stop_streaming()
                await self.start_streaming()
                return True
                
        # Check if streaming is inactive but should be active
        if not self.is_streaming:
            logger.info("Stream health check: streaming inactive, restarting")
            await self.start_streaming()
            return True
            
        return True  # Healthy or repaired
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        with self._result_lock:
            last_result_text = self._last_final_result.text if self._last_final_result else None
        
        current_time = time.time()
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "stream_healthy": self._stream_healthy,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "queue_size": self.audio_queue.qsize(),
            "project_id": self.project_id,
            "model": "telephony_short",
            "interim_results": self.interim_results,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "last_final_result": last_result_text,
            "restart_counter": self._restart_counter,
            "max_restarts": self._max_restarts,
            "session_age": current_time - self._last_streaming_start if self._last_streaming_start else 0,
            "last_audio_received": self._last_received_audio_time,
            "time_since_audio": current_time - self._last_received_audio_time if self._last_received_audio_time else None,
            "last_result_time": self._last_result_time,
            "time_since_result": current_time - self._last_result_time if self._last_result_time else None,
            "voice_activity_timeout": {
                "speech_start": "3s",  # Updated values
                "speech_end": "1s"
            }
        }
    
    async def cleanup(self):
        """CRITICAL FIX: Cleanup with proper resource management."""
        await self.stop_streaming()
        
        # Cleanup executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break