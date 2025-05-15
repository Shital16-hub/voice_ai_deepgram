"""
Enhanced Google Cloud Speech-to-Text v2 with better timeout handling.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator
from dataclasses import dataclass

# Import Speech-to-Text v2 API
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf import duration_pb2

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    session_id: str = ""

class GoogleCloudStreamingSTT:
    """
    Enhanced Google Cloud Speech-to-Text v2 client with timeout handling.
    Automatically manages stream restarts for continuous conversation.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        project_id: Optional[str] = None,
        location: str = "global",
        **kwargs  # Accept other args but ignore them
    ):
        """Initialize with telephony settings and timeout handling."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        
        # Get project ID
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            # Try to extract from credentials file
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        self.project_id = creds_data.get('project_id')
                except Exception:
                    pass
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize client
        self.client = SpeechClient()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Setup configuration
        self._setup_config()
        
        # State tracking
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=200)  # Increased buffer
        self.result_queue = asyncio.Queue()
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        self._stream_lock = threading.Lock()
        
        # Timeout handling
        self.last_activity = time.time()
        self.stream_timeout = 55.0  # Google's timeout is ~60s, restart at 55s
        self.audio_timeout = 10.0   # Restart if no audio for 10s
        
        # Stats
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.start_time = None
        self.stream_restarts = 0
        
        logger.info(f"Enhanced Speech v2 - Project: {self.project_id}")
    
    def _setup_config(self):
        """Setup recognition configuration with optimized settings."""
        # Audio encoding
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Create configs with timeout optimizations
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony",
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_word_time_offsets=False,  # Disable to reduce latency
                enable_word_confidence=False,     # Disable to reduce latency
            ),
        )
        
        # Enhanced streaming features with voice activity detection
        voice_activity_timeout = cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
            speech_start_timeout=duration_pb2.Duration(seconds=3),  # Reduced from 5
            speech_end_timeout=duration_pb2.Duration(seconds=1),    # Must be integer
        )
        
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                voice_activity_timeout=voice_activity_timeout,
                enable_voice_activity_events=True,
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Generate requests for streaming with timeout monitoring."""
        # Send initial config
        yield self.config_request
        
        # Send audio chunks
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                    
                # Update activity time
                self.last_activity = time.time()
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Check for timeouts
                current_time = time.time()
                
                # Stream too old - let it timeout naturally
                if current_time - self.last_activity > self.stream_timeout:
                    logger.debug("Stream approaching timeout, letting it end naturally")
                    break
                    
                continue
    
    def _run_streaming(self):
        """Run streaming with enhanced error handling and timeout management."""
        try:
            logger.info(f"Starting streaming (session: {self.session_id})")
            self.last_activity = time.time()
            
            # Create streaming call
            responses = self.client.streaming_recognize(
                requests=self._request_generator()
            )
            
            # Process responses with timeout detection
            response_count = 0
            for response in responses:
                if self.stop_event.is_set():
                    break
                
                response_count += 1
                self.last_activity = time.time()
                
                # Handle voice activity events
                if hasattr(response, 'speech_event_type'):
                    event_type = response.speech_event_type
                    if event_type:
                        logger.debug(f"Speech event: {event_type}")
                
                # Process results
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence if result.is_final else 0.0,
                            session_id=self.session_id,
                        )
                        
                        # Add to result queue
                        try:
                            self.result_queue.put_nowait(transcription_result)
                        except asyncio.QueueFull:
                            # If queue is full, remove oldest and add new
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait(transcription_result)
                            except asyncio.QueueEmpty:
                                pass
                        
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Final: '{alternative.transcript}' (conf: {alternative.confidence:.2f})")
            
            logger.info(f"Streaming ended naturally (session: {self.session_id}, responses: {response_count})")
            
        except Exception as e:
            # Common Google Cloud timeout errors
            if "timed out" in str(e).lower() or "stream" in str(e).lower():
                logger.info(f"Stream timeout (expected): {e}")
            else:
                logger.error(f"Streaming error: {e}")
        finally:
            logger.info(f"Streaming ended (session: {self.session_id})")
    
    async def start_streaming(self) -> None:
        """Start streaming session with automatic restart capability."""
        with self._stream_lock:
            if self.is_streaming:
                logger.debug("Already streaming, stopping previous session")
                await self.stop_streaming()
            
            self.is_streaming = True
            self.stop_event.clear()
            self.session_id = str(uuid.uuid4())
            self.start_time = time.time()
            self.last_activity = time.time()
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Start thread
            self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
            self.stream_thread.start()
            
            logger.info(f"Started streaming session: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming and get final result."""
        with self._stream_lock:
            if not self.is_streaming:
                return "", 0.0
            
            self.is_streaming = False
            self.stop_event.set()
            
            # Signal end
            try:
                self.audio_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            
            # Wait for thread
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
                if self.stream_thread.is_alive():
                    logger.warning("Stream thread did not terminate gracefully")
            
            # Get final results
            final_transcript = ""
            duration = time.time() - self.start_time if self.start_time else 0.0
            
            final_results = []
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if result.is_final:
                        final_results.append(result)
                except asyncio.QueueEmpty:
                    break
            
            if final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                final_transcript = best_result.text
            
            logger.info(f"Stopped streaming: final='{final_transcript}', duration: {duration:.2f}s")
            return final_transcript, duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with automatic stream management."""
        # Check if stream needs restart due to timeout
        current_time = time.time()
        
        if (self.is_streaming and 
            current_time - self.last_activity > self.audio_timeout):
            logger.debug("Restarting stream due to audio timeout")
            await self.start_streaming()
        
        if not self.is_streaming:
            logger.debug("Starting stream for audio processing")
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert to bytes if needed
            if hasattr(audio_chunk, 'tobytes'):
                audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = bytes(audio_chunk)
            
            # Skip tiny chunks
            if len(audio_bytes) < 40:
                return None
            
            # Add to queue
            try:
                self.audio_queue.put(audio_bytes, block=False)
            except queue.Full:
                # Remove oldest chunk and add new one
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_bytes, block=False)
                except queue.Empty:
                    pass
            
            # Check for results
            final_result = None
            results_processed = 0
            while not self.result_queue.empty() and results_processed < 5:
                try:
                    result = self.result_queue.get_nowait()
                    results_processed += 1
                    
                    if callback:
                        await callback(result)
                    if result.is_final:
                        final_result = result
                except asyncio.QueueEmpty:
                    break
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Try to restart stream on error
            if self.is_streaming:
                await self.start_streaming()
            return None
    
    async def is_stream_healthy(self) -> bool:
        """Check if the current stream is healthy and responsive."""
        if not self.is_streaming:
            return False
        
        current_time = time.time()
        
        # Stream is considered unhealthy if:
        # 1. No activity for too long
        # 2. Stream has been running for too long
        # 3. Thread is not alive
        
        if current_time - self.last_activity > self.audio_timeout:
            return False
        
        if current_time - self.start_time > self.stream_timeout:
            return False
        
        if not self.stream_thread or not self.stream_thread.is_alive():
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        current_time = time.time()
        duration = current_time - self.start_time if self.start_time else 0.0
        
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "project_id": self.project_id,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "stream_restarts": self.stream_restarts,
            "success_rate": round((self.successful_transcriptions / max(self.total_chunks, 1)) * 100, 2),
            "session_duration": round(duration, 2),
            "last_activity_ago": round(current_time - self.last_activity, 2),
            "audio_queue_size": self.audio_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "stream_healthy": asyncio.run(self.is_stream_healthy()) if self.is_streaming else False
        }
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_streaming()
        logger.info("Enhanced STT cleanup completed")