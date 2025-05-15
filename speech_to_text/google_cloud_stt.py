"""
Fixed Google Cloud Speech-to-Text v2 implementation with proper async handling
and better session management for continuous conversation.
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
from google.oauth2 import service_account

# Import Duration from protobuf directly
from google.protobuf.duration_pb2 import Duration

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
    Google Cloud Speech-to-Text v2 client optimized for continuous telephony conversations.
    Fixed async handling and improved session management for multi-turn conversations.
    """
    
    # Constants for session management
    STREAMING_LIMIT = 240000  # 4 minutes - safely under 5min limit with buffer
    CHUNK_TIMEOUT = 30.0      # Timeout for receiving audio chunks
    RECONNECT_DELAY = 1.0     # Delay before attempting reconnection
    
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
        **kwargs
    ):
        """Initialize with telephony settings optimized for continuous conversation."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        
        # Get project ID
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            # Try to extract from credentials file
            credentials_file_to_check = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file_to_check and os.path.exists(credentials_file_to_check):
                try:
                    with open(credentials_file_to_check, 'r') as f:
                        creds_data = json.load(f)
                        self.project_id = creds_data.get('project_id')
                        logger.info(f"Extracted project ID from credentials: {self.project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize client with explicit credentials like TTS
        self._initialize_client()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Setup configuration
        self._setup_config()
        
        # State tracking for continuous operation
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # Session management for handling 5-minute limit
        self.stream_start_time = None
        self.reconnection_in_progress = False
        self.current_stream = None
        
        # Voice activity tracking
        self.last_audio_time = time.time()
        self.speech_detected = False
        
        # Create a proper event loop for async callbacks
        self.callback_loop = None
        self.callback_thread = None
        
        # Stats
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.session_count = 0
        
        logger.info(f"Initialized Speech v2 for continuous conversation - Project: {self.project_id}")
    
    def _initialize_client(self):
        """Initialize the Google Cloud Speech client with explicit credentials handling."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use service account credentials (same as TTS)
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.client = SpeechClient(credentials=credentials)
                logger.info(f"Initialized Speech client with credentials from {self.credentials_file}")
            else:
                # Use default credentials (ADC)
                self.client = SpeechClient()
                logger.info("Initialized Speech client with default credentials")
                
        except Exception as e:
            logger.error(f"Error initializing Speech client: {e}")
            raise
    
    def _setup_config(self):
        """Setup recognition configuration optimized for continuous conversation."""
        # Audio encoding
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Create recognition config (v2 style)
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony",  # Optimal for phone conversations
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
            ),
        )
        
        # Streaming configuration with voice activity detection
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    speech_start_timeout=Duration(seconds=8),   # Wait up to 8s for speech
                    speech_end_timeout=Duration(seconds=2)      # Wait 2s after speech ends
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _create_callback_loop(self):
        """Create a separate event loop for handling async callbacks."""
        def run_callback_loop():
            self.callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.callback_loop)
            try:
                self.callback_loop.run_forever()
            except Exception as e:
                logger.error(f"Error in callback loop: {e}")
            finally:
                self.callback_loop.close()
        
        self.callback_thread = threading.Thread(target=run_callback_loop, daemon=True)
        self.callback_thread.start()
        logger.debug("Started callback event loop thread")
    
    def _stop_callback_loop(self):
        """Stop the callback event loop."""
        if self.callback_loop and not self.callback_loop.is_closed():
            self.callback_loop.call_soon_threadsafe(self.callback_loop.stop)
        if self.callback_thread and self.callback_thread.is_alive():
            self.callback_thread.join(timeout=1.0)
        logger.debug("Stopped callback event loop thread")
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Generate requests for streaming with proper session management."""
        # Send initial config
        yield self.config_request
        
        # Send audio chunks
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                
                # Check if we need to reconnect due to time limit
                if self._should_reconnect():
                    logger.info("Stream approaching time limit, will reconnect after current session")
                    break
                
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                self.audio_queue.task_done()
                self.last_audio_time = time.time()
                
            except queue.Empty:
                # Check for inactivity timeout
                if time.time() - self.last_audio_time > self.CHUNK_TIMEOUT:
                    logger.debug("Audio timeout - continuing...")
                continue
    
    def _should_reconnect(self) -> bool:
        """Check if we should reconnect due to session limits."""
        if not self.stream_start_time:
            return False
        
        # Reconnect if we're approaching the time limit
        elapsed_time = (time.time() - self.stream_start_time) * 1000
        return elapsed_time > self.STREAMING_LIMIT
    
    def _handle_reconnection(self):
        """Handle reconnection to avoid the 5-minute limit."""
        if self.reconnection_in_progress:
            return
        
        logger.info("Initiating STT session reconnection for continuous conversation")
        self.reconnection_in_progress = True
        
        # Start a new stream in a separate thread
        def reconnect():
            time.sleep(self.RECONNECT_DELAY)  # Brief pause
            try:
                self._create_new_stream()
            except Exception as e:
                logger.error(f"Error during reconnection: {e}")
            finally:
                self.reconnection_in_progress = False
        
        threading.Thread(target=reconnect, daemon=True).start()
    
    def _create_new_stream(self):
        """Create a new streaming session."""
        try:
            # Update session info
            self.session_count += 1
            old_session_id = self.session_id
            self.session_id = str(uuid.uuid4())
            self.stream_start_time = time.time()
            
            logger.info(f"Creating new STT session: {self.session_id} (replacing {old_session_id})")
            
            # The old stream will be replaced by the new one in _run_streaming
            
        except Exception as e:
            logger.error(f"Error creating new stream: {e}")
            self.reconnection_in_progress = False
    
    def _run_streaming(self):
        """Run streaming in background thread with proper v2 API handling and auto-reconnection."""
        while self.is_streaming and not self.stop_event.is_set():
            try:
                logger.info(f"Starting streaming session: {self.session_id}")
                self.stream_start_time = time.time()
                
                # Create streaming call with proper v2 API
                self.current_stream = self.client.streaming_recognize(
                    requests=self._request_generator()
                )
                
                # Process responses
                for response in self.current_stream:
                    if self.stop_event.is_set():
                        break
                    
                    self._process_response(response)
                
                # If we reach here and still streaming, it means we need to reconnect
                if self.is_streaming and not self.stop_event.is_set():
                    logger.info("Stream ended, reconnecting for continuous conversation...")
                    time.sleep(self.RECONNECT_DELAY)
                    # Create new session ID and continue
                    self.session_count += 1
                    self.session_id = str(uuid.uuid4())
                    continue
                    
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                
                # If still streaming and not a stop event, try to reconnect
                if self.is_streaming and not self.stop_event.is_set():
                    logger.info("Attempting to recover from streaming error")
                    time.sleep(self.RECONNECT_DELAY)
                    # Create new session ID and continue
                    self.session_count += 1
                    self.session_id = str(uuid.uuid4())
                    continue
                else:
                    break
            
        logger.info(f"Streaming thread ended (session: {self.session_id})")
    
    def _process_response(self, response):
        """Process streaming response and handle voice activity events."""
        # Handle voice activity events
        if hasattr(response, 'speech_event_type') and response.speech_event_type:
            speech_event = response.speech_event_type
            logger.debug(f"Voice activity event: {speech_event}")
            
            if speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                self.speech_detected = True
                logger.debug("Speech activity detected")
            elif speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                logger.debug("Speech activity ended")
                self.speech_detected = False
        
        # Process transcription results
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
                
                # Handle callbacks properly with async
                if hasattr(self, '_current_callback') and self._current_callback:
                    if self.callback_loop and not self.callback_loop.is_closed():
                        # Schedule callback in the callback event loop
                        asyncio.run_coroutine_threadsafe(
                            self._current_callback(transcription_result),
                            self.callback_loop
                        )
                
                if result.is_final:
                    self.successful_transcriptions += 1
                    logger.info(f"Final: '{alternative.transcript}' (conf: {alternative.confidence:.2f})")
    
    async def start_streaming(self) -> None:
        """Start streaming session with continuous conversation support."""
        if self.is_streaming:
            logger.debug("Stream already active, keeping existing session")
            return
        
        # Create callback event loop
        self._create_callback_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        self.reconnection_in_progress = False
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started streaming session: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming and get final result."""
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
            self.stream_thread.join(timeout=3.0)
        
        # Cancel current stream
        if self.current_stream:
            try:
                self.current_stream.cancel()
            except:
                pass
        
        # Stop callback loop
        self._stop_callback_loop()
        
        duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
        logger.info(f"Stopped streaming, duration: {duration:.2f}s")
        return "", duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with automatic session management."""
        # Store callback for use in response processing
        self._current_callback = callback
        
        if not self.is_streaming:
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
            
            # Add to queue (with timeout to prevent blocking)
            try:
                self.audio_queue.put(audio_bytes, block=False)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
                return None
            
            return None  # Results come through callbacks
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "project_id": self.project_id,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "session_count": self.session_count,
            "speech_detected": self.speech_detected,
            "success_rate": round((self.successful_transcriptions / max(self.total_chunks, 1)) * 100, 2),
            "reconnection_in_progress": self.reconnection_in_progress
        }
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_streaming()
        logger.info("STT cleanup completed")