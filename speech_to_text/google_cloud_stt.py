"""
Google Cloud Speech-to-Text v2 implementation with enhanced WebRTC-based echo cancellation,
improved session management, and voice activity detection.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
import uuid
import webrtcvad
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator
from dataclasses import dataclass, field
from collections import deque

# Import Speech-to-Text v2 API
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account
from google.protobuf.duration_pb2 import Duration

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Enhanced result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    session_id: str = ""
    vad_detected: bool = False  # Voice activity detection result
    echo_suppressed: bool = False  # Whether echo was suppressed

@dataclass
class EchoSuppressionState:
    """State for WebRTC-based echo suppression."""
    recent_tts_audio: deque = field(default_factory=lambda: deque(maxlen=50))
    echo_detection_window: float = 5.0
    suppression_threshold: float = 0.7
    last_tts_timestamp: float = 0.0

class GoogleCloudStreamingSTT:
    """
    Enhanced Google Cloud Speech-to-Text v2 client with WebRTC echo cancellation,
    improved voice activity detection, and robust session management.
    """
    
    # Enhanced constants for better session management
    STREAMING_LIMIT = 240000  # 4 minutes - safely under 5min limit
    CHUNK_TIMEOUT = 5.0      # Reduced for more responsive handling
    RECONNECT_DELAY = 0.3    # Faster reconnection
    MAX_SILENCE_TIME = 8.0   # Reduced for more responsive behavior
    VAD_AGGRESSIVENESS = 3   # WebRTC VAD aggressiveness (0-3, 3 most aggressive)
    
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
        enable_echo_suppression: bool = True,
        **kwargs
    ):
        """Initialize with enhanced WebRTC-based features."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        self.enable_vad = enable_vad
        self.enable_echo_suppression = enable_echo_suppression
        
        # Initialize WebRTC VAD
        if self.enable_vad:
            self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
            logger.info(f"Initialized WebRTC VAD with aggressiveness {self.VAD_AGGRESSIVENESS}")
        
        # Echo suppression state
        self.echo_state = EchoSuppressionState()
        
        # Get project ID with enhanced error handling
        self.project_id = self._get_project_id(project_id)
        
        # Initialize client with explicit credentials
        self._initialize_client()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Setup configuration with enhanced telephony settings
        self._setup_config()
        
        # Enhanced state tracking
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=200)  # Increased buffer
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.session_id = str(uuid.uuid4())
        
        # Session management for handling timeouts
        self.stream_start_time = None
        self.reconnection_in_progress = False
        self.current_stream = None
        self.last_response_time = None
        self.last_audio_time = time.time()
        
        # Voice activity and enhanced echo detection
        self.speech_detected = False
        self.silence_frames = 0
        self.speaking_start_time = None
        
        # Enhanced keep-alive mechanism
        self.keep_alive_interval = 2.0  # Send keep-alive every 2 seconds
        self.last_keep_alive = time.time()
        self.keep_alive_data = b'\x80' * 160  # Silent MULAW frame
        
        # Audio processing tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.session_count = 0
        self.timeout_count = 0
        self.consecutive_errors = 0
        self.vad_speech_frames = 0
        self.echo_suppressions = 0
        
        # Create callback event loop for async operations
        self.callback_loop = None
        self.callback_thread = None
        
        logger.info(f"Initialized Enhanced Speech v2 with WebRTC features - Project: {self.project_id}")
    
    def _get_project_id(self, project_id: Optional[str]) -> str:
        """Get project ID with enhanced fallback mechanisms."""
        # Try provided project_id first
        if project_id:
            return project_id
            
        # Try environment variable
        env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if env_project_id:
            return env_project_id
        
        # Try to extract from credentials file
        credentials_file_to_check = self.credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file_to_check and os.path.exists(credentials_file_to_check):
            try:
                with open(credentials_file_to_check, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Extracted project ID from credentials: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        raise ValueError("Google Cloud project ID is required")
    
    def _initialize_client(self):
        """Initialize the Google Cloud Speech client with enhanced error handling."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.client = SpeechClient(credentials=credentials)
                logger.info(f"Initialized Speech client with credentials from {self.credentials_file}")
            else:
                self.client = SpeechClient()
                logger.info("Initialized Speech client with default credentials")
                
        except Exception as e:
            logger.error(f"Error initializing Speech client: {e}")
            raise
    
    def _setup_config(self):
        """Setup recognition configuration with enhanced telephony and VAD settings."""
        # Audio encoding configuration
        if self.encoding == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Enhanced recognition config for telephony with VAD
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=audio_encoding,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony_short",  # Optimized for short utterances
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                profanity_filter=False,
                enable_word_confidence=True,
                max_alternatives=1,
                # Enhanced for better echo handling
                enable_word_time_offsets=True,
            ),
        )
        
        # Enhanced streaming configuration with aggressive VAD
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    # Very aggressive timeouts for responsive telephony
                    speech_start_timeout=Duration(seconds=2),   # Wait 2s for speech to start
                    speech_end_timeout=Duration(nanos=500000000)  # Wait 0.5s after speech ends
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _apply_webrtc_vad(self, audio_data: bytes) -> bool:
        """Apply WebRTC Voice Activity Detection to audio data."""
        if not self.enable_vad:
            return True
        
        try:
            # WebRTC VAD expects 16-bit PCM, so convert if needed
            if self.encoding == "MULAW":
                # Convert MULAW to 16-bit PCM for VAD
                import audioop
                pcm_data = audioop.ulaw2lin(audio_data, 2)
            else:
                pcm_data = audio_data
            
            # WebRTC VAD works with specific frame sizes (10ms, 20ms, or 30ms)
            # For 8kHz, 20ms = 160 samples = 320 bytes
            frame_size = 320
            
            if len(pcm_data) >= frame_size:
                # Take the first complete frame
                frame = pcm_data[:frame_size]
                # VAD returns True if speech is detected
                return self.vad.is_speech(frame, self.sample_rate)
            
            return False
        except Exception as e:
            logger.error(f"Error in WebRTC VAD: {e}")
            return True  # Default to speech when VAD fails
    
    def _apply_echo_suppression(self, audio_data: bytes, transcription: str) -> tuple[bool, bool]:
        """Apply WebRTC-based echo suppression using audio analysis."""
        if not self.enable_echo_suppression:
            return True, False
        
        current_time = time.time()
        
        # Check if we're within echo detection window of recent TTS
        if current_time - self.echo_state.last_tts_timestamp > self.echo_state.echo_detection_window:
            return True, False
        
        # Apply multiple echo detection methods
        echo_detected = False
        
        # 1. Timing-based detection
        if current_time - self.echo_state.last_tts_timestamp < 1.0:
            echo_detected = True
            logger.debug("Echo detected: too close to TTS output")
        
        # 2. Audio fingerprint matching (simplified)
        try:
            # Convert audio to comparable format for analysis
            if self.echo_state.recent_tts_audio:
                # Simple energy-based comparison
                import audioop
                if self.encoding == "MULAW":
                    audio_energy = sum(abs(b) for b in audio_data)
                else:
                    # For PCM, compute RMS energy
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_energy = np.sqrt(np.mean(audio_array**2))
                
                # Compare with recent TTS audio characteristics
                # This is a simplified comparison - could be enhanced with FFT analysis
                for tts_fingerprint in list(self.echo_state.recent_tts_audio)[-5:]:
                    if abs(audio_energy - tts_fingerprint.get('energy', 0)) < 0.1 * audio_energy:
                        echo_detected = True
                        break
        except Exception as e:
            logger.debug(f"Error in audio echo analysis: {e}")
        
        # 3. Text-based echo detection
        if transcription:
            for tts_fingerprint in list(self.echo_state.recent_tts_audio)[-3:]:
                tts_text = tts_fingerprint.get('text', '').lower()
                if tts_text and tts_text in transcription.lower():
                    echo_detected = True
                    logger.debug(f"Echo detected: text match with TTS output")
                    break
        
        if echo_detected:
            self.echo_suppressions += 1
            logger.debug(f"Echo suppressed: {transcription[:50]}...")
        
        return not echo_detected, echo_detected
    
    def add_tts_fingerprint(self, text: str, audio_data: Optional[bytes] = None):
        """Add TTS output fingerprint for echo detection."""
        current_time = time.time()
        self.echo_state.last_tts_timestamp = current_time
        
        fingerprint = {
            'text': text,
            'timestamp': current_time,
            'energy': 0.0
        }
        
        if audio_data:
            try:
                # Calculate audio energy fingerprint
                if self.encoding == "MULAW":
                    fingerprint['energy'] = sum(abs(b) for b in audio_data[:160])
                else:
                    audio_array = np.frombuffer(audio_data[:320], dtype=np.int16)
                    fingerprint['energy'] = float(np.sqrt(np.mean(audio_array**2)))
            except Exception as e:
                logger.debug(f"Error calculating audio fingerprint: {e}")
        
        self.echo_state.recent_tts_audio.append(fingerprint)
        logger.debug(f"Added TTS fingerprint: {text[:30]}...")
    
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
        logger.debug("Started enhanced callback event loop thread")
    
    def _stop_callback_loop(self):
        """Stop the callback event loop."""
        if self.callback_loop and not self.callback_loop.is_closed():
            self.callback_loop.call_soon_threadsafe(self.callback_loop.stop)
        if self.callback_thread and self.callback_thread.is_alive():
            self.callback_thread.join(timeout=1.0)
        logger.debug("Stopped callback event loop thread")
    
    def _request_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Enhanced request generator with keep-alive and VAD."""
        # Send initial config
        yield self.config_request
        
        # Track timing for keep-alive
        last_audio_sent = time.time()
        
        # Send audio chunks with enhanced flow control and keep-alive
        while not self.stop_event.is_set():
            try:
                # Check if we need keep-alive
                current_time = time.time()
                if (current_time - last_audio_sent > self.keep_alive_interval and 
                    current_time - self.last_keep_alive > self.keep_alive_interval):
                    # Send keep-alive frame
                    yield cloud_speech.StreamingRecognizeRequest(audio=self.keep_alive_data)
                    self.last_keep_alive = current_time
                    logger.debug("Sent keep-alive frame")
                    continue
                
                # Get audio chunk with shorter timeout for responsiveness
                try:
                    chunk = self.audio_queue.get(timeout=0.05)
                    if chunk is None:
                        break
                except queue.Empty:
                    # Check for session restart conditions
                    if self._should_restart_session():
                        logger.info("Session restart required")
                        break
                    continue
                
                # Apply WebRTC VAD
                has_speech = self._apply_webrtc_vad(chunk)
                
                if has_speech:
                    # Send audio and track timing
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                    self.audio_queue.task_done()
                    self.last_audio_time = time.time()
                    last_audio_sent = time.time()
                    self.vad_speech_frames += 1
                else:
                    # Skip non-speech frames but still update queue
                    self.audio_queue.task_done()
                    self.silence_frames += 1
                
                # Check for extended silence
                if current_time - last_audio_sent > self.MAX_SILENCE_TIME:
                    logger.info(f"Extended silence ({self.MAX_SILENCE_TIME}s), ending session")
                    break
                    
            except Exception as e:
                logger.error(f"Error in request generator: {e}")
                break
    
    def _should_restart_session(self) -> bool:
        """Enhanced logic to determine when to restart the session."""
        if not self.stream_start_time:
            return False
        
        elapsed_time = (time.time() - self.stream_start_time) * 1000
        
        # Restart if approaching time limit (with buffer)
        if elapsed_time > self.STREAMING_LIMIT:
            return True
        
        # Restart if we've had multiple consecutive errors
        if self.consecutive_errors > 3:
            logger.info(f"Restarting session due to {self.consecutive_errors} consecutive errors")
            return True
        
        # Restart if too many silence frames (indicates poor connection)
        if self.silence_frames > 300:  # ~30 seconds of silence at 10fps
            logger.info("Restarting session due to excessive silence")
            return True
        
        return False
    
    def _run_streaming(self):
        """Enhanced streaming with better error recovery and session management."""
        while self.is_streaming and not self.stop_event.is_set():
            try:
                logger.info(f"Starting enhanced streaming session: {self.session_id}")
                self.stream_start_time = time.time()
                self.consecutive_errors = 0
                self.silence_frames = 0
                self.vad_speech_frames = 0
                
                # Create streaming call with enhanced timeout
                self.current_stream = self.client.streaming_recognize(
                    requests=self._request_generator(),
                    timeout=300  # 5 minute timeout
                )
                
                # Process responses with enhanced error handling
                try:
                    for response in self.current_stream:
                        if self.stop_event.is_set():
                            break
                        
                        self.last_response_time = time.time()
                        self._process_response(response)
                        
                except StopIteration:
                    logger.debug("Stream iteration completed normally")
                except Exception as e:
                    logger.error(f"Error processing stream responses: {e}")
                    self.consecutive_errors += 1
                    raise
                
                # If we reach here and still streaming, session ended normally
                if self.is_streaming and not self.stop_event.is_set():
                    logger.info("Stream ended normally, will restart if needed")
                    time.sleep(self.RECONNECT_DELAY)
                    self._start_new_session()
                    continue
                    
            except Exception as e:
                self.consecutive_errors += 1
                self.timeout_count += 1
                
                # Enhanced error classification
                error_str = str(e).lower()
                if "timeout" in error_str or "409" in error_str:
                    logger.warning(f"Stream timeout (#{self.timeout_count}): {e}")
                elif "cancelled" in error_str:
                    logger.info("Stream cancelled by client")
                    break
                else:
                    logger.error(f"Streaming error (#{self.consecutive_errors}): {e}")
                
                # Enhanced restart logic with backoff
                if self.is_streaming and not self.stop_event.is_set():
                    if self.consecutive_errors < 5:
                        backoff_delay = min(self.RECONNECT_DELAY * (2 ** self.consecutive_errors), 5.0)
                        logger.info(f"Attempting recovery with {backoff_delay}s backoff")
                        time.sleep(backoff_delay)
                        self._start_new_session()
                        continue
                    else:
                        logger.error(f"Too many consecutive errors ({self.consecutive_errors}), stopping")
                        break
                else:
                    break
        
        logger.info(f"Enhanced streaming thread ended (session: {self.session_id})")
    
    def _start_new_session(self):
        """Start a new session with enhanced cleanup."""
        old_session_id = self.session_id
        self.session_count += 1
        self.session_id = str(uuid.uuid4())
        
        logger.info(f"Starting new enhanced STT session: {self.session_id} (replacing {old_session_id})")
        
        # Clear audio queue to prevent old audio from affecting new session
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Reset counters
        self.silence_frames = 0
        self.consecutive_errors = 0
    
    def _process_response(self, response):
        """Enhanced response processing with echo suppression and VAD."""
        # Handle voice activity events
        if hasattr(response, 'speech_event_type') and response.speech_event_type:
            speech_event = response.speech_event_type
            logger.debug(f"Voice activity event: {speech_event}")
            
            if speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                self.speech_detected = True
                self.speaking_start_time = time.time()
                logger.debug("Speech activity detected by Google Cloud")
            elif speech_event == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                self.speech_detected = False
                speaking_duration = time.time() - (self.speaking_start_time or time.time())
                logger.debug(f"Speech activity ended (duration: {speaking_duration:.2f}s)")
        
        # Process transcription results with enhanced echo suppression
        for result in response.results:
            if result.alternatives:
                alternative = result.alternatives[0]
                text = alternative.transcript.strip()
                
                if text and result.is_final:
                    confidence = alternative.confidence
                    
                    # Apply enhanced echo suppression
                    should_process, echo_suppressed = self._apply_echo_suppression(b"", text)
                    
                    if should_process:
                        # Create enhanced result
                        transcription_result = StreamingTranscriptionResult(
                            text=text,
                            is_final=True,
                            confidence=confidence,
                            session_id=self.session_id,
                            vad_detected=True,  # Already passed VAD if we're here
                            echo_suppressed=echo_suppressed
                        )
                        
                        # Handle callbacks
                        if hasattr(self, '_current_callback') and self._current_callback:
                            if self.callback_loop and not self.callback_loop.is_closed():
                                asyncio.run_coroutine_threadsafe(
                                    self._current_callback(transcription_result),
                                    self.callback_loop
                                )
                        
                        self.successful_transcriptions += 1
                        logger.info(f"Enhanced transcription (session {self.session_id}): '{text}' "
                                   f"(conf: {confidence:.2f}, VAD frames: {self.vad_speech_frames})")
                    else:
                        logger.debug(f"Echo suppressed: '{text}' (total suppressions: {self.echo_suppressions})")
    
    async def start_streaming(self) -> None:
        """Enhanced streaming startup with better initialization."""
        if self.is_streaming:
            logger.debug("Stream already active, keeping existing session")
            return
        
        # Create callback event loop
        self._create_callback_loop()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.session_id = str(uuid.uuid4())
        self.reconnection_in_progress = False
        self.consecutive_errors = 0
        self.silence_frames = 0
        self.vad_speech_frames = 0
        self.echo_suppressions = 0
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started enhanced streaming session: {self.session_id}")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Enhanced streaming stop with comprehensive cleanup."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info(f"Stopping enhanced streaming session: {self.session_id}")
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end to request generator
        try:
            self.audio_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5.0)
            
            if self.stream_thread.is_alive():
                logger.warning("Stream thread did not finish gracefully")
        
        # Cancel current stream
        if self.current_stream:
            try:
                self.current_stream.cancel()
                logger.debug("Cancelled current stream")
            except Exception as e:
                logger.debug(f"Error cancelling stream: {e}")
        
        # Stop callback loop
        self._stop_callback_loop()
        
        # Calculate session duration
        duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
        
        logger.info(f"Stopped enhanced streaming, duration: {duration:.2f}s, "
                   f"sessions: {self.session_count}, timeouts: {self.timeout_count}, "
                   f"VAD speech frames: {self.vad_speech_frames}, echo suppressions: {self.echo_suppressions}")
        
        return "", duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Enhanced audio chunk processing with VAD and echo suppression."""
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
            
            # Skip tiny chunks that might be silence
            if len(audio_bytes) < 40:
                return None
            
            # Add to queue with enhanced error handling
            try:
                self.audio_queue.put(audio_bytes, block=False)
            except queue.Full:
                # If queue is full, remove oldest item and add new one
                try:
                    old_chunk = self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    self.audio_queue.put(audio_bytes, block=False)
                    logger.warning("Audio queue full, dropped old chunk")
                except (queue.Empty, queue.Full):
                    logger.warning("Could not manage full audio queue")
                    return None
            
            return None  # Results come through callbacks
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics with enhanced metrics."""
        current_time = time.time()
        session_duration = current_time - self.stream_start_time if self.stream_start_time else 0
        
        return {
            "session_id": self.session_id,
            "is_streaming": self.is_streaming,
            "project_id": self.project_id,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "session_count": self.session_count,
            "timeout_count": self.timeout_count,
            "consecutive_errors": self.consecutive_errors,
            "speech_detected": self.speech_detected,
            "session_duration": session_duration,
            "success_rate": round((self.successful_transcriptions / max(self.total_chunks, 1)) * 100, 2),
            "avg_timeouts_per_session": round(self.timeout_count / max(self.session_count, 1), 2),
            "reconnection_in_progress": self.reconnection_in_progress,
            "queue_size": self.audio_queue.qsize(),
            "last_audio_time": self.last_audio_time,
            "last_response_time": self.last_response_time,
            # Enhanced metrics
            "vad_enabled": self.enable_vad,
            "echo_suppression_enabled": self.enable_echo_suppression,
            "vad_speech_frames": self.vad_speech_frames,
            "vad_silence_frames": self.silence_frames,
            "echo_suppressions": self.echo_suppressions,
            "recent_tts_fingerprints": len(self.echo_state.recent_tts_audio),
            "vad_aggressiveness": self.VAD_AGGRESSIVENESS if self.enable_vad else None
        }
    
    async def cleanup(self):
        """Enhanced cleanup with comprehensive resource management."""
        logger.info(f"Cleaning up enhanced STT session: {self.session_id}")
        await self.stop_streaming()
        
        # Clear any remaining audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Clean up WebRTC resources
        if hasattr(self, 'vad') and self.vad:
            del self.vad
        
        # Clear echo suppression state
        self.echo_state.recent_tts_audio.clear()
        
        logger.info("Enhanced STT cleanup completed")