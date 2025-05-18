"""
Infinite streaming implementation for Google Cloud Speech-to-Text v2.
Provides uninterrupted streaming for entire call duration without session restarts.

FIXED VERSION: Properly handles event loop in threading context
"""
import os
import asyncio
import queue
import threading
import time
import logging
import uuid
from typing import Dict, Any, Optional, AsyncIterator, List, Callable, Awaitable, Union

# Import Google Cloud Speech v2
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.duration_pb2 import Duration
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class StreamingTranscriptionResult:
    """Result from streaming transcription with extended fields."""
    
    def __init__(
        self,
        text: str,
        is_final: bool,
        confidence: float = 0.0,
        session_id: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0,
        alternatives: List[Dict[str, Any]] = None,
        stability: float = 0.0
    ):
        """Initialize with comprehensive result data."""
        self.text = text
        self.is_final = is_final
        self.confidence = confidence
        self.session_id = session_id
        self.start_time = start_time
        self.end_time = end_time
        self.alternatives = alternatives or []
        self.stability = stability

class InfiniteStreamingSTT:
    """
    Implementation of infinite streaming with Google Cloud Speech-to-Text v2.
    
    This class provides truly continuous speech recognition by managing multiple
    overlapping streaming sessions that ensure uninterrupted transcription
    throughout the entire call duration.
    """
    
    def __init__(
        self,
        project_id: str,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = True,
        location: str = "global",
        credentials_file: Optional[str] = None,
        session_overlap_seconds: int = 30,  # Overlap between sessions
        session_max_duration: int = 240     # 4 minutes (under Google's ~5 min limit)
    ):
        """
        Initialize infinite streaming STT.
        
        Args:
            project_id: Google Cloud project ID
            language: Language code for recognition
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results
            location: Google Cloud location
            credentials_file: Path to credentials file
            session_overlap_seconds: Seconds of overlap between sessions
            session_max_duration: Maximum duration for each session in seconds
        """
        self.project_id = project_id
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.location = location
        self.credentials_file = credentials_file
        self.session_overlap_seconds = session_overlap_seconds
        self.session_max_duration = session_max_duration
        
        # Initialize client with proper credentials
        self._initialize_client()
        
        # Create recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Set up streaming configuration
        self.streaming_config = self._create_streaming_config()
        
        # Session management
        self.is_streaming = False
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
        self.active_sessions = []  # List of active sessions
        self.session_counter = 0
        
        # Set maximum active sessions (primary + backup during transition)
        self.max_active_sessions = 2
        
        # Audio buffer for smooth transition between sessions
        self.audio_buffer = []
        self.buffer_max_size = 50  # Approximately 5 seconds of audio at 10 chunks/sec
        
        # Track last received audio/result time for health monitoring
        self.last_audio_time = None
        self.last_result_time = None
        
        # Metrics tracking
        self.total_audio_chunks = 0
        self.total_results = 0
        self.successful_transcriptions = 0
        self.errors = 0
        self.session_starts = 0
        self.session_ends = 0
        
        # CRITICAL FIX: Store main event loop reference for thread safety
        self._main_event_loop = None
        
        logger.info(f"Initialized InfiniteStreamingSTT with {session_max_duration}s sessions and {session_overlap_seconds}s overlap")
    
    def _initialize_client(self):
        """Initialize Google Cloud Speech client with proper credentials."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use service account credentials
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
    
    def _create_streaming_config(self) -> cloud_speech.StreamingRecognitionConfig:
        """
        Create a streaming config optimized for infinite streaming.
        
        Returns:
            StreamingRecognitionConfig for Google Cloud Speech-to-Text v2
        """
        # Determine audio encoding
        if self.encoding.upper() == "MULAW":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        elif self.encoding.upper() == "LINEAR16":
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        else:
            # Default to MULAW for telephony
            audio_encoding = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
            logger.warning(f"Unknown encoding {self.encoding}, using MULAW")
        
        # Create explicit decoding config
        decoding_config = cloud_speech.ExplicitDecodingConfig(
            encoding=audio_encoding,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels
        )
        
        # Recognition config with telephony optimization
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=decoding_config,
            language_codes=[self.language],
            model="telephony_short",  # Optimized for phone calls
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
        
        # Streaming features with extended timeouts for continuous streaming
        streaming_features = cloud_speech.StreamingRecognitionFeatures(
            interim_results=self.interim_results,
            # Critical for continuous streaming:
            enable_voice_activity_events=True,
            # Extended timeouts for continuous streaming:
            voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                speech_start_timeout=Duration(seconds=20),  # Longer timeout for speech start
                speech_end_timeout=Duration(seconds=5)      # Longer timeout for speech end
            )
        )
        
        # Create final streaming config
        return cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=streaming_features,
        )
    
    async def start_streaming(self) -> None:
        """
        Start infinite streaming with session overlap.
        
        This initiates the first streaming session and sets up continuous
        session management to ensure uninterrupted transcription.
        """
        if self.is_streaming:
            logger.info("Already streaming, not starting again")
            return
        
        logger.info("Starting infinite streaming session")
        self.is_streaming = True
        self.stop_event.clear()
        
        # CRITICAL FIX: Store reference to current event loop for thread safety
        try:
            self._main_event_loop = asyncio.get_running_loop()
            logger.info("Captured main event loop for thread safety")
        except RuntimeError:
            logger.warning("Could not get running event loop, async operations from threads may fail")
            # Try to get or create an event loop as fallback
            try:
                self._main_event_loop = asyncio.get_event_loop()
            except RuntimeError:
                # If we're not in the main thread, this might still fail
                logger.error("Failed to get event loop - async operations from threads will fail")
                self._main_event_loop = None
        
        # Reset metrics
        self.total_audio_chunks = 0
        self.total_results = 0
        self.successful_transcriptions = 0
        self.errors = 0
        self.session_starts = 0
        self.session_ends = 0
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                await self.result_queue.get_nowait()
                self.result_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Start initial session
        await self._start_new_session()
        
        # Start automatic session management for continuous streaming
        asyncio.create_task(self._manage_sessions())
        
        # Start monitoring task
        asyncio.create_task(self._monitor_health())
        
        logger.info("Infinite streaming started successfully")
    
    async def _start_new_session(self) -> None:
        """
        Start a new streaming session that overlaps with existing ones.
        
        This creates a new session thread that processes audio from the shared
        queue, allowing multiple sessions to run concurrently during transitions.
        """
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        
        # Create session thread
        session_thread = threading.Thread(
            target=self._run_streaming_session,
            args=(session_id,),
            daemon=True
        )
        
        # Track session
        session_info = {
            "id": session_id,
            "thread": session_thread,
            "start_time": time.time(),
            "active": True,
            "results_received": 0,
            "last_result_time": None
        }
        self.active_sessions.append(session_info)
        
        # Start session
        session_thread.start()
        self.session_starts += 1
        logger.info(f"Started new STT session: {session_id}")
        
        # Prune old sessions if needed
        if len(self.active_sessions) > self.max_active_sessions:
            # Mark oldest session for cleanup (but let it finish naturally)
            oldest_session = self.active_sessions[0]
            oldest_session["active"] = False
            logger.info(f"Marked old session for retirement: {oldest_session['id']}")
    
    def _run_streaming_session(self, session_id: str) -> None:
        """
        Run a single streaming session with auto-termination.
        
        This method runs in a separate thread and handles the communication
        with Google Cloud Speech-to-Text API for a single streaming session.
        
        Args:
            session_id: Unique identifier for this session
        """
        try:
            # Create initial request with config
            initial_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=self.recognizer_path,
                streaming_config=self.streaming_config
            )
            
            # Session duration tracking
            session_start = time.time()
            
            # Find our session info
            session_info = next(s for s in self.active_sessions if s["id"] == session_id)
            
            # Create request generator
            def request_generator():
                # Yield initial config request
                yield initial_request
                
                # Add any buffered audio for smooth transition
                if hasattr(self, 'audio_buffer') and self.audio_buffer:
                    for chunk in self.audio_buffer:
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                
                # Process audio chunks until session should end
                while not self.stop_event.is_set() and session_info["active"]:
                    # Check if session has been running too long
                    if time.time() - session_start > self.session_max_duration:
                        logger.info(f"Session {session_id} reached max duration, ending")
                        break
                        
                    try:
                        # Get audio chunk with timeout
                        chunk = self.audio_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                            
                        # Create and yield request with audio
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                        self.audio_queue.task_done()
                        
                    except queue.Empty:
                        # No audio available, continue waiting
                        continue
                    except Exception as e:
                        logger.error(f"Error in request generator: {e}")
                        break
            
            # Start streaming with Google Cloud Speech-to-Text v2
            responses = self.client.streaming_recognize(request_generator())
            
            # Process streaming responses
            for response in responses:
                if self.stop_event.is_set() or not session_info["active"]:
                    break
                
                # Update session time
                session_info["last_result_time"] = time.time()
                
                # Process results
                for result in response.results:
                    streaming_result = self._convert_to_result(result, session_id)
                    if streaming_result:
                        # Update session stats
                        session_info["results_received"] += 1
                        
                        # Add to result queue for processing
                        if self._main_event_loop:
                            asyncio.run_coroutine_threadsafe(
                                self.result_queue.put(streaming_result),
                                self._main_event_loop
                            )
                        else:
                            logger.error("Cannot put result in queue - no event loop available")
                
                # Process voice activity events (for debugging)
                if hasattr(response, 'speech_event_type') and response.speech_event_type:
                    event_type = response.speech_event_type
                    logger.debug(f"Speech event in {session_id}: {event_type}")
            
            logger.info(f"Session {session_id} completed normally with {session_info['results_received']} results")
            
        except Exception as e:
            logger.error(f"Error in streaming session {session_id}: {e}")
            self.errors += 1
        finally:
            # Mark session as completed
            session_info["active"] = False
            self.session_ends += 1
            
            # CRITICAL FIX: Use stored event loop reference for cleanup
            if self._main_event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._cleanup_session(session_id), 
                    self._main_event_loop
                )
            else:
                logger.error(f"Cannot clean up session {session_id} - no event loop available")
    
    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up completed session after delay to ensure smooth transition."""
        await asyncio.sleep(2.0)  # Wait to ensure no more results are coming
        self.active_sessions = [s for s in self.active_sessions if s["id"] != session_id]
        logger.info(f"Removed completed session {session_id}, {len(self.active_sessions)} sessions active")
    
    def _convert_to_result(self, result: Any, session_id: str) -> Optional[StreamingTranscriptionResult]:
        """
        Convert Google Cloud result to StreamingTranscriptionResult.
        
        Args:
            result: Google Cloud Speech API result
            session_id: Current session ID
            
        Returns:
            StreamingTranscriptionResult or None if invalid
        """
        if not result.alternatives:
            return None
            
        alternative = result.alternatives[0]
        
        # Create result object
        streaming_result = StreamingTranscriptionResult(
            text=alternative.transcript,
            is_final=result.is_final,
            confidence=alternative.confidence if result.is_final else 0.0,
            session_id=session_id,
            start_time=0.0,  # Not provided in streaming responses
            end_time=0.0,    # Not provided in streaming responses
            stability=getattr(result, 'stability', 0.0) if hasattr(result, 'stability') else 0.0
        )
        
        # Add alternative results if available
        if len(result.alternatives) > 1:
            streaming_result.alternatives = [
                {"text": alt.transcript, "confidence": alt.confidence}
                for alt in result.alternatives[1:]
            ]
        
        return streaming_result
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio chunk for all active sessions with result callback.
        
        Args:
            audio_chunk: Audio data chunk
            callback: Optional async callback for results
            
        Returns:
            Final transcription result if available
        """
        if not self.is_streaming:
            logger.warning("Called process_audio_chunk but not streaming")
            return None
            
        if not audio_chunk or len(audio_chunk) == 0:
            logger.warning("Empty audio chunk received")
            return None
        
        # Track metrics
        self.total_audio_chunks += 1
        self.last_audio_time = time.time()
        
        # Add to buffer for smooth transitions
        self.audio_buffer.append(audio_chunk)
        if len(self.audio_buffer) > self.buffer_max_size:
            self.audio_buffer.pop(0)  # Remove oldest chunk
        
        # Add to queue for all active sessions
        self.audio_queue.put(audio_chunk)
        
        # Return value tracking
        final_result = None
        
        # Process any results
        processed = 0
        while not self.result_queue.empty() and processed < 10:  # Limit batch size
            try:
                result = await self.result_queue.get()
                
                # Update metrics
                self.total_results += 1
                self.last_result_time = time.time()
                if result.is_final:
                    self.successful_transcriptions += 1
                    final_result = result  # Store last final result
                
                # Call callback if provided
                if callback:
                    await callback(result)
                
                self.result_queue.task_done()
                processed += 1
                
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                self.errors += 1
        
        return final_result
    
    async def _manage_sessions(self) -> None:
        """
        Continuously manage sessions to ensure uninterrupted streaming.
        
        This task monitors active sessions and starts new ones before
        existing sessions reach their maximum duration limit.
        """
        while self.is_streaming and not self.stop_event.is_set():
            try:
                # Find oldest active session
                oldest_active = None
                for session in self.active_sessions:
                    if session["active"]:
                        if oldest_active is None or session["start_time"] < oldest_active["start_time"]:
                            oldest_active = session
                
                # If oldest active session is approaching timeout, start a new session
                if oldest_active:
                    session_age = time.time() - oldest_active["start_time"]
                    
                    # Start new session before current one expires
                    # The overlap is session_max_duration - session_overlap_seconds
                    if session_age > (self.session_max_duration - self.session_overlap_seconds):
                        logger.info(f"Session {oldest_active['id']} age: {session_age:.1f}s, starting new overlapping session")
                        await self._start_new_session()
                        
                        # Add extra delay after starting a new session
                        await asyncio.sleep(10)
                        continue
                else:
                    # No active sessions, start one
                    logger.warning("No active sessions found, starting new session")
                    await self._start_new_session()
                
                # Check again after delay
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in session management: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _monitor_health(self) -> None:
        """Monitor health of streaming system and recover from issues."""
        while self.is_streaming and not self.stop_event.is_set():
            try:
                # Check for active sessions
                active_count = sum(1 for s in self.active_sessions if s["active"])
                
                if active_count == 0:
                    logger.warning("No active sessions, starting new session")
                    await self._start_new_session()
                
                # Check for session inactivity (no results for too long)
                now = time.time()
                for session in self.active_sessions:
                    if session["active"] and session["last_result_time"]:
                        inactive_time = now - session["last_result_time"]
                        if inactive_time > 30 and session["results_received"] > 0:
                            logger.warning(f"Session {session['id']} inactive for {inactive_time:.1f}s, marking as inactive")
                            session["active"] = False
                
                # Log health stats periodically
                logger.info(f"Health stats: {active_count} active sessions, "
                           f"{self.total_audio_chunks} chunks, {self.total_results} results, "
                           f"{self.successful_transcriptions} transcriptions, {self.errors} errors")
                
                await asyncio.sleep(15)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop all streaming sessions and return final result.
        
        Returns:
            Tuple of (final_text, duration)
        """
        logger.info("Stopping infinite streaming")
        
        # Set stop flag
        self.stop_event.set()
        self.is_streaming = False
        
        # Signal end of audio
        try:
            self.audio_queue.put(None)
        except:
            pass
        
        # Wait briefly for sessions to process remaining audio
        await asyncio.sleep(0.5)
        
        # Get any final results
        final_results = []
        while not self.result_queue.empty():
            try:
                result = await self.result_queue.get_nowait()
                if result and result.is_final:
                    final_results.append(result)
                self.result_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error getting final results: {e}")
        
        # Return best final result if available
        if final_results:
            best_result = max(final_results, key=lambda r: r.confidence if hasattr(r, 'confidence') and r.confidence is not None else 0)
            logger.info(f"Final result: '{best_result.text}' (confidence: {best_result.confidence:.2f})")
            return best_result.text, 0.0
        
        # No final results
        logger.info("No final results available")
        return "", 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats about the infinite streaming system."""
        active_sessions = [s for s in self.active_sessions if s["active"]]
        
        stats = {
            "is_streaming": self.is_streaming,
            "active_sessions": len(active_sessions),
            "total_sessions_started": self.session_starts,
            "total_sessions_ended": self.session_ends,
            "total_audio_chunks": self.total_audio_chunks,
            "total_results": self.total_results,
            "successful_transcriptions": self.successful_transcriptions,
            "errors": self.errors,
            "buffer_size": len(self.audio_buffer),
            "session_max_duration": self.session_max_duration,
            "session_overlap_seconds": self.session_overlap_seconds,
            "current_sessions": [
                {
                    "id": s["id"],
                    "age": time.time() - s["start_time"],
                    "results": s["results_received"],
                    "last_result": time.time() - (s["last_result_time"] or time.time())
                }
                for s in self.active_sessions
            ],
            "event_loop_available": self._main_event_loop is not None
        }
        
        # Add timing information
        if self.last_audio_time:
            stats["time_since_last_audio"] = time.time() - self.last_audio_time
        
        if self.last_result_time:
            stats["time_since_last_result"] = time.time() - self.last_result_time
        
        return stats