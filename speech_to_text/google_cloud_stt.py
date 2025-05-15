"""
Google Cloud Speech-to-Text v2 implementation with proper streaming support.
Fixed for Google Cloud Speech v2 API method signatures.
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
    Handles session limits and automatic reconnection for seamless multi-turn interactions.
    """
    
    # Constants for session management
    STREAMING_LIMIT = 280000  # 4m40s - safely under 5min limit
    CHUNK_TIMEOUT = 30.0      # Timeout for receiving audio chunks
    
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
        """Initialize with telephony settings optimized for continuous conversation."""
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
        
        # State tracking for continuous operation
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
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
        
        # Stats
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.session_count = 0
        
        logger.info(f"Initialized Speech v2 for continuous conversation - Project: {self.project_id}")
    
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
                    speech_start_timeout=cloud_speech.Duration(seconds=8),   # Wait up to 8s for speech to start
                    speech_end_timeout=cloud_speech.Duration(seconds=3)      # Wait 3s after speech ends
                ),
            ),
        )
        
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
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
                    # Signal to start reconnection process
                    self._handle_reconnection()
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
        
        # Reconnect if we're approaching the 5-minute limit
        elapsed_time = (time.time() - self.stream_start_time) * 1000
        return elapsed_time > self.STREAMING_LIMIT
    
    def _handle_reconnection(self):
        """Handle reconnection to avoid the 5-minute limit."""
        if self.reconnection_in_progress:
            return
        
        logger.info("Initiating proactive STT session reconnection")
        self.reconnection_in_progress = True
        
        # Start a new stream in a separate thread
        def reconnect():
            time.sleep(0.1)  # Brief pause
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
            old_stream = self.current_stream
            
            # Update session info
            self.session_count += 1
            self.session_id = str(uuid.uuid4())
            self.stream_start_time = time.time()
            
            # Create new stream with proper v2 API
            self.current_stream = self.client.streaming_recognize(
                requests=self._request_generator()
            )
            
            # Close old stream
            if old_stream:
                try:
                    old_stream.cancel()
                except:
                    pass
            
            logger.info(f"STT session reconnected (session #{self.session_count})")
            
        except Exception as e:
            logger.error(f"Error during STT reconnection: {e}")
            self.reconnection_in_progress = False
    
    def _run_streaming(self):
        """Run streaming in background thread with proper v2 API handling."""
        try:
            logger.info(f"Starting streaming (session: {self.session_id})")
            self.session_count += 1
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
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            if not self.stop_event.is_set() and not self.reconnection_in_progress:
                # Try to reconnect if we haven't been stopped
                logger.info("Attempting to recover from streaming error")
                self._handle_reconnection()
        finally:
            logger.info(f"Streaming ended (session: {self.session_id})")
    
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
                
                # Add to result queue
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.result_queue.put(transcription_result),
                        asyncio.get_event_loop()
                    )
                except asyncio.QueueFull:
                    logger.warning("Result queue full")
                except RuntimeError:
                    # No event loop in current thread, create one
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    asyncio.get_event_loop().run_until_complete(
                        self.result_queue.put(transcription_result)
                    )
                
                if result.is_final:
                    self.successful_transcriptions += 1
                    logger.info(f"Final: '{alternative.transcript}' (conf: {alternative.confidence:.2f})")
    
    async def start_streaming(self) -> None:
        """Start streaming session with continuous conversation support."""
        if self.is_streaming:
            logger.debug("Stream already active, keeping existing session")
            return
        
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
        
        # Get final results
        final_transcript = ""
        duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
        
        final_results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                if result.is_final:
                    final_results.append(result)
            except asyncio.QueueEmpty:
                break
        
        if final_results:
            # Get the most recent final result
            best_result = final_results[-1]
            final_transcript = best_result.text
        
        logger.info(f"Stopped streaming: '{final_transcript}', duration: {duration:.2f}s")
        return final_transcript, duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, bytearray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with automatic session management."""
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
            
            # Check for results
            final_result = None
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if callback:
                        await callback(result)
                    if result.is_final:
                        final_result = result
                except asyncio.QueueEmpty:
                    break
            
            return final_result
            
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