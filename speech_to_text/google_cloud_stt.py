"""
Google Cloud Speech-to-Text v2 implementation for Twilio telephony.
Based on official Google Cloud examples with proper async implementation.
"""
import logging
import asyncio
import time
import os
import json
import queue
import threading
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Iterator
import numpy as np
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
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None

class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text v2 client optimized for telephony.
    Uses proper async/sync boundary management for real-time streaming.
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
        enhanced_model: bool = True
    ):
        """Initialize with optimal settings for Twilio telephony."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.location = location
        
        # Get project ID
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        self.project_id = creds_data.get('project_id')
                        logger.info(f"Auto-extracted project ID: {self.project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials: {e}")
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize client
        self.client = SpeechClient()
        
        # Setup recognizer
        self._setup_recognizer()
        
        # State tracking
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
        self.stream_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.total_chunks = 0
        self.successful_transcriptions = 0
        
        logger.info(f"Initialized Speech v2: {sample_rate}Hz, {encoding}, project: {self.project_id}")
    
    def _setup_recognizer(self):
        """Setup recognizer configuration based on official examples."""
        # Recognizer path
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        
        # Explicit decoding config
        if self.encoding == "MULAW":
            encoding_type = cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW
        else:
            encoding_type = cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        
        # Recognition config (optimized for telephony)
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=self.sample_rate,
                encoding=encoding_type,
                audio_channel_count=self.channels,
            ),
            language_codes=[self.language],
            model="telephony",  # Best for phone calls
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
            ),
        )
        
        # Streaming config
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    speech_start_timeout=duration_pb2.Duration(seconds=5),
                    speech_end_timeout=duration_pb2.Duration(seconds=1),
                ),
            ),
        )
        
        # Initial config request
        self.config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
    
    def _requests_generator(self) -> Iterator[cloud_speech.StreamingRecognizeRequest]:
        """Generate streaming requests following official examples."""
        # Yield initial config
        yield self.config_request
        
        # Yield audio chunks
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:  # End marker
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
    
    def _run_streaming(self):
        """Run streaming recognition in separate thread."""
        try:
            logger.info("Starting streaming recognition thread")
            
            # Create the streaming recognition call
            responses = self.client.streaming_recognize(
                requests=self._requests_generator()
            )
            
            # Process responses
            for response in responses:
                if self.stop_event.is_set():
                    break
                
                # Process results
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result object
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence if result.is_final else 0.0,
                        )
                        
                        # Put result in async queue
                        try:
                            self.result_queue.put_nowait(transcription_result)
                        except asyncio.QueueFull:
                            logger.warning("Result queue full, dropping result")
                        
                        # Log final results
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Final: '{alternative.transcript}' ({alternative.confidence:.2f})")
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
        finally:
            logger.info("Streaming recognition thread ended")
    
    async def start_streaming(self) -> None:
        """Start streaming session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        self.is_streaming = True
        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._run_streaming, daemon=True)
        self.stream_thread.start()
        
        logger.info("Started streaming session")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.stop_event.set()
        
        # Signal end of audio
        try:
            self.audio_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        logger.info("Stopped streaming session")
        return "", 0.0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk."""
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert numpy to bytes if needed
            if isinstance(audio_chunk, np.ndarray):
                if audio_chunk.dtype == np.float32:
                    import audioop
                    pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
                    audio_bytes = audioop.lin2ulaw(pcm_data, 2)
                else:
                    audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Skip tiny chunks
            if len(audio_bytes) < 40:
                return None
            
            # Add to queue
            try:
                self.audio_queue.put(audio_bytes, timeout=0.1)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
                return None
            
            # Check for results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if callback:
                        await callback(result)
                    if result.is_final:
                        return result
                except asyncio.QueueEmpty:
                    break
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        return {
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "telephony",
            "api_version": "v2",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "project_id": self.project_id,
            "location": self.location,
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2)
        }