# speech_to_text/google_cloud_stt.py

"""
Google Cloud Speech-to-Text v2 implementation following official documentation.
Fixed MULAW encoding support for telephony streaming.
"""
import logging
import asyncio
import time
import os
import json
import threading
import queue
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, AsyncIterator
import numpy as np
from dataclasses import dataclass

# Import Speech-to-Text v2 API
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from google.protobuf.duration_pb2 import Duration
import grpc
from concurrent.futures import ThreadPoolExecutor

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
    alternatives: List[str] = None

class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text v2 client optimized for MULAW telephony audio.
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
        enhanced_model: bool = True,
        recognizer_id: str = "_"
    ):
        """Initialize Google Cloud STT v2 with explicit MULAW support."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.location = location
        self.recognizer_id = recognizer_id
        
        # Get project ID
        self.project_id = project_id or self._get_project_id()
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize v2 client
        self._initialize_client()
        
        # Construct recognizer path for v2 API
        self.recognizer_path = f"projects/{self.project_id}/locations/{location}/recognizers/{recognizer_id}"
        
        # Streaming state
        self.is_streaming = False
        self._streaming_responses = None
        self._stream_thread = None
        self._stream_queue = queue.Queue(maxsize=100)
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.last_result = None
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Initialized Google Cloud STT v2: {sample_rate}Hz, {encoding}, project: {self.project_id}")
    
    def _get_project_id(self) -> Optional[str]:
        """Auto-extract project ID from environment or credentials."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Auto-extracted project ID: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        return None
    
    def _initialize_client(self):
        """Initialize v2 client with proper configuration."""
        # Set up client options for location if not global
        if self.location != "global":
            endpoint = f"{self.location}-speech.googleapis.com"
            client_options = ClientOptions(api_endpoint=endpoint)
        else:
            client_options = None
        
        # Initialize the client
        self.client = speech_v2.SpeechClient(client_options=client_options)
        logger.info(f"Initialized v2 client for location: {self.location}")
    
    def _get_recognition_config(self) -> speech_v2.RecognitionConfig:
        """Get recognition configuration with explicit MULAW decoding."""
        
        # Use ExplicitDecodingConfig for MULAW - more reliable than auto-detect
        explicit_config = speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels,
        )
        
        # Create advanced recognition features  
        features = speech_v2.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            profanity_filter=False,
            max_alternatives=1,
        )
        
        # Use telephony_short model for 8kHz MULAW (better for telephony)
        model = "telephony_short"  # Better than latest_long for 8kHz MULAW
        
        # Create recognition config with explicit decoding
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=explicit_config,  # Use explicit instead of auto
            language_codes=[self.language],
            model=model,
            features=features,
        )
        
        return config
    
    def _get_streaming_config(self) -> speech_v2.StreamingRecognitionConfig:
        """Get streaming configuration optimized for telephony - simplified version."""
        
        # Create simplified streaming config without voice activity timeouts
        # This avoids the voice activity event requirement
        config = speech_v2.StreamingRecognitionConfig(
            config=self._get_recognition_config(),
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                # Remove voice activity timeouts to avoid the error
            ),
        )
        return config
    
    def _generate_requests(self):
        """Generate streaming recognition requests based on Google Cloud documentation."""
        # First request - configuration only
        config_request = speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self._get_streaming_config(),
        )
        
        logger.info(f"Sending initial config request with recognizer: {self.recognizer_path}")
        logger.info(f"Config: model=telephony_short, encoding=MULAW, sample_rate={self.sample_rate}")
        yield config_request
        
        # Subsequent requests - audio data only
        while not self._stop_event.is_set():
            try:
                # Get audio chunk with timeout
                audio_chunk = self._stream_queue.get(timeout=1.0)
                
                if audio_chunk is None:  # Sentinel to stop
                    break
                
                # Log details about the audio chunk being sent
                logger.debug(f"Sending audio chunk: {len(audio_chunk)} bytes")
                if len(audio_chunk) > 0:
                    # Sample the first few bytes for debugging
                    sample_bytes = audio_chunk[:min(10, len(audio_chunk))]
                    logger.debug(f"Audio sample bytes: {list(sample_bytes)}")
                
                # Create audio request - just the audio bytes
                audio_request = speech_v2.StreamingRecognizeRequest(audio=audio_chunk)
                yield audio_request
                
                self._stream_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error generating request: {e}")
                break
    
    def _stream_recognition(self):
        """Run streaming recognition in a separate thread."""
        try:
            # Create streaming recognition call
            logger.info("Creating streaming recognize call...")
            self._streaming_responses = self.client.streaming_recognize(
                self._generate_requests()
            )
            
            logger.info("Started streaming recognition, waiting for responses...")
            
            # Process responses
            response_count = 0
            for response in self._streaming_responses:
                if self._stop_event.is_set():
                    break
                
                response_count += 1
                logger.debug(f"Received response #{response_count} with {len(response.results)} results")
                
                # Process each result in the response
                for result_idx, result in enumerate(response.results):
                    logger.debug(f"Processing result {result_idx}: is_final={result.is_final}, "
                                f"alternatives={len(result.alternatives)}")
                    
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create transcription result
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence if result.is_final else 0.0,
                            chunk_id=self.total_chunks
                        )
                        
                        # Store latest result
                        self.last_result = transcription_result
                        
                        # Put in result queue
                        self._result_queue.put(transcription_result)
                        
                        # Log results
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Final transcription: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
                        else:
                            logger.debug(f"Interim: '{alternative.transcript}'")
                    else:
                        logger.debug("Received result with no alternatives")
            
            logger.info(f"Finished processing streaming responses, total responses: {response_count}")
                        
        except grpc.RpcError as e:
            logger.error(f"gRPC error in streaming recognition: {e}")
            logger.error(f"gRPC error details: {e.details()}")
            logger.error(f"gRPC error code: {e.code()}")
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logger.debug("Streaming recognition thread finished")
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        try:
            # Clear queues
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                    self._stream_queue.task_done()
                except queue.Empty:
                    break
            
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                    self._result_queue.task_done()
                except queue.Empty:
                    break
            
            # Reset state
            self.is_streaming = True
            self._stop_event.clear()
            
            # Start streaming thread
            self._stream_thread = threading.Thread(
                target=self._stream_recognition,
                daemon=True
            )
            self._stream_thread.start()
            
            logger.info("Started v2 streaming session with telephony_short model for MULAW")
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.is_streaming = False
            raise
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with proper MULAW handling."""
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        # Handle different input types for MULAW audio
        if isinstance(audio_chunk, np.ndarray):
            # If numpy array, assume it's already MULAW encoded bytes as uint8
            if audio_chunk.dtype == np.uint8:
                audio_bytes = audio_chunk.tobytes()
            else:
                # If float32, it should already be MULAW converted by telephony layer
                # Just convert to bytes
                audio_bytes = audio_chunk.astype(np.uint8).tobytes()
        else:
            # Already bytes - should be MULAW encoded
            audio_bytes = audio_chunk
        
        # Follow Google's 25KB limit per chunk
        MAX_CHUNK_SIZE = 24000  # Keep under 25KB limit
        
        if len(audio_bytes) > MAX_CHUNK_SIZE:
            logger.warning(f"Audio chunk too large: {len(audio_bytes)} bytes, splitting")
            # Split large chunks
            for i in range(0, len(audio_bytes), MAX_CHUNK_SIZE):
                chunk = audio_bytes[i:i+MAX_CHUNK_SIZE]
                if chunk:
                    try:
                        self._stream_queue.put(chunk, block=False)
                    except queue.Full:
                        logger.warning("Stream queue full, dropping chunk")
        else:
            # Send normal-sized chunk
            try:
                self._stream_queue.put(audio_bytes, block=False)
                logger.debug(f"Queued {len(audio_bytes)} bytes for streaming (MULAW)")
            except queue.Full:
                logger.warning("Stream queue is full, dropping audio chunk")
        
        # Process any results
        final_result = None
        while not self._result_queue.empty():
            try:
                result = self._result_queue.get_nowait()
                self._result_queue.task_done()
                
                if callback:
                    await callback(result)
                
                if result.is_final:
                    final_result = result
            except queue.Empty:
                break
        
        return final_result
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session and get final transcription."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info("Stopping v2 streaming session")
        
        try:
            # Signal stop
            self._stop_event.set()
            
            # Send sentinel to stop request generator
            try:
                self._stream_queue.put(None, block=False)
            except queue.Full:
                pass
            
            # Wait for streaming thread to finish
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=2.0)
            
            # Cancel streaming call
            if self._streaming_responses:
                try:
                    self._streaming_responses.cancel()
                except:
                    pass
                self._streaming_responses = None
            
            # Collect any final results
            final_text = ""
            duration = 0.0
            
            # Process any remaining results
            while not self._result_queue.empty():
                try:
                    result = self._result_queue.get_nowait()
                    self._result_queue.task_done()
                    if result.is_final and result.text:
                        final_text = result.text
                        duration = result.end_time - result.start_time
                except queue.Empty:
                    break
            
            # If no final results, check the last result
            if not final_text and self.last_result and self.last_result.text:
                final_text = self.last_result.text
                duration = 0.0
            
            logger.info(f"Streaming session ended. Processed {self.total_chunks} chunks, "
                       f"{self.successful_transcriptions} successful transcriptions")
            
            # Reset state
            self.is_streaming = False
            self._stream_thread = None
            
            return final_text, duration
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return "", 0.0
        finally:
            # Ensure state is reset
            self.is_streaming = False
            self._streaming_responses = None
            self._stream_thread = None
            self.last_result = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        
        return {
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2),
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "telephony_short",
            "api_version": "v2",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "project_id": self.project_id,
            "location": self.location,
            "stream_queue_size": self._stream_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._executor:
            self._executor.shutdown(wait=False)