# speech_to_text/google_cloud_stt.py

"""
Fixed Google Cloud Speech-to-Text v2 implementation for Twilio with proper recognizer configuration.
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
    Fixed Google Cloud Speech-to-Text v2 client with proper Twilio integration.
    Key fix: Using recognizer resource instead of default recognizer.
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
        """Initialize Google Cloud STT v2 with proper Twilio configuration."""
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
        
        # CRITICAL FIX: Create a proper recognizer instead of using default
        # The "_" recognizer often has issues with Twilio
        self.recognizer_path = self._create_or_get_recognizer()
        
        # Streaming state
        self.is_streaming = False
        self._streaming_responses = None
        self._stream_thread = None
        self._stream_queue = queue.Queue(maxsize=50)  # Reduced queue size
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.last_result = None
        
        # Audio validation
        self.min_chunk_size = 320   # 40ms at 8kHz (reduced)
        self.max_chunk_size = 20480 # 20KB limit
        
        logger.info(f"Initialized Google Cloud STT v2: {sample_rate}Hz, {encoding}, project: {self.project_id}")
        logger.info(f"Using recognizer: {self.recognizer_path}")
    
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
    
    def _create_or_get_recognizer(self) -> str:
        """Create or get a proper recognizer for Twilio."""
        # For Twilio, we need to create a specific recognizer optimized for telephony
        recognizer_name = f"twilio-telephony-{int(time.time())}"
        recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/{recognizer_name}"
        
        try:
            # Create recognition config
            recognition_config = self._get_recognition_config()
            
            # Create recognizer request
            recognizer = speech_v2.Recognizer(
                name=recognizer_path,
                language_codes=[self.language],
                model=recognition_config.model,
                default_recognition_config=recognition_config
            )
            
            # Create the recognizer
            create_request = speech_v2.CreateRecognizerRequest(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                recognizer_id=recognizer_name,
                recognizer=recognizer
            )
            
            logger.info(f"Creating new recognizer for Twilio: {recognizer_path}")
            operation = self.client.create_recognizer(request=create_request)
            
            # Wait for operation to complete
            recognizer_result = operation.result(timeout=30)
            logger.info(f"Created recognizer: {recognizer_result.name}")
            
            return recognizer_result.name
            
        except Exception as e:
            logger.warning(f"Could not create custom recognizer, using default: {e}")
            # Fall back to default recognizer with proper path
            return f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
    
    def _get_recognition_config(self) -> speech_v2.RecognitionConfig:
        """Get recognition configuration optimized for Twilio MULAW."""
        
        # Create explicit decoding config for MULAW
        explicit_config = speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels,
        )
        
        # Create recognition features - minimal for better performance
        features = speech_v2.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,  # Disabled for performance
            enable_word_confidence=False,    # Disabled for performance  
            profanity_filter=False,
            max_alternatives=1,
        )
        
        # CRITICAL: Use the correct model for Twilio telephony
        # telephony_short is specifically designed for 8kHz MULAW audio
        model = "telephony_short"
        
        # Create recognition config
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=explicit_config,
            language_codes=[self.language],
            model=model,
            features=features,
        )
        
        return config
    
    def _get_streaming_config(self) -> speech_v2.StreamingRecognitionConfig:
        """Get streaming configuration optimized for Twilio."""
        
        # Get recognition config
        recognition_config = self._get_recognition_config()
        
        # Create streaming config with optimized settings
        config = speech_v2.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                # No voice activity events for Twilio (causes issues)
                enable_voice_activity_events=False,
                # Use shorter timeout for better responsiveness
                end_pointer_events=speech_v2.StreamingRecognitionFeatures.EndpointerEvents.END_OF_UTTERANCE
            ),
        )
        return config
    
    def _validate_audio_chunk(self, audio_chunk: bytes) -> bool:
        """Validate audio chunk before sending to STT."""
        if not audio_chunk:
            return False
        
        # Check size limits
        if len(audio_chunk) < self.min_chunk_size:
            logger.debug(f"Audio chunk too small: {len(audio_chunk)} bytes")
            return False
            
        if len(audio_chunk) > self.max_chunk_size:
            logger.warning(f"Audio chunk too large: {len(audio_chunk)} bytes")
            return False
        
        # MULAW validation - check for variation
        sample = audio_chunk[:min(100, len(audio_chunk))]
        non_zero_count = sum(1 for b in sample if b != 0)
        if non_zero_count < len(sample) * 0.1:
            logger.debug("Audio chunk appears to be silence")
            return False
            
        # Check for reasonable MULAW distribution
        mean_val = sum(sample) / len(sample)
        if mean_val < 50 or mean_val > 200:
            logger.debug(f"Unusual MULAW distribution, mean: {mean_val}")
            # Don't reject, just log
        
        return True
    
    def _generate_requests(self):
        """Generate streaming recognition requests."""
        # First request - configuration only
        config_request = speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self._get_streaming_config(),
        )
        
        logger.info(f"Sending config request: recognizer={self.recognizer_path}")
        logger.info(f"Config: model=telephony_short, encoding=MULAW, rate={self.sample_rate}")
        yield config_request
        
        # Subsequent requests - audio data only
        while not self._stop_event.is_set():
            try:
                # Get audio chunk with shorter timeout for responsiveness
                audio_chunk = self._stream_queue.get(timeout=0.5)
                
                if audio_chunk is None:  # Sentinel to stop
                    break
                
                # Validate audio chunk
                if not self._validate_audio_chunk(audio_chunk):
                    logger.debug(f"Skipping invalid audio chunk: {len(audio_chunk)} bytes")
                    self._stream_queue.task_done()
                    continue
                
                # Log first few bytes for debugging
                if len(audio_chunk) >= 10:
                    sample_bytes = list(audio_chunk[:10])
                    logger.debug(f"Sending MULAW audio: {len(audio_chunk)} bytes, "
                               f"sample: {sample_bytes}")
                
                # Create audio request with raw MULAW bytes
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
            logger.info("Starting streaming recognition thread...")
            
            # Create streaming recognition call
            self._streaming_responses = self.client.streaming_recognize(
                self._generate_requests()
            )
            
            logger.info("Started streaming recognition, waiting for responses...")
            
            # Process responses
            response_count = 0
            result_count = 0
            
            for response in self._streaming_responses:
                if self._stop_event.is_set():
                    break
                
                response_count += 1
                logger.debug(f"Received response #{response_count}")
                
                # Check for error in response
                if hasattr(response, 'error') and response.error:
                    logger.error(f"STT API error: {response.error}")
                    continue
                
                # Process each result in the response
                for result_idx, result in enumerate(response.results):
                    result_count += 1
                    logger.info(f"Processing result {result_idx}: is_final={result.is_final}, "
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
                            logger.info(f"FINAL transcription: '{alternative.transcript}' "
                                       f"(confidence: {alternative.confidence:.2f})")
                        else:
                            logger.debug(f"Interim: '{alternative.transcript}'")
                    else:
                        logger.warning(f"Result {result_idx} has no alternatives")
            
            logger.info(f"Finished streaming recognition - responses: {response_count}, "
                       f"results: {result_count}")
                        
        except grpc.RpcError as e:
            logger.error(f"gRPC error in streaming: {e}")
            logger.error(f"gRPC details: {e.details()}")
            logger.error(f"gRPC code: {e.code()}")
            
            # Check for common errors
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                logger.error("Invalid argument - check audio format and recognizer configuration")
            elif e.code() == grpc.StatusCode.PERMISSION_DENIED:
                logger.error("Permission denied - check your Google Cloud credentials and permissions")
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                logger.error("Resource exhausted - you may have exceeded quota limits")
                
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}", exc_info=True)
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
            
            logger.info("Started v2 streaming session with telephony_short model")
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.is_streaming = False
            raise
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk - expects raw MULAW bytes from Twilio."""
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        # Ensure we have raw bytes (MULAW audio should already be bytes)
        if isinstance(audio_chunk, np.ndarray):
            # Convert numpy array to bytes if needed
            audio_bytes = audio_chunk.tobytes()
        else:
            # Should already be raw MULAW bytes from Twilio
            audio_bytes = audio_chunk
        
        # Validate and send to queue
        if self._validate_audio_chunk(audio_bytes):
            try:
                self._stream_queue.put(audio_bytes, block=False)
                logger.debug(f"Queued {len(audio_bytes)} bytes of MULAW audio")
            except queue.Full:
                logger.warning("Stream queue full, dropping audio chunk")
                # Clear some older items from queue
                try:
                    for _ in range(5):
                        self._stream_queue.get_nowait()
                        self._stream_queue.task_done()
                    self._stream_queue.put(audio_bytes, block=False)
                except (queue.Empty, queue.Full):
                    pass
        
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
            
            logger.info(f"Session ended. Processed {self.total_chunks} chunks, "
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
            "recognizer_path": self.recognizer_path,
            "stream_queue_size": self._stream_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            # Try to delete the custom recognizer if we created one
            if hasattr(self, 'recognizer_path') and 'twilio-telephony' in self.recognizer_path:
                try:
                    delete_request = speech_v2.DeleteRecognizerRequest(
                        name=self.recognizer_path
                    )
                    self.client.delete_recognizer(request=delete_request)
                    logger.info(f"Deleted recognizer: {self.recognizer_path}")
                except Exception as e:
                    logger.debug(f"Could not delete recognizer: {e}")
        except Exception:
            pass