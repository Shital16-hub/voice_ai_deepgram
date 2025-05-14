"""
Google Cloud Speech-to-Text v2 implementation optimized for telephony with proper streaming.
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
    Google Cloud Speech-to-Text v2 client optimized for telephony.
    Uses a ping-based approach to keep the stream alive and handle intermittent audio.
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
        """
        Initialize with optimal settings for telephony using v2 streaming API.
        """
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
        
        # Initialize v2 client with proper endpoint
        self._initialize_client()
        
        # Construct recognizer path for v2 API
        self.recognizer_path = self.client.recognizer_path(
            project=self.project_id,
            location=location,
            recognizer=recognizer_id
        )
        
        # Streaming state
        self.is_streaming = False
        self.streaming_call = None
        self.request_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.result_queue = queue.Queue()
        self.response_thread = None
        self.sender_thread = None
        self.keep_alive_thread = None
        
        # Performance tracking
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.last_result = None
        self.callback_function = None
        
        # Keep alive mechanism
        self.last_audio_time = 0
        self.keep_alive_interval = 5.0  # Send keep-alive every 5 seconds
        
        logger.info(f"Initialized Google Cloud STT v2: {sample_rate}Hz, {encoding}, project: {project_id}")
    
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
        """Get optimized recognition configuration for telephony."""
        # Create explicit decoding config for the audio format
        if self.encoding == "MULAW":
            encoding_enum = speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW
        elif self.encoding == "LINEAR16":
            encoding_enum = speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16
        else:
            encoding_enum = speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW
        
        decoding_config = speech_v2.ExplicitDecodingConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels,
        )
        
        # Create advanced recognition features
        features = speech_v2.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            enable_spoken_punctuation=False,
            enable_spoken_emojis=False,
            max_alternatives=1,
            profanity_filter=False,
        )
        
        # Select optimal model for telephony
        if self.enhanced_model:
            model = "telephony_short"  # Best for short telephony utterances
        else:
            model = "latest_short"
        
        # Create recognition config
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=decoding_config,
            language_codes=[self.language],
            model=model,
            features=features,
        )
        
        return config
    
    def _get_streaming_config(self) -> speech_v2.StreamingRecognitionConfig:
        """Get streaming configuration optimized for real-time telephony."""
        config = speech_v2.StreamingRecognitionConfig(
            config=self._get_recognition_config(),
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,  # Required for voice activity timeouts
                voice_activity_timeout=speech_v2.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    speech_start_timeout=Duration(seconds=5),
                    speech_end_timeout=Duration(seconds=1),
                ),
            ),
        )
        return config
    
    def _request_sender(self):
        """Send requests to the streaming call in a separate thread."""
        try:
            # First request - configuration
            config_request = speech_v2.StreamingRecognizeRequest(
                recognizer=self.recognizer_path,
                streaming_config=self._get_streaming_config(),
            )
            self.streaming_call.send(config_request)
            logger.debug("Sent configuration request")
            
            # Continuous loop to send audio and keep-alive
            while self.is_streaming:
                try:
                    # Try to get audio chunk
                    audio_chunk = self.request_queue.get(timeout=1.0)
                    
                    if audio_chunk is None:  # Sentinel to stop
                        break
                    
                    # Create audio request
                    audio_request = speech_v2.StreamingRecognizeRequest(audio=audio_chunk)
                    self.streaming_call.send(audio_request)
                    self.last_audio_time = time.time()
                    
                    self.request_queue.task_done()
                    logger.debug(f"Sent audio chunk: {len(audio_chunk)} bytes")
                    
                except queue.Empty:
                    # No audio available - send keep-alive if needed
                    current_time = time.time()
                    if current_time - self.last_audio_time > self.keep_alive_interval:
                        # Send empty audio chunk as keep-alive
                        try:
                            keep_alive = speech_v2.StreamingRecognizeRequest(audio=b'')
                            self.streaming_call.send(keep_alive)
                            self.last_audio_time = current_time
                            logger.debug("Sent keep-alive")
                        except Exception as e:
                            logger.error(f"Error sending keep-alive: {e}")
                    continue
                    
                except Exception as e:
                    logger.error(f"Error sending request: {e}")
                    break
            
            # Close the send side
            self.streaming_call.close()
            logger.debug("Request sender finished")
            
        except Exception as e:
            logger.error(f"Error in request sender: {e}")
    
    def _response_processor(self):
        """Process streaming responses in a separate thread."""
        try:
            for response in self.streaming_call:
                # Process each result in the response
                for result in response.results:
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
                        self.result_queue.put(transcription_result)
                        
                        # Log results
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Final transcription: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
                        else:
                            logger.debug(f"Interim: '{alternative.transcript}'")
                        
                        # Call callback if provided
                        if self.callback_function and result.is_final:
                            try:
                                # Run callback in event loop if available
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self.callback_function(transcription_result),
                                        loop
                                    )
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error in response processor: {e}")
        except Exception as e:
            logger.error(f"Error processing responses: {e}")
        finally:
            logger.debug("Response processor finished")
    
    async def start_streaming(self) -> None:
        """Start a new streaming session with proper v2 implementation."""
        if self.is_streaming:
            await self.stop_streaming()
        
        try:
            # Clear any existing state
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                    self.request_queue.task_done()
                except queue.Empty:
                    break
            
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.task_done()
                except queue.Empty:
                    break
            
            # Start streaming
            self.is_streaming = True
            self.last_audio_time = time.time()
            
            # Create the streaming call
            self.streaming_call = self.client.streaming_recognize(
                iter([])  # We'll send requests manually
            )
            
            # Start sender thread
            self.sender_thread = threading.Thread(
                target=self._request_sender,
                daemon=True
            )
            self.sender_thread.start()
            
            # Start response processing thread
            self.response_thread = threading.Thread(
                target=self._response_processor,
                daemon=True
            )
            self.response_thread.start()
            
            logger.info("Started v2 streaming session with separate sender and receiver threads")
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.is_streaming = False
            raise
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio chunk with optimized buffering for v2 API.
        """
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        self.callback_function = callback
        
        # Convert numpy array to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            if audio_chunk.dtype == np.float32:
                # Convert float32 to MULAW for telephony
                import audioop
                # First convert to 16-bit PCM
                pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
                audio_bytes = audioop.lin2ulaw(pcm_data, 2)
            else:
                audio_bytes = audio_chunk.tobytes()
        else:
            audio_bytes = audio_chunk
        
        # Skip tiny chunks
        if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
            return None
        
        # Send to streaming API
        try:
            self.request_queue.put(audio_bytes, block=False)
            logger.debug(f"Queued {len(audio_bytes)} bytes for streaming")
        except queue.Full:
            logger.warning("Request queue is full, dropping audio chunk")
            # Try to clear some old items
            try:
                self.request_queue.get_nowait()
                self.request_queue.task_done()
                self.request_queue.put(audio_bytes, block=False)
            except:
                pass
        
        # Check for any results
        try:
            # Non-blocking check for results
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.result_queue.task_done()
                if result.is_final:
                    return result
        except queue.Empty:
            pass
        
        return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session and get final transcription."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info("Stopping v2 streaming session")
        
        try:
            # Stop the streaming flag
            self.is_streaming = False
            
            # Send sentinel to stop request sender
            try:
                self.request_queue.put(None, block=False)
            except queue.Full:
                pass
            
            # Wait for threads to finish
            if self.sender_thread and self.sender_thread.is_alive():
                self.sender_thread.join(timeout=2.0)
            
            if self.response_thread and self.response_thread.is_alive():
                self.response_thread.join(timeout=2.0)
            
            # Close streaming call
            if self.streaming_call:
                try:
                    self.streaming_call.cancel()
                except:
                    pass
            
            # Collect any final results
            final_text = ""
            duration = 0.0
            
            # Process any remaining results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    self.result_queue.task_done()
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
            
            return final_text, duration
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return "", 0.0
        finally:
            # Reset state
            self.streaming_call = None
            self.response_thread = None
            self.sender_thread = None
            self.last_result = None
            self.callback_function = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        
        return {
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2),
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "telephony_short" if self.enhanced_model else "latest_short",
            "api_version": "v2",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "project_id": self.project_id,
            "location": self.location,
            "request_queue_size": self.request_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "last_audio_time": self.last_audio_time
        }