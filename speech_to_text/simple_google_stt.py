"""
Fixed Google Cloud Speech client with corrected streaming API usage.
"""
import logging
import asyncio
import queue
import threading
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass

try:
    from google.cloud import speech
    from google.api_core import retry as google_retry
    from google.api_core import exceptions as google_exceptions
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

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
    """Google Cloud Speech client optimized for Twilio with direct MULAW support."""
    
    def __init__(
        self, 
        language="en-US", 
        sample_rate=8000,  # 8kHz for Twilio
        encoding="MULAW",  # MULAW for Twilio
        channels=1, 
        interim_results=True,
        enhanced_model=True,
        speech_context_phrases=None
    ):
        """Initialize the Google Cloud Speech client for MULAW support."""
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech API not available")
        
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.speech_context_phrases = speech_context_phrases or [
            "pricing", "plan", "cost", "subscription", "service", "features",
            "support", "upgrade", "payment", "account", "question", "help"
        ]
        
        # Create client
        try:
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech client created successfully with MULAW support")
        except Exception as e:
            logger.error(f"Failed to create Google Cloud Speech client: {e}")
            raise
        
        # State tracking
        self.is_streaming = False
        self.utterance_id = 0
        
        # For improved streaming
        self.audio_queue = queue.Queue(maxsize=50)  # Limit queue size
        self.result_queue = asyncio.Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self._current_loop = None
    
    def _get_config(self) -> speech.RecognitionConfig:
        """Get the recognition config optimized for MULAW telephony."""
        try:
            # Set encoding
            if self.encoding == "MULAW":
                encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
            else:
                encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
            
            config = speech.RecognitionConfig(
                encoding=encoding_enum,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language,
                enable_automatic_punctuation=True,
                use_enhanced=self.enhanced_model,
                model="telephony",  # Optimized for phone calls
                audio_channel_count=self.channels,
                enable_word_time_offsets=True,
                profanity_filter=False,
            )
            
            # Add speech contexts for better recognition
            if self.speech_context_phrases:
                speech_context = speech.SpeechContext(
                    phrases=self.speech_context_phrases,
                    boost=15.0  # High boost for telephony
                )
                config.speech_contexts.append(speech_context)
            
            return config
        except Exception as e:
            logger.error(f"Error creating recognition config: {e}")
            raise
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get the streaming recognition config."""
        recognition_config = self._get_config()
        
        return speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            single_utterance=False
        )
    
    def _process_audio_loop(self):
        """Process audio chunks in a separate thread using correct API."""
        while not self.stop_processing.is_set():
            try:
                # Get audio chunk with timeout
                audio_bytes = self.audio_queue.get(timeout=1.0)
                
                # Create streaming config
                streaming_config = self._get_streaming_config()
                
                # Create requests generator - CRITICAL FIX HERE
                def request_generator():
                    # First request: send config
                    yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                    # Second request: send audio
                    yield speech.StreamingRecognizeRequest(audio_content=audio_bytes)
                
                # FIXED: Pass the generator function result, not the function itself
                responses = self.client.streaming_recognize(request_generator())
                
                # Process responses
                for response in responses:
                    for result in response.results:
                        if result.alternatives:
                            alt = result.alternatives[0]
                            
                            # Create result object
                            self.utterance_id += 1
                            transcription_result = StreamingTranscriptionResult(
                                text=alt.transcript,
                                is_final=result.is_final,
                                confidence=alt.confidence if hasattr(alt, "confidence") else 0.8,
                                chunk_id=self.utterance_id
                            )
                            
                            # Put result in queue safely
                            try:
                                if self._current_loop and not self._current_loop.is_closed():
                                    future = asyncio.run_coroutine_threadsafe(
                                        self.result_queue.put(transcription_result),
                                        self._current_loop
                                    )
                                    future.result(timeout=1.0)
                            except Exception as e:
                                logger.debug(f"Error putting result in queue: {e}")
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                continue
    
    async def start_streaming(self):
        """Start streaming session using improved approach."""
        if self.is_streaming:
            await self.stop_streaming()
        
        self.is_streaming = True
        self.stop_processing.clear()
        self._current_loop = asyncio.get_event_loop()
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                await self.result_queue.get()
                self.result_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Started Google Cloud Speech streaming session with MULAW")
    
    async def process_audio_chunk(
        self, 
        audio_chunk, 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ):
        """Process a chunk of audio using the improved queue-based approach."""
        if not self.is_streaming:
            logger.warning("Called process_audio_chunk but streaming is not active")
            await self.start_streaming()
            return None
        
        try:
            # Convert audio chunk to proper format
            if isinstance(audio_chunk, np.ndarray):
                if self.encoding == "MULAW":
                    # For MULAW, we need uint8 bytes, with proper offset
                    # MULAW is unsigned 8-bit, range 0-255
                    audio_bytes = (audio_chunk * 127 + 128).clip(0, 255).astype(np.uint8).tobytes()
                else:
                    # For LINEAR16, we need int16 bytes
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Check minimum chunk size
            if len(audio_bytes) < 320:  # Less than 40ms at 8kHz
                logger.debug(f"Audio chunk too small: {len(audio_bytes)} bytes")
                return None
            
            # Add to processing queue with timeout
            try:
                self.audio_queue.put(audio_bytes, timeout=0.1)
            except queue.Full:
                logger.warning("Audio queue is full, dropping chunk")
                # Remove oldest item and try again
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                    self.audio_queue.put(audio_bytes, timeout=0.1)
                except (queue.Empty, queue.Full):
                    return None
            
            # Check for results with very short timeout
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.05)
                
                if callback:
                    await callback(result)
                
                self.result_queue.task_done()
                
                # Return final results
                if result.is_final:
                    return result
                
                return None
                
            except asyncio.TimeoutError:
                # No results available yet
                return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        self.stop_processing.set()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
            
        # Process any remaining audio in the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Get any remaining results
        final_text = ""
        duration = 0.0
        
        while not self.result_queue.empty():
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                if result.is_final and result.text:
                    final_text = result.text
                    duration = result.end_time - result.start_time if result.end_time > result.start_time else 0.0
                self.result_queue.task_done()
            except asyncio.TimeoutError:
                break
            except:
                break
        
        self._current_loop = None
        logger.info(f"Stopped Google Cloud Speech streaming session. Final text: '{final_text}'")
        return final_text, duration
    
    async def transcribe_file(
        self,
        file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Transcribe an audio file."""
        try:
            # Read audio file based on encoding
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Create RecognitionAudio
            audio = speech.RecognitionAudio(content=content)
            
            # Get config
            config = self._get_config()
            
            # Perform recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Process results
            transcription = ""
            confidence = 0.0
            
            for result in response.results:
                for alt in result.alternatives:
                    transcription += alt.transcript + " "
                    confidence = max(confidence, alt.confidence)
            
            transcription = transcription.strip()
            
            return {
                "transcription": transcription,
                "confidence": confidence,
                "is_final": True
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return {
                "error": str(e),
                "transcription": "",
                "confidence": 0.0,
                "is_final": True
            }

# Alias for backward compatibility
class SimpleGoogleSTT(GoogleCloudStreamingSTT):
    """Alias for backward compatibility."""
    pass