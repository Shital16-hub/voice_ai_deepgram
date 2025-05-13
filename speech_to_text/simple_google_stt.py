# speech_to_text/simple_google_stt.py

"""
Enhanced Google Cloud Speech client with proper telephony configuration.
Addresses core issues without requiring major import changes.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass
import io
import time
import audioop

try:
    from google.cloud import speech
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

class SimpleGoogleSTT:
    """
    Enhanced Google Cloud Speech client optimized for telephony.
    """
    
    def __init__(
        self, 
        language_code="en-US", 
        sample_rate=8000,
        enable_automatic_punctuation=True,
        enhanced_telephony=True
    ):
        """Initialize the Google Cloud Speech client with optimized settings."""
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech API not available")
        
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enhanced_telephony = enhanced_telephony
        
        # Create client
        try:
            self.client = speech.SpeechClient()
            logger.info("Successfully created Google Cloud Speech client")
        except Exception as e:
            logger.error(f"Failed to create Google Cloud Speech client: {e}")
            raise
        
        # State tracking
        self.is_streaming = False
        self.utterance_id = 0
        
        # Enhanced audio accumulation strategy
        self.audio_buffer = bytearray()
        self.min_buffer_size = 6400    # 800ms at 8kHz - much better than 3200
        self.max_buffer_size = 32000   # 4 seconds max
        
        # Quality tracking
        self.total_chunks = 0
        self.successful_recognitions = 0
        
        logger.info(f"Initialized Google Cloud STT: {sample_rate}Hz, enhanced={enhanced_telephony}")
        logger.info(f"Buffer strategy: min={self.min_buffer_size}, max={self.max_buffer_size}")
    
    def _get_config(self):
        """Get the recognition config with enhanced telephony optimization."""
        try:
            # Enhanced configuration for telephony
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                model="phone_call",  # Best model for telephony
                use_enhanced=True,  # Use premium model for better accuracy
                audio_channel_count=1,
                enable_spoken_punctuation=True,  # Better for natural speech
                enable_spoken_digits=True,      # Better for numbers
                enable_word_time_offsets=False, # Disable for lower latency
                max_alternatives=1,             # Single best result
                profanity_filter=False,
                # Add telephony-specific metadata
                metadata=speech.RecognitionMetadata(
                    interaction_type=speech.RecognitionMetadata.InteractionType.VOICE_SEARCH,
                    microphone_distance=speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                    recording_device_type=speech.RecognitionMetadata.RecordingDeviceType.PHONE_LINE,
                )
            )
            
            # No hardcoded business phrases - let the model work naturally
            # This makes the system domain-agnostic
            
            logger.info(f"Created config: encoding={config.encoding}, sample_rate={config.sample_rate_hertz}")
            logger.info(f"Model: {config.model}, enhanced: {config.use_enhanced}")
            return config
            
        except Exception as e:
            logger.error(f"Error creating recognition config: {e}")
            raise
    
    def _analyze_audio_content(self, audio_data: bytes) -> Dict[str, float]:
        """
        Analyze audio content to determine if it contains speech.
        """
        if len(audio_data) < 160:
            return {"has_speech": False}
        
        try:
            # Convert small sample to analyze
            sample = audio_data[:320]  # 40ms sample
            linear_sample = audioop.ulaw2lin(sample, 2)
            audio_array = np.frombuffer(linear_sample, dtype=np.int16)
            
            # Calculate energy metrics
            energy = np.mean(np.square(audio_array.astype(np.float32)))
            rms = np.sqrt(energy)
            
            # Zero crossing rate (good indicator of speech)
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr = zero_crossings / len(audio_array)
            
            # Speech typically has energy > 100 and reasonable ZCR
            has_speech = rms > 100 and 0.01 < zcr < 0.3
            
            return {
                "has_speech": has_speech,
                "rms": float(rms),
                "zcr": float(zcr),
                "energy": float(energy)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"has_speech": True}  # Assume speech if analysis fails
    
    async def start_streaming(self):
        """Start streaming session with enhanced initialization."""
        self.is_streaming = True
        self.utterance_id = 0
        self.audio_buffer = bytearray()
        logger.info("Started Google Cloud Speech streaming session")
    
    async def stop_streaming(self):
        """Stop streaming session and process final buffer."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        
        # Process any remaining audio
        final_result = ""
        if len(self.audio_buffer) > 1600:  # At least 200ms of audio
            logger.info(f"Processing final audio buffer: {len(self.audio_buffer)} bytes")
            try:
                final_result = await self._process_audio_buffer()
            except Exception as e:
                logger.error(f"Error processing final buffer: {e}")
        
        logger.info(f"Stopped streaming. Final result: '{final_result}'")
        return final_result, 0.0
    
    async def _process_audio_buffer(self) -> str:
        """Process the accumulated audio buffer with enhanced error handling."""
        if len(self.audio_buffer) == 0:
            return ""
        
        try:
            # Create config and audio objects
            config = self._get_config()
            audio_data = bytes(self.audio_buffer)
            
            logger.info(f"Processing {len(audio_data)} bytes of audio with Google Cloud STT")
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform recognition with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.recognize(config=config, audio=audio)
                    ),
                    timeout=10.0  # Increased timeout for better reliability
                )
            except asyncio.TimeoutError:
                logger.warning("Google Cloud STT request timed out")
                return ""
            
            logger.info(f"Got {len(response.results)} results from Google Cloud STT")
            
            # Process results with enhanced handling
            for i, result in enumerate(response.results):
                logger.info(f"Result {i}: is_final={getattr(result, 'is_final', 'N/A')}, "
                           f"alternatives={len(result.alternatives)}")
                
                if result.alternatives:
                    alternative = result.alternatives[0]
                    confidence = getattr(alternative, 'confidence', 0.9)
                    text = alternative.transcript.strip()
                    
                    logger.info(f"  Alternative 0: text='{text}', confidence={confidence}")
                    
                    # Accept any result with reasonable confidence or if final
                    if text and (confidence >= 0.5 or result.is_final):
                        logger.info(f"Accepting transcription: '{text}' (confidence: {confidence})")
                        self.successful_recognitions += 1
                        return text
            
            logger.warning("No valid transcription results found")
            return ""
            
        except Exception as e:
            logger.error(f"Error in _process_audio_buffer: {e}", exc_info=True)
            return ""
    
    async def process_audio_chunk(
        self, 
        audio_chunk: Union[bytes, np.ndarray], 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with enhanced buffering strategy."""
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        # Convert input to bytes
        if isinstance(audio_chunk, np.ndarray):
            logger.debug(f"Converting numpy array: dtype={audio_chunk.dtype}, shape={audio_chunk.shape}")
            
            if audio_chunk.dtype == np.float32:
                # Convert float32 to mulaw
                audio_bytes = audioop.lin2ulaw(
                    (audio_chunk * 32767).astype(np.int16).tobytes(), 2
                )
            else:
                audio_bytes = audio_chunk.tobytes()
        else:
            audio_bytes = audio_chunk
        
        logger.debug(f"Received audio chunk: {len(audio_bytes)} bytes")
        
        # Skip tiny chunks
        if len(audio_bytes) < 160:
            logger.debug("Skipping tiny audio chunk")
            return None
        
        # Analyze audio content
        analysis = self._analyze_audio_content(audio_bytes)
        
        # Add to buffer
        self.audio_buffer.extend(audio_bytes)
        
        # Limit buffer size
        if len(self.audio_buffer) > self.max_buffer_size:
            # Remove from beginning to maintain recent audio
            excess = len(self.audio_buffer) - int(self.max_buffer_size * 0.9)
            self.audio_buffer = self.audio_buffer[excess:]
            logger.debug(f"Trimmed audio buffer by {excess} bytes")
        
        # Process when we have enough audio
        current_buffer_size = len(self.audio_buffer)
        
        # Adaptive processing based on content
        should_process = False
        
        if current_buffer_size >= self.max_buffer_size:
            should_process = True
            logger.debug("Processing due to max buffer size")
        elif current_buffer_size >= self.min_buffer_size:
            if analysis.get("has_speech", False):
                should_process = True
                logger.debug("Processing due to detected speech")
            elif current_buffer_size >= self.min_buffer_size * 1.5:
                should_process = True
                logger.debug("Processing due to buffer size threshold")
        
        if should_process:
            logger.info(f"Processing audio buffer: {current_buffer_size} bytes")
            
            try:
                result_text = await self._process_audio_buffer()
                
                # Clear buffer after processing
                self.audio_buffer = bytearray()
                
                if result_text:
                    self.utterance_id += 1
                    result = StreamingTranscriptionResult(
                        text=result_text,
                        is_final=True,
                        confidence=0.9,  # Assume good confidence for synchronous recognition
                        chunk_id=self.utterance_id
                    )
                    
                    logger.info(f"Created result: text='{result.text}', is_final={result.is_final}")
                    
                    if callback:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
                    
                    return result
                else:
                    logger.debug("No text returned from processing")
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
        else:
            logger.debug(f"Not processing: buffer_size={current_buffer_size}, "
                        f"has_speech={analysis.get('has_speech', False)}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_chunks": self.total_chunks,
            "successful_recognitions": self.successful_recognitions,
            "success_rate": round((self.successful_recognitions / max(self.total_chunks, 1)) * 100, 2),
            "buffer_size": len(self.audio_buffer),
            "config": {
                "model": "phone_call",
                "sample_rate": self.sample_rate,
                "language": self.language_code,
                "enhanced": True
            }
        }