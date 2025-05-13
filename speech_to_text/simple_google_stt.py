"""
Complete rewrite of Google Cloud Speech client with better error handling and debugging.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass
import io

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
    """Simplified Google Cloud Speech client with extensive debugging."""
    
    def __init__(
        self, 
        language_code="en-US", 
        sample_rate=8000,
        enable_automatic_punctuation=True,
        enhanced_telephony=True
    ):
        """Initialize the Google Cloud Speech client."""
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
        
        # Audio accumulation
        self.audio_buffer = bytearray()
        self.min_buffer_size = 3200  # 400ms at 8kHz - larger for better accuracy
        self.max_buffer_size = 16000  # 2 seconds max
        
        logger.info(f"Initialized Google Cloud STT: {sample_rate}Hz, enhanced={enhanced_telephony}")
    
    def _get_config(self):
        """Get the recognition config with extensive debugging."""
        try:
            # For Twilio mulaw audio
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                model="phone_call",  # Telephony model is crucial
                use_enhanced=True,
                audio_channel_count=1,
                enable_word_time_offsets=False,
                max_alternatives=1,
                profanity_filter=False,
            )
            
            # Add speech contexts
            if self.enhanced_telephony:
                phrases = [
                    "VoiceAssist", "features", "pricing", "plan", "cost", 
                    "subscription", "service", "support", "upgrade", 
                    "payment", "account", "question", "help", "information",
                    "what are the features", "how much does it cost", "pricing plans"
                ]
                
                speech_context = speech.SpeechContext(
                    phrases=phrases,
                    boost=20.0  # Higher boost for better recognition
                )
                config.speech_contexts.append(speech_context)
                logger.info(f"Added {len(phrases)} speech context phrases")
            
            logger.info(f"Created config: encoding={config.encoding}, sample_rate={config.sample_rate_hertz}, model={config.model}")
            return config
            
        except Exception as e:
            logger.error(f"Error creating recognition config: {e}")
            raise
    
    async def start_streaming(self):
        """Start streaming session."""
        self.is_streaming = True
        self.utterance_id = 0
        self.audio_buffer = bytearray()
        logger.info("Started Google Cloud Speech streaming session")
    
    async def stop_streaming(self):
        """Stop streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        
        # Process any remaining audio
        final_result = ""
        if len(self.audio_buffer) > 160:
            logger.info(f"Processing final audio buffer: {len(self.audio_buffer)} bytes")
            try:
                final_result = await self._process_audio_buffer()
            except Exception as e:
                logger.error(f"Error processing final buffer: {e}")
        
        logger.info(f"Stopped streaming. Final result: '{final_result}'")
        return final_result, 0.0
    
    async def _process_audio_buffer(self) -> str:
        """Process the accumulated audio buffer."""
        if len(self.audio_buffer) == 0:
            return ""
        
        try:
            # Create config and audio objects
            config = self._get_config()
            audio_data = bytes(self.audio_buffer)
            
            logger.info(f"Processing {len(audio_data)} bytes of audio with Google Cloud STT")
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform recognition
            response = self.client.recognize(config=config, audio=audio)
            
            logger.info(f"Got {len(response.results)} results from Google Cloud STT")
            
            # Process results
            for i, result in enumerate(response.results):
                logger.info(f"Result {i}: is_final={getattr(result, 'is_final', 'N/A')}, alternatives={len(result.alternatives)}")
                
                if result.alternatives:
                    alternative = result.alternatives[0]
                    logger.info(f"  Alternative 0: text='{alternative.transcript}', confidence={getattr(alternative, 'confidence', 'N/A')}")
                    
                    if alternative.transcript:
                        return alternative.transcript
            
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
        """Process audio chunk with extensive debugging."""
        if not self.is_streaming:
            await self.start_streaming()
        
        # Convert input to bytes
        if isinstance(audio_chunk, np.ndarray):
            logger.debug(f"Converting numpy array: dtype={audio_chunk.dtype}, shape={audio_chunk.shape}")
            
            if audio_chunk.dtype == np.float32:
                # Convert float32 to mulaw-like bytes
                audio_bytes = (audio_chunk * 255).astype(np.uint8).tobytes()
            else:
                audio_bytes = audio_chunk.tobytes()
        else:
            audio_bytes = audio_chunk
        
        logger.debug(f"Received audio chunk: {len(audio_bytes)} bytes")
        
        # Skip tiny chunks
        if len(audio_bytes) < 160:
            logger.debug("Skipping tiny audio chunk")
            return None
        
        # Add to buffer
        self.audio_buffer.extend(audio_bytes)
        
        # Limit buffer size
        if len(self.audio_buffer) > self.max_buffer_size:
            excess = len(self.audio_buffer) - self.max_buffer_size
            self.audio_buffer = self.audio_buffer[excess:]
            logger.debug(f"Trimmed audio buffer by {excess} bytes")
        
        # Process when we have enough audio
        if len(self.audio_buffer) >= self.min_buffer_size:
            logger.info(f"Processing audio buffer: {len(self.audio_buffer)} bytes")
            
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
                    logger.warning("No text returned from processing")
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
        
        return None