"""
Enhanced Google Cloud Speech client for Voice AI Agent.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass

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
    """Simple Google Cloud Speech client with enhanced error handling."""
    
    def __init__(
        self, 
        language_code="en-US", 
        sample_rate=16000, 
        enable_automatic_punctuation=True
    ):
        """Initialize the Google Cloud Speech client."""
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech API not available")
        
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation
        
        # Create client
        try:
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech client created successfully")
        except Exception as e:
            logger.error(f"Failed to create Google Cloud Speech client: {e}")
            raise
        
        # State tracking
        self.is_streaming = False
        self.utterance_id = 0
    
    def _get_config(self):
        """Get the recognition config."""
        try:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                use_enhanced=True,
                model="phone_call",
                audio_channel_count=1
            )
            
            return config
        except Exception as e:
            logger.error(f"Error creating recognition config: {e}")
            raise
    
    async def start_streaming(self):
        """Start streaming session."""
        try:
            self.is_streaming = True
            self.utterance_id = 0
            logger.info("Started Google Cloud Speech streaming session")
        except Exception as e:
            logger.error(f"Error starting streaming session: {e}")
            raise
    
    async def stop_streaming(self):
        """Stop streaming session."""
        try:
            self.is_streaming = False
            logger.info("Stopped Google Cloud Speech streaming session")
            return "", 0.0
        except Exception as e:
            logger.error(f"Error stopping streaming session: {e}")
            return "", 0.0
    
    async def process_audio_chunk(
        self, 
        audio_chunk, 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ):
        """Process a chunk of audio with synchronous recognition."""
        if not self.is_streaming:
            logger.warning("Called process_audio_chunk but streaming is not active")
            await self.start_streaming()
        
        try:
            # Convert numpy array to bytes if needed
            if isinstance(audio_chunk, np.ndarray):
                # Ensure the data is float32 in [-1.0, 1.0] range
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                # Convert to int16
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Check if we have enough audio data
            if len(audio_bytes) < 320:  # Less than 10ms at 16kHz
                logger.debug(f"Audio chunk too small: {len(audio_bytes)} bytes")
                return None
            
            # Instead of trying to use streaming recognition, just use synchronous recognition
            try:
                # Create recognition config
                config = self._get_config()
                
                # Create recognition audio
                audio = speech.RecognitionAudio(content=audio_bytes)
                
                # Perform recognition
                response = self.client.recognize(config=config, audio=audio)
                
                # Process results
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alt = result.alternatives[0]
                    
                    # Create a result object
                    self.utterance_id += 1
                    transcription_result = StreamingTranscriptionResult(
                        text=alt.transcript,
                        is_final=True,
                        confidence=alt.confidence if hasattr(alt, "confidence") else 0.8,
                        chunk_id=self.utterance_id
                    )
                    
                    # Call callback if provided
                    if callback:
                        await callback(transcription_result)
                        
                    logger.info(f"Speech recognition result: {alt.transcript}")
                    return transcription_result
                
                # No results
                logger.debug("No speech recognition results")
                return None
                
            except Exception as e:
                logger.error(f"Error in Google Cloud Speech recognition: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

# For compatibility with the existing code that imports GoogleCloudStreamingSTT
class GoogleCloudStreamingSTT(SimpleGoogleSTT):
    """Alias for compatibility with existing code."""
    
    def __init__(
        self,
        language="en-US",
        sample_rate=16000,
        encoding="LINEAR16",
        channels=1,
        interim_results=True,
        enhanced_model=True,
        speech_context_phrases=None
    ):
        """Initialize with compatibility for the existing interface."""
        super().__init__(
            language_code=language,
            sample_rate=sample_rate,
            enable_automatic_punctuation=True
        )
        
        # Store additional parameters (even if not used in this simple implementation)
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.speech_context_phrases = speech_context_phrases or []