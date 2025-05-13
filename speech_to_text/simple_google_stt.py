"""
Simplified Google Cloud Speech client for Voice AI Agent.
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
    """Simple Google Cloud Speech client optimized for telephony."""
    
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
        self.client = speech.SpeechClient()
        
        # State tracking
        self.is_streaming = False
        self.utterance_id = 0
        
        # Buffer for accumulating audio
        self.audio_buffer = bytearray()
        self.buffer_size = sample_rate * 2  # 2 seconds of audio
        
        logger.info("Initialized SimpleGoogleSTT with optimized telephony settings")
    
    def _get_config(self):
        """Get the recognition config optimized for telephony."""
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            max_alternatives=1,
            enable_automatic_punctuation=self.enable_automatic_punctuation,
            enable_word_time_offsets=True,
            profanity_filter=False,
            use_enhanced=True,
            model="phone_call",  # Optimized for telephony
            audio_channel_count=1
        )
        
        # Add common telephony phrases for better recognition
        speech_context = speech.SpeechContext(
            phrases=[
                "price", "cost", "plan", "subscription", "service", "features",
                "support", "help", "information", "details", "cancel", "upgrade"
            ],
            boost=15.0  # Boost these phrases
        )
        config.speech_contexts.append(speech_context)
        
        return config
    
    async def start_streaming(self):
        """Start streaming session (placeholder for compatibility)."""
        self.is_streaming = True
        self.utterance_id = 0
        logger.info("Started SimpleGoogleSTT streaming session")
    
    async def stop_streaming(self):
        """Stop streaming session (placeholder for compatibility)."""
        self.is_streaming = False
        logger.info("Stopped SimpleGoogleSTT streaming session")
        return "", 0.0
    
    async def process_audio_chunk(
        self, 
        audio_chunk: Union[bytes, np.ndarray], 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process a chunk of audio with batch recognition optimized for real-time."""
        if not self.is_streaming:
            logger.debug("Called process_audio_chunk but streaming is not active")
            await self.start_streaming()
        
        # Convert numpy array to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            # Ensure the data is float32 in [-1.0, 1.0] range
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            # Convert to int16
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_chunk
        
        # Add to buffer
        self.audio_buffer.extend(audio_bytes)
        
        # Only process when we have enough audio
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        # Get audio to process
        audio_to_process = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        
        try:
            # Create recognition config
            config = self._get_config()
            
            # Create recognition audio
            audio = speech.RecognitionAudio(content=audio_to_process)
            
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
                    
                return transcription_result
            
            # No results
            return None
            
        except Exception as e:
            logger.error(f"Error in Google Cloud Speech recognition: {e}")
            return None