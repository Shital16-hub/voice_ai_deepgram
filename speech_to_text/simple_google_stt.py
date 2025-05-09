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
    """Simple Google Cloud Speech client."""
    
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
    
    def _get_config(self):
        """Get the recognition config."""
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=self.enable_automatic_punctuation,
            use_enhanced=True,
            model="phone_call",
            audio_channel_count=1
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )
        
        return streaming_config
    
    async def start_streaming(self):
        """Start streaming session (placeholder)."""
        self.is_streaming = True
        self.utterance_id = 0
        logger.info("Started Google Cloud Speech streaming session")
    
    async def stop_streaming(self):
        """Stop streaming session (placeholder)."""
        self.is_streaming = False
        logger.info("Stopped Google Cloud Speech streaming session")
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
        
        # Convert numpy array to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            # Ensure the data is float32 in [-1.0, 1.0] range
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            # Convert to int16
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_chunk
        
        # Instead of trying to use streaming recognition, just use synchronous recognition
        try:
            # Create recognition config
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                use_enhanced=True,
                model="phone_call",
                audio_channel_count=1
            )
            
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
                    
                return transcription_result
            
            # No results
            return None
            
        except Exception as e:
            logger.error(f"Error in Google Cloud Speech recognition: {e}")
            return None