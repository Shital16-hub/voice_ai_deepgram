"""
Optimized Google Cloud Speech-to-Text client with telephony optimizations.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
import re
from dataclasses import dataclass

try:
    from google.cloud import speech
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None


class GoogleCloudStreamingSTT:
    """
    Optimized Google Cloud Speech-to-Text client for telephony applications.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,  # Use 8kHz for telephony
        encoding: str = "MULAW",   # Direct mulaw support
        channels: int = 1,
        interim_results: bool = False,  # Disabled for lower latency
        speech_context_phrases: Optional[List[str]] = None,
        enhanced_model: bool = True
    ):
        """
        Initialize the Google Cloud STT client optimized for telephony.
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech modules not installed")

        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        
        # Telephony-optimized phrases
        self.speech_context_phrases = speech_context_phrases or [
            "pricing", "plan", "cost", "subscription", "service", "features",
            "support", "upgrade", "payment", "account", "question", "help",
            "information", "details", "monthly", "annually"
        ]
        
        # State management
        self.is_streaming = False
        self.utterance_id = 0
        
        # Create the client
        try:
            self.client = speech.SpeechClient()
            logger.info("Initialized optimized Google Cloud STT for telephony")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud STT: {e}")
            raise
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get optimized recognition configuration for telephony."""
        # Get encoding enum
        if self.encoding == "MULAW":
            encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
        elif self.encoding == "LINEAR16":
            encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
        else:
            encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
        
        # Create optimized config for telephony
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            max_alternatives=1,  # Reduce latency
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,  # Disable for lower latency
            profanity_filter=False,  # Faster processing
            model="phone_call",  # Telephony-optimized model
            use_enhanced=self.enhanced_model,
        )
        
        # Add speech contexts for better recognition
        if self.speech_context_phrases:
            speech_context = speech.SpeechContext(
                phrases=self.speech_context_phrases,
                boost=15.0
            )
            config.speech_contexts.append(speech_context)
            
        return config
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get optimized streaming configuration."""
        recognition_config = self._get_recognition_config()
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            single_utterance=False,
            enable_voice_activity_events=True  # Helps with detection
        )
        
        return streaming_config
    
    async def start_streaming(self) -> None:
        """Start a new streaming recognition session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        self.utterance_id = 0
        self.is_streaming = True
        
        logger.info("Started optimized Google Cloud STT streaming session")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and return final text.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        logger.info("Stopped Google Cloud STT streaming session")
        
        return "", 0.0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of audio with optimized telephony settings.
        
        Args:
            audio_chunk: Audio chunk as mulaw bytes or numpy array
            callback: Optional async callback for results
            
        Returns:
            Transcription result or None
        """
        if not self.is_streaming:
            logger.warning("Not streaming - call start_streaming() first")
            return None
        
        try:
            # Ensure audio_chunk is bytes in mulaw format
            if isinstance(audio_chunk, np.ndarray):
                # Convert numpy array to mulaw bytes if needed
                if audio_chunk.dtype == np.float32:
                    # Convert float32 to mulaw
                    audio_bytes = (audio_chunk * 255).astype(np.uint8).tobytes()
                else:
                    audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Skip very small chunks
            if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
                return None
            
            # Use synchronous recognition for better reliability
            recognition_config = self._get_recognition_config()
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Process with Google Cloud STT
            response = self.client.recognize(config=recognition_config, audio=audio)
            
            # Process results
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alt = result.alternatives[0]
                
                # Create result object
                self.utterance_id += 1
                transcription_result = StreamingTranscriptionResult(
                    text=alt.transcript,
                    is_final=True,  # Synchronous recognition is always final
                    confidence=getattr(alt, 'confidence', 0.9),
                    chunk_id=self.utterance_id
                )
                
                # Call callback if provided
                if callback:
                    await callback(transcription_result)
                    
                return transcription_result
            
            return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None