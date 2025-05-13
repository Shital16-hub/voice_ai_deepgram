# speech_to_text/google_cloud_stt.py

"""
Optimized Google Cloud Speech-to-Text implementation for Twilio telephony.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass
import json

from google.cloud import speech

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
    """
    Optimized Google Cloud Speech-to-Text client for telephony with v2.32.0.
    Uses telephony-specific models and minimal preprocessing for best accuracy.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        speech_context_phrases: Optional[List[str]] = None,
        enhanced_model: bool = True
    ):
        """
        Initialize with optimal settings for Twilio telephony.
        """
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        
        # Don't hardcode speech contexts - use Google's telephony model instead
        self.speech_context_phrases = speech_context_phrases if speech_context_phrases else []
        
        # Initialize client
        self.client = speech.SpeechClient()
        
        # State tracking
        self.is_streaming = False
        self.chunk_count = 0
        self.total_chunks = 0
        self.successful_transcriptions = 0
        
        logger.info(f"Initialized GoogleCloudStreamingSTT: {sample_rate}Hz, {encoding}, telephony-optimized")
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get optimized recognition configuration for telephony."""
        # Map encoding string to enum
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
        }
        
        encoding_enum = encoding_map.get(self.encoding, speech.RecognitionConfig.AudioEncoding.MULAW)
        
        # Optimal configuration for telephony
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            audio_channel_count=self.channels,
            
            # Key telephony optimizations
            model="phone_call",  # Use telephony-optimized model
            use_enhanced=self.enhanced_model,
            
            # Enable features that improve accuracy
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            max_alternatives=1,  # Focus on best result
            
            # Instead of hardcoded speech contexts, use adaptation if needed
            # speech_contexts=[...] - Removed to avoid hardcoding
        )
        
        # Only add speech contexts if specifically provided
        if self.speech_context_phrases:
            speech_context = speech.SpeechContext(
                phrases=self.speech_context_phrases,
                boost=15.0
            )
            config.speech_contexts.append(speech_context)
        
        return config
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        self.is_streaming = True
        self.chunk_count = 0
        logger.info("Started Google Cloud Speech streaming session")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        logger.info(f"Stopped streaming session. Processed {self.chunk_count} chunks")
        return "", 0.0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio chunk with minimal preprocessing for optimal accuracy.
        """
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert numpy array to bytes if needed - NO preprocessing
            if isinstance(audio_chunk, np.ndarray):
                # Don't modify the audio - just convert format
                if audio_chunk.dtype == np.float32:
                    if self.encoding == "MULAW":
                        # Convert to mulaw directly
                        import audioop
                        # Convert float32 to 16-bit PCM first
                        pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
                        audio_bytes = audioop.lin2ulaw(pcm_data, 2)
                    else:
                        # For LINEAR16, keep as PCM
                        audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = audio_chunk
            
            logger.debug(f"Processing audio chunk #{self.total_chunks}: {len(audio_bytes)} bytes")
            
            # Skip tiny chunks
            if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
                logger.debug("Skipping tiny audio chunk")
                return None
            
            # Get recognition config
            config = self._get_recognition_config()
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Perform recognition
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.recognize(config=config, audio=audio)
            )
            
            # Process results
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result object
                        self.chunk_count += 1
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=True,
                            confidence=getattr(alternative, 'confidence', 0.9),
                            chunk_id=self.chunk_count
                        )
                        
                        self.successful_transcriptions += 1
                        logger.info(f"Transcription: '{alternative.transcript}' (confidence: {transcription_result.confidence:.2f})")
                        
                        # Call callback if provided
                        if callback:
                            await callback(transcription_result)
                        
                        return transcription_result
            
            logger.debug("No transcription results from Google Cloud Speech")
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        
        return {
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2),
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "phone_call",
            "enhanced": self.enhanced_model,
            "encoding": self.encoding
        }