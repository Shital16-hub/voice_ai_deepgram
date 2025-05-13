# speech_to_text/google_cloud_stt_v2.py

"""
Google Cloud Speech-to-Text using synchronous recognition for reliability.
"""
import os
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass
import numpy as np

from google.cloud import speech

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None

class GoogleCloudStreamingSTT_V2:
    """
    Google Cloud Speech-to-Text using synchronous recognition for reliability.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = True,
        enhanced_model: bool = True,
        timeout: float = 30.0
    ):
        """Initialize Google Cloud STT with synchronous recognition."""
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.timeout = timeout
        
        # Initialize client
        self.client = speech.SpeechClient()
        
        # State tracking
        self.is_streaming = False
        self.chunk_count = 0
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.utterance_id = 0
        
        logger.info(f"Initialized GoogleCloudStreamingSTT_V2: {sample_rate}Hz, {encoding}")
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get recognition configuration for synchronous calls."""
        # Map encoding string to enum
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
        }
        
        encoding_enum = encoding_map.get(self.encoding, speech.RecognitionConfig.AudioEncoding.MULAW)
        
        # Create configuration optimized for telephony
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            audio_channel_count=self.channels,
            
            # Essential telephony optimizations
            model="phone_call",  # Best for telephony
            use_enhanced=self.enhanced_model,  # Premium model for accuracy
            
            # Automatic features - no hardcoded patterns!
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
        )
        
        return config
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        logger.info("Starting Google Cloud Speech streaming session")
        self.is_streaming = True
        self.chunk_count = 0
        self.utterance_id = 0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process audio chunk with synchronous recognition."""
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert to proper format
            if isinstance(audio_chunk, np.ndarray):
                if audio_chunk.dtype == np.float32:
                    # For MULAW, convert through linear PCM
                    if self.encoding == "MULAW":
                        # Convert float32 to int16 PCM
                        linear_audio = (audio_chunk * 32767).astype(np.int16).tobytes()
                        # Don't convert to mulaw - keep as linear for recognition
                        audio_bytes = linear_audio
                        # Update config to LINEAR16 for this recognition
                        temp_encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
                    else:
                        # For LINEAR16, just convert to int16
                        audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                        temp_encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
                else:
                    audio_bytes = audio_chunk.tobytes()
                    temp_encoding = speech.RecognitionConfig.AudioEncoding.MULAW
            else:
                audio_bytes = audio_chunk
                temp_encoding = speech.RecognitionConfig.AudioEncoding.MULAW
            
            # Skip very small chunks
            if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
                logger.debug("Skipping tiny audio chunk")
                return None
            
            # Get recognition config
            config = speech.RecognitionConfig(
                encoding=temp_encoding,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language,
                audio_channel_count=self.channels,
                model="phone_call",
                use_enhanced=self.enhanced_model,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
            )
            
            # Create recognition audio
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Perform synchronous recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Process results
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                
                # Extract words if available
                words = []
                if hasattr(alternative, 'words') and alternative.words:
                    for word_info in alternative.words:
                        words.append({
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds(),
                            "confidence": getattr(word_info, 'confidence', alternative.confidence)
                        })
                
                # Create result object
                self.utterance_id += 1
                transcription_result = StreamingTranscriptionResult(
                    text=alternative.transcript,
                    is_final=True,  # Synchronous results are always final
                    confidence=alternative.confidence if hasattr(alternative, 'confidence') else 0.8,
                    chunk_id=self.utterance_id,
                    words=words
                )
                
                self.successful_transcriptions += 1
                logger.info(f"Transcription: '{alternative.transcript}' (confidence: {transcription_result.confidence:.2f})")
                
                # Call callback if provided
                if callback:
                    await callback(transcription_result)
                
                return transcription_result
            
            # No results
            logger.debug(f"No transcription results for chunk {self.total_chunks}")
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info("Stopping Google Cloud Speech streaming session")
        self.is_streaming = False
        
        # Log final stats
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        logger.info(f"Session complete. Success rate: {success_rate:.1f}%")
        
        # Return empty for synchronous implementation
        return "", 0.0
    
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
            "encoding": self.encoding,
            "sample_rate": self.sample_rate
        }