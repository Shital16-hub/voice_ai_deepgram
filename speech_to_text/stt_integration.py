"""
Enhanced Speech-to-Text integration module optimized for telephony.
"""
import logging
import time
import asyncio
import re
import numpy as np
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Enhanced Speech-to-Text integration optimized for telephony with minimal processing.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Keep minimal cleanup patterns - let Google handle most of it
        self.cleanup_patterns = [
            # Only remove obvious technical artifacts
            (re.compile(r'\[.*?\]'), ''),  # [inaudible]
            (re.compile(r'\<.*?\>'), ''),  # <music>
            # Keep filler words - they're part of natural speech
        ]
    
    async def init(self, api_key: Optional[str] = None) -> None:
        """Initialize the STT component if not already initialized."""
        if self.initialized:
            return
            
        try:
            # Create a new Google Cloud streaming client with optimal settings
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=8000,  # Match Twilio
                encoding="MULAW",   # Match Twilio
                channels=1,
                interim_results=False,  # Disable for better accuracy
                enhanced_model=True
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud API (telephony-optimized)")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Minimal cleanup - let Google's telephony model handle most of it.
        """
        if not text:
            return ""
        
        # Apply minimal cleanup patterns
        cleaned = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        return cleaned
    
    def is_valid_transcription(self, text: str, min_words: int = 1) -> bool:
        """
        Validate transcription with minimal requirements.
        """
        cleaned = self.cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Must have at least one word
        words = cleaned.split()
        return len(words) >= min_words
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio data with minimal processing.
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed - no preprocessing
            if isinstance(audio_data, bytes):
                # Keep as bytes for direct processing
                pass
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            # For numpy arrays, let the STT handle conversion
            
            # Get results directly from STT
            final_results = []
            
            # Define a callback to collect results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Start streaming session
            await self.speech_recognizer.start_streaming()
            
            # Process the audio
            await self.speech_recognizer.process_audio_chunk(audio_data, store_result)
            
            # Stop streaming to get final results
            final_text, duration = await self.speech_recognizer.stop_streaming()
            
            # Get the best result
            if final_text:
                transcription = final_text
                confidence = 0.9
            elif final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
            else:
                logger.warning("No transcription results obtained")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": 0.0,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
            # Minimal cleanup
            cleaned_text = self.cleanup_transcription(transcription)
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": confidence,
                "duration": duration if duration > 0 else 0.0,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def start_streaming(self) -> None:
        """Start a new streaming transcription session."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return
        
        await self.speech_recognizer.start_streaming()
        logger.debug("Started streaming transcription session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of streaming audio with minimal processing.
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Don't preprocess - pass directly to STT
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_chunk,
            callback=callback
        )
    
    async def end_streaming(self) -> Tuple[str, float]:
        """
        End the streaming session and get final transcription.
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming session
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Minimal cleanup
        cleaned_text = self.cleanup_transcription(final_text)
        
        if final_text != cleaned_text:
            logger.debug(f"Cleaned final transcription: '{final_text}' -> '{cleaned_text}'")
        
        return cleaned_text, duration
    
    def optimize_for_telephony(self):
        """Already optimized - this is a no-op."""
        logger.info("STT integration already optimized for telephony")
        pass