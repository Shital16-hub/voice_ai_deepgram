"""
Enhanced Speech-to-Text integration module optimized for telephony with v2 API.
Uses latest_long model for better conversation handling.
"""
import logging
import time
import asyncio
import re
import numpy as np
import os
import json
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Enhanced Speech-to-Text integration optimized for telephony with minimal processing.
    Uses Google Cloud Speech-to-Text v2 API with latest_long model.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en-US"
    ):
        """
        Initialize the STT integration.
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Keep minimal cleanup patterns - let Google v2 handle most of it
        self.cleanup_patterns = [
            # Only remove obvious technical artifacts
            (re.compile(r'\[.*?\]'), ''),  # [inaudible]
            (re.compile(r'\<.*?\>'), ''),  # <music>
            # Remove excessive whitespace
            (re.compile(r'\s+'), ' '),
        ]
    
    async def init(self, project_id: Optional[str] = None) -> None:
        """Initialize the STT component if not already initialized."""
        if self.initialized:
            return
        
        # Get project ID with automatic extraction
        final_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not provided, try to extract from credentials file
        if not final_project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        final_project_id = creds_data.get('project_id')
                        logger.info(f"STTIntegration: Auto-extracted project ID from credentials: {final_project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
        
        if not final_project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
            
        try:
            # Create a new Google Cloud v2 streaming client with optimal settings
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=8000,      # Match Twilio
                encoding="MULAW",      # Keep MULAW for compatibility
                channels=1,
                interim_results=False, # Disabled for better accuracy
                project_id=final_project_id,
                enhanced_model=True,
                location="global"      # Use global for better model access
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud v2 API (conversation-optimized)")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Minimal cleanup - let Google's v2 telephony model handle most of it.
        """
        if not text:
            return ""
        
        # Apply minimal cleanup patterns
        cleaned = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Basic normalization
        cleaned = cleaned.strip()
        
        # Ensure proper capitalization
        if cleaned and len(cleaned) > 0:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
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
        Transcribe audio data with minimal processing using v2 API.
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed - minimal preprocessing
            if isinstance(audio_data, bytes):
                # Keep as bytes for direct processing
                pass
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            # For numpy arrays, let the STT handle conversion
            
            # Get results directly from STT v2
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
                "is_valid": self.is_valid_transcription(cleaned_text),
                "api_version": "v2",
                "model": "latest_long"
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "api_version": "v2"
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
        
        # Don't preprocess - pass directly to STT v2
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
        """Already optimized for telephony with v2 API - this is a no-op."""
        logger.info("STT integration already optimized for telephony with v2 API and latest_long model")
        pass