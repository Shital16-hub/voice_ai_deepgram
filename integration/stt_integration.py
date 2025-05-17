"""
Speech-to-Text integration module optimized for telephony with minimal processing.
IMPROVED: Ultra-low latency speech detection.
"""
import logging
import time
import os
import json
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Speech-to-Text integration optimized for telephony with ultra-fast processing.
    Uses Google Cloud Speech-to-Text v2 API optimally for telephony.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en-US"
    ):
        """Initialize the STT integration."""
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # IMPROVED: Simplified pattern for faster processing 
        import re
        self.non_speech_pattern = re.compile(r'\[.*?\]|\(.*?\)|\<.*?\>|music playing|background noise|static|\b(um|uh|hmm|mmm)\b')
        
        # IMPROVED: Track performance
        self.transcription_count = 0
        self.valid_transcription_count = 0
        self.processing_time_total = 0
        
        logger.info("STTIntegration initialized for telephony with minimal processing")
    
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
            # Create Google Cloud v2 streaming client with optimal telephony settings
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=8000,  # Match Twilio exactly
                encoding="MULAW",   # Match Twilio exactly
                channels=1,
                interim_results=True,  # IMPROVED: Enable for debugging and faster results
                project_id=final_project_id,
                enhanced_model=True,    # Use telephony-enhanced model
                location="global"
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud v2 API (telephony-optimized)")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def cleanup_transcription(self, text: str) -> str:
        """IMPROVED: Ultra-minimal cleanup for faster processing."""
        if not text:
            return ""
        
        # Remove non-speech annotations with a single regex (faster)
        cleaned = self.non_speech_pattern.sub('', text)
        
        # Only strip whitespace and ensure proper capitalization
        cleaned = cleaned.strip()
        
        # Capitalize first letter if needed (with fast check)
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def is_valid_transcription(self, text: str, min_words: int = 1) -> bool:
        """IMPROVED: Ultra-lenient validation with minimal requirements."""
        cleaned = self.cleanup_transcription(text)
        
        if not cleaned:
            return False
        
        # Must have at least one word
        words = cleaned.split()
        if len(words) < min_words:
            return False
        
        # Must have at least one alphabetic character
        if not any(c.isalpha() for c in cleaned):
            return False
        
        return True
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """IMPROVED: Ultra-fast transcription optimized for telephony."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert list to bytes if needed (no other processing)
            if isinstance(audio_data, list):
                # Convert list of numbers to bytes (assume they're already MULAW samples)
                audio_data = bytes(audio_data)
            
            # IMPROVED: Ensure we have a fresh streaming session for better results
            if (self.speech_recognizer and 
                hasattr(self.speech_recognizer, '_last_streaming_start') and 
                self.speech_recognizer._last_streaming_start and 
                time.time() - self.speech_recognizer._last_streaming_start > 60):  # Reduced from 120s to 60s
                logger.info("Refreshing STT streaming session for better results")
                await self.speech_recognizer.stop_streaming()
                await self.speech_recognizer.start_streaming()
            
            # Get results directly from STT v2 - no preprocessing
            final_results = []
            
            # Define a callback to collect results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Start streaming session if needed
            if not self.speech_recognizer.is_streaming:
                logger.info("Starting new streaming session")
                await self.speech_recognizer.start_streaming()
            
            # Process the audio directly
            await self.speech_recognizer.process_audio_chunk(audio_data, store_result)
            
            # Stop streaming to get final results
            final_text, duration = await self.speech_recognizer.stop_streaming()
            
            # IMPROVED: Immediately restart streaming for continuous listening
            await self.speech_recognizer.start_streaming()
            
            # Get the best result
            if final_text:
                transcription = final_text
                confidence = 0.9  # Default confidence for final results
            elif final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
            else:
                logger.info("No transcription results obtained")
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
            
            # Track performance metrics
            self.transcription_count += 1
            if self.is_valid_transcription(cleaned_text):
                self.valid_transcription_count += 1
                
            self.processing_time_total += (time.time() - start_time)
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": confidence,
                "duration": duration if duration > 0 else 0.0,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text),
                "api_version": "v2"
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
        audio_chunk: Union[bytes, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """IMPROVED: Ultra-fast stream processing with better error handling."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # IMPROVED: Ensure we have an active streaming session
        if not self.speech_recognizer.is_streaming:
            logger.info("No active streaming session, starting one")
            await self.speech_recognizer.start_streaming()
        
        # Convert list to bytes if needed
        if isinstance(audio_chunk, list):
            audio_chunk = bytes(audio_chunk)
        
        # Pass directly to STT v2 without any processing
        try:
            return await self.speech_recognizer.process_audio_chunk(
                audio_chunk=audio_chunk,
                callback=callback
            )
        except Exception as e:
            logger.error(f"Error processing stream chunk: {e}")
            return None
    
    async def end_streaming(self) -> Tuple[str, float]:
        """End the streaming session with automatic restart."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming session
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # IMPROVED: Immediately restart streaming
        await self.speech_recognizer.start_streaming()
        
        # Minimal cleanup
        cleaned_text = self.cleanup_transcription(final_text)
        
        if final_text != cleaned_text:
            logger.debug(f"Cleaned final transcription: '{final_text}' -> '{cleaned_text}'")
        
        return cleaned_text, duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get STT performance statistics."""
        return {
            "initialized": self.initialized,
            "language": self.language,
            "transcription_count": self.transcription_count,
            "valid_transcription_count": self.valid_transcription_count,
            "validation_rate": self.valid_transcription_count / max(1, self.transcription_count),
            "avg_processing_time": self.processing_time_total / max(1, self.transcription_count),
            "api_type": "google_cloud_v2"
        }