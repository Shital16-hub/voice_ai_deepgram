"""
Speech-to-Text integration module for telephony using infinite streaming.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union

# Import the new infinite streaming implementation
from speech_to_text.infinite_streaming_stt import InfiniteStreamingSTT, StreamingTranscriptionResult

logger = logging.getLogger(__name__)

class STTIntegration:
    """
    Speech-to-Text integration with infinite streaming for uninterrupted telephony.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[InfiniteStreamingSTT] = None,
        language: str = "en-US"
    ):
        """Initialize the STT integration with infinite streaming."""
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = speech_recognizer is not None
        
        # Define patterns for non-speech annotations (same as before)
        import re
        self.non_speech_pattern = re.compile('|'.join([
            r'\[.*?\]',           # Anything in square brackets
            r'\(.*?\)',           # Anything in parentheses
            r'\<.*?\>',           # Anything in angle brackets
            r'music playing',     # Common transcription
            r'background noise',  # Common transcription
            r'static',            # Common transcription
            r'\b(um|uh|hmm|mmm)\b',  # Common filler words
        ]))
        
        logger.info("STTIntegration initialized with infinite streaming support")
    
    async def init(self, project_id: Optional[str] = None) -> None:
        """Initialize the STT component with infinite streaming if not already initialized."""
        if self.initialized:
            return
        
        # Get project ID with automatic extraction
        final_project_id = project_id or self._get_project_id()
        
        if not final_project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
            
        try:
            # Create infinite streaming STT client
            self.speech_recognizer = InfiniteStreamingSTT(
                project_id=final_project_id,
                language=self.language,
                sample_rate=8000,
                encoding="MULAW",
                channels=1,
                interim_results=True,
                location="global",
                credentials_file=None  # Use default
            )
            
            self.initialized = True
            logger.info(f"Initialized infinite streaming STT with project ID: {final_project_id}")
            
        except Exception as e:
            logger.error(f"Error initializing STT with infinite streaming: {e}")
            raise
    
    def _get_project_id(self) -> Optional[str]:
        """Get project ID from environment variable or credentials file."""
        import os
        import json
        
        # Try environment variable
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
            
        # Try credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    return creds_data.get('project_id')
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        return None
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription by removing non-speech annotations."""
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
        import re
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def is_valid_transcription(self, text: str, min_words: int = 2) -> bool:
        """Check if a transcription is valid and worth processing."""
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Transcribe audio data with infinite streaming."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert to bytes if needed (no other processing)
            if isinstance(audio_data, list):
                # Convert list to bytes
                audio_data = bytes(audio_data)
            
            # Ensure we have a streaming session running
            if not self.speech_recognizer.is_streaming:
                logger.info("Starting infinite streaming session")
                await self.speech_recognizer.start_streaming()
            
            # Define a callback to collect results
            final_results = []
            
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Process the audio directly with infinite streaming
            result = await self.speech_recognizer.process_audio_chunk(audio_data, store_result)
            
            # If we have a final result from this chunk, use it
            if result and result.is_final:
                transcription = result.text
                confidence = result.confidence
            # Otherwise check if we collected any final results
            elif final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
            else:
                # No transcription results
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": 0.0,
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False,
                    "streaming_active": self.speech_recognizer.is_streaming
                }
            
            # Clean up the transcription
            cleaned_text = self.cleanup_transcription(transcription)
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": confidence,
                "duration": 0.0,  # Not applicable for streaming
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text),
                "streaming_active": self.speech_recognizer.is_streaming
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "streaming_active": getattr(self.speech_recognizer, 'is_streaming', False)
            }
    
    async def start_streaming(self) -> None:
        """Start a new infinite streaming session that will run for the duration of the call."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return
        
        await self.speech_recognizer.start_streaming()
        logger.info("Started infinite streaming session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process a chunk of streaming audio with infinite streaming."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Convert to bytes if needed
        if isinstance(audio_chunk, list):
            audio_chunk = bytes(audio_chunk)
        
        # Ensure infinite streaming is active
        if not self.speech_recognizer.is_streaming:
            logger.info("Streaming not active, starting infinite streaming")
            await self.speech_recognizer.start_streaming()
        
        # Process with infinite streaming
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_chunk,
            callback=callback
        )
    
    async def end_streaming(self) -> Tuple[str, float]:
        """End the infinite streaming session and get final transcription."""
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop infinite streaming with final results
        return await self.speech_recognizer.stop_streaming()