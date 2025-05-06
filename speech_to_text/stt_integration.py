"""
Enhanced Speech-to-Text integration module for Voice AI Agent.

This module provides classes and functions for integrating Google Cloud Speech-to-Text
capabilities with the Voice AI Agent system.
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
    Enhanced Speech-to-Text integration for Voice AI Agent with Google Cloud.
    
    Provides an abstraction layer for speech recognition functionality,
    handling audio processing and transcription.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en"
    ):
        """
        Initialize the STT integration.
        
        Args:
            speech_recognizer: Initialized GoogleCloudStreamingSTT instance
            language: Language code for speech recognition
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Keep track of average audio levels for adaptive thresholding
        self.noise_samples = []
        self.max_samples = 20
        self.ambient_noise_level = 0.01  # Starting threshold
        
        # Agent speaking state for barge-in detection
        self.agent_is_speaking = False
    
    async def init(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the STT component if not already initialized.
        
        Args:
            api_key: API key (not used for Google Cloud STT, rely on environment variable)
        """
        if self.initialized:
            return
            
        try:
            # Create a new Google Cloud streaming client
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=16000,
                encoding="LINEAR16",
                channels=1,
                interim_results=True,
                vad_enabled=True,  # Enable voice activity detection
                barge_in_threshold=0.02  # Set barge-in sensitivity
            )
            
            self.initialized = True
            logger.info(f"Initialized STT with Google Cloud API and language: {self.language}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def set_agent_speaking(self, speaking: bool) -> None:
        """
        Set whether the agent is currently speaking, for barge-in detection.
        
        Args:
            speaking: True if agent is speaking, False otherwise
        """
        self.agent_is_speaking = speaking
        
        # Update underlying speech recognizer if available
        if self.speech_recognizer:
            self.speech_recognizer.set_agent_speaking(speaking)
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Enhanced cleanup of transcription text by removing non-speech annotations and filler words.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if self.speech_recognizer:
            return self.speech_recognizer.cleanup_transcription(text)
            
        # Fall back to minimal cleanup if no speech recognizer
        if not text:
            return ""
            
        # Remove common filler words at beginning of sentences
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', text, flags=re.IGNORECASE)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str, min_words: int = 2) -> bool:
        """
        Check if a transcription is valid and worth processing.
        
        Args:
            text: Transcription text
            min_words: Minimum word count to be considered valid
            
        Returns:
            True if the transcription is valid
        """
        if self.speech_recognizer:
            return self.speech_recognizer.is_valid_transcription(text, min_words)
            
        # Fall back to simple check if no speech recognizer
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            return False
            
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            return False
            
        return True
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file with improved noise handling.
        
        Args:
            audio_file_path: Path to audio file
            callback: Optional callback for interim results
            
        Returns:
            Dictionary with transcription results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Use Google Cloud streaming for file transcription
            results = await self.speech_recognizer.stream_file(
                audio_file_path, 
                callback=callback
            )
            
            # Get the best final result
            if results:
                # Sort by confidence and get the best one
                best_result = max(results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
                duration = best_result.end_time - best_result.start_time 
            else:
                transcription = ""
                confidence = 0.0
                duration = 0.0
            
            # Clean up the transcription
            cleaned_text = self.cleanup_transcription(transcription)
            
            # Check if it's valid
            is_valid = self.is_valid_transcription(cleaned_text)
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": confidence,
                "duration": duration,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": is_valid
            }
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def transcribe_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray, List[float]],
        is_short_audio: bool = False,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio data with improved noise handling.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            is_short_audio: Flag to indicate short audio for optimized handling
            callback: Optional callback for interim results
            
        Returns:
            Dictionary with transcription results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return {"error": "STT integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assume 16-bit PCM format
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, list):
                audio = np.array(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Start streaming session
            await self.speech_recognizer.start_streaming()
            
            # Process the audio with callback
            final_results = []
            
            # Define a custom callback to collect final results
            async def store_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                
                # Call the original callback if provided
                if callback:
                    await callback(result)
            
            # Process the audio data
            result = await self.speech_recognizer.process_audio_chunk(
                audio_data=audio,
                callback=store_result
            )
            
            # Stop streaming to get final results
            final_text, duration = await self.speech_recognizer.stop_streaming()
            
            # Get transcription from various sources
            if final_text:
                transcription = final_text
                confidence = 0.9  # Assume high confidence for final result
            elif final_results:
                # Use the best final result (with highest confidence)
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                confidence = best_result.confidence
            elif result and result.is_final:
                transcription = result.text
                confidence = result.confidence
            else:
                logger.warning("No transcription results obtained")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "duration": len(audio) / 16000,  # Assuming 16kHz
                    "processing_time": time.time() - start_time,
                    "is_final": True,
                    "is_valid": False
                }
            
            # Clean up the transcription
            cleaned_text = self.cleanup_transcription(transcription)
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": confidence,
                "duration": duration if duration > 0 else len(audio) / 16000,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text),
                "barge_in_detected": result.barge_in_detected if result else False
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
        Process a chunk of streaming audio with improved noise handling.
        
        Args:
            audio_chunk: Audio chunk data
            callback: Optional callback for results
            
        Returns:
            Transcription result or None for interim results
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return None
        
        # Process the audio chunk with barge-in detection capability
        return await self.speech_recognizer.process_audio_chunk(
            audio_chunk=audio_chunk,
            callback=callback
        )
    
    async def end_streaming(self) -> Tuple[str, float]:
        """
        End the streaming session and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            return "", 0.0
        
        # Stop streaming session
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Clean up the transcription
        cleaned_text = self.cleanup_transcription(final_text)
        
        return cleaned_text, duration
    
    async def process_realtime_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        silence_frames_threshold: int = 30
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process real-time audio stream and detect utterances with improved noise handling.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            callback: Optional callback for interim results
            silence_frames_threshold: Number of silence frames to consider end of utterance
            
        Yields:
            Transcription results for each detected utterance
        """
        if not self.initialized:
            logger.error("STT integration not properly initialized")
            yield {"error": "STT integration not initialized"}
            return
        
        # Start streaming
        await self.speech_recognizer.start_streaming()
        
        # Track state
        is_speaking = False
        silence_frames = 0
        max_silence_frames = silence_frames_threshold
        
        try:
            async for audio_chunk in audio_stream:
                # Process the audio chunk
                result = await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_chunk,
                    callback=callback
                )
                
                # Check for barge-in
                if result and result.barge_in_detected:
                    # Yield a barge-in notification
                    yield {
                        "barge_in_detected": True,
                        "energy_level": result.energy_level,
                        "is_final": False
                    }
                
                # Check for speech activity
                if result and result.is_final and result.text:
                    # Clean up transcription
                    transcription = self.cleanup_transcription(result.text)
                    
                    # Validate transcription
                    if transcription and self.is_valid_transcription(transcription):
                        logger.info(f"Utterance detected: {transcription}")
                        
                        # Yield the result
                        yield {
                            "transcription": transcription,
                            "original_transcription": result.text,
                            "duration": result.end_time - result.start_time,
                            "confidence": result.confidence,
                            "is_final": True,
                            "is_valid": True,
                            "barge_in_detected": result.barge_in_detected
                        }
                        
                        # Reset streaming for next utterance
                        await self.speech_recognizer.stop_streaming()
                        await self.speech_recognizer.start_streaming()
        
        except Exception as e:
            logger.error(f"Error in real-time audio processing: {e}")
            yield {"error": str(e)}
        
        finally:
            # Clean up
            await self.speech_recognizer.stop_streaming()