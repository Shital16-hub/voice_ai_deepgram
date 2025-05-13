"""
Optimized STT integration with direct MULAW support for Google Cloud Speech.
"""
import logging
import time
import asyncio
import re
import numpy as np
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union, AsyncIterator

# Import Google Cloud STT with MULAW support
from speech_to_text.simple_google_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.utils.audio_utils import load_audio_file

logger = logging.getLogger(__name__)

# Enhanced patterns for telephony noise
NON_SPEECH_PATTERNS = [
    r'\[.*?\]',           # [music], [noise], etc.
    r'\(.*?\)',           # (background noise), etc.
    r'\<.*?\>',           # <unclear>, etc.
    r'music playing',
    r'background noise',
    r'static',
    r'beep',
    r'tone',
    r'dial tone',
    r'busy signal',
    r'\b(um|uh|hmm|mmm|er|ah)\b',  # Filler words
]

class STTIntegration:
    """
    Optimized STT integration with Google Cloud and MULAW support.
    """
    
    def __init__(
        self,
        speech_recognizer: Optional[GoogleCloudStreamingSTT] = None,
        language: str = "en-US"
    ):
        """
        Initialize STT integration with MULAW optimization.
        
        Args:
            speech_recognizer: Google Cloud STT client
            language: Language code
        """
        self.speech_recognizer = speech_recognizer
        self.language = language
        self.initialized = True if speech_recognizer else False
        
        # Compile patterns for efficiency
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Adaptive noise handling
        self.noise_samples = []
        self.max_samples = 20
        self.ambient_noise_level = 0.01
        self.confidence_threshold = 0.7  # Lowered for telephony
        
        # Performance tracking
        self.total_transcriptions = 0
        self.valid_transcriptions = 0
        self.error_count = 0
    
    async def init(self, model_path: Optional[str] = None) -> None:
        """Initialize STT with MULAW support."""
        if self.initialized:
            return
            
        try:
            # Create Google Cloud STT with MULAW configuration
            self.speech_recognizer = GoogleCloudStreamingSTT(
                language=self.language,
                sample_rate=8000,  # 8kHz for MULAW
                encoding="MULAW",  # Direct MULAW support
                channels=1,
                interim_results=True,
                enhanced_model=True
            )
            
            self.initialized = True
            logger.info(f"STT initialized with MULAW support, language: {self.language}")
        except Exception as e:
            logger.error(f"Error initializing STT: {e}")
            raise
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """Update ambient noise level for adaptive processing."""
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # Update noise samples with quiet audio
        if energy < 0.02:  # Very quiet
            self.noise_samples.append(energy)
            if len(self.noise_samples) > self.max_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level
            if self.noise_samples:
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 95) * 2.0
                )
                logger.debug(f"Ambient noise level: {self.ambient_noise_level:.6f}")
    
    def _preprocess_audio_for_mulaw(self, audio_data: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing for MULAW audio."""
        try:
            from scipy import signal
            
            # Simple high-pass filter for MULAW (already compressed)
            b, a = signal.butter(3, 100/(8000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Gentle noise gate for MULAW
            noise_gate_threshold = max(0.02, self.ambient_noise_level)
            noise_gate = np.where(np.abs(filtered_audio) < noise_gate_threshold, 
                                  filtered_audio * 0.1, filtered_audio)
            
            # Light normalization (MULAW is already compressed)
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.9
            else:
                normalized = noise_gate
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data
    
    def cleanup_transcription(self, text: str) -> str:
        """Enhanced cleanup for telephony transcriptions."""
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove filler words at sentence start
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Fix punctuation spacing
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Collapse multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log significant changes
        if text != cleaned_text and len(text) > 10:
            logger.debug(f"Cleaned: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str, min_words: int = 1) -> bool:
        """Check if transcription is valid with telephony adjustments."""
        # Clean first
        cleaned_text = self.cleanup_transcription(text)
        
        if not cleaned_text:
            logger.debug("Empty transcription after cleaning")
            return False
        
        # Lower confidence threshold for telephony
        confidence_estimate = 1.0
        if any(marker in text for marker in ["?", "[", "(", "<"]):
            confidence_estimate = 0.6
            
        if confidence_estimate < self.confidence_threshold:
            logger.debug(f"Low confidence estimate: {confidence_estimate}")
            return False
            
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            logger.debug(f"Too few words: {word_count}")
            return False
            
        return True
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file with MULAW optimization."""
        if not self.initialized:
            return {"error": "STT not initialized"}
        
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)
            
            # For MULAW, minimal preprocessing
            if self.speech_recognizer.encoding == "MULAW":
                # Convert float32 to MULAW-like representation
                audio = self._preprocess_audio_for_mulaw(audio)
                # Convert to uint8 for MULAW
                audio_bytes = (audio * 127 + 128).astype(np.uint8).tobytes()
            else:
                # Regular processing for LINEAR16
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Process through streaming
            final_results = []
            
            async def collect_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                if callback:
                    await callback(result)
            
            # Start streaming and process
            await self.speech_recognizer.start_streaming()
            
            # Send audio in chunks
            chunk_size = 800  # 100ms for 8kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                await self.speech_recognizer.process_audio_chunk(chunk, collect_result)
            
            # Get final result
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # Use best result
            if transcription:
                pass
            elif final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
            
            # Clean and validate
            cleaned_text = self.cleanup_transcription(transcription)
            
            self.total_transcriptions += 1
            if self.is_valid_transcription(cleaned_text):
                self.valid_transcriptions += 1
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": 0.9,
                "duration": duration if duration > 0 else len(audio) / 8000,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            self.error_count += 1
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
        """Transcribe audio data with MULAW optimization."""
        if not self.initialized:
            return {"error": "STT not initialized"}
        
        start_time = time.time()
        
        try:
            # Convert to numpy array
            if isinstance(audio_data, bytes):
                # For MULAW bytes, convert directly
                if self.speech_recognizer.encoding == "MULAW":
                    audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
                    audio = (audio - 128) / 127.0  # Convert to float32 range
                else:
                    # For LINEAR16 bytes
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, list):
                audio = np.array(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Update noise level
            self._update_ambient_noise_level(audio)
            
            # Preprocess based on encoding
            if self.speech_recognizer.encoding == "MULAW":
                audio = self._preprocess_audio_for_mulaw(audio)
                # Convert back to MULAW bytes
                audio_bytes = (audio * 127 + 128).astype(np.uint8).tobytes()
            else:
                # Regular LINEAR16 processing
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Collect results
            final_results = []
            
            async def collect_result(result: StreamingTranscriptionResult):
                if result.is_final:
                    final_results.append(result)
                if callback:
                    await callback(result)
            
            # Process through streaming
            await self.speech_recognizer.start_streaming()
            await self.speech_recognizer.process_audio_chunk(audio_bytes, collect_result)
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # Get best result
            if not transcription and final_results:
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                
            # Clean and validate
            cleaned_text = self.cleanup_transcription(transcription)
            
            self.total_transcriptions += 1
            if self.is_valid_transcription(cleaned_text):
                self.valid_transcriptions += 1
            
            return {
                "transcription": cleaned_text,
                "original_transcription": transcription,
                "confidence": 0.9,
                "duration": duration if duration > 0 else len(audio) / 8000,
                "processing_time": time.time() - start_time,
                "is_final": True,
                "is_valid": self.is_valid_transcription(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            self.error_count += 1
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def start_streaming(self) -> None:
        """Start streaming session."""
        if not self.initialized:
            return
        
        await self.speech_recognizer.start_streaming()
        logger.debug("Started streaming transcription session")
    
    async def process_stream_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray, List[float]],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """Process streaming audio chunk with MULAW support."""
        if not self.initialized:
            return None
        
        try:
            # Convert to proper format
            if isinstance(audio_chunk, bytes):
                if self.speech_recognizer.encoding == "MULAW":
                    # Direct MULAW bytes
                    audio_bytes = audio_chunk
                else:
                    # Convert int16 bytes to float32
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            else:
                # Convert array to bytes
                if isinstance(audio_chunk, list):
                    audio_data = np.array(audio_chunk, dtype=np.float32)
                else:
                    audio_data = audio_chunk
                
                if self.speech_recognizer.encoding == "MULAW":
                    # Convert to MULAW bytes
                    audio_bytes = (audio_data * 127 + 128).astype(np.uint8).tobytes()
                else:
                    # Convert to LINEAR16 bytes
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Custom callback for cleaning
            async def clean_callback(result: StreamingTranscriptionResult):
                if result and result.text:
                    original_text = result.text
                    result.text = self.cleanup_transcription(result.text)
                    
                    if original_text != result.text:
                        logger.debug(f"Cleaned streaming: '{original_text}' -> '{result.text}'")
                    
                    if result.text and self.is_valid_transcription(result.text) and callback:
                        await callback(result)
            
            # Process chunk
            return await self.speech_recognizer.process_audio_chunk(
                audio_bytes, clean_callback
            )
            
        except Exception as e:
            logger.error(f"Error processing stream chunk: {e}")
            self.error_count += 1
            return None
    
    async def end_streaming(self) -> Tuple[str, float]:
        """End streaming and get final result."""
        if not self.initialized:
            return "", 0.0
        
        final_text, duration = await self.speech_recognizer.stop_streaming()
        
        # Clean the final result
        cleaned_text = self.cleanup_transcription(final_text)
        
        if final_text != cleaned_text:
            logger.info(f"Cleaned final: '{final_text}' -> '{cleaned_text}'")
        
        return cleaned_text, duration
    
    def optimize_for_telephony(self):
        """Optimize settings for telephony with MULAW."""
        # Adjust for telephony environment
        self.min_words_for_valid_query = 1  # More permissive
        self.confidence_threshold = 0.6  # Lower for telephony
        
        # Add telephony-specific patterns
        telephony_patterns = [
            r'\(.*?static.*?\)',
            r'\(.*?dial tone.*?\)',
            r'\(.*?busy signal.*?\)',
            r'\(.*?ring.*?\)',
            r'\(.*?beep.*?\)',
        ]
        
        # Update pattern
        import re
        all_patterns = NON_SPEECH_PATTERNS + telephony_patterns
        self.non_speech_pattern = re.compile('|'.join(all_patterns))
        
        logger.info("STT optimized for telephony with MULAW")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = (self.valid_transcriptions / max(1, self.total_transcriptions)) * 100
        
        return {
            "total_transcriptions": self.total_transcriptions,
            "valid_transcriptions": self.valid_transcriptions,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "ambient_noise_level": self.ambient_noise_level,
            "encoding": self.speech_recognizer.encoding if self.speech_recognizer else "Unknown",
            "sample_rate": self.speech_recognizer.sample_rate if self.speech_recognizer else "Unknown"
        }