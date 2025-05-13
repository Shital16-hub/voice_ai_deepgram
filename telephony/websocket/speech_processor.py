"""
Speech recognition and transcription processing.
"""
import logging
import re
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
import numpy as np

from speech_to_text.simple_google_stt import SimpleGoogleSTT
from telephony.websocket.audio_manager import AudioManager

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Handles speech recognition, transcription cleanup, and validation."""
    
    def __init__(self, pipeline):
        """Initialize speech processor."""
        self.pipeline = pipeline
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=16000,
            enable_automatic_punctuation=True
        )
        self.google_speech_active = False
        
        # Transcription patterns
        self.non_speech_patterns = self._compile_non_speech_patterns()
        self.echo_detection_history = []
        self.max_echo_history = 5
        self.min_words_for_valid_query = 1
    
    def _compile_non_speech_patterns(self) -> re.Pattern:
        """Compile regex patterns for non-speech annotations."""
        patterns = [
            r'\(.*?music.*?\)', r'\(.*?wind.*?\)', r'\(.*?engine.*?\)',
            r'\(.*?noise.*?\)', r'\(.*?sound.*?\)', r'\(.*?silence.*?\)',
            r'\[.*?silence.*?\]', r'\[.*?BLANK.*?\]', r'\(.*?applause.*?\)',
            r'\(.*?laughter.*?\)', r'\(.*?footsteps.*?\)', r'\(.*?breathing.*?\)',
            r'\(.*?static.*?\)', r'\[.*?unclear.*?\]', r'\(.*?inaudible.*?\)',
            r'music playing', r'background noise', r'static'
        ]
        return re.compile('|'.join(patterns))
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio data through speech recognition.
        
        Args:
            audio_data: Audio data as bytes
            callback: Optional callback for interim results
            
        Returns:
            Final transcription if available
        """
        try:
            # Convert to PCM format
            from telephony.audio_processor import AudioProcessor
            audio_processor = AudioProcessor()
            pcm_audio = audio_processor.mulaw_to_pcm(audio_data)
            
            # Preprocess audio
            pcm_audio = self._preprocess_audio(pcm_audio)
            
            # Convert to bytes for Google Speech
            audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
            
            # Initialize speech session if needed
            if not self.google_speech_active:
                await self.speech_client.start_streaming()
                self.google_speech_active = True
            
            # Collect results
            transcription_results = []
            
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.debug(f"Received final Google Speech result: {result.text}")
            
            # Process audio chunk
            await self.speech_client.process_audio_chunk(
                audio_chunk=audio_bytes,
                callback=transcription_callback
            )
            
            # Wait for results
            await asyncio.sleep(0.5)
            
            # Get best transcription
            if transcription_results:
                best_result = max(transcription_results, key=lambda r: getattr(r, 'confidence', 0))
                transcription = best_result.text
            else:
                # Try to get final results
                if self.google_speech_active:
                    final_transcription, _ = await self.speech_client.stop_streaming()
                    await self.speech_client.start_streaming()
                    self.google_speech_active = True
                    transcription = final_transcription
                else:
                    transcription = ""
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for better recognition."""
        try:
            from scipy import signal
            
            # High-pass filter
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Simple noise gate
            noise_gate_threshold = 0.015
            noise_gate = np.where(np.abs(filtered_audio) < noise_gate_threshold, 0, filtered_audio)
            
            # Normalize
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription text."""
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_patterns.sub('', text)
        
        # Remove filler words at beginning
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation and spaces
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """Check if transcription is valid and worth processing."""
        cleaned_text = self.cleanup_transcription(text)
        
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Check for question patterns
        question_starters = ["what", "who", "where", "when", "why", "how", "can", "could", "do", "does", "is", "are"]
        lowered_text = cleaned_text.lower()
        
        for starter in question_starters:
            if lowered_text.startswith(starter):
                logger.info(f"Allowing question pattern: {text}")
                return True
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Check if transcription is an echo of system's own speech."""
        if not transcription:
            return False
        
        for phrase in self.echo_detection_history:
            clean_phrase = self.cleanup_transcription(phrase)
            
            if clean_phrase and len(clean_phrase) > 5:
                if clean_phrase in transcription or transcription in clean_phrase:
                    similarity_ratio = len(clean_phrase) / max(len(transcription), 1)
                    
                    if similarity_ratio > 0.5:
                        logger.info(f"Detected echo: '{clean_phrase}' similar to '{transcription}'")
                        return True
                    
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo detection history."""
        self.echo_detection_history.append(response)
        if len(self.echo_detection_history) > self.max_echo_history:
            self.echo_detection_history.pop(0)
    
    async def stop_speech_session(self) -> None:
        """Stop the Google Speech session."""
        if self.google_speech_active:
            try:
                await self.speech_client.stop_streaming()
                self.google_speech_active = False
                logger.info("Stopped Google Cloud Speech streaming session")
            except Exception as e:
                logger.error(f"Error stopping Google Speech session: {e}")