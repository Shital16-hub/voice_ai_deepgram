"""
TTS integration with Google Cloud TTS and improved cancellation support for barge-in.
"""
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List, Union

from text_to_speech import GoogleCloudTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration with Google Cloud TTS and improved barge-in support.
    """
    
    def __init__(
        self,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        enable_caching: bool = True,
        voice_type: str = "NEURAL2",  # Options: NEURAL2, CHIRP_3_HD, STANDARD, WAVENET, STUDIO
        container_format: str = "mulaw"  # For Twilio compatibility
    ):
        """
        Initialize the TTS integration with Google Cloud TTS.
        
        Args:
            voice_name: Voice name to use for Google Cloud TTS
            voice_gender: Voice gender (MALE, FEMALE, NEUTRAL)
            language_code: Language code (defaults to en-US)
            enable_caching: Whether to enable TTS caching
            voice_type: Type of voice to use (NEURAL2, CHIRP_3_HD, etc.)
            container_format: Audio format for Twilio compatibility
        """
        self.voice_name = voice_name
        self.voice_gender = voice_gender
        self.language_code = language_code
        self.enable_caching = enable_caching
        self.voice_type = voice_type
        self.container_format = container_format
        self.tts_client = None
        self.initialized = False
        
        # Parameters for better conversation flow and barge-in support
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # Reduced from 800ms to 500ms for better responsiveness
        
        # Track current TTS task for barge-in support
        self.current_tts_task = None
        self.barge_in_detected = False
        self.speech_interrupted = False
        
        # Voice mapping for different types
        self._setup_voice_mapping()
    
    def _setup_voice_mapping(self):
        """Setup voice mappings based on voice type."""
        # Default voices for different types
        if self.voice_type == "CHIRP_3_HD":
            # New Chirp 3 HD voices (latest and highest quality)
            self.default_voices = {
                "en-US": "en-US-Chirp3-HD-C",  # High-quality Chirp 3 voice
                "en-GB": "en-GB-Chirp3-HD-B",
                "es-ES": "es-ES-Chirp3-HD-A",
                "fr-FR": "fr-FR-Chirp3-HD-A",
                "de-DE": "de-DE-Chirp3-HD-A"
            }
        elif self.voice_type == "NEURAL2":
            # Neural2 voices (custom voice technology)
            self.default_voices = {
                "en-US": "en-US-Neural2-C",
                "en-US-MALE": "en-US-Neural2-D",
                "en-US-FEMALE": "en-US-Neural2-F",
                "en-GB": "en-GB-Neural2-A",
                "es-ES": "es-ES-Neural2-A",
                "fr-FR": "fr-FR-Neural2-A",
                "de-DE": "de-DE-Neural2-A"
            }
        elif self.voice_type == "STUDIO":
            # Studio voices (professional quality)
            self.default_voices = {
                "en-US": "en-US-Studio-Q",
                "en-GB": "en-GB-Studio-B",
                "en-AU": "en-AU-Studio-A"
            }
        elif self.voice_type == "WAVENET":
            # WaveNet voices (still good quality)
            self.default_voices = {
                "en-US": "en-US-Wavenet-C",
                "en-US-MALE": "en-US-Wavenet-D",
                "en-US-FEMALE": "en-US-Wavenet-F",
                "en-GB": "en-GB-Wavenet-A",
                "es-ES": "es-ES-Wavenet-A",
                "fr-FR": "fr-FR-Wavenet-A",
                "de-DE": "de-DE-Wavenet-A"
            }
        else:  # STANDARD
            # Standard voices (basic quality)
            self.default_voices = {
                "en-US": "en-US-Standard-C",
                "en-US-MALE": "en-US-Standard-D",
                "en-US-FEMALE": "en-US-Standard-F",
                "en-GB": "en-GB-Standard-A",
                "es-ES": "es-ES-Standard-A",
                "fr-FR": "fr-FR-Standard-A",
                "de-DE": "de-DE-Standard-A"
            }
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Get credentials from environment
            import os
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            
            # Determine the voice name if not provided
            if not self.voice_name:
                # Try to find a voice based on language and gender
                if self.voice_gender and self.voice_gender.upper() in ["MALE", "FEMALE"]:
                    key = f"{self.language_code}-{self.voice_gender.upper()}"
                    self.voice_name = self.default_voices.get(key, self.default_voices.get(self.language_code))
                else:
                    self.voice_name = self.default_voices.get(self.language_code)
            
            # Initialize the Google Cloud TTS client with Twilio-optimized settings
            self.tts_client = GoogleCloudTTS(
                credentials_file=credentials_file,
                voice_name=self.voice_name,
                voice_gender=self.voice_gender or "NEUTRAL",
                language_code=self.language_code or "en-US",
                container_format=self.container_format,  # mulaw for Twilio
                sample_rate=8000,  # 8kHz for Twilio telephony
                enable_caching=self.enable_caching
            )
            
            self.initialized = True
            logger.info(f"Initialized Google Cloud TTS with voice: {self.voice_name} "
                       f"({self.voice_type}) for {self.language_code}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            raise

    async def cancel_ongoing_tts(self) -> bool:
        """
        Cancel any ongoing TTS generation for barge-in support with immediate effect.
        
        Returns:
            True if an ongoing response was interrupted
        """
        if hasattr(self, 'current_tts_task') and self.current_tts_task:
            logger.info("Canceling ongoing TTS generation due to barge-in")
            
            # Set both flags immediately
            self.barge_in_detected = True
            self.speech_interrupted = True
            
            # Cancel the task
            self.current_tts_task.cancel()
            try:
                # Set a shorter timeout for cancellation
                await asyncio.wait_for(self.current_tts_task, timeout=0.05)  # Reduced from 0.1s for faster response
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("Timeout canceling TTS task, forcing cancellation")
            except Exception as e:
                logger.error(f"Error canceling TTS task: {e}")
            
            # Reset task reference immediately
            self.current_tts_task = None
            
            # Add a very short silence pause after interruption
            await asyncio.sleep(0.02)  # Reduced from 0.1s for faster response
            
            return True
        
        return False
    
    def _split_text_at_sentence_boundaries(self, text: str) -> List[str]:
        """
        Split text at sentence boundaries for improved speech pacing and barge-in opportunities.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence chunks
        """
        # Split on sentence endings, keeping the punctuation with the sentence
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences and combine very short ones
        result = []
        current = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If current sentence is very short, combine with next
            if len(current) > 0 and len(current) < 30:
                current += " " + sentence
            elif len(current) > 0:
                result.append(current)
                current = sentence
            else:
                current = sentence
        
        # Add the last sentence if there is one
        if current:
            result.append(current)
            
        return result
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech with barge-in support and enhanced silence for better
        conversation flow.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (mulaw format for Twilio)
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Reset barge-in and interruption flags
            self.barge_in_detected = False
            self.speech_interrupted = False
            
            # Store the TTS task reference for potential cancellation
            self.current_tts_task = asyncio.create_task(self._generate_speech_with_enhanced_quality(text))
            
            try:
                # Wait for TTS completion or interruption
                audio_data = await self.current_tts_task
                self.current_tts_task = None
                
                # If barge-in or interruption was detected, return empty audio
                if self.barge_in_detected or self.speech_interrupted:
                    logger.info("Returning empty audio due to barge-in/interruption")
                    return b''
                
                return audio_data
            except asyncio.CancelledError:
                logger.info("TTS generation was cancelled for barge-in")
                return b''  # Return empty audio if cancelled
                
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            raise

    async def _generate_speech_with_enhanced_quality(self, text: str) -> bytes:
        """
        Generate speech with enhanced quality for telephony with improved barge-in support.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (mulaw format)
        """
        # Split text at sentence boundaries for better pacing and barge-in opportunities
        sentences = self._split_text_at_sentence_boundaries(text)
        
        # For short text, process normally
        if len(sentences) <= 1 or len(text) < 100:
            # Get audio data from TTS client (already in mulaw format)
            audio_data = await self.tts_client.synthesize(text)
            
            # Add a pause after speech for better conversation flow
            if self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                
                # Append silence to audio data
                audio_data = audio_data + silence_data
            
            return audio_data
        
        # For multi-sentence text, process with pauses for natural speech
        all_audio = []
        
        for i, sentence in enumerate(sentences):
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            # Check for barge-in/interruption after each sentence
            if self.barge_in_detected or self.speech_interrupted:
                logger.info("Barge-in detected during speech generation, stopping")
                break
                
            # Generate speech for this sentence
            sentence_audio = await self.tts_client.synthesize(sentence)
            all_audio.append(sentence_audio)
            
            # Add shorter inter-sentence pauses for all but the last sentence
            if i < len(sentences) - 1:
                # Add a shorter pause between sentences (150ms) - reduced from 300ms
                sentence_pause_ms = 150
                inter_sentence_silence = b'\x00' * int(8000 * (sentence_pause_ms / 1000))
                all_audio.append(inter_sentence_silence)
                
                # Add a yield point for barge-in detection
                await asyncio.sleep(0.01)
        
        # Combine all audio chunks
        combined_audio = b''.join(all_audio)
        
        # Add the final pause after the complete speech if no interruption
        if self.add_pause_after_speech and not self.barge_in_detected and not self.speech_interrupted:
            # Generate silence based on pause_duration_ms
            silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
            silence_data = b'\x00' * silence_size
            
            # Append silence to audio data
            combined_audio = combined_audio + silence_data
        
        return combined_audio
    
    async def text_to_speech_streaming(
        self, 
        text_stream: List[str]
    ) -> bytes:
        """
        Convert a stream of text chunks to speech using Google Cloud TTS streaming.
        
        Args:
            text_stream: List of text chunks to synthesize
            
        Returns:
            Combined audio data as bytes
        """
        if not self.initialized:
            await self.init()
        
        try:
            # Use Google Cloud TTS streaming synthesis
            audio_chunks = []
            
            async for audio_chunk in self.tts_client.synthesize_streaming(
                self._text_generator(text_stream)
            ):
                if self.barge_in_detected or self.speech_interrupted:
                    logger.info("Barge-in detected during streaming TTS")
                    break
                    
                audio_chunks.append(audio_chunk)
            
            # If interrupted, return empty audio
            if self.barge_in_detected or self.speech_interrupted:
                return b''
            
            # Combine all chunks
            combined_audio = b''.join(audio_chunks)
            
            # Add final pause
            if self.add_pause_after_speech:
                silence_size = int(8000 * (self.pause_duration_ms / 1000))
                silence_data = b'\x00' * silence_size
                combined_audio = combined_audio + silence_data
            
            return combined_audio
            
        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            raise
    
    async def _text_generator(self, text_stream: List[str]):
        """Generate text chunks for streaming synthesis."""
        for text_chunk in text_stream:
            if not self.barge_in_detected and not self.speech_interrupted:
                yield text_chunk
            else:
                break
    
    def get_available_voices(self) -> Dict[str, List[str]]:
        """
        Get available voices for the current voice type and language.
        
        Returns:
            Dictionary mapping language codes to available voice names
        """
        return {
            "current_voice": self.voice_name,
            "voice_type": self.voice_type,
            "language": self.language_code,
            "available_voices": self.default_voices
        }