"""
TTS integration with improved cancellation support for barge-in.
"""
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List, Union

from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration with improved barge-in support.
    """
    
    def __init__(
        self,
        voice_id: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the TTS integration.
        
        Args:
            voice_id: Voice ID to use for ElevenLabs TTS
            enable_caching: Whether to enable TTS caching
        """
        self.voice_id = voice_id
        self.enable_caching = enable_caching
        self.tts_client = None
        self.initialized = False
        
        # Parameters for better conversation flow and barge-in support
        self.add_pause_after_speech = True
        self.pause_duration_ms = 500  # Reduced from 800ms to 500ms for better responsiveness
        
        # Track current TTS task for barge-in support
        self.current_tts_task = None
        self.barge_in_detected = False
        self.speech_interrupted = False
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Get API key from environment
            import os
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")  # Use the latest model
            
            # Initialize the ElevenLabs TTS client with settings optimized for Twilio
            self.tts_client = ElevenLabsTTS(
                api_key=api_key,
                voice_id=self.voice_id, 
                enable_caching=self.enable_caching,
                container_format="mulaw",  # Use mulaw for Twilio compatibility
                sample_rate=8000,  # Set sample rate for telephony
                model_id=model_id,
                optimize_streaming_latency=4  # Maximum optimization for real-time performance
            )
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs, voice: {self.voice_id or 'default'}, model: {model_id}")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
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
            Audio data as bytes
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
            Audio data as bytes with optimized quality
        """
        # Split text at sentence boundaries for better pacing and barge-in opportunities
        sentences = self._split_text_at_sentence_boundaries(text)
        
        # For short text, process normally
        if len(sentences) <= 1 or len(text) < 100:
            # Get audio data from TTS client with quality parameters
            params = {
                "optimize_streaming_latency": 2,  # Lower for better quality (0-4)
                "stability": 0.5,  # Balanced stability
                "clarity": 0.75,  # Improved clarity
                "style": 0.25,  # Some speaking style variation
            }
            
            audio_data = await self.tts_client.synthesize(text, **params)
            
            # Ensure the audio data has an even number of bytes
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
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
                
            # Generate speech for this sentence with enhanced quality
            params = {
                "optimize_streaming_latency": 2,
                "stability": 0.5,
                "clarity": 0.75,
                "style": 0.25,
            }
            sentence_audio = await self.tts_client.synthesize(sentence, **params)
            
            # Ensure even byte count
            if len(sentence_audio) % 2 != 0:
                sentence_audio = sentence_audio + b'\x00'
            
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