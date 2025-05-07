"""
Text-to-Speech integration module for Voice AI Agent.

This module provides classes and functions for integrating text-to-speech
capabilities with the Voice AI Agent system using ElevenLabs.
"""
import os
import logging
import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union, AsyncIterator

from text_to_speech import ElevenLabsTTS, RealTimeResponseHandler, AudioProcessor

logger = logging.getLogger(__name__)

class TTSIntegration:
    """
    Text-to-Speech integration for Voice AI Agent.
    
    Provides an abstraction layer for TTS functionality, handling initialization,
    single-text processing, and streaming capabilities with enhanced silence 
    handling for better barge-in performance.
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
        self.tts_handler = None
        self.initialized = False
        
        # Parameters for better conversation flow and barge-in support
        self.add_pause_after_speech = True
        self.pause_duration_ms = 800  # Increased from 500ms to 800ms for better barge-in opportunity
        
        # Track recently generated audio for echo detection
        self.recent_tts_outputs = []
        self.max_recent_outputs = 5
        
        # Track current TTS task for barge-in support
        self.current_tts_task = None
        self.barge_in_detected = False
    
    async def init(self) -> None:
        """Initialize the TTS components."""
        if self.initialized:
            return
            
        try:
            # Initialize the ElevenLabs TTS client with settings optimized for Twilio
            self.tts_client = ElevenLabsTTS(
                voice_id=self.voice_id, 
                enable_caching=self.enable_caching,
                container_format="mulaw",  # Use mulaw for Twilio compatibility
                sample_rate=8000,  # Set sample rate for telephony
                model_id=os.getenv("TTS_MODEL_ID", "eleven_turbo_v2"),  # Use the latest model
                optimize_streaming_latency=4  # Maximum optimization for real-time performance
            )
            
            # Initialize the RealTimeResponseHandler
            self.tts_handler = RealTimeResponseHandler(tts_streamer=None, tts_client=self.tts_client)
            
            self.initialized = True
            logger.info(f"Initialized TTS with ElevenLabs, voice: {self.voice_id or 'default'}, model: {self.tts_client.model_id}, format: mulaw for Twilio")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise

    async def cancel_ongoing_tts(self) -> bool:
        """
        Cancel any ongoing TTS generation for barge-in support.
        
        Returns:
            True if an ongoing response was interrupted
        """
        if hasattr(self, 'current_tts_task') and self.current_tts_task:
            logger.info("Canceling ongoing TTS generation due to barge-in")
            
            # Set barge-in flag
            self.barge_in_detected = True
            
            # Cancel the task
            self.current_tts_task.cancel()
            try:
                # Set a shorter timeout for cancellation
                await asyncio.wait_for(self.current_tts_task, timeout=0.1)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("Timeout canceling TTS task, forcing cancellation")
            except Exception as e:
                logger.error(f"Error canceling TTS task: {e}")
            
            # Reset task reference
            self.current_tts_task = None
            
            # Add a short silence pause after interruption
            await asyncio.sleep(0.1)
            
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
            # Reset barge-in flag
            self.barge_in_detected = False
            
            # Store the TTS task reference for potential cancellation
            self.current_tts_task = asyncio.create_task(self._generate_speech_with_enhanced_quality(text))
            
            try:
                # Wait for TTS completion or interruption
                audio_data = await self.current_tts_task
                self.current_tts_task = None
                
                # If barge-in was detected, return empty audio
                if self.barge_in_detected:
                    logger.info("Returning empty audio due to barge-in")
                    return b''
                
                # Track this text and audio for echo detection
                speech_info = {
                    "text": text,
                    "timestamp": time.time()
                }
                self.recent_tts_outputs.append(speech_info)
                
                # Limit the size of recent outputs
                if len(self.recent_tts_outputs) > self.max_recent_outputs:
                    self.recent_tts_outputs.pop(0)
                
                return audio_data
            except asyncio.CancelledError:
                logger.info("TTS generation was cancelled for barge-in")
                return b''  # Return empty audio if cancelled
                
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {e}")
            raise

    async def _generate_speech_with_enhanced_quality(self, text: str) -> bytes:
        """
        Generate speech with enhanced quality for telephony.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes with optimized quality
        """
        # Split text at sentence boundaries for better pacing
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
                logger.debug("Padded audio data to make even length")
            
            # Add a more substantial pause after speech for better conversation flow
            if self.add_pause_after_speech:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                
                # Append silence to audio data
                audio_data = audio_data + silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause after speech")
            
            return audio_data
        
        # For multi-sentence text, process with pauses for natural speech
        all_audio = []
        
        for i, sentence in enumerate(sentences):
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            # Check for barge-in after each sentence
            if self.barge_in_detected:
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
                # Add a shorter pause between sentences (300ms)
                sentence_pause_ms = 300
                inter_sentence_silence = b'\x00' * int(8000 * (sentence_pause_ms / 1000))
                all_audio.append(inter_sentence_silence)
        
        # Combine all audio chunks
        combined_audio = b''.join(all_audio)
        
        # Add the final pause after the complete speech
        if self.add_pause_after_speech and not self.barge_in_detected:
            # Generate silence based on pause_duration_ms
            silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
            silence_data = b'\x00' * silence_size
            
            # Append silence to audio data
            combined_audio = combined_audio + silence_data
            logger.debug(f"Added {self.pause_duration_ms}ms pause after multi-sentence speech")
        
        return combined_audio
    
    async def text_to_speech_streaming(
        self, 
        text_generator: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech conversion with barge-in support.
        
        Args:
            text_generator: Async generator producing text chunks
            
        Yields:
            Audio data chunks as they are generated
        """
        if not self.initialized:
            await self.init()
        
        # Reset barge-in flag
        self.barge_in_detected = False
        
        try:
            # Track if we need to add the final pause
            needs_final_pause = False
            accumulated_text = ""
            
            async for text_chunk in text_generator:
                # Check for barge-in
                if self.barge_in_detected:
                    logger.info("Barge-in detected during streaming, stopping")
                    break
                
                if not text_chunk:
                    continue
                
                # Accumulate text for better speech synthesis
                accumulated_text += text_chunk
                
                # Process text when we have enough or hit sentence boundaries
                if len(accumulated_text) >= 100 or any(c in accumulated_text for c in ['.', '!', '?']):
                    # Create a task for speech generation to allow cancellation
                    generation_task = asyncio.create_task(
                        self._generate_speech_with_enhanced_quality(accumulated_text)
                    )
                    
                    try:
                        # Wait for generation or cancellation
                        audio_data = await generation_task
                        
                        # Check for barge-in again
                        if self.barge_in_detected:
                            break
                        
                        # Track for echo detection
                        speech_info = {
                            "text": accumulated_text,
                            "timestamp": time.time()
                        }
                        self.recent_tts_outputs.append(speech_info)
                        if len(self.recent_tts_outputs) > self.max_recent_outputs:
                            self.recent_tts_outputs.pop(0)
                        
                        # Reset accumulated text
                        accumulated_text = ""
                        
                        # Only the last chunk should get the pause
                        needs_final_pause = True
                        yield audio_data
                    except asyncio.CancelledError:
                        logger.info("Speech generation cancelled during streaming")
                        break
            
            # Process any remaining text if no barge-in occurred
            if accumulated_text and not self.barge_in_detected:
                try:
                    # Create a task for speech generation
                    generation_task = asyncio.create_task(
                        self._generate_speech_with_enhanced_quality(accumulated_text)
                    )
                    
                    # Wait for generation or cancellation
                    audio_data = await generation_task
                    
                    # Track for echo detection
                    speech_info = {
                        "text": accumulated_text,
                        "timestamp": time.time()
                    }
                    self.recent_tts_outputs.append(speech_info)
                    
                    yield audio_data
                except asyncio.CancelledError:
                    logger.info("Final speech generation cancelled during streaming")
            
            # Add a pause at the end of the complete audio stream if needed
            if needs_final_pause and self.add_pause_after_speech and not self.barge_in_detected:
                # Generate silence based on pause_duration_ms
                silence_size = int(8000 * (self.pause_duration_ms / 1000))  # 8kHz for Twilio
                silence_data = b'\x00' * silence_size
                yield silence_data
                logger.debug(f"Added {self.pause_duration_ms}ms pause at end of streaming audio")
                
        except asyncio.CancelledError:
            logger.info("TTS streaming cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in streaming text to speech: {e}")
            raise
    
    def is_recent_speech(self, text: str, max_age_seconds: float = 5.0) -> bool:
        """
        Check if text is similar to recently generated speech (for echo detection).
        
        Args:
            text: Text to check
            max_age_seconds: Maximum age to consider
            
        Returns:
            True if the text is similar to recent speech
        """
        if not text or not self.recent_tts_outputs:
            return False
            
        current_time = time.time()
        
        # Check each recent output
        for output in self.recent_tts_outputs:
            # Skip if too old
            if current_time - output["timestamp"] > max_age_seconds:
                continue
                
            # Get similarity with recent output
            similarity = self._get_text_similarity(text, output["text"])
            
            # If high similarity, consider it recently spoken
            if similarity > 0.7:  # 70% similarity threshold
                logger.info(f"Detected echo of recent speech (similarity: {similarity:.2f})")
                return True
                
        return False
    
    def _get_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to lowercase for better comparison
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Simple substring check
        if text1 in text2 or text2 in text1:
            return 0.9
        
        # Count matching words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union