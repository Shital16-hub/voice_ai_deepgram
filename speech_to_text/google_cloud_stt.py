"""
Enhanced Google Cloud Speech-to-Text client for Voice AI Agent.

This module provides real-time streaming speech recognition with advanced
features like noise handling, barge-in detection, and adaptive thresholds.
"""
import os
import logging
import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union, AsyncGenerator
import numpy as np
import re
from dataclasses import dataclass, field
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from google.cloud import speech
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enhanced patterns for non-speech annotations
NON_SPEECH_PATTERNS = [
    r'\(.*?music.*?\)',         # (music), (tense music), etc.
    r'\(.*?wind.*?\)',          # (wind), (wind blowing), etc.
    r'\(.*?engine.*?\)',        # (engine), (engine revving), etc.
    r'\(.*?noise.*?\)',         # (noise), (background noise), etc.
    r'\(.*?sound.*?\)',         # (sound), (sounds), etc.
    r'\(.*?silence.*?\)',       # (silence), etc.
    r'\[.*?silence.*?\]',       # [silence], etc.
    r'\[.*?BLANK.*?\]',         # [BLANK_AUDIO], etc.
    r'\(.*?applause.*?\)',      # (applause), etc.
    r'\(.*?laughter.*?\)',      # (laughter), etc.
    r'\(.*?footsteps.*?\)',     # (footsteps), etc.
    r'\(.*?breathing.*?\)',     # (breathing), etc.
    r'\(.*?growling.*?\)',      # (growling), etc.
    r'\(.*?coughing.*?\)',      # (coughing), etc.
    r'\(.*?clap.*?\)',          # (clap), etc.
    r'\(.*?laugh.*?\)',         # (laughing), etc.
    r'\[.*?noise.*?\]',         # [noise], etc.
    r'\(.*?background.*?\)',    # (background), etc.
    r'\[.*?music.*?\]',         # [music], etc.
    r'\(.*?static.*?\)',        # (static), etc.
    r'\[.*?unclear.*?\]',       # [unclear], etc.
    r'\(.*?inaudible.*?\)',     # (inaudible), etc.
    r'\<.*?noise.*?\>',         # <noise>, etc.
    r'music playing',           # Common transcription
    r'background noise',        # Common transcription
    r'static',                  # Common transcription
    r'\b(um|uh|hmm|mmm)\b',     # Common filler words
]

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription with enhanced metadata."""
    text: str
    is_final: bool
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None
    alternatives: List[str] = field(default_factory=list)
    barge_in_detected: bool = False
    energy_level: float = 0.0  # Audio energy level for diagnostics
    
    @classmethod
    def from_google_result(cls, result, energy_level: float = 0.0):
        """Create a StreamingTranscriptionResult from a Google Cloud Speech result."""
        if not result.alternatives:
            return cls(
                text="",
                is_final=result.is_final,
                energy_level=energy_level
            )
            
        alt = result.alternatives[0]
        alternatives = [a.transcript for a in result.alternatives[1:]] if len(result.alternatives) > 1 else []
        
        # Extract word timing information if available
        words = None
        if hasattr(alt, 'words') and alt.words:
            words = []
            for word_info in alt.words:
                word = {
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds() if hasattr(word_info.start_time, 'total_seconds') else 0,
                    "end_time": word_info.end_time.total_seconds() if hasattr(word_info.end_time, 'total_seconds') else 0,
                    "confidence": alt.confidence if hasattr(alt, 'confidence') else 0.7
                }
                words.append(word)
                
        # Get timing information
        start_time = 0.0
        end_time = 0.0
        if words:
            start_time = words[0]["start_time"]
            end_time = words[-1]["end_time"]
        
        return cls(
            text=alt.transcript,
            is_final=result.is_final,
            confidence=alt.confidence if hasattr(alt, 'confidence') else 0.7,
            start_time=start_time,
            end_time=end_time,
            words=words,
            alternatives=alternatives,
            energy_level=energy_level
        )


class GoogleCloudStreamingSTT:
    """
    Enhanced Google Cloud Speech-to-Text streaming client for Voice AI Agent.

    This class provides real-time streaming speech recognition with:
    - Barge-in detection for interruptions
    - Adaptive noise floor tracking
    - Improved result filtering and formatting
    - Robust error recovery
    - Optimized for telephony applications
    """
    
    def __init__(
        self, 
        language: str = "en-US",
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        channels: int = 1,
        interim_results: bool = True,
        speech_context_phrases: Optional[List[str]] = None,
        enhanced_model: bool = True,
        vad_enabled: bool = True,  # Voice Activity Detection
        barge_in_threshold: float = 0.02  # Energy threshold for barge-in detection
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            language: Language code (BCP-47)
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results
            speech_context_phrases: Phrases to boost recognition
            enhanced_model: Whether to use enhanced telephony model
            vad_enabled: Whether to use Voice Activity Detection
            barge_in_threshold: Energy threshold for detecting user interruptions
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech modules not installed. Please run: pip install google-cloud-speech")

        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.speech_context_phrases = speech_context_phrases or [
            "pricing", "plan", "cost", "subscription", "service", "features",
            "support", "upgrade", "payment", "account", "question", "help"
        ]
        self.enhanced_model = enhanced_model
        self.vad_enabled = vad_enabled
        self.barge_in_threshold = barge_in_threshold
        
        # State management
        self.is_streaming = False
        self.stream = None
        self.streaming_client = None
        self.streaming_config = None
        self.utterance_id = 0
        self.last_result = None
        
        # Ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.01  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Audio buffer for more context in processing
        self.audio_buffer = []
        self.max_buffer_frames = 10  # Maximum number of frames to keep
        
        # Barge-in detection state
        self.agent_is_speaking = False
        self.potential_barge_in = False
        self.barge_in_frame_count = 0
        self.min_barge_in_frames = 3  # Minimum frames of speech to confirm barge-in
        
        # Compile the non-speech pattern for efficient filtering
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Create the client
        try:
            self.client = speech.SpeechClient()
            logger.info("Initialized Google Cloud Speech-to-Text client")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Speech-to-Text client: {e}")
            raise
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get enhanced recognition configuration for Google Cloud Speech API."""
        # Get audio encoding enum
        encoding_enum = getattr(speech.RecognitionConfig.AudioEncoding, self.encoding)
        
        # Create RecognitionConfig
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            max_alternatives=2,  # Get multiple alternatives for better results
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            profanity_filter=False,
            model="telephony" if self.enhanced_model else "command_and_search",
            use_enhanced=self.enhanced_model,
        )
        
        # Add speech contexts for better recognition
        if self.speech_context_phrases:
            speech_context = speech.SpeechContext(
                phrases=self.speech_context_phrases,
                boost=15.0  # Boost these phrases
            )
            config.speech_contexts.append(speech_context)
            
        return config
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get enhanced streaming configuration for Google Cloud Speech API."""
        recognition_config = self._get_recognition_config()
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            single_utterance=False  # Allow continuous recognition
        )
        
        return streaming_config
    
    def _update_noise_floor(self, audio_data: np.ndarray) -> None:
        """
        Update ambient noise level based on audio energy.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # If audio is silence (very low energy), use it to update noise floor
        if energy < 0.02:  # Very quiet audio
            self.noise_samples.append(energy)
            # Keep only recent samples
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level (with safety floor)
            if self.noise_samples:
                # Use 95th percentile to avoid outliers
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 95) * 2.0  # Set threshold just above noise
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
    
    def _detect_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Detect if user is interrupting the agent (barge-in).
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        if not self.agent_is_speaking:
            return False
            
        # Calculate audio energy
        energy = np.mean(np.abs(audio_data))
        
        # Check if energy exceeds barge-in threshold
        threshold = max(self.barge_in_threshold, self.ambient_noise_level * 3.0)
        if energy > threshold:
            # Count consecutive frames above threshold
            self.barge_in_frame_count += 1
            if self.barge_in_frame_count >= self.min_barge_in_frames:
                logger.info(f"Barge-in detected! Energy: {energy:.4f}, Threshold: {threshold:.4f}")
                self.potential_barge_in = True
                return True
        else:
            # Reset counter if energy drops below threshold
            self.barge_in_frame_count = 0
            
        return False
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data to reduce noise.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            from scipy import signal
            
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            b, a = signal.butter(4, 80/(self.sample_rate/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply band-pass filter for speech frequencies (300-3400 Hz)
            b, a = signal.butter(3, [300/(self.sample_rate/2), 3400/(self.sample_rate/2)], 'band')
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply pre-emphasis to boost higher frequencies
            pre_emphasized = np.append(filtered_audio[0], filtered_audio[1:] - 0.97 * filtered_audio[:-1])
            
            # 4. Simple noise gate (suppress very low amplitudes)
            noise_gate_threshold = max(0.015, self.ambient_noise_level * 2)
            noise_gate = np.where(np.abs(pre_emphasized) < noise_gate_threshold, 0, pre_emphasized)
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            # Log audio statistics for debugging
            orig_energy = np.mean(np.abs(audio_data))
            proc_energy = np.mean(np.abs(normalized))
            
            # Detect barge-in if agent is speaking
            if self.agent_is_speaking:
                self._detect_barge_in(normalized)
                
            # Update noise floor if energy is very low
            if orig_energy < 0.01:
                self._update_noise_floor(audio_data)
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data  # Return original audio if preprocessing fails
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Enhanced cleanup of transcription text by removing non-speech annotations and filler words.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log what was cleaned if substantial
        if text != cleaned_text and len(text) > len(cleaned_text) + 10:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
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
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
        
        # Estimate confidence based on presence of uncertainty markers
        confidence_estimate = 1.0
        if "?" in text or "[" in text or "(" in text or "<" in text:
            confidence_estimate = 0.6  # Lower confidence if it contains uncertainty markers
            
        if confidence_estimate < 0.7:
            logger.info(f"Transcription confidence too low: {confidence_estimate}")
            return False
            
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < min_words:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    async def start_streaming(self) -> None:
        """Start a new streaming recognition session."""
        if self.is_streaming:
            # Make sure to properly close existing session first
            await self.stop_streaming()
        
        logger.info("Starting new Google Cloud Speech streaming session")
        
        # Create streaming client
        self.streaming_client = speech.SpeechClient()
        
        # Create streaming config
        self.streaming_config = self._get_streaming_config()
        
        # Initialize request generator
        self.request_generator = self._create_request_generator()
        
        # Prime the generator by sending the config
        next(self.request_generator)
        
        # Create bidirectional streaming RPC - Fixed to properly pass requests
        self.stream = self.streaming_client.streaming_recognize(
            requests=self.request_generator
        )
        
        # Reset state
        self.utterance_id = 0
        self.last_result = None
        self.is_streaming = True
        self.audio_buffer = []
        self.barge_in_frame_count = 0
        self.potential_barge_in = False
    
    def _create_request_generator(self):
        """Create a generator for streaming requests."""
        # First, yield the streaming config
        yield speech.StreamingRecognizeRequest(streaming_config=self.streaming_config)
        
        # This is a generator that will be used to send audio data
        while True:
            # This will block until audio data is sent
            data = yield
            if data is None:
                break
            
            # Yield audio content
            yield speech.StreamingRecognizeRequest(audio_content=data)
    
    def set_agent_speaking(self, speaking: bool) -> None:
        """
        Set whether the agent is currently speaking, for barge-in detection.
        
        Args:
            speaking: True if agent is speaking, False otherwise
        """
        # Only log if the state changes
        if speaking != self.agent_is_speaking:
            logger.info(f"Agent speaking state changed: {speaking}")
            self.agent_is_speaking = speaking
            
            # Reset barge-in detection state when agent starts speaking
            if speaking:
                self.potential_barge_in = False
                self.barge_in_frame_count = 0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of streaming audio.
        
        Args:
            audio_chunk: Audio chunk as bytes or numpy array
            callback: Optional async callback for results
            
        Returns:
            Transcription result or None
        """
        if not self.is_streaming:
            logger.warning("Not streaming - call start_streaming() first")
            try:
                await self.start_streaming()
            except Exception as e:
                logger.error(f"Error starting streaming session: {e}")
                return None
        
        try:
            # Ensure audio_chunk is numpy array for preprocessing
            if isinstance(audio_chunk, bytes):
                # Convert from 16-bit PCM bytes to float32 array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_data = audio_chunk
                
            # Save original audio data for barge-in detection and VAD
            energy_level = np.mean(np.abs(audio_data))
            
            # Process audio for better recognition and detect barge-in
            processed_audio = self._preprocess_audio(audio_data)
            
            # Keep buffer of recent audio for context
            self.audio_buffer.append(processed_audio)
            if len(self.audio_buffer) > self.max_buffer_frames:
                self.audio_buffer.pop(0)
            
            # Convert to bytes for Google Cloud
            audio_bytes = (processed_audio * 32767).astype(np.int16).tobytes()
            
            # Send audio to Google Cloud Speech API
            try:
                self.request_generator.send(audio_bytes)
            except StopIteration:
                # If generator has stopped, restart streaming session
                logger.warning("Request generator stopped, restarting streaming session")
                await self.start_streaming()
                if not self.is_streaming:
                    return None
                    
                # Try sending audio again
                self.request_generator.send(audio_bytes)
            
            # Process available responses (non-blocking)
            results = []
            
            def process_responses():
                try:
                    # Get responses from the stream
                    for response in self.stream.responses:
                        # Only process if still streaming
                        if not self.is_streaming:
                            break
                            
                        # Process each result in the response
                        for result in response.results:
                            self.utterance_id += 1
                            
                            # Create result object with energy level for barge-in detection
                            transcription_result = StreamingTranscriptionResult.from_google_result(
                                result, 
                                energy_level=energy_level
                            )
                            
                            # Set barge_in_detected flag if we detected a barge-in
                            transcription_result.barge_in_detected = self.potential_barge_in
                            
                            # Add chunk_id for tracking
                            transcription_result.chunk_id = self.utterance_id
                            
                            # Add to results
                            results.append(transcription_result)
                            
                            # Save final results for stop_streaming
                            if result.is_final:
                                self.last_result = transcription_result
                            
                            # Call callback asynchronously
                            if callback:
                                loop = asyncio.get_event_loop()
                                asyncio.run_coroutine_threadsafe(
                                    callback(transcription_result), 
                                    loop
                                )
                                
                except Exception as e:
                    logger.error(f"Error processing responses: {e}")
                finally:
                    return results
            
            # Run in thread pool to avoid blocking
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = asyncio.get_event_loop().run_in_executor(
                    executor, 
                    process_responses
                )
                
                # Wait a short time for any results
                try:
                    response_results = await asyncio.wait_for(future, timeout=0.05)
                    if response_results:
                        # Return last result (most recent)
                        return response_results[-1]
                except asyncio.TimeoutError:
                    # No results yet, that's ok
                    pass
            
            # If no results from process_responses but barge-in detected, create a result
            if self.potential_barge_in and self.agent_is_speaking:
                # Create a fake result to indicate barge-in detection
                barge_in_result = StreamingTranscriptionResult(
                    text="",  # No text yet
                    is_final=False,
                    confidence=0.0,
                    chunk_id=self.utterance_id,
                    barge_in_detected=True,
                    energy_level=energy_level
                )
                return barge_in_result
                
            return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and return final text.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        final_text = ""
        duration = 0.0
        
        try:
            # Close the stream by sending None to the generator
            if self.request_generator:
                try:
                    self.request_generator.send(None)
                except (StopIteration, ValueError):
                    pass
            
            # Get the final result if available
            if self.last_result:
                final_text = self.last_result.text
                duration = self.last_result.end_time - self.last_result.start_time
                
                # Clean up the text
                final_text = self.cleanup_transcription(final_text)
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
        finally:
            self.is_streaming = False
            self.stream = None
            self.request_generator = None
            self.last_result = None
            logger.info(f"Stopped Google Cloud STT streaming session. Final text: '{final_text}'")
        
        return final_text, duration
    
    async def stream_file(
        self,
        file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None,
        chunk_size: int = 4096  # ~128ms at 16kHz, 16-bit mono
    ) -> List[StreamingTranscriptionResult]:
        """
        Stream audio file through Google Cloud Speech.
        
        Args:
            file_path: Path to audio file
            callback: Optional callback for results
            chunk_size: Size of audio chunks to process
            
        Returns:
            List of final recognition results
        """
        # Start streaming session
        await self.start_streaming()
        
        results = []
        try:
            # Open and read audio file
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                        
                    # Process chunk
                    result = await self.process_audio_chunk(chunk, callback)
                    
                    # Save final results
                    if result and result.is_final:
                        results.append(result)
                        
                    # Simulate real-time audio
                    await asyncio.sleep(0.02)  # Small delay for stability
                        
            # Stop streaming
            final_text, duration = await self.stop_streaming()
            
            return results
        except Exception as e:
            logger.error(f"Error streaming file: {e}")
            await self.stop_streaming()
            return results
    
    def set_speech_context(self, phrases: List[str], boost: float = 15.0) -> None:
        """
        Update speech context phrases for improved recognition.
        
        Args:
            phrases: List of phrases to boost
            boost: Boost value (0-20)
        """
        self.speech_context_phrases = phrases
        
        # This will take effect next time streaming is started
        logger.info(f"Updated speech context with {len(phrases)} phrases")