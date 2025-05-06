"""
Google Cloud Speech-to-Text streaming implementation for real-time transcription.
"""
import os
import logging
import asyncio
import queue
import threading
import re
from typing import Dict, Any, Optional, Union, List, AsyncGenerator, Callable, Awaitable, Iterator
from dataclasses import dataclass, field

from google.cloud import speech
import numpy as np

from speech_to_text.config import config
from speech_to_text.google_cloud.models import TranscriptionConfig
from speech_to_text.google_cloud.exceptions import STTError, STTStreamingError

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
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
    Google Cloud Speech-to-Text streaming client.
    
    This class provides real-time streaming speech recognition
    optimized for telephony applications.
    """
    
    def __init__(
        self, 
        credentials_file: Optional[str] = None,
        model_name: Optional[str] = None,
        language: str = "en-US",
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        channels: int = 1,
        interim_results: bool = True,
        speech_context_phrases: Optional[List[str]] = None,
        enhanced_model: bool = True,  # Voice Activity Detection
        vad_enabled: bool = True,  # Voice Activity Detection
        barge_in_threshold: float = 0.02  # Energy threshold for barge-in detection
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            model_name: STT model to use (defaults to config)
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
        self.credentials_file = credentials_file
        
        # Set environment variable for credentials if provided
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            
        self.model_name = model_name or "latest_long"
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
        self.client = None
        self.streaming_config = None
        self.utterance_id = 0
        self.last_result = None
        
        # Audio processing
        self.audio_queue = None
        self.stop_event = None
        self.stream = None
        self.responses_thread = None
        self.result_callbacks = []
        
        # Ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.01  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Barge-in detection state
        self.agent_is_speaking = False
        self.potential_barge_in = False
        self.barge_in_frame_count = 0
        self.min_barge_in_frames = 3  # Minimum frames of speech to confirm barge-in
        
        # Compile the non-speech pattern for efficient filtering
        self.non_speech_pattern = re.compile('|'.join([
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
        ]))
        
        # Create the speech client
        try:
            self.client = speech.SpeechClient()
            logger.info("Initialized Google Cloud Speech-to-Text client")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Speech-to-Text client: {e}")
            raise
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get enhanced recognition configuration for Google Cloud Speech API."""
        # Get audio encoding enum
        encoding_enum = None
        if self.encoding.upper() == "LINEAR16":
            encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif self.encoding.upper() == "MULAW":
            encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
        else:
            encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Create RecognitionConfig
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            max_alternatives=2,  # Get multiple alternatives for better results
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            profanity_filter=False,
            model=self.model_name,
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
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
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
            # Telephone lines often have low frequency hum
            b, a = signal.butter(4, 80/(self.sample_rate/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply a mild de-emphasis filter to reduce hissing sounds in phone calls
            b, a = signal.butter(1, 3000/(self.sample_rate/2), 'low')
            de_emphasis = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply a simple noise gate to remove background noise
            noise_threshold = max(0.005, self.ambient_noise_level)
            noise_gate = np.where(np.abs(de_emphasis) < noise_threshold, 0, de_emphasis)
            
            # 4. Apply pre-emphasis filter to boost higher frequencies (for better speech detection)
            pre_emphasis = np.append(noise_gate[0], noise_gate[1:] - 0.97 * noise_gate[:-1])
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(pre_emphasis)) > 0:
                normalized = pre_emphasis / np.max(np.abs(pre_emphasis)) * 0.95
            else:
                normalized = pre_emphasis
            
            # 6. Apply a mild compression to even out volumes
            # Compression ratio 2:1 for values above threshold
            threshold = 0.2
            ratio = 0.5  # 2:1 compression
            
            def compressor(x, threshold, ratio):
                # If below threshold, leave it alone
                # If above threshold, compress it
                mask = np.abs(x) > threshold
                sign = np.sign(x)
                mag = np.abs(x)
                compressed = np.where(
                    mask,
                    threshold + (mag - threshold) * ratio,
                    mag
                )
                return sign * compressed
            
            compressed = compressor(normalized, threshold, ratio)
            
            # Re-normalize after compression
            if np.max(np.abs(compressed)) > 0:
                result = compressed / np.max(np.abs(compressed)) * 0.95
            else:
                result = compressed
            
            # Detect barge-in if agent is speaking
            if self.agent_is_speaking:
                self._detect_barge_in(result)
                
            # Update noise floor if energy is very low
            energy = np.mean(np.abs(audio_data))
            if energy < 0.01:
                self._update_ambient_noise_level(audio_data)
                
            return result
            
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
            logger.info(f"Reduced confidence due to uncertainty markers: {text}")
            
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
        
        try:
            # Reset state
            self.is_streaming = True
            self.audio_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.result_callbacks = []
            
            # Create the streaming configuration
            streaming_config = self._get_streaming_config()
            
            # Create the initial request with streaming config
            initial_request = speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )
            
            # Start a thread to handle streaming
            self.responses_thread = threading.Thread(
                target=self._stream_audio_data,
                args=(initial_request,)
            )
            self.responses_thread.daemon = True
            self.responses_thread.start()
            
            logger.info("Google Cloud Speech streaming session started successfully")
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            raise
    
    def _stream_audio_data(self, initial_request):
        """Handle streaming in a background thread to avoid blocking."""
        try:
            # Create a request generator function
            def request_generator():
                # First, yield the initial config request
                yield initial_request
                
                # Then keep yielding audio chunks from the queue
                while not self.stop_event.is_set():
                    try:
                        chunk = self.audio_queue.get(timeout=0.5)
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                        self.audio_queue.task_done()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in request generator: {e}")
                        break
            
            # Start streaming recognize with the request generator
            # Use only the requests parameter
            responses = self.client.streaming_recognize(
                requests=request_generator()
            )
            
            # Process results
            for response in responses:
                if self.stop_event.is_set():
                    break
                
                # Process each result in the response
                for result in response.results:
                    transcription_result = StreamingTranscriptionResult.from_google_result(
                        result,
                        energy_level=0.0
                    )
                    
                    # Set barge-in flag if detected
                    transcription_result.barge_in_detected = self.potential_barge_in
                    
                    # Save final results for later use
                    if result.is_final:
                        self.last_result = transcription_result
                    
                    # Process callbacks
                    for callback in self.result_callbacks:
                        try:
                            # Call callback in the event loop
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    callback(transcription_result),
                                    loop
                                )
                        except Exception as e:
                            logger.error(f"Error calling callback: {e}")
                
        except Exception as e:
            logger.error(f"Error in streaming audio: {e}")
    
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
            # Add callback if provided
            if callback and callback not in self.result_callbacks:
                self.result_callbacks.append(callback)
            
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
            
            # Convert to bytes for Google Cloud
            audio_bytes = (processed_audio * 32767).astype(np.int16).tobytes()
            
            # Add to queue if streaming is active
            if self.is_streaming and not self.stop_event.is_set():
                self.audio_queue.put(audio_bytes)
            
            # If barge-in detected, create a fake result to signal it immediately
            if self.potential_barge_in and self.agent_is_speaking:
                # Create a fake result for immediate barge-in handling
                barge_in_result = StreamingTranscriptionResult(
                    text="",  # No text yet
                    is_final=False,
                    confidence=0.0,
                    chunk_id=self.utterance_id,
                    barge_in_detected=True,
                    energy_level=energy_level
                )
                return barge_in_result
                
            # No immediate result to return
            return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and get final transcription.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        final_text = ""
        duration = 0.0
        
        try:
            # Set stop event to signal request generator to stop
            self.stop_event.set()
            
            # Wait for the responses thread to finish
            if self.responses_thread and self.responses_thread.is_alive():
                self.responses_thread.join(timeout=2.0)
            
            # Get the final result if available
            if self.last_result:
                final_text = self.last_result.text
                duration = self.last_result.end_time - self.last_result.start_time
                
                # Clean up the text
                final_text = self.cleanup_transcription(final_text)
                
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
        finally:
            # Reset state
            self.is_streaming = False
            self.stream = None
            self.responses_thread = None
            self.result_callbacks = []
            self.audio_queue = None
            self.stop_event = None
            self.last_result = None
            logger.info(f"Stopped Google Cloud STT streaming session. Final text: '{final_text}'")
        
        return final_text, duration