"""
WebSocket handler for Twilio media streams with Google Cloud Speech integration
and ElevenLabs TTS.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
from typing import Dict, Any, Callable, Awaitable, Optional, List
from scipy import signal
from google.cloud import speech
from google.cloud.speech import SpeechClient, StreamingRecognitionConfig, RecognitionConfig, StreamingRecognizeRequest
from google.cloud.speech_v1p1beta1 import SpeechAsyncClient
from google.api_core.exceptions import GoogleAPIError

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.simple_google_stt import SimpleGoogleSTT

from telephony.audio_processor import AudioProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_BUFFER_SIZE
import concurrent.futures
import threading

# Import ElevenLabs TTS
from text_to_speech import ElevenLabsTTS

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
    # Additional noise patterns
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
]

class StreamingRecognitionResult:
    """A wrapper for Google Cloud Speech results to maintain API compatibility."""
    
    def __init__(self, text="", is_final=False, confidence=0.0, alternatives=None):
        self.text = text
        self.is_final = is_final
        self.confidence = confidence
        self.alternatives = alternatives or []
        self.start_time = 0.0
        self.end_time = 0.0
        self.chunk_id = 0
        self.barge_in_detected = False
        
    @classmethod
    def from_google_result(cls, result):
        """Create a StreamingRecognitionResult from a Google Cloud Speech result."""
        if not result.alternatives:
            return cls(is_final=result.is_final)
            
        alt = result.alternatives[0]
        return cls(
            text=alt.transcript,
            is_final=result.is_final,
            confidence=alt.confidence if hasattr(alt, 'confidence') else 0.7,
            alternatives=[a.transcript for a in result.alternatives[1:]]
        )

class GoogleCloudSpeechHandler:
    """Handler for Google Cloud Speech API streaming recognition."""
    
    def __init__(self, language_code="en-US", sample_rate=16000, enable_automatic_punctuation=True):
        """
        Initialize the Google Cloud Speech client.
        
        Args:
            language_code: Language code for recognition
            sample_rate: Audio sample rate in Hz
            enable_automatic_punctuation: Whether to enable automatic punctuation
        """
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.enable_automatic_punctuation = enable_automatic_punctuation
        
        # Create a speech client
        self.client = speech.SpeechClient()
        
        # State tracking
        self.is_streaming = False
        self.streaming_config = None
        self.result_callbacks = []
        
        # Audio streaming data
        self.audio_queue = asyncio.Queue()
        self.stream_task = None
        self._stop_event = asyncio.Event()
        
    async def start_streaming(self):
        """Start a new streaming recognition session."""
        if self.is_streaming:
            await self.stop_streaming()
            
        self.is_streaming = True
        self._stop_event.clear()
        
        # Configure recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=self.enable_automatic_punctuation,
            use_enhanced=True,  # Use enhanced model for better accuracy
            model="phone_call",  # Optimized for phone calls
            audio_channel_count=1  # Mono audio
        )
        
        # Create streaming config
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False  # Allow multiple utterances in a stream
        )
        
        # Start the streaming task
        self.stream_task = asyncio.create_task(self._stream_recognition())
        
        logger.info("Started Google Cloud Speech streaming session")
        
    async def process_audio_chunk(self, audio_chunk, callback=None):
        """
        Process an audio chunk through Google Cloud Speech.
        
        Args:
            audio_chunk: Audio data as bytes or numpy array
            callback: Optional callback for results
            
        Returns:
            None, as results are delivered via callback
        """
        if not self.is_streaming:
            logger.warning("Called process_audio_chunk but streaming is not active")
            try:
                await self.start_streaming()
            except Exception as e:
                logger.error(f"Error starting streaming session: {e}")
                return None
            
        # Convert numpy array to bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            # Ensure the data is float32 in [-1.0, 1.0] range
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            # Convert to int16
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_chunk
            
        # Add callback if provided
        if callback and callback not in self.result_callbacks:
            self.result_callbacks.append(callback)
            
        # Add audio chunk to the queue
        if self.is_streaming and not self._stop_event.is_set():
            await self.audio_queue.put(audio_bytes)
            
        return None  # Results come via callback
            
    async def stop_streaming(self):
        """Stop the streaming recognition session and get final transcription."""
        if not self.is_streaming:
            return "", 0.0
            
        try:
            # Signal that we're done adding audio
            self._stop_event.set()
            
            # Cancel the stream task if it's running
            if self.stream_task and not self.stream_task.done():
                try:
                    # Give it a moment to complete gracefully
                    await asyncio.wait_for(self.stream_task, timeout=2.0)
                except asyncio.TimeoutError:
                    # Cancel if it takes too long
                    self.stream_task.cancel()
                    try:
                        await self.stream_task
                    except asyncio.CancelledError:
                        pass
                        
            # Cleanup
            self.is_streaming = False
            self.result_callbacks = []
            
            # Return empty string as this matches the expected interface
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            self.is_streaming = False
            return "", 0.0
    
    async def _stream_recognition(self):
        """Main function to handle the streaming recognition."""
        try:
            # Create a generator that yields requests
            def generate_requests():
                # First request contains only the config
                yield speech.StreamingRecognizeRequest(
                    streaming_config=self.streaming_config
                )
                
                # Create a buffer for audio data
                audio_buffer = []
                
                # Set up a signal for stopping
                stop_event = threading.Event()
                
                # Thread function to collect audio data from the queue
                def audio_collector():
                    try:
                        while not stop_event.is_set() and not self._stop_event.is_set():
                            try:
                                # Get audio from queue with timeout
                                loop = asyncio.get_event_loop()
                                future = asyncio.run_coroutine_threadsafe(
                                    self.audio_queue.get(), loop
                                )
                                # Wait with timeout
                                audio_data = future.result(timeout=0.5)
                                
                                # Add to buffer
                                audio_buffer.append(audio_data)
                                
                                # Mark as done
                                loop.call_soon_threadsafe(self.audio_queue.task_done)
                            except concurrent.futures.TimeoutError:
                                # No data available yet
                                continue
                            except Exception as e:
                                logger.error(f"Error in audio collector: {e}")
                                break
                    except Exception as e:
                        logger.error(f"Error in audio collector thread: {e}")
                
                # Start the collector thread
                collector_thread = threading.Thread(target=audio_collector)
                collector_thread.daemon = True
                collector_thread.start()
                
                try:
                    # Yield audio data as it becomes available
                    while not self._stop_event.is_set():
                        # Check if we have audio data to send
                        if audio_buffer:
                            # Get the next audio chunk
                            audio_chunk = audio_buffer.pop(0)
                            
                            # Create and yield the request
                            req = speech.StreamingRecognizeRequest(
                                audio_content=audio_chunk
                            )
                            yield req
                        else:
                            # No audio available, wait a bit
                            time.sleep(0.1)
                finally:
                    # Signal thread to stop
                    stop_event.set()
                    collector_thread.join(timeout=1.0)
            
            # Create the requests
            request_generator = generate_requests()
            
            # Start streaming recognition
            responses = self.client.streaming_recognize(
                    requests=request_generator,
                    config=self.streaming_config
            )
            
            # Process responses
            for response in responses:
                if self._stop_event.is_set():
                    break
                    
                if not response.results:
                    continue
                    
                for result in response.results:
                    # Convert Google result to our format
                    streaming_result = StreamingRecognitionResult.from_google_result(result)
                    
                    # Call all registered callbacks
                    for callback in self.result_callbacks:
                        try:
                            # Create a task for each callback
                            loop = asyncio.get_event_loop()
                            loop.create_task(callback(streaming_result))
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
            
        except Exception as e:
            logger.error(f"Error in streaming recognition: {e}")
        finally:
            self.is_streaming = False

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with Google Cloud Speech integration
    and ElevenLabs TTS.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        self.audio_processor = AudioProcessor()
        
        # Audio buffers
        self.input_buffer = bytearray()
        self.output_buffer = bytearray()
        
        # State tracking
        self.is_speaking = False
        self.silence_start_time = None
        self.is_processing = False
        self.conversation_active = True
        self.sequence_number = 0  # For Twilio media sequence tracking
        
        # Connection state tracking
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Transcription tracker to avoid duplicate processing
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management
        self.pause_after_response = 2.0  # Wait 2 seconds after response before processing new input
        self.min_words_for_valid_query = 2  # Minimum words for a valid query
        
        # Add ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.01  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Set up Google Cloud Speech with barge-in detection
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=16000,
            encoding="LINEAR16",
            channels=1,
            interim_results=True,
            vad_enabled=True,  # Enable voice activity detection
            barge_in_threshold=0.02  # Set barge-in sensitivity
        )
        
        # Track agent speaking state for barge-in detection
        self.agent_is_speaking = False
        self.response_interrupted = False
        
        # Ensure we start with a fresh speech recognition session
        self.google_speech_active = False
        
        # Set up ElevenLabs TTS with optimized settings for Twilio
        self.elevenlabs_tts = None
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with Google Cloud Speech and ElevenLabs TTS")
    
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
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
            
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """
        Check if a transcription is valid and worth processing.
        
        Args:
            text: Transcription text
            
        Returns:
            True if the transcription is valid
        """
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if it's empty after cleaning
        if not cleaned_text:
            logger.info("Transcription contains only non-speech annotations")
            return False
            
        # Check if it matches any non-speech patterns
        if self.non_speech_pattern.search(text):
            logger.info(f"Transcription contains non-speech patterns: {text}")
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
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        return True
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        if not message:
            logger.warning("Received empty message")
            return
            
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            # Handle different event types
            if event_type == 'connected':
                await self._handle_connected(data, ws)
            elif event_type == 'start':
                await self._handle_start(data, ws)
            elif event_type == 'media':
                await self._handle_media(data, ws)
            elif event_type == 'stop':
                await self._handle_stop(data)
            elif event_type == 'mark':
                await self._handle_mark(data)
            else:
                logger.warning(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """
        Handle connected event.
        
        Args:
            data: Connected event data
            ws: WebSocket connection
        """
        logger.info(f"WebSocket connected for call {self.call_sid}")
        logger.info(f"Connected data: {data}")
        
        # Set connection state
        self.connected = True
        self.connection_active.set()
        
        # Start keep-alive task
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """
        Handle stream start event.
        
        Args:
            data: Start event data
            ws: WebSocket connection
        """
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
        
        # Reset state for new stream
        self.input_buffer.clear()
        self.output_buffer.clear()
        self.is_speaking = False
        self.is_processing = False
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.noise_samples = []  # Reset noise samples
        self.google_speech_active = False  # Reset Google Speech session state
        self.agent_is_speaking = False  # Reset agent speaking state
        self.response_interrupted = False  # Reset interruption state
        
        # Initialize ElevenLabs TTS if not already
        if self.elevenlabs_tts is None:
            try:
                # Get API key from environment if not explicitly provided
                import os
                api_key = os.environ.get("ELEVENLABS_API_KEY")
                voice_id = os.environ.get("TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Default to Bella voice
                model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")  # Use the latest model
                
                # Create ElevenLabs TTS client with improved parameters for telephony
                self.elevenlabs_tts = ElevenLabsTTS(
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=model_id,
                    container_format="mulaw",  # For Twilio compatibility
                    sample_rate=8000,  # For Twilio compatibility
                    optimize_streaming_latency=4  # Maximum optimization for real-time performance
                )
                logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}, model ID: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing ElevenLabs TTS: {e}")
                # Will fall back to pipeline TTS integration
        
        # Send a welcome message
        await self.send_text_response("I'm listening. How can I help you today?", ws)
        
        # Initialize Google Cloud Speech streaming session
        try:
            await self.speech_client.start_streaming()
            self.google_speech_active = True
            logger.info("Started Google Cloud Speech streaming session")
        except Exception as e:
            logger.error(f"Error starting Google Cloud Speech streaming session: {e}")
            self.google_speech_active = False
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """
        Handle media event with audio data.
        
        Args:
            data: Media event data
            ws: WebSocket connection
        """
        if not self.conversation_active:
            logger.debug("Conversation not active, ignoring media")
            return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Add to input buffer
            self.input_buffer.extend(audio_data)
            
            # Limit buffer size to prevent memory issues
            if len(self.input_buffer) > MAX_BUFFER_SIZE:
                # Keep the most recent portion
                excess = len(self.input_buffer) - MAX_BUFFER_SIZE
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            # Check if we should process based on time since last response
            time_since_last_response = time.time() - self.last_response_time
            if time_since_last_response < self.pause_after_response and not self.agent_is_speaking:
                # Still in pause period after last response, wait before processing new input
                logger.debug(f"In pause period after response ({time_since_last_response:.1f}s < {self.pause_after_response:.1f}s)")
                return
            
            # Process buffer when it's large enough and not already processing
            if len(self.input_buffer) >= AUDIO_BUFFER_SIZE and not self.is_processing:
                async with self.processing_lock:
                    if not self.is_processing:  # Double-check within lock
                        self.is_processing = True
                        try:
                            logger.info(f"Processing audio buffer of size: {len(self.input_buffer)} bytes")
                            await self._process_audio(ws)
                        finally:
                            self.is_processing = False
            
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """
        Handle stream stop event.
        
        Args:
            data: Stop event data
        """
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Close Google Cloud Speech streaming session
        if self.google_speech_active:
            try:
                await self.speech_client.stop_streaming()
                logger.info("Stopped Google Cloud Speech streaming session")
                self.google_speech_active = False
            except Exception as e:
                logger.error(f"Error stopping Google Cloud Speech streaming session: {e}")
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """
        Handle mark event for audio playback tracking.
        
        Args:
            data: Mark event data
        """
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data to reduce noise.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Simple noise gate (suppress very low amplitudes)
            noise_gate_threshold = max(0.015, self.ambient_noise_level)
            noise_gate = np.where(np.abs(filtered_audio) < noise_gate_threshold, 0, filtered_audio)
            
            # 3. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            # Log stats about the audio
            orig_energy = np.mean(np.abs(audio_data))
            proc_energy = np.mean(np.abs(normalized))
            logger.debug(f"Audio preprocessing: original energy={orig_energy:.4f}, processed energy={proc_energy:.4f}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}", exc_info=True)
            return audio_data  # Return original audio if preprocessing fails
    
    async def _process_audio(self, ws) -> None:
        """
        Process accumulated audio data through the pipeline with Google Cloud Speech
        and ElevenLabs TTS.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert buffer to PCM with enhanced processing
            try:
                mulaw_bytes = bytes(self.input_buffer)
                
                # Convert using the enhanced audio processing
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                
                # Additional processing to improve recognition
                pcm_audio = self._preprocess_audio(pcm_audio)
                
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
                
            # Add some checks for audio quality
            if len(pcm_audio) < 1000:  # Very small audio chunk
                logger.warning(f"Audio chunk too small: {len(pcm_audio)} samples")
                return
            
            # Create a list to collect transcription results
            transcription_results = []
            
            # Define a callback to collect results and handle barge-in
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.debug(f"Received final Google Speech result: {result.text}")
                
                # Check for barge-in detection
                if hasattr(result, 'barge_in_detected') and result.barge_in_detected and self.agent_is_speaking:
                    logger.info("Barge-in detected! User interrupted the agent.")
                    self.response_interrupted = True
                    # Stop current audio playback
                    self.output_buffer.clear()
                    # Process the interruption immediately
                    if hasattr(result, 'text') and result.text:
                        # Process this partial transcription immediately
                        await self._handle_barge_in(result.text, ws)
            
            # Set agent speaking state for barge-in detection
            self.speech_client.set_agent_speaking(self.agent_is_speaking)
            
            # Process audio through Google Cloud Speech
            try:
                # Convert to bytes format for Google Cloud Speech
                audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                
                # Make sure the Google Speech streaming session is active
                if not self.google_speech_active:
                    logger.info("Starting new Google Cloud Speech streaming session")
                    await self.speech_client.start_streaming()
                    self.google_speech_active = True
                
                # Process chunk with Google Cloud Speech
                await self.speech_client.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=transcription_callback
                )
                
                # Wait a short time for any pending results
                await asyncio.sleep(0.5)
                
                # Get transcription if we have results
                if transcription_results:
                    # Use the best result based on confidence
                    best_result = max(transcription_results, key=lambda r: getattr(r, 'confidence', 0))
                    transcription = best_result.text
                else:
                    # If no results, try stopping and restarting the session to get final results
                    if self.google_speech_active:
                        final_transcription, _ = await self.speech_client.stop_streaming()
                        await self.speech_client.start_streaming()
                        self.google_speech_active = True
                        transcription = final_transcription
                    else:
                        transcription = ""
                
                # Log before cleanup for debugging
                logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
                
                # Clean up transcription
                transcription = self.cleanup_transcription(transcription)
                logger.info(f"CLEANED TRANSCRIPTION: '{transcription}'")
                
                # Only process if it's a valid transcription
                if transcription and self.is_valid_transcription(transcription):
                    logger.info(f"Complete transcription: {transcription}")
                    
                    # Now clear the input buffer since we have a valid transcription
                    self.input_buffer.clear()
                    
                    # Don't process duplicate transcriptions
                    if transcription == self.last_transcription:
                        logger.info("Duplicate transcription, not processing again")
                        return
                    
                    # Process through knowledge base
                    try:
                        if hasattr(self.pipeline, 'query_engine'):
                            query_result = await self.pipeline.query_engine.query(transcription)
                            response = query_result.get("response", "")
                            
                            logger.info(f"Generated response: {response}")
                            
                            # Convert to speech with ElevenLabs TTS
                            if response:
                                # Try using direct ElevenLabs TTS first, fall back to pipeline TTS integration
                                try:
                                    if self.elevenlabs_tts:
                                        speech_audio = await self.elevenlabs_tts.synthesize(response)
                                        logger.info(f"Generated speech with ElevenLabs TTS: {len(speech_audio)} bytes")
                                    else:
                                        # Fall back to pipeline's TTS integration
                                        speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                        logger.info(f"Generated speech with pipeline TTS: {len(speech_audio)} bytes")
                                                                        
                                    # Convert to mulaw for Twilio if needed
                                    if not self.elevenlabs_tts or self.elevenlabs_tts.container_format != "mulaw":
                                        mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                                    else:
                                        # Already in mulaw format from ElevenLabs
                                        mulaw_audio = speech_audio
                                    
                                    # Send back to Twilio
                                    logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                    await self._send_audio(mulaw_audio, ws)
                                    
                                    # Update state
                                    self.last_transcription = transcription
                                    self.last_response_time = time.time()
                                except Exception as tts_error:
                                    logger.error(f"Error with ElevenLabs TTS, falling back to pipeline TTS: {tts_error}")
                                    
                                    # Fall back to pipeline's TTS integration
                                    speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                    
                                    # Convert to mulaw for Twilio
                                    mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                                    
                                    # Send back to Twilio
                                    logger.info(f"Sending fallback audio response ({len(mulaw_audio)} bytes)")
                                    await self._send_audio(mulaw_audio, ws)
                                    
                                    # Update state
                                    self.last_transcription = transcription
                                    self.last_response_time = time.time()
                    except Exception as e:
                        logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                        
                        # Try to send a fallback response
                        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
                        try:
                            if self.elevenlabs_tts:
                                fallback_audio = await self.elevenlabs_tts.synthesize(fallback_message)
                            else:
                                fallback_audio = await self.pipeline.tts_integration.text_to_speech(fallback_message)
                                
                            # Convert to mulaw for Twilio if needed
                            if not self.elevenlabs_tts or self.elevenlabs_tts.container_format != "mulaw":
                                mulaw_fallback = self.audio_processor.convert_to_mulaw(fallback_audio)
                            else:
                                mulaw_fallback = fallback_audio
                                
                            await self._send_audio(mulaw_fallback, ws)
                            self.last_response_time = time.time()
                        except Exception as e2:
                            logger.error(f"Failed to send fallback response: {e2}")
                else:
                    # If no valid transcription, reduce buffer size but keep some for context
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.debug(f"No valid transcription, reduced buffer to {len(self.input_buffer)} bytes")
            
            except Exception as e:
                logger.error(f"Error during Google Speech processing: {e}", exc_info=True)
                # If error, clear part of buffer and continue
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                
                # If we had a Google Speech session error, reset the session
                if self.google_speech_active:
                    try:
                        logger.info("Resetting Google Speech session after error")
                        await self.speech_client.stop_streaming()
                        await self.speech_client.start_streaming()
                        self.google_speech_active = True
                    except Exception as session_error:
                        logger.error(f"Error resetting Google Speech session: {session_error}")
                        self.google_speech_active = False
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)

    async def _handle_barge_in(self, partial_text: str, ws) -> None:
        """
        Handle user interruption (barge-in) with partial text.
        
        Args:
            partial_text: Partial transcription from the interruption
            ws: WebSocket connection
        """
        logger.info(f"Handling barge-in with partial text: '{partial_text}'")
        
        # Stop any current speech output
        self.output_buffer.clear()
        
        # Set flag that agent is no longer speaking
        self.agent_is_speaking = False
        self.speech_client.set_agent_speaking(False)
        
        # Clear the input buffer to start fresh
        self.input_buffer.clear()
        
        # Optional: Send a small audio cue that the agent was interrupted
        try:
            interrupt_message = "I'm listening."
            interrupt_audio = await self.elevenlabs_tts.synthesize(interrupt_message)
            await self._send_audio(interrupt_audio, ws)
        except Exception as e:
            logger.error(f"Error sending interruption cue: {e}")
    
    def _contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection with better noise filtering.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio contains potential speech
        """
        if len(audio_data) < 100:
            return False
            
        # Calculate RMS energy
        energy = np.sqrt(np.mean(np.square(audio_data)))
        
        # Calculate zero-crossing rate (helps distinguish speech from noise)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
        
        # Calculate spectral centroid (speech typically has higher centroids than noise)
        fft_data = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
        spectral_centroid = np.sum(freqs * fft_data) / (np.sum(fft_data) + 1e-10)
        
        # Get threshold based on ambient noise
        speech_threshold = max(0.01, self.ambient_noise_level * 2.5)
        
        # Log values for debugging
        logger.debug(f"Audio energy: {energy:.6f} (threshold: {speech_threshold:.6f}), "
                    f"zero crossings: {zero_crossings:.6f}, "
                    f"spectral centroid: {spectral_centroid:.2f}Hz")
        
        # Combined speech detection logic
        is_speech = (energy > speech_threshold) and \
                   (zero_crossings > 0.01) and \
                   (zero_crossings < 0.15) and \
                   (spectral_centroid > 500)  # Speech typically > 500Hz
        
        return is_speech
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """
        Send audio data to Twilio.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
        """
        try:
            # Ensure the audio data is valid
            if not audio_data or len(audio_data) == 0:
                logger.warning("Attempted to send empty audio data")
                return
                
            # Check connection status
            if not self.connected:
                logger.warning("WebSocket connection is closed, cannot send audio")
                return
            
            # Split audio into smaller chunks to avoid timeouts
            chunk_size = 4000  # Smaller chunks (250ms of audio at 8kHz mono)
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                try:
                    # Encode audio to base64
                    audio_base64 = base64.b64encode(chunk).decode('utf-8')
                    
                    # Create media message
                    message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": audio_base64
                        }
                    }
                    
                    # Send message
                    ws.send(json.dumps(message))
                    
                    # Add a small delay between chunks to prevent flooding
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.02)  # 20ms delay between chunks
                    
                except Exception as e:
                    if "Connection closed" in str(e):
                        logger.warning(f"WebSocket connection closed while sending chunk {i+1}/{len(chunks)}")
                        self.connected = False
                        self.connection_active.clear()
                        return
                    else:
                        logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                        return
            
            logger.debug(f"Sent {len(chunks)} audio chunks ({len(audio_data)} bytes total)")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
            if "Connection closed" in str(e):
                self.connected = False
                self.connection_active.clear()
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send a text response by converting to speech with ElevenLabs first.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Set flag that agent is speaking for barge-in detection
            self.agent_is_speaking = True
            self.speech_client.set_agent_speaking(True)
            self.response_interrupted = False
            
            # Convert text to speech with ElevenLabs
            if self.elevenlabs_tts:
                try:
                    # Use direct ElevenLabs TTS
                    speech_audio = await self.elevenlabs_tts.synthesize(text)
                    logger.info(f"Generated speech with ElevenLabs TTS: {len(speech_audio)} bytes")
                    
                    # If already in mulaw format, send directly
                    if self.elevenlabs_tts.container_format == "mulaw":
                        mulaw_audio = speech_audio
                    else:
                        # Convert to mulaw for Twilio
                        mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                        
                    # Send audio
                    await self._send_audio(mulaw_audio, ws)
                    logger.info(f"Sent text response using ElevenLabs: '{text}'")
                    
                    # Update last response time to add pause
                    self.last_response_time = time.time()
                    return
                except Exception as e:
                    logger.error(f"Error with ElevenLabs TTS, falling back to pipeline TTS: {e}")
            
            # Fall back to pipeline's TTS integration
            if hasattr(self.pipeline, 'tts_integration'):
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw for Twilio
                mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                logger.info(f"Sent text response using pipeline TTS: '{text}'")
                
                # Update last response time to add pause
                self.last_response_time = time.time()
            else:
                logger.error("TTS integration not available")
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
        finally:
            # Reset speaking state
            self.agent_is_speaking = False
            self.speech_client.set_agent_speaking(False)
    
    async def _keep_alive_loop(self, ws) -> None:
        """
        Send periodic keep-alive messages to maintain the WebSocket connection.
        """
        try:
            while self.conversation_active:
                await asyncio.sleep(10)  # Send every 10 seconds
                
                # Only send if we have a valid stream
                if not self.stream_sid or not self.connected:
                    continue
                    
                try:
                    message = {
                        "event": "ping",
                        "streamSid": self.stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        self.conversation_active = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")