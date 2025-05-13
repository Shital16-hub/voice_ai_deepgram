"""
WebSocket handler for Twilio media streams with Google Cloud Speech integration
and ElevenLabs TTS without barge-in detection.
"""
import json
import base64
import asyncio
import logging
import time
import numpy as np
import re
from typing import Dict, Any, Callable, Awaitable, Optional, List, Tuple, Union
from scipy import signal
from google.cloud import speech
from google.cloud.speech import SpeechClient, StreamingRecognitionConfig, RecognitionConfig, StreamingRecognizeRequest
from google.cloud.speech_v1p1beta1 import SpeechAsyncClient
from google.api_core.exceptions import GoogleAPIError

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.simple_google_stt import SimpleGoogleSTT
from speech_to_text.utils.speech_detector import SpeechActivityDetector

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_BUFFER_SIZE

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

class RealTimeAudioBuffer:
    """Specialized buffer for real-time audio processing that prioritizes recent audio."""
    
    def __init__(self, max_size=32000):
        self.buffer = bytearray()
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def add(self, data):
        """Add data to the buffer, keeping only the most recent data if size exceeded."""
        async with self.lock:
            self.buffer.extend(data)
            # Keep only the most recent data
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]
    
    async def get(self, size=None):
        """Get data from the buffer."""
        async with self.lock:
            if size is None:
                return bytes(self.buffer)
            else:
                return bytes(self.buffer[-size:])
    
    async def clear(self):
        """Clear the buffer."""
        async with self.lock:
            self.buffer = bytearray()

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

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with Google Cloud Speech integration
    and ElevenLabs TTS without barge-in detection.
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
        
        # Add Mulaw buffer processor to address small mulaw data warnings
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)  # 80ms at 8kHz
        
        # State tracking
        self.is_speaking = False
        self.speech_interrupted = False
        self.current_audio_chunks = []
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
        self.last_audio_output_time = 0  # Track when we last sent audio
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Add real-time audio buffer for improved processing
        self.rt_audio_buffer = RealTimeAudioBuffer(max_size=48000)  # 3 seconds at 16kHz
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management
        self.pause_after_response = 0.3  # Reduced from 0.5 for faster responsiveness
        self.min_words_for_valid_query = 1  # Reduced from 2 for better sensitivity
        
        # Add ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.008  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20

        # Tracking for recent system responses
        self.recent_system_responses = []
        
        # Set up Google Cloud Speech
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=16000,
            enable_automatic_punctuation=True
        )
        
        # Ensure we start with a fresh speech recognition session
        self.google_speech_active = False
        
        # Set up ElevenLabs TTS with optimized settings for Twilio
        self.elevenlabs_tts = None
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with mulaw buffering")
    
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
    
    def _is_echo_of_system_speech(self, transcription: str) -> bool:
        """
        Check if a transcription appears to be an echo of the system's own speech.
        
        Args:
            transcription: The transcription to check
            
        Returns:
            True if the transcription appears to be an echo of the system's own speech
        """
        # No transcription, no echo
        if not transcription:
            return False
        
        # Check recent responses for similarity
        for phrase in self.recent_system_responses:
            # Clean up response text for comparison
            clean_phrase = self.cleanup_transcription(phrase)
            
            # Check for substring match
            if clean_phrase and len(clean_phrase) > 5:
                # If transcription contains a significant part of our recent speech
                if clean_phrase in transcription or transcription in clean_phrase:
                    similarity_ratio = len(clean_phrase) / max(len(transcription), 1)
                    
                    if similarity_ratio > 0.5:  # At least 50% match
                        logger.info(f"Detected echo of system speech: '{clean_phrase}' similar to '{transcription}'")
                        return True
                    
        return False
    
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
            
        # Basic checks for valid question patterns that should always pass
        question_starters = ["what", "who", "where", "when", "why", "how", "can", "could", "do", "does", "is", "are"]
        lowered_text = cleaned_text.lower()
        
        # Allow questions even if they contain uncertainty markers
        for starter in question_starters:
            if lowered_text.startswith(starter):
                logger.info(f"Allowing question pattern: {text}")
                return True
        
        # Check word count - more permissive (1 word can be valid)
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
        self.speech_interrupted = False
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0  # Reset audio output tracking
        self.conversation_active = True
        self.noise_samples = []  # Reset noise samples
        self.recent_system_responses = []  # Reset system responses
        self.google_speech_active = False  # Reset Google Speech session state
        
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
            
            # Skip processing if system is currently speaking
            if self.is_speaking:
                return
            
            # Process with MulawBufferProcessor to solve "Very small mulaw data" warnings
            processed_data = self.mulaw_processor.process(audio_data)
            
            # Skip if still buffering
            if processed_data is None:
                return
            
            # Add to input buffer
            self.input_buffer.extend(processed_data)
            
            # Limit buffer size to prevent memory issues
            if len(self.input_buffer) > MAX_BUFFER_SIZE:
                # Keep the most recent portion
                excess = len(self.input_buffer) - MAX_BUFFER_SIZE
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            # Convert to PCM for speech detection
            pcm_audio = self.audio_processor.mulaw_to_pcm(processed_data)
            
            # Update ambient noise level (adaptive threshold)
            self._update_ambient_noise_level(pcm_audio)
            
            # Check if we should process based on time since last response
            time_since_last_response = time.time() - self.last_response_time
            if time_since_last_response < self.pause_after_response:
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
    
    def _split_audio_into_chunks(self, audio_data: bytes) -> list:
        """
        Split audio into smaller chunks for processing.
        
        Args:
            audio_data: Audio data to split
            
        Returns:
            List of chunks
        """
        # Simple chunking without silence detection
        chunk_size = 800  # 100ms at 8kHz
        chunks = []
        
        # Split into equal-sized chunks
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
            
        return chunks
    
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
            
            # Define a callback to collect results
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.debug(f"Received final Google Speech result: {result.text}")
            
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
                
                # Check if this is an echo of the system's own speech
                if self._is_echo_of_system_speech(transcription):
                    logger.info("Detected echo of system speech, ignoring")
                    # Clear input buffer and return
                    self.input_buffer.clear()
                    return
                
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
                                    
                                    # Track this response for echo detection
                                    self.recent_system_responses.append(response)
                                    if len(self.recent_system_responses) > 5:  # Keep last 5 responses
                                        self.recent_system_responses.pop(0)
                                    
                                    # Set that we're speaking
                                    self.is_speaking = True
                                    
                                    # Send back to Twilio
                                    logger.info(f"Sending audio response ({len(mulaw_audio)} bytes)")
                                    await self._send_audio(mulaw_audio, ws)
                                    
                                    # Update state - we're no longer speaking
                                    self.is_speaking = False
                                    
                                    # Update state
                                    self.last_transcription = transcription
                                    self.last_response_time = time.time()
                                except Exception as tts_error:
                                    logger.error(f"Error with ElevenLabs TTS, falling back to pipeline TTS: {tts_error}")
                                    
                                    # Fall back to pipeline's TTS integration
                                    speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                    
                                    # Convert to mulaw for Twilio
                                    mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                                    
                                    # Track this response for echo detection
                                    self.recent_system_responses.append(response)
                                    if len(self.recent_system_responses) > 5:
                                        self.recent_system_responses.pop(0)
                                    
                                    # Set that we're speaking
                                    self.is_speaking = True
                                    
                                    # Send back to Twilio
                                    logger.info(f"Sending fallback audio response ({len(mulaw_audio)} bytes)")
                                    await self._send_audio(mulaw_audio, ws)
                                    
                                    # Update state - we're no longer speaking
                                    self.is_speaking = False
                                    
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
                            
                            # Set that we're speaking
                            self.is_speaking = True
                            
                            # Send the audio
                            await self._send_audio(mulaw_fallback, ws)
                            
                            # Update state - we're no longer speaking
                            self.is_speaking = False
                            
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
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """
        Send audio data to Twilio without concern for barge-in.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
        """
        try:
            # Update timing
            self.last_audio_output_time = time.time()
            
            # Ensure the audio data is valid
            if not audio_data or len(audio_data) == 0:
                logger.warning("Attempted to send empty audio data")
                self.is_speaking = False
                return
                    
            # Check connection status
            if not self.connected:
                logger.warning("WebSocket connection is closed, cannot send audio")
                self.is_speaking = False
                return
            
            # Split audio into smaller chunks
            chunks = self._split_audio_into_chunks(audio_data)
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks for playback")
            
            # Send the chunks
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
                    
                except Exception as e:
                    if "Connection closed" in str(e):
                        logger.warning(f"WebSocket connection closed while sending chunk {i+1}/{len(chunks)}")
                        self.connected = False
                        self.connection_active.clear()
                        self.is_speaking = False
                        return
                    else:
                        logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                        self.is_speaking = False
                        return
            
            logger.debug(f"Sent {i+1}/{len(chunks)} audio chunks ({len(audio_data)} bytes total)")
            
            # We're no longer speaking
            self.is_speaking = False
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
            self.is_speaking = False
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
            # Track this response for echo detection
            self.recent_system_responses.append(text)
            if len(self.recent_system_responses) > 5:  # Keep last 5 responses
                self.recent_system_responses.pop(0)
                
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
                    
                    # Set that we're speaking
                    self.is_speaking = True
                    
                    # Send audio
                    await self._send_audio(mulaw_audio, ws)
                    
                    # Update state - we're no longer speaking
                    self.is_speaking = False
                        
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
                
                # Set that we're speaking
                self.is_speaking = True
                
                # Send audio
                await self._send_audio(mulaw_audio, ws)
                
                # Update state - we're no longer speaking
                self.is_speaking = False
                
                logger.info(f"Sent text response using pipeline TTS: '{text}'")
                
                # Update last response time to add pause
                self.last_response_time = time.time()
            else:
                logger.error("TTS integration not available")
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
    
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