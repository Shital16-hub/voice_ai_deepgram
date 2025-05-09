"""
WebSocket handler for Twilio media streams with improved feedback loop prevention
and optimized response generation.
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

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.simple_google_stt import SimpleGoogleSTT
from speech_to_text.utils.speech_detector import SpeechActivityDetector

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, SILENCE_THRESHOLD

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

class WebSocketHandler:
    """
    Handles WebSocket connections for Twilio media streams with improved
    feedback loop prevention and optimized response generation.
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
        
        # Real-time audio buffer for improved processing
        self.rt_audio_buffer = asyncio.Queue(maxsize=100)  # Buffer for audio chunks
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management - optimized delays
        self.pause_after_response = 0.2  # Reduced from 0.3 for faster responsiveness
        self.min_words_for_valid_query = 1  # Minimum word count for valid query
        
        # Add ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.008  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20

        # Track recent system responses to detect echo
        self.recent_system_responses = []
        self.echo_detection_window = 5000  # 5 seconds in milliseconds
        
        # Add track identification (critical for feedback loop prevention)
        self.current_track = "unknown"
        
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
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with feedback loop prevention")

    def _dump_audio_buffer(self, filename: str = "debug_audio.wav") -> None:
        """
        Save the current audio buffer to a file for debugging.
        
        Args:
            filename: Filename to save the audio buffer
        """
        try:
            if not self.input_buffer:
                logger.warning("No audio in buffer to dump")
                return
                
            # Convert mulaw to PCM
            mulaw_bytes = bytes(self.input_buffer)
            pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
            
            # Convert to int16
            audio_int16 = (pcm_audio * 32767).astype(np.int16)
            
            # Save as WAV file
            import wave
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"Saved audio buffer to {filename}")
        except Exception as e:
            logger.error(f"Error saving audio buffer: {e}")
    
    def _test_google_speech(self) -> None:
        """Run a test on Google Cloud Speech to verify it's working."""
        try:
            # Create a simple tone for testing
            import numpy as np
            from scipy import signal
            
            # Generate a 1-second 1kHz tone
            sample_rate = 16000
            t = np.linspace(0, 1, sample_rate, endpoint=False)
            tone = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz tone at half amplitude
            
            # Convert to int16
            tone_int16 = (tone * 32767).astype(np.int16)
            tone_bytes = tone_int16.tobytes()
            
            logger.info(f"Generated test tone of {len(tone_bytes)} bytes")
            
            # Reset Google Speech session
            if self.google_speech_active:
                self.speech_client.stop_streaming()
            
            self.speech_client.start_streaming()
            self.google_speech_active = True
            
            # Send the tone
            logger.info("Sending test tone to Google Speech")
            self.speech_client.process_audio_chunk(tone_bytes)
            
            # Log completion
            logger.info("Test tone sent to Google Speech")
        except Exception as e:
            logger.error(f"Error in Google Speech test: {e}")
    
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
        
        # If we've recently been speaking, this might be echo
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_audio_output_time < self.echo_detection_window:
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
            
        # Be more lenient during troubleshooting - lower word count threshold
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.info(f"Transcription too short: {word_count} words")
            return False
            
        # More permissive during troubleshooting
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
        self.input_buffer = bytearray()
        self.output_buffer = bytearray()
        self.is_speaking = False
        self.is_processing = False
        self.speech_interrupted = False
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
                    optimize_streaming_latency=3  # Reduced to 3 for better quality
                )
                logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}, model ID: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing ElevenLabs TTS: {e}")
                # Will fall back to pipeline TTS integration
        
        # Send a welcome message
        await self.send_text_response("How can I help you today?", ws)
        
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
            
        # Extract track information
        track = data.get('track', 'unknown')
        self.current_track = track
        
        # Add explicit enhanced logging for ALL media packets
        logger.info(f"Media event received: {data}")
        logger.info(f"Received media packet: track={track}, size={len(data.get('media', {}).get('payload', ''))}")
        
        # Process all audio data during testing - remove track restriction
        # Only process inbound audio - MODIFIED to be more permissive during debugging
        # if track not in ['inbound', 'inbound_track']:
        #     logger.info(f"Received {track} track (not processing during normal operation)")
        #     return
            
        # Temporarily disable speaking check during troubleshooting
        # Skip processing if system is currently speaking - ADDED check to ensure this isn't causing issues
        # if self.is_speaking:
        #     logger.info("System is speaking, not processing inbound audio")
        #     return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        # Add more detailed logging
        logger.info(f"Processing media payload of size: {len(payload)}")
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Add direct logging of payload size
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")
            
            # Process with MulawBufferProcessor to solve "Very small mulaw data" warnings
            processed_data = self.mulaw_processor.process(audio_data)
            
            # Skip if still buffering
            if processed_data is None:
                # Added more informative log
                logger.info(f"Buffering audio chunk of size {len(audio_data)}, not enough for processing yet")
                return
            
            # Add to input buffer
            self.input_buffer.extend(processed_data)
            
            # Log buffer stats
            logger.info(f"Input buffer size now: {len(self.input_buffer)} bytes")
            
            # Limit buffer size to prevent memory issues
            if len(self.input_buffer) > AUDIO_BUFFER_SIZE * 2:
                # Keep the most recent portion
                excess = len(self.input_buffer) - AUDIO_BUFFER_SIZE
                self.input_buffer = self.input_buffer[excess:]
                logger.info(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            # Convert to PCM for speech detection
            pcm_audio = self.audio_processor.mulaw_to_pcm(processed_data)
            
            # Log PCM conversion and audio levels
            if len(pcm_audio) > 0:
                audio_level = np.mean(np.abs(pcm_audio)) * 100
                logger.info(f"Converted to PCM: {len(pcm_audio)} samples, Audio level: {audio_level:.2f}%")
            
            # Update ambient noise level (adaptive threshold)
            self._update_ambient_noise_level(pcm_audio)
            
            # Check if we should process based on time since last response
            time_since_last_response = time.time() - self.last_response_time
            if time_since_last_response < self.pause_after_response:
                # Still in pause period after last response, wait before processing new input
                logger.info(f"In pause period after response ({time_since_last_response:.1f}s < {self.pause_after_response:.1f}s)")
                return
            
            # Process buffer when it's large enough and not already processing
            # REDUCED minimum buffer size from AUDIO_BUFFER_SIZE to quarter
            min_buffer_size = AUDIO_BUFFER_SIZE // 4
            if len(self.input_buffer) >= min_buffer_size and not self.is_processing:
                logger.info(f"Processing audio buffer of size: {len(self.input_buffer)} bytes")
                async with self.processing_lock:
                    if not self.is_processing:  # Double-check within lock
                        self.is_processing = True
                        try:
                            await self._process_audio(ws)
                        finally:
                            self.is_processing = False
            else:
                logger.info(f"Not processing yet: buffer={len(self.input_buffer)}/{min_buffer_size}, is_processing={self.is_processing}")
            
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
        # Save audio buffer periodically for debugging
        if random.random() < 0.05:  # 5% chance to dump audio
            self._dump_audio_buffer(f"debug_audio_{int(time.time())}.wav")
    
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
        
        # Handle speaking markers for feedback prevention
        if name == "speaking_started":
            self.is_speaking = True
            self.last_audio_output_time = time.time() * 1000  # milliseconds
            logger.debug("System started speaking")
        elif name == "speaking_ended":
            # Add a short delay before clearing speaking flag
            await asyncio.sleep(0.1)
            self.is_speaking = False
            logger.debug("System stopped speaking")
    
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
        # More optimal chunking for telephony
        chunk_size = 640  # 80ms at 8kHz
        chunks = []
        
        # Split into equal-sized chunks
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
            
        return chunks
    
    async def _process_audio(self, ws) -> None:
        """
        Process accumulated audio data through the pipeline.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert buffer to PCM with enhanced processing
            try:
                mulaw_bytes = bytes(self.input_buffer)
                
                # Added log for debugging
                logger.info(f"Processing audio buffer of {len(mulaw_bytes)} bytes")
                
                # Convert using the enhanced audio processing
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                
                # Additional processing to improve recognition
                pcm_audio = self._preprocess_audio(pcm_audio)
                
                # Log audio levels for debugging
                audio_level = np.mean(np.abs(pcm_audio)) * 100
                logger.info(f"Preprocessed PCM audio level: {audio_level:.2f}%")
                
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
            
            # Reduced minimum threshold for processing
            if len(pcm_audio) < 320:  # Reduced from 640 to 320 samples (20ms at 16kHz)
                logger.warning(f"Audio chunk too small: {len(pcm_audio)} samples, accumulating more")
                return
    
            
            # Create a list to collect transcription results
            transcription_results = []
            
            # Define a callback to collect results
            async def transcription_callback(result):
                if hasattr(result, 'is_final') and result.is_final:
                    transcription_results.append(result)
                    logger.info(f"Received final Google Speech result: {result.text}")
                else:
                    # Log interim results too
                    logger.info(f"Received interim result: {result.text if hasattr(result, 'text') else str(result)}")
            
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
                logger.info(f"Sending {len(audio_bytes)} bytes to Google Speech")
                await self.speech_client.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=transcription_callback
                )
                
                # Wait a short time for any pending results
                await asyncio.sleep(0.3)  # Reduced from 0.5 for faster response
                
                # Get transcription if we have results
                if transcription_results:
                    # Use the best result based on confidence
                    best_result = max(transcription_results, key=lambda r: getattr(r, 'confidence', 0))
                    transcription = best_result.text
                    logger.info(f"Best transcription result: {transcription}")
                else:
                    # If no results, try stopping and restarting the session to get final results
                    if self.google_speech_active:
                        logger.info("No transcription results, stopping and restarting Google Speech")
                        final_transcription, _ = await self.speech_client.stop_streaming()
                        logger.info(f"Final transcription from session stop: {final_transcription}")
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
                            logger.info("Sending query to knowledge base")
                            query_result = await self.pipeline.query_engine.query(transcription)
                            response = query_result.get("response", "")
                            
                            # Optimize response for telephony if too long
                            if len(response.split()) > 50:  # If response is too long
                                logger.info(f"Original response too long ({len(response.split())} words), optimizing for telephony")
                                optimized_response = self._optimize_response_for_telephony(response)
                                response = optimized_response
                            
                            logger.info(f"Generated response: {response}")
                            
                            # Convert to speech with ElevenLabs TTS
                            if response:
                                # Use the send_media method with mark events for feedback prevention
                                await self.send_media_with_marks(response, ws)
                                
                                # Update state
                                self.last_transcription = transcription
                                self.last_response_time = time.time()
                        else:
                            logger.error("Pipeline does not have query_engine attribute")
                    except Exception as e:
                        logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                        
                        # Try to send a fallback response
                        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
                        await self.send_media_with_marks(fallback_message, ws)
                        self.last_response_time = time.time()
                else:
                    # If no valid transcription, reduce buffer size but keep some for context
                    half_size = len(self.input_buffer) // 2
                    self.input_buffer = self.input_buffer[half_size:]
                    logger.info(f"No valid transcription, reduced buffer to {len(self.input_buffer)} bytes")
            
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

    def _optimize_response_for_telephony(self, response: str) -> str:
        """
        Optimize a response for telephony by making it shorter and more direct.
        
        Args:
            response: Original response
            
        Returns:
            Optimized response
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        # If only 1-2 sentences, keep as is if not too long
        if len(sentences) <= 2 and len(response.split()) <= 30:
            return response
            
        # For longer responses, keep only the most important sentences
        if len(sentences) > 2:
            # Keep first sentence (usually the most direct answer)
            optimized = sentences[0]
            
            # If first sentence is very short, add another
            if len(optimized.split()) < 10 and len(sentences) > 1:
                optimized += " " + sentences[1]
                
            return optimized
            
        # Fallback - truncate to ~25 words
        words = response.split()
        if len(words) > 25:
            trunc_response = " ".join(words[:25])
            
            # Make sure we have ending punctuation
            if not trunc_response.endswith(('.', '!', '?')):
                trunc_response += "."
                
            return trunc_response
            
        return response
    
    async def send_media_with_marks(self, text: str, ws) -> None:
        """
        Send a text response using marks to enable feedback prevention.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Track this response for echo detection
            self.recent_system_responses.append(text)
            if len(self.recent_system_responses) > 5:  # Keep last 5 responses
                self.recent_system_responses.pop(0)
                
            # Send mark event to indicate speaking has started
            mark_start = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": "speaking_started"}
            }
            ws.send(json.dumps(mark_start))
            
            # Set speaking flag
            self.is_speaking = True
            self.last_audio_output_time = time.time() * 1000  # milliseconds
            
            # Convert text to speech with ElevenLabs
            try:
                if self.elevenlabs_tts:
                    # Use direct ElevenLabs TTS
                    speech_audio = await self.elevenlabs_tts.synthesize(text)
                    logger.info(f"Generated speech with ElevenLabs TTS: {len(speech_audio)} bytes")
                    
                    # If already in mulaw format, send directly
                    if self.elevenlabs_tts.container_format == "mulaw":
                        mulaw_audio = speech_audio
                    else:
                        # Convert to mulaw for Twilio
                        mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                else:
                    # Fall back to pipeline's TTS integration
                    speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                    mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                
                # Split into smaller chunks for better streaming
                chunks = self._split_audio_into_chunks(mulaw_audio)
                
                # Send the chunks
                for chunk in chunks:
                    # Create media message
                    media_msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": base64.b64encode(chunk).decode('utf-8')
                        }
                    }
                    ws.send(json.dumps(media_msg))
                    
                    # Small delay between chunks for better streaming
                    await asyncio.sleep(0.01)
                
                # Send mark event to indicate speaking has ended
                mark_end = {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "speaking_ended"}
                }
                ws.send(json.dumps(mark_end))
                
                # Add a small delay before clearing speaking flag
                await asyncio.sleep(0.1)
                self.is_speaking = False
                
                logger.info(f"Sent text response: '{text}'")
            except Exception as e:
                logger.error(f"Error with TTS: {e}", exc_info=True)
                
                # Clear speaking flag on error
                self.is_speaking = False
                
                # Send mark to end speaking state
                mark_end = {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "speaking_ended"}
                }
                ws.send(json.dumps(mark_end))
                
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
            self.is_speaking = False
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Legacy method to send a text response by converting to speech.
        Now uses the mark-based approach.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        await self.send_media_with_marks(text, ws)
    
    async def handle_interruption(self, ws) -> None:
        """
        Handle user interruption while system is speaking.
        
        Args:
            ws: WebSocket connection
        """
        if not self.is_speaking:
            return
            
        try:
            # Send clear event to stop current audio playback
            clear_event = {
                "event": "clear",
                "streamSid": self.stream_sid
            }
            ws.send(json.dumps(clear_event))
            
            # Reset speaking state
            self.is_speaking = False
            
            logger.info("Sent clear event to handle interruption")
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")
            self.is_speaking = False
    
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