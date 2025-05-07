"""
WebSocket handler for Twilio media streams with Google Cloud Speech integration
and ElevenLabs TTS with enhanced barge-in detection.
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
from speech_to_text.utils.speech_detector import SpeechActivityDetector

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor
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

class AudioFingerprinter:
    """Audio fingerprinting to recognize system's own speech."""
    
    def __init__(self, max_fingerprints=10):
        self.fingerprints = []
        self.max_fingerprints = max_fingerprints
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio fingerprint of outgoing speech."""
        if len(audio_data) < 1000:
            return
            
        # Create spectral fingerprint (simplified)
        fingerprint = self._extract_fingerprint(audio_data)
        timestamp = time.time()
        
        # Store with timestamp
        self.fingerprints.append({
            'fingerprint': fingerprint,
            'timestamp': timestamp,
            'duration': len(audio_data) / 16000  # Assuming 16kHz
        })
        
        # Keep only recent fingerprints
        if len(self.fingerprints) > self.max_fingerprints:
            self.fingerprints.pop(0)
    
    def _extract_fingerprint(self, audio_data):
        """Extract frequency domain fingerprint."""
        from scipy import signal
        
        # Create spectrogram
        f, t, Sxx = signal.spectrogram(audio_data, fs=16000, nperseg=512)
        
        # Reduce dimensions for efficient comparison
        signature = np.mean(Sxx, axis=1)
        
        # Normalize
        if np.max(signature) > 0:
            signature = signature / np.max(signature)
            
        return signature
        
    def is_echo(self, audio_data: np.ndarray, max_age_seconds=2.0) -> bool:
        """Check if incoming audio matches any recent outgoing audio."""
        if not self.fingerprints or len(audio_data) < 1000:
            return False
            
        # Create fingerprint of incoming audio
        incoming_fp = self._extract_fingerprint(audio_data)
        
        # Get current time for age calculation
        current_time = time.time()
        
        # Compare against stored fingerprints
        for fp_data in self.fingerprints:
            # Skip old fingerprints
            age = current_time - fp_data['timestamp']
            if age > max_age_seconds:
                continue
                
            # Compare fingerprints
            stored_fp = fp_data['fingerprint']
            if len(incoming_fp) == len(stored_fp):
                similarity = np.corrcoef(incoming_fp, stored_fp)[0, 1]
                
                # If similarity is high, it's likely an echo
                if similarity > 0.7:  # Threshold to be tuned
                    return True
        
        return False

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
    and ElevenLabs TTS with enhanced barge-in detection.
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
        # IMPORTANT: Always enable barge-in for better user experience
        self.barge_in_enabled = True  # Force enable barge-in
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
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Conversation flow management
        self.pause_after_response = 0.5  # Increased from 0.3 to 0.5 for better barge-in behavior
        self.min_words_for_valid_query = 2  # Minimum words for a valid query
        
        # Add ambient noise tracking for adaptive thresholds
        self.ambient_noise_level = 0.008  # Starting threshold
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Add fingerprinting for echo detection
        self.audio_fingerprinter = AudioFingerprinter(max_fingerprints=20)
        
        # Set up Google Cloud Speech
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=16000,
            enable_automatic_punctuation=True
        )
        
        # Create speech detector for barge-in
        self.speech_detector = SpeechActivityDetector(
            energy_threshold=0.04,  # Adjusted threshold for better detection
            consecutive_frames=2,   # Reduced for faster detection
            frame_duration=0.02     # 20ms frames
        )
        
        # Ensure we start with a fresh speech recognition session
        self.google_speech_active = False
        
        # Set up ElevenLabs TTS with optimized settings for Twilio
        self.elevenlabs_tts = None
        
        # Add barge-in sensitivity settings - INCREASED THRESHOLD
        self.barge_in_energy_threshold = 0.015  # Increased from 0.005 to reduce false detections
        self.barge_in_check_enabled = True
        
        # Track recent audio segments to detect echo
        self.recent_audio_energy = []
        self.max_recent_audio = 5  # Track last 5 audio segments
        
        # Track own audio output timestamps for echo detection
        self.own_audio_segments = []  # List of (timestamp, duration) tuples
        self.max_audio_segments = 10  # Keep track of last 10 segments
        
        # Track recent system responses for content-based echo detection
        self.recent_system_responses = []
        
        logger.info(f"WebSocketHandler initialized for call {call_sid} with barge-in support (FORCED ENABLED) and mulaw buffering")
    
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

    def _is_likely_echo(self, audio_data: np.ndarray, time_since_output: float) -> bool:
        """
        Determine if incoming audio is likely an echo of our own speech.
        
        Args:
            audio_data: Audio data as numpy array
            time_since_output: Time since our last audio output
            
        Returns:
            True if the audio is likely an echo
        """
        # If we haven't sent audio recently, it's not an echo
        if time_since_output > 2.0:
            return False
            
        # Very recent output (within 200ms) is almost certainly echo
        if time_since_output < 0.2:
            return True
        
        # Use audio fingerprinting for more reliable echo detection
        if hasattr(self, 'audio_fingerprinter'):
            if self.audio_fingerprinter.is_echo(audio_data):
                return True
        
        # Legacy fallback logic
        # Compute energy level
        audio_energy = np.mean(np.abs(audio_data))
        
        # Dynamic threshold based on time since last output
        # Higher threshold right after our speech, gradually lowering
        decay_factor = 1.0 - (time_since_output / 2.0)  # 1.0 at 0s, 0.0 at 2.0s
        dynamic_threshold = self.barge_in_energy_threshold * (1.0 + 19.0 * decay_factor)  # 20x to 1x
        
        # It's an echo if energy is below our dynamic threshold
        is_echo = audio_energy < dynamic_threshold
        
        logger.debug(f"Echo check: energy={audio_energy:.6f}, threshold={dynamic_threshold:.6f}, "
                   f"time_since_output={time_since_output:.3f}s, is_echo={is_echo}")
        
        return is_echo

    def _detect_barge_in_during_speech(self, audio_data: np.ndarray, time_since_output: float) -> bool:
        """
        Special barge-in detection during system speech with higher confidence requirements.
        
        Args:
            audio_data: Incoming audio data
            time_since_output: Time since last audio output
            
        Returns:
            True if confident this is a real barge-in, not an echo
        """
        # Check for speech activity using our specialized detector
        if hasattr(self, 'speech_detector'):
            is_speech = self.speech_detector.detect(audio_data)
            if is_speech:
                logger.info(f"Speech detector detected active speech during agent output")
                return True
        
        # Apply more stringent tests during speech
        
        # 1. Energy must be significantly higher than typical echo
        energy = np.mean(np.abs(audio_data))
        min_energy_threshold = 0.1  # Higher energy requirement during speech
        
        # 2. Must have speech-like patterns (not just noise or echo)
        frame_size = min(len(audio_data), 320)  # 20ms at 16kHz
        if frame_size <= 0:
            return False
            
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        frame_energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
        
        if len(frame_energies) >= 3:
            # Calculate energy variance (speech has more variance than steady noise/echo)
            energy_std = np.std(frame_energies)
            energy_mean = np.mean(frame_energies)
            variation_ratio = energy_std / energy_mean if energy_mean > 0 else 0
            
            # 3. Check spectral variety (more speech-like)
            has_speech_pattern = variation_ratio > 0.4  # Much higher than echo (typically <0.2)
            
            # 4. Has strong peak-to-average ratio (speech has peaks)
            peak = np.max(np.abs(audio_data))
            peak_ratio = peak / energy if energy > 0 else 0
            has_peaks = peak_ratio > 5.0  # Speech has stronger peaks than echo
            
            # 5. Dynamic energy growth (speech tends to increase in volume)
            has_energy_growth = False
            if len(frame_energies) > 5:
                first_half = frame_energies[:len(frame_energies)//2]
                second_half = frame_energies[len(frame_energies)//2:]
                first_avg = np.mean(first_half)
                second_avg = np.mean(second_half)
                has_energy_growth = second_avg > (first_avg * 1.5)  # 50% growth from first to second half
            
            # Log detailed detection info
            logger.debug(f"Speech barge-in check: energy={energy:.4f}, min_threshold={min_energy_threshold:.4f}, "
                        f"variation={variation_ratio:.3f}, peak_ratio={peak_ratio:.1f}, "
                        f"energy_growth={has_energy_growth}")
            
            # COMBINED DECISION - need multiple factors to confirm real interruption
            is_interruption = (
                energy > min_energy_threshold and
                has_speech_pattern and
                (has_peaks or has_energy_growth)
            )
            
            # Check against audio fingerprinting for additional confirmation
            if is_interruption and hasattr(self, 'audio_fingerprinter'):
                if self.audio_fingerprinter.is_echo(audio_data):
                    logger.info("Fingerprint matched our own speech - not a true interruption")
                    return False  # It matched our fingerprint, so it's not a barge-in
            
            return is_interruption
        
        return False  # Not enough frames to analyze

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
            
            # Check for substring match (more strict than before)
            if clean_phrase and len(clean_phrase) > 5:
                # If transcription contains a significant part of our recent speech
                if clean_phrase in transcription or transcription in clean_phrase:
                    similarity_ratio = len(clean_phrase) / max(len(transcription), 1)
                    
                    if similarity_ratio > 0.5:  # At least 50% match
                        logger.info(f"Detected echo of system speech: '{clean_phrase}' similar to '{transcription}'")
                        return True
                    
        return False

    def _contains_human_speech_pattern(self, audio_data: np.ndarray) -> bool:
        """
        More specialized detector that identifies human speech patterns vs. echoes.
        Uses speech-specific features that distinguish it from echoes and noise.
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            True if the audio contains likely human speech patterns
        """
        if len(audio_data) < 1600:  # Need at least 100ms at 16kHz
            return False
        
        # 1. Check overall energy
        energy = np.mean(np.abs(audio_data))
        if energy < 0.01:  # Very quiet
            return False
        
        # 2. Get spectral features that distinguish speech
        from scipy import signal
        
        # Generate spectrum
        freqs, power = signal.welch(audio_data, fs=16000, nperseg=512)
        
        # Define speech-relevant frequency bands
        # Focus on bands where human speech is concentrated (formants)
        formant1_band = (300, 1000)   # First formant region
        formant2_band = (1000, 2500)  # Second formant region
        high_band = (2500, 4000)      # Higher frequencies
        
        # Extract power in each band
        f1_power = np.mean(power[(freqs >= formant1_band[0]) & (freqs <= formant1_band[1])])
        f2_power = np.mean(power[(freqs >= formant2_band[0]) & (freqs <= formant2_band[1])])
        high_power = np.mean(power[(freqs >= high_band[0]) & (freqs <= high_band[1])])
        
        # Calculate ratios that are characteristic of speech
        f1_f2_ratio = f1_power / f2_power if f2_power > 0 else 0
        speech_high_ratio = (f1_power + f2_power) / high_power if high_power > 0 else 0
        
        # 3. Check spectral balance - speech has specific formant patterns
        is_speech_spectrum = (
            f1_f2_ratio > 0.8 and f1_f2_ratio < 3.0 and  # Typical range for speech formants
            speech_high_ratio > 2.0                       # Speech has more energy in formant regions
        )
        
        # 4. Check for energy variations over time (syllable-like patterns)
        frame_size = 320  # 20ms
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        frame_energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
        
        if len(frame_energies) >= 5:
            # Speech typically has syllabic pattern (energy goes up and down)
            peaks = 0
            for i in range(1, len(frame_energies)-1):
                if (frame_energies[i] > frame_energies[i-1] and 
                    frame_energies[i] > frame_energies[i+1] and
                    frame_energies[i] > 0.015):  # Real peak, not just noise
                    peaks += 1
            
            has_syllabic_pattern = peaks >= 2  # At least 2 energy peaks (syllables)
        else:
            has_syllabic_pattern = False
        
        # Combine spectral and temporal features
        return is_speech_spectrum and has_syllabic_pattern
        
    def _get_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using a simple approach."""
        # Convert to lowercase for better comparison
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Very simple similarity check - see if one text is contained in the other
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
        Modified to handle questions properly for barge-in scenarios.
        
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
            
        # Basic checks for valid question patterns that should always pass
        question_starters = ["what", "who", "where", "when", "why", "how", "can", "could", "do", "does", "is", "are"]
        lowered_text = cleaned_text.lower()
        
        # Allow questions even if they contain uncertainty markers
        for starter in question_starters:
            if lowered_text.startswith(starter):
                logger.info(f"Allowing question pattern: {text}")
                return True
        
        # Estimate confidence based on presence of uncertainty markers
        # BUT DON'T COUNT QUESTION MARKS as uncertainty markers
        confidence_estimate = 1.0
        uncertainty_markers = ["[", "(", "<"]  # Removed '?' from this list
        if any(marker in text for marker in uncertainty_markers):
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
        Handle incoming WebSocket message with barge-in support.
        
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
            elif event_type == 'bargein':  # Handler for explicit barge-in events
                await self._handle_bargein(data, ws)
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
        Handle stream start event with barge-in configuration.
        
        Args:
            data: Start event data
            ws: WebSocket connection
        """
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
        
        # Extract barge-in configuration
        start_data = data.get('start', {})
        custom_params = start_data.get('customParameters', {})
        twilio_barge_in = custom_params.get('bargeIn', 'false')
        
        # IMPORTANT: Force enable barge-in regardless of Twilio setting
        # But respect Twilio's setting by logging the discrepancy
        self.barge_in_enabled = True
        
        if twilio_barge_in.lower() != 'true':
            logger.info(f"Twilio bargeIn is '{twilio_barge_in}' but ENABLING anyway for improved experience")
            
            # Send a Mark event to enable barge-in if not already enabled
            try:
                mark_message = {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {
                        "name": "enable_barge_in"
                    }
                }
                ws.send(json.dumps(mark_message))
                logger.info("Sent mark to enable barge-in")
            except Exception as e:
                logger.error(f"Error sending barge-in mark: {e}")
        else:
            logger.info("Twilio bargeIn is already enabled")
        
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
        self.own_audio_segments = []  # Reset audio segment tracking
        self.google_speech_active = False  # Reset Google Speech session state
        self.recent_system_responses = []  # Reset system responses
        
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
        Handle media event with audio data, with enhanced barge-in detection.
        
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
            
            # Calculate time since last audio output
            time_since_output = time.time() - self.last_audio_output_time
            
            # Convert to PCM for speech detection
            pcm_audio = self.audio_processor.mulaw_to_pcm(processed_data)
            
            # Update ambient noise level (adaptive threshold)
            self._update_ambient_noise_level(pcm_audio)
            
            # SPECIAL HANDLING DURING SYSTEM SPEECH
            if self.is_speaking and self.barge_in_enabled:
                # Apply echo detection first - if it's an echo, ignore it
                if self._is_likely_echo(pcm_audio, time_since_output):
                    logger.debug("Detected echo during system speech, ignoring")
                    return
                    
                # Use speech detector for more reliable barge-in detection during speech
                if hasattr(self, 'speech_detector'):
                    is_speech = self.speech_detector.detect(pcm_audio)
                    if is_speech:
                        logger.info("BARGE-IN DETECTED by speech detector! Interrupting...")
                        self.speech_interrupted = True
                        # Clear current audio chunks to stop sending
                        self.current_audio_chunks = []
                        # Reset the state
                        self.is_speaking = False
                        # Process this interrupting audio immediately
                        await self._process_audio(ws)
                        return
                
                # Apply stronger barge-in detection with higher threshold for true interruptions
                # We need to be more certain this is an interruption during our speech
                barge_in_detected = self._detect_barge_in_during_speech(pcm_audio, time_since_output)
                
                if barge_in_detected:
                    logger.info("BARGE-IN DETECTED during system speech! Interrupting...")
                    self.speech_interrupted = True
                    # Clear current audio chunks to stop sending
                    self.current_audio_chunks = []
                    # Reset the state
                    self.is_speaking = False
                    # Process this interrupting audio immediately
                    await self._process_audio(ws)
                    return
            
            # Check if we should process based on time since last response
            time_since_last_response = time.time() - self.last_response_time
            if time_since_last_response < self.pause_after_response and not self.speech_interrupted:
                # Still in pause period after last response, wait before processing new input
                # But continue if we detected a barge-in
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

    async def _handle_bargein(self, data: Dict[str, Any], ws) -> None:
        """
        Handle explicit barge-in event from Twilio.
        
        Args:
            data: Barge-in event data
            ws: WebSocket connection
        """
        logger.info(f"EXPLICIT TWILIO BARGE-IN EVENT RECEIVED: {data}")
        
        # Immediate stop any ongoing speech
        self.speech_interrupted = True
        
        # Clear current audio chunks to stop sending
        self.current_audio_chunks = []
        
        # Update state
        self.is_speaking = False
        self.is_processing = False
        
        # Reset time tracking to allow immediate processing
        self.last_response_time = 0
        
        # Add a small wait to ensure audio buffers are cleared
        await asyncio.sleep(0.1)
        
        logger.info("Explicit barge-in processed - ready for new user input")
    
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
    
    def _split_audio_into_chunks_with_silence_detection(self, audio_data: bytes, chunk_size: int = 800) -> list:
        """
        Split audio into chunks with silence detection for improved barge-in.
        Add longer pauses at sentence boundaries for better barge-in opportunities.
        
        Args:
            audio_data: Audio data to split
            chunk_size: Base chunk size
            
        Returns:
            List of chunks with appropriate silences
        """
        # Convert to PCM for analysis
        pcm_data = self.audio_processor.mulaw_to_pcm(audio_data)
        
        # Simple sentence boundary detection from audio (rough approximation)
        # Looking for prolonged drops in energy that might represent pauses
        frame_size = 1600  # 100ms at 16kHz
        frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size)]
        frame_energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
        
        # Detect potential sentence boundaries as places where energy drops significantly
        boundaries = []
        for i in range(1, len(frame_energies)):
            if frame_energies[i] < frame_energies[i-1] * 0.3:  # 70% drop in energy
                boundaries.append(i * frame_size)
        
        # Now split the audio with extended silence at these boundaries
        chunks = []
        last_pos = 0
        
        for boundary in boundaries:
            # Add chunks up to this boundary
            for i in range(last_pos, boundary, chunk_size):
                end = min(i + chunk_size, boundary)
                chunks.append(audio_data[i:end])
            
            # Add explicit silence at sentence boundaries (100ms)
            silence_chunk = b'\x00' * 800  # 100ms of silence at 8kHz
            chunks.append(silence_chunk)
            
            last_pos = boundary
        
        # Add any remaining chunks
        for i in range(last_pos, len(audio_data), chunk_size):
            end = min(i + chunk_size, len(audio_data))
            chunks.append(audio_data[i:end])
        
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
                                    
                                    # Track this response for echo detection
                                    self.recent_system_responses.append(response)
                                    if len(self.recent_system_responses) > 5:
                                        self.recent_system_responses.pop(0)
                                    
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
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """
        Send audio data to Twilio with improved barge-in handling.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
        """
        try:
            # Mark that we're speaking and update timing
            self.is_speaking = True
            self.speech_interrupted = False
            self.last_audio_output_time = time.time()
            
            # Add to fingerprinting database
            try:
                pcm_sample = self.audio_processor.mulaw_to_pcm(audio_data[:min(len(audio_data), 4000)])
                self.audio_fingerprinter.add_audio(pcm_sample)
            except Exception as e:
                logger.error(f"Error fingerprinting outgoing audio: {e}")
            
            # Add to own audio segments tracking
            audio_duration = len(audio_data) / 8000  # Assuming 8kHz mono mulaw
            self.own_audio_segments.append((self.last_audio_output_time, audio_duration))
            
            # Keep only recent segments
            if len(self.own_audio_segments) > self.max_audio_segments:
                self.own_audio_segments.pop(0)
            
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
            
            # Split audio into smaller chunks with silence detection for better responsiveness and barge-in
            chunks = self._split_audio_into_chunks_with_silence_detection(audio_data)
            
            logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks for playback")
            
            # Store chunks for potential interruption
            self.current_audio_chunks = chunks.copy()
            
            # Add a short delay after starting speech to allow for system to stabilize
            # This helps avoid echo detection immediately after starting to speak
            await asyncio.sleep(0.05)
            
            for i, chunk in enumerate(chunks):
                # Check if we've been interrupted before sending each chunk
                if self.speech_interrupted:
                    logger.info(f"Speech playback interrupted after sending {i}/{len(chunks)} chunks")
                    break
                    
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
                    
                    # Add a small delay between chunks to allow for interruption
                    # This ensures we can detect barge-in and stop quickly
                    if i < len(chunks) - 1:  # Don't delay after the last chunk
                        await asyncio.sleep(0.05)  # 50ms delay between chunks for better barge-in opportunity
                    
                    # Check for barge-in every 10 chunks
                    if i % 10 == 0 and len(self.input_buffer) > 4000:
                        # Process any incoming audio for barge-in detection
                        try:
                            # Sample a small portion of the input buffer
                            sample_size = min(len(self.input_buffer), 4000)
                            sample_data = self.audio_processor.mulaw_to_pcm(bytes(self.input_buffer[-sample_size:]))
                            
                            # Use the specialized speech detector for more reliable detection
                            if hasattr(self, 'speech_detector'):
                                is_speech = self.speech_detector.detect(sample_data)
                                if is_speech:
                                    logger.info(f"BARGE-IN DETECTED by speech detector during playback at chunk {i}/{len(chunks)}")
                                    self.speech_interrupted = True
                                    break
                            
                            # Check for speech but with a higher threshold during playback to avoid false detections
                            # Use the special barge-in detection during speech
                            if self._detect_barge_in_during_speech(sample_data, 0.1):  # Small time since output for strictness
                                logger.info(f"AUDIO-BASED BARGE-IN DETECTED during audio playback at chunk {i}/{len(chunks)}")
                                self.speech_interrupted = True
                                break
                        except Exception as e:
                            logger.error(f"Error checking for barge-in during playback: {e}")
                    
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