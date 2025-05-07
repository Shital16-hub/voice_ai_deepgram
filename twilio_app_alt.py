#!/usr/bin/env python3
"""
Enhanced Twilio application using WebSocket streaming with ElevenLabs TTS integration
and improved barge-in detection with advanced echo cancellation.
"""
import os
import sys
import asyncio
import logging
import json
import base64
import requests
import time
import threading
import numpy as np
from scipy import signal
from flask import Flask, request, Response, jsonify
from simple_websocket import Server, ConnectionClosed
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client

from speech_to_text.utils.speech_detector import SpeechActivityDetector
from telephony.audio_processor import MulawBufferProcessor, AudioProcessor

# Load environment variables
load_dotenv()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from telephony.audio_processor import AudioProcessor
from voice_ai_agent import VoiceAIAgent
from integration.tts_integration import TTSIntegration
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech import ElevenLabsTTS  # Import ElevenLabs TTS

# Get Twilio credentials from environment
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Get ElevenLabs credentials from environment
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL')  # Default to Bella voice
ELEVENLABS_MODEL_ID = os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2')

def configure_logging():
    """Configure logging to reduce noise from small audio chunks."""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to higher levels to reduce noise
    logging.getLogger('telephony.audio_processor').setLevel(logging.ERROR)  # Only show errors
    
    # Create a filter to ignore specific messages
    class IgnoreSmallMulawFilter(logging.Filter):
        def filter(self, record):
            return not (record.getMessage().startswith("Very small mulaw data"))
    
    # Add filter to the handlers
    for handler in root_logger.handlers:
        handler.addFilter(IgnoreSmallMulawFilter())
    
    # Optionally create a file handler for full logs
    file_handler = logging.FileHandler('full_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # Don't apply the filter to the file handler so we have complete logs
    root_logger.addHandler(file_handler)
    
    return root_logger

# Configure logging at application start
configure_logging()
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Global instances
twilio_handler = None
voice_ai_pipeline = None
base_url = None

# Dictionary to store event loops and state for each call
call_event_loops = {}

# Audio fingerprinting class for echo detection
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

# Enhanced helper function to detect speech vs echo
def contains_human_speech_pattern(audio_data: np.ndarray) -> bool:
    """
    Specialized detector that identifies human speech patterns vs. echoes.
    Uses speech-specific features that distinguish it from echoes and noise.
    """
    if len(audio_data) < 1600:  # Need at least 100ms at 16kHz
        return False
    
    # 1. Check overall energy
    energy = np.mean(np.abs(audio_data))
    if energy < 0.01:  # Very quiet
        return False
    
    # 2. Get spectral features that distinguish speech
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
            if (frame_energies[i] > frame_energies[i-1] * 1.2 and 
                frame_energies[i] > frame_energies[i+1] * 1.2 and
                frame_energies[i] > 0.015):  # Real peak, not just noise
                peaks += 1
        
        has_syllabic_pattern = peaks >= 2  # At least 2 energy peaks (syllables)
    else:
        has_syllabic_pattern = False
    
    # Combine spectral and temporal features
    return is_speech_spectrum and has_syllabic_pattern

# Improved function to detect barge-in during speech
def detect_barge_in_during_speech(audio_data: np.ndarray, time_since_output: float) -> bool:
    """
    Special barge-in detection during system speech with improved thresholds.
    
    Args:
        audio_data: Incoming audio data
        time_since_output: Time since last audio output
        
    Returns:
        True if confident this is a real barge-in, not an echo
    """
    # Ignore very small audio samples
    if len(audio_data) < 1600:  # Need at least 100ms at 16kHz
        return False
        
    # 1. Energy must be significantly higher than typical echo
    energy = np.mean(np.abs(audio_data))
    min_energy_threshold = 0.08  # REDUCED from 0.1 for better sensitivity
    
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
        has_speech_pattern = variation_ratio > 0.3  # REDUCED from 0.4 for better sensitivity
        
        # 4. Has strong peak-to-average ratio (speech has peaks)
        peak = np.max(np.abs(audio_data))
        peak_ratio = peak / energy if energy > 0 else 0
        has_peaks = peak_ratio > 4.0  # REDUCED from 5.0 for better sensitivity
        
        # 5. Dynamic energy growth (speech tends to increase in volume)
        has_energy_growth = False
        if len(frame_energies) > 5:
            first_half = frame_energies[:len(frame_energies)//2]
            second_half = frame_energies[len(frame_energies)//2:]
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            has_energy_growth = second_avg > (first_avg * 1.3)  # REDUCED from 1.5 for better sensitivity
        
        # Log detailed detection info
        logger.debug(f"Speech barge-in check: energy={energy:.4f}, min_threshold={min_energy_threshold:.4f}, "
                    f"variation={variation_ratio:.3f}, peak_ratio={peak_ratio:.1f}, "
                    f"energy_growth={has_energy_growth}")
        
        # COMBINED DECISION - need multiple factors to confirm real interruption
        # RELAXED from requiring all conditions to requiring only 2 out of 3
        conditions_met = sum([has_speech_pattern, has_peaks, has_energy_growth])
        is_interruption = (
            energy > min_energy_threshold and
            conditions_met >= 2
        )
        
        return is_interruption
    
    return False  # Not enough frames to analyze

# Helper function to split audio with improved pausing
def split_audio_into_chunks_with_silence_detection(audio_data: bytes) -> list:
    """
    Split audio into chunks with silence detection for improved barge-in.
    Add longer pauses at sentence boundaries for better barge-in opportunities.
    """
    # Create an AudioProcessor instance
    audio_processor = AudioProcessor()
    
    # Convert to PCM for analysis
    pcm_data = audio_processor.mulaw_to_pcm(audio_data)
    
    # Check if we have enough data to process
    if len(pcm_data) < 3200:  # Less than 200ms at 16kHz
        return [audio_data]  # Return original if too short
    
    # Simple sentence boundary detection from audio (rough approximation)
    # Looking for prolonged drops in energy that might represent pauses
    frame_size = 1600  # 100ms at 16kHz
    frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size) if i+frame_size <= len(pcm_data)]
    
    if not frames:
        return [audio_data]  # Return original if too short
        
    frame_energies = [np.mean(np.abs(frame)) for frame in frames]
    
    # Detect potential sentence boundaries as places where energy drops significantly
    boundaries = []
    for i in range(1, len(frame_energies)):
        if frame_energies[i] < frame_energies[i-1] * 0.3:  # 70% drop in energy
            boundaries.append(i * frame_size)
    
    # Now split the audio with extended silence at these boundaries
    chunk_size = 800  # Base chunk size (100ms at 8kHz)
    chunks = []
    last_pos = 0
    
    for boundary in boundaries:
        # Add chunks up to this boundary
        for i in range(last_pos, min(boundary, len(audio_data)), chunk_size):
            end = min(i + chunk_size, boundary, len(audio_data))
            chunks.append(audio_data[i:end])
        
        # Add explicit silence at sentence boundaries (100ms)
        silence_chunk = b'\x00' * 800  # 100ms of silence at 8kHz
        chunks.append(silence_chunk)
        
        last_pos = min(boundary, len(audio_data))
    
    # Add any remaining chunks
    for i in range(last_pos, len(audio_data), chunk_size):
        end = min(i + chunk_size, len(audio_data))
        chunks.append(audio_data[i:end])
    
    return chunks

async def initialize_system():
    """Initialize all system components with ElevenLabs TTS integration."""
    global twilio_handler, voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with ElevenLabs TTS integration...")
    
    # Verify ElevenLabs API key is set
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY not set in environment, attempting to proceed without it")
    
    # Define a generic telephony-optimized prompt that works with any knowledge base
    telephony_prompt = (
        "This is a telephone conversation with a customer. "
        "The customer may ask questions about products, services, pricing, or features. "
        "Transcribe exactly what is spoken, filtering out noise, static, and line interference. "
        "Common terms in business conversations include: pricing, plan, cost, features, "
        "monthly, subscription, support, upgrade, details, information."
    )
    
    # Get base URL from environment
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set in environment")
        raise ValueError("BASE_URL must be set")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Voice AI Agent with enhanced parameters that are knowledge-base agnostic
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
        # Pass generic telephony-optimized parameters
        whisper_initial_prompt=telephony_prompt,
        whisper_temperature=0.0,  # Greedy decoding for more reliable transcription
        whisper_no_context=True,  # Each utterance is independent
        whisper_preset="default",
        # Pass ElevenLabs parameters
        elevenlabs_api_key=ELEVENLABS_API_KEY,
        elevenlabs_voice_id=ELEVENLABS_VOICE_ID,
        elevenlabs_model_id=ELEVENLABS_MODEL_ID
    )
    await agent.init()
    
    # Initialize TTS integration with ElevenLabs
    tts = TTSIntegration(
        voice_id=ELEVENLABS_VOICE_ID,  # Use ElevenLabs voice ID
        enable_caching=True
    )
    await tts.init()
    
    # Create pipeline
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    # Initialize Twilio handler
    twilio_handler = TwilioHandler(voice_ai_pipeline, base_url)
    await twilio_handler.start()
    
    logger.info("System initialized successfully with ElevenLabs TTS integration")

@app.route('/', methods=['GET'])
def index():
    """Simple test endpoint."""
    return "Voice AI Agent is running with ElevenLabs TTS integration!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls using WebSocket stream with improved barge-in support."""
    logger.info("Received incoming call request")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request form data: {request.form}")
    
    if not twilio_handler:
        logger.error("System not initialized")
        fallback_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>System is not initialized. Please try again later.</Say>
</Response>'''
        return Response(fallback_twiml, mimetype='text/xml')
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    try:
        # Add call to manager
        twilio_handler.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response with improved barge-in configuration
        response = VoiceResponse()
        
        # Use shorter initial greeting to reduce echo confusion
        response.say("Welcome to the Voice AI Agent.", 
                    voice='alice', 
                    bargeIn="true")  # Use string "true" for explicit attribute
        response.pause(length=0.5)  # Shorter pause
        
        # Use WebSocket streaming with explicit parameters
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"Setting up WebSocket stream at: {ws_url}")
        
        # Create the stream with explicit parameters
        connect = Connect()
        stream = Stream(
            url=ws_url, 
            bargeIn="true",  # Explicit true
            track="inbound_track"
        )
        
        # Add extra parameters to ensure best audio quality
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        stream.parameter(name="bargeInEnabled", value="true")  # Redundant but explicit
        connect.append(stream)
        response.append(connect)
        
        # Very minimal TwiML after stream to avoid confusion
        response.pause(length=1)
        
        # Log the TwiML for verification
        logger.info(f"Generated TwiML for WebSocket streaming with barge-in: {response}")
        return Response(str(response), mimetype='text/xml')
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        fallback_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>An error occurred. Please try again later.</Say>
</Response>'''
        return Response(fallback_twiml, mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks."""
    logger.info("Received status callback")
    logger.info(f"Status data: {request.form}")
    
    if not twilio_handler:
        logger.error("System not initialized")
        return Response('', status=204)
    
    # Get status parameters
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    try:
        # Handle status update
        twilio_handler.handle_status_callback(call_sid, call_status)
        
        # Clean up event loop if call is completed and exists in call_event_loops
        if call_status in ['completed', 'failed', 'busy', 'no-answer'] and call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            # Signal termination
            if 'terminate_flag' in loop_info:
                loop_info['terminate_flag'].set()
                
            # Remove from dictionary
            if loop_info.get('thread'):
                # Wait for thread to join with timeout
                thread = loop_info['thread']
                thread.join(timeout=1.0)
                
            # Remove from tracking
            try:
                del call_event_loops[call_sid]
                logger.info(f"Cleaned up event loop resources for call {call_sid}")
            except KeyError:
                logger.warning(f"Call {call_sid} not found in event loops - already cleaned up")
                
        return Response('', status=204)
    except Exception as e:
        logger.error(f"Error handling status callback: {e}")
        return Response('', status=204)
        
def run_event_loop_in_thread(call_sid, ws):
    """
    Run WebSocket event loop in a separate thread.
    
    Args:
        call_sid: Twilio call SID
        ws: WebSocket connection
    """
    try:
        # Create a new asyncio event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create speech detector for barge-in support
        speech_detector = SpeechActivityDetector(
            energy_threshold=0.04,  # Adjust based on your audio levels
            consecutive_frames=2,   # Detect quickly for better responsiveness
            frame_duration=0.02     # 20ms frames
        )
        
        # Initialize WebSocketHandler with pipeline and speech detector
        # Use the global voice_ai_pipeline variable
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Assign the speech detector to the handler
        ws_handler.speech_detector = speech_detector
        
        # Create mulaw buffer processor to handle small audio chunks
        ws_handler.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)  # 80ms at 8kHz
        
        # Start keep-alive loop
        keep_alive_task = asyncio.ensure_future(ws_handler._keep_alive_loop(ws))
        
        # Setup on_message callback
        async def on_message(message):
            await ws_handler.handle_message(message, ws)
        
        # Setup on_close callback
        async def on_close():
            logger.info(f"WebSocket closed for call {call_sid}")
            # Cancel keep-alive task
            if keep_alive_task and not keep_alive_task.done():
                keep_alive_task.cancel()
                try:
                    await keep_alive_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup
            if ws_handler.google_speech_active:
                try:
                    await ws_handler.speech_client.stop_streaming()
                    logger.info("Stopped Google Cloud Speech streaming session")
                except Exception as e:
                    logger.error(f"Error stopping Google Speech session: {e}")
        
        # Register callbacks
        ws.on_message = on_message
        ws.on_close = on_close
        
        # Run the event loop
        loop.run_forever()
    except Exception as e:
        logger.error(f"Error in event loop thread for call {call_sid}: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            # Close the loop
            loop.close()
            logger.info(f"Event loop closed for call {call_sid}")
        except Exception as e:
            logger.error(f"Error closing event loop for call {call_sid}: {e}")

@app.route('/diagnostic/barge-in', methods=['POST'])
def barge_in_diagnostic():
    """
    Endpoint for testing barge-in detection parameters.
    Send audio samples for analysis without actual call handling.
    """
    try:
        # Get audio data from request
        audio_data = request.data
        if not audio_data:
            return jsonify({"error": "No audio data provided"}), 400
            
        # Create audio processor
        processor = AudioProcessor()
        
        # Convert mulaw to PCM
        pcm_audio = processor.mulaw_to_pcm(audio_data)
        
        # Calculate audio metrics
        audio_energy = float(np.mean(np.abs(pcm_audio)) if len(pcm_audio) > 0 else 0)
        
        # Test speech detection methods
        speech_detected = contains_human_speech_pattern(pcm_audio)
        is_barge_in = detect_barge_in_during_speech(pcm_audio, 0.5)
        
        # Create test fingerprinter to check echo detection
        fingerprinter = AudioFingerprinter(max_fingerprints=10)
        is_echo = fingerprinter.is_echo(pcm_audio)
        
        # Analyze using lower threshold for comparison
        is_barge_in_sensitive = False
        if len(pcm_audio) >= 1600:
            energy = np.mean(np.abs(pcm_audio))
            is_barge_in_sensitive = energy > 0.05  # Lower threshold for comparison
        
        results = {
            "audio_size": len(audio_data),
            "pcm_size": len(pcm_audio),
            "audio_energy": audio_energy,
            "speech_detected": speech_detected,
            "is_barge_in": is_barge_in,
            "is_barge_in_sensitive": is_barge_in_sensitive,
            "is_echo": is_echo
        }
            
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error in barge-in diagnostic: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with enhanced barge-in detection."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create speech detector for barge-in support
        speech_detector = SpeechActivityDetector(
            energy_threshold=0.04,
            consecutive_frames=2,
            frame_duration=0.02
        )
        
        # Initialize WebSocketHandler with pipeline
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Assign the speech detector to the handler
        ws_handler.speech_detector = speech_detector
        
        # Add the audio fingerprinter for echo detection
        ws_handler.audio_fingerprinter = AudioFingerprinter(max_fingerprints=20)
        
        # Force enable barge-in for better user experience
        ws_handler.barge_in_enabled = True
        ws_handler.barge_in_check_enabled = True
        
        # Lower threshold for faster response
        ws_handler.barge_in_energy_threshold = 0.05
        
        # Set longer pause after system speech to avoid echo confusion  
        ws_handler.pause_after_response = 0.5
        
        # Increase buffer size for better detection
        ws_handler.min_words_for_valid_query = 1
        
        # Create a new event loop for this WebSocket connection
        loop = asyncio.new_event_loop()
        
        # Create a thread to run the event loop
        def run_ws_handler_loop():
            asyncio.set_event_loop(loop)
            try:
                # Create tasks for handling the initial connection and message processing
                loop.create_task(ws_handler.handle_message(json.dumps({
                    "event": "connected",
                    "protocol": "Call",
                    "version": "1.0.0"
                }), ws))
                
                # Run the event loop
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in WebSocket handler loop: {e}", exc_info=True)
            finally:
                loop.close()
        
        # Start the thread
        thread = threading.Thread(target=run_ws_handler_loop, daemon=True)
        thread.start()
        
        # Store the thread and loop for cleanup
        call_event_loops[call_sid] = {
            'loop': loop,
            'thread': thread
        }
        
        # Process messages in the main thread and send them to the event loop
        while True:
            try:
                message = ws.receive(timeout=5)
                if message is None:
                    continue
                
                # Schedule the message to be processed in the event loop
                asyncio.run_coroutine_threadsafe(
                    ws_handler.handle_message(message, ws),
                    loop
                )
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}", exc_info=True)
    finally:
        # Clean up
        if call_sid in call_event_loops:
            info = call_event_loops[call_sid]
            if 'loop' in info and info['loop'].is_running():
                info['loop'].call_soon_threadsafe(info['loop'].stop)
            
            # Remove from tracking
            del call_event_loops[call_sid]
        
        if ws:
            try:
                ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        logger.info(f"WebSocket connection cleanup complete for call {call_sid}")
        return ""

# Function to initialize the system synchronously
def init_system():
    """Run the async initialization in a synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_system())
    finally:
        loop.close()

# Starting point of the application
if __name__ == '__main__':
    print("Starting Voice AI Agent with ElevenLabs TTS integration...")
    
    # Initialize the system before starting the Flask app
    init_system()
    
    # Run the Flask app
    print(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)