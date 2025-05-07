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
from flask import Flask, request, Response
from simple_websocket import Server, ConnectionClosed
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client

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

# Configure logging with more debug info
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
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
                logger.error(f"Error handling status callback: '{call_sid}'")
                
        return Response('', status=204)
    except Exception as e:
        logger.error(f"Error handling status callback: {e}")
        return Response('', status=204)
        
def run_event_loop_in_thread(loop, ws_handler, ws, call_sid, terminate_flag):
    """Run event loop in a separate thread."""
    try:
        # Set this loop as the event loop for this thread
        asyncio.set_event_loop(loop)
        
        # Create a task for the keep-alive mechanism
        keep_alive_task = asyncio.ensure_future(ws_handler._keep_alive_loop(ws))
        
        # Run the loop until the terminate flag is set or an error occurs
        while not terminate_flag.is_set():
            try:
                # Run the loop for a short duration
                loop.run_until_complete(asyncio.sleep(0.1))
            except Exception as e:
                logger.error(f"Error in event loop for call {call_sid}: {e}")
                break
        
        # Cancel the keep-alive task
        keep_alive_task.cancel()
        try:
            loop.run_until_complete(keep_alive_task)
        except (asyncio.CancelledError, Exception):
            pass
            
        # Close the loop
        loop.close()
        logger.info(f"Event loop for call {call_sid} has been closed")
        
    except Exception as e:
        logger.error(f"Error in event loop thread for call {call_sid}: {e}", exc_info=True)
    finally:
        # Clean up call_event_loops entry if it still exists
        if call_sid in call_event_loops:
            del call_event_loops[call_sid]

# Helper function to detect speech vs echo
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

# Helper function to split audio with improved pausing
def split_audio_into_chunks_with_silence_detection(audio_data: bytes, audio_processor: AudioProcessor) -> list:
    """
    Split audio into chunks with silence detection for improved barge-in.
    Add longer pauses at sentence boundaries for better barge-in opportunities.
    """
    # Convert to PCM for analysis
    pcm_data = audio_processor.mulaw_to_pcm(audio_data)
    
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
        
        # Create WebSocket handler with enhanced barge-in detection
        ws_handler = WebSocketHandler(call_sid, voice_ai_pipeline)
        
        # Add the audio fingerprinter for echo detection
        ws_handler.audio_fingerprinter = AudioFingerprinter(max_fingerprints=20)
        
        # Provide our enhanced speech detection function
        ws_handler._contains_human_speech_pattern = contains_human_speech_pattern
        
        # Provide improved audio splitting function
        ws_handler._split_audio_into_chunks_with_silence_detection = lambda audio_data: split_audio_into_chunks_with_silence_detection(
            audio_data, ws_handler.audio_processor)
        
        # Force enable barge-in for better user experience
        ws_handler.barge_in_enabled = True
        ws_handler.barge_in_check_enabled = True
        
        # Lower threshold for faster response
        ws_handler.barge_in_energy_threshold = 0.005  # Reduced for better sensitivity
        
        # Set longer pause after system speech to avoid echo confusion  
        ws_handler.pause_after_response = 0.5  # Increased from default
        
        # Create an event loop for this connection
        loop = asyncio.new_event_loop()
        
        # Create a flag for termination
        terminate_flag = threading.Event()
        
        # Store the event loop and related info
        call_event_loops[call_sid] = {
            'loop': loop,
            'terminate_flag': terminate_flag,
            'handler': ws_handler
        }
        
        # Create and start a thread for the event loop
        loop_thread = threading.Thread(
            target=run_event_loop_in_thread,
            args=(loop, ws_handler, ws, call_sid, terminate_flag),
            daemon=True
        )
        loop_thread.start()
        
        # Add thread to tracking
        call_event_loops[call_sid]['thread'] = loop_thread
        
        # Process connected event in the event loop
        connected_message = json.dumps({
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        })
        
        # Use asyncio.run_coroutine_threadsafe but without waiting for result
        asyncio.run_coroutine_threadsafe(
            ws_handler.handle_message(connected_message, ws),
            loop
        )
        
        # Process messages until connection closed
        while True:
            try:
                # Use shorter timeout
                message = ws.receive(timeout=5)
                if message is None:
                    logger.warning(f"Received None message for call {call_sid}")
                    break
                
                # Process the message in the dedicated event loop
                # Don't wait for the result to avoid blocking
                asyncio.run_coroutine_threadsafe(
                    ws_handler.handle_message(message, ws),
                    loop
                )
                
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                # Don't break the loop on message processing errors
                
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket cleanup for call {call_sid}")
        
        # Signal termination
        if call_sid in call_event_loops:
            call_event_loops[call_sid]['terminate_flag'].set()
        
        try:
            if ws:
                ws.close()
        except Exception as close_error:
            logger.error(f"Error closing WebSocket: {close_error}")
        
        # Return empty response
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