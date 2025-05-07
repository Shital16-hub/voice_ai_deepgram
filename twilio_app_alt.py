#!/usr/bin/env python3
"""
Optimized Twilio application with enhanced barge-in detection.
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
from flask import Flask, request, Response, jsonify
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# Import your modules
from speech_to_text.utils.speech_detector import SpeechActivityDetector
from telephony.audio_processor import MulawBufferProcessor, AudioProcessor
from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler, AudioFingerprinter
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech import ElevenLabsTTS

# Load environment variables
load_dotenv()

# Configure logging with reduced noise from small audio chunks
def configure_logging():
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
    root_logger.addHandler(file_handler)
    
    return root_logger

# Configure logging
logger = configure_logging()

# Flask app setup
app = Flask(__name__)

# Global instances
twilio_handler = None
voice_ai_pipeline = None
base_url = None

# Dictionary to store event loops and state for each call
call_event_loops = {}

# Helper function to detect speech vs echo
def detect_barge_in_during_speech(audio_data: np.ndarray, time_since_output: float) -> bool:
    """
    Enhanced barge-in detection during system speech with lower thresholds for faster detection.
    """
    # Ignore very small audio samples
    if len(audio_data) < 1000:  # Reduced from 1600
        return False
        
    # 1. Energy must be higher than typical noise/echo
    energy = np.mean(np.abs(audio_data))
    min_energy_threshold = 0.06  # Reduced from 0.08/0.1
    
    # Calculate frame statistics if possible
    frame_size = min(len(audio_data), 320)  # 20ms at 16kHz
    frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
    frame_energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
    
    # Simplified condition with lower thresholds
    if len(frame_energies) >= 2:  # Reduced from 3
        # Energy growth check (looser requirement)
        has_energy_growth = False
        if len(frame_energies) > 3:
            first_half = frame_energies[:len(frame_energies)//2]
            second_half = frame_energies[len(frame_energies)//2:]
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            has_energy_growth = second_avg > (first_avg * 1.2)  # Reduced from 1.3/1.5
        
        # Variation check
        energy_std = np.std(frame_energies)
        energy_mean = np.mean(frame_energies)
        variation_ratio = energy_std / energy_mean if energy_mean > 0 else 0
        has_variation = variation_ratio > 0.25  # Reduced from 0.3/0.4
        
        # Check peak ratio
        peak = np.max(np.abs(audio_data))
        peak_ratio = peak / energy if energy > 0 else 0
        has_peaks = peak_ratio > 3.5  # Reduced from 4.0/5.0
        
        # Require fewer conditions (1 out of 3 instead of 2 out of 3)
        conditions_met = sum([has_variation, has_peaks, has_energy_growth])
        is_interruption = (
            energy > min_energy_threshold and
            conditions_met >= 1  # Reduced from 2
        )
        
        # Add a super aggressive emergency check for strong interruptions
        emergency_interrupt = energy > 0.15 and peak > 0.5  # High-energy obvious interruption
        
        return is_interruption or emergency_interrupt
    
    # Fallback for minimal data
    return energy > 0.15  # Simple energy threshold as fallback

# Helper function to split audio with improved pausing
def split_audio_into_chunks_with_silence_detection(audio_data: bytes) -> list:
    """
    Split audio into smaller chunks with silence detection for improved barge-in opportunities.
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
    chunk_size = 400  # Reduced from 800 for more frequent checks
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
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        logger.warning("ELEVENLABS_API_KEY not set in environment, attempting to proceed without it")
    
    # Define a generic telephony-optimized prompt
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
    
    # Initialize Voice AI Agent with optimized parameters
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
        # Pass optimized telephony parameters
        whisper_initial_prompt=telephony_prompt,
        whisper_temperature=0.0,  # Greedy decoding for more reliable transcription
        whisper_no_context=True,  # Each utterance is independent
        whisper_preset="default",
        # Pass ElevenLabs parameters
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),  # Default to Bella voice
        elevenlabs_model_id=os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2')  # Latest model
    )
    await agent.init()
    
    # Initialize TTS integration with ElevenLabs
    from integration.tts_integration import TTSIntegration
    tts = TTSIntegration(
        voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
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
    return "Voice AI Agent is running with ElevenLabs TTS integration and enhanced barge-in support!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls with improved Twilio barge-in integration."""
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
        
        # Create TwiML response with improved barge-in handling
        response = VoiceResponse()
        
        # Add a 1 second silence before starting - this helps avoid initial echo problems
        response.pause(length=1)
        
        # Create WebSocket URL for streaming
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream for bi-directional audio
        connect = Connect()
        
        # Create Stream with explicit parameters for optimal Twilio barge-in support
        stream = Stream(
            name="stream", 
            url=ws_url, 
            track="inbound_track"
        )
        
        # IMPROVED TWILIO BARGE-IN PARAMETERS
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        stream.parameter(name="bargeInEnabled", value="true") 
        
        # More conservative barge-in settings
        stream.parameter(name="sensitivity", value="low")  # Lower sensitivity to avoid false positives
        stream.parameter(name="bargeInMinimumDuration", value="600")  # 600ms minimum duration
        stream.parameter(name="detectVoiceDuringPlayback", value="true")  # Critical for barge-in
        
        # Add stream to connect
        connect.append(stream)
        
        # Add connect to response
        response.append(connect)
        
        # Return TwiML
        twiml = str(response)
        return Response(twiml, mimetype='text/xml')
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        # Fallback response
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

@app.route('/test/barge-in', methods=['POST'])
def test_barge_in_detection():
    """Test barge-in detection with audio samples."""
    try:
        # Get audio data
        audio_data = request.data
        if not audio_data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Process with both original and new detection algorithms
        audio_processor = AudioProcessor()
        pcm_audio = audio_processor.mulaw_to_pcm(audio_data)
        
        # Create a speech detector
        speech_detector = SpeechActivityDetector(
            energy_threshold=0.04,
            consecutive_frames=2,
            frame_duration=0.02
        )
        
        # Test both detection methods
        speech_detected = speech_detector.detect(pcm_audio)
        
        # Original detection
        original_detection = detect_barge_in_during_speech(pcm_audio, 0.5)
        
        # New detection parameters
        energy = np.mean(np.abs(pcm_audio))
        peak = np.max(np.abs(pcm_audio))
        new_detection = energy > 0.06 and peak > 0.2
        
        # Emergency detection
        emergency_detection = energy > 0.15 and peak > 0.5
        
        return jsonify({
            "speech_detected": speech_detected,
            "original_detection": original_detection,
            "new_detection": new_detection,
            "emergency_detection": emergency_detection,
            "energy": float(energy),
            "peak": float(peak),
            "audio_size": len(audio_data),
            "pcm_size": len(pcm_audio)
        }), 200
    except Exception as e:
        logger.error(f"Error in barge-in test: {e}")
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
        
        # Create improved speech detector with conservative settings
        speech_detector = SpeechActivityDetector(
            energy_threshold=0.10,  # Increased threshold
            consecutive_frames=5,   # Require more frames
            frame_duration=0.02     # 20ms frames
        )
        
        # Initialize WebSocketHandler with better barge-in handling
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # IMPROVED CONFIGURATION
        # Set speech detector
        ws_handler.speech_detector = speech_detector
        
        # Enhance echo detection
        ws_handler.audio_fingerprinter = AudioFingerprinter(max_fingerprints=50)
        ws_handler.audio_fingerprinter.similarity_threshold = 0.65
        
        # Conservative barge-in settings
        ws_handler.barge_in_enabled = True
        ws_handler.barge_in_energy_threshold = 0.12
        ws_handler.post_speech_dead_zone = 0.5  # 500ms dead zone after speech
        ws_handler.barge_in_min_duration = 0.6  # 600ms minimum speech duration
        ws_handler.barge_in_debounce_time = 6.0  # 6s between barge-in attempts
        
        # Increase pause after system response
        ws_handler.pause_after_response = 0.8  # Longer pause between turns
        
        # Create speech pattern detector
        ws_handler.speech_pattern_detector = SpeechPatternDetector()
        
        # Set minimum words for a valid query
        ws_handler.min_words_for_valid_query = 2  # Increased from 1 for more reliable detection
        
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
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
                break
        
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
    print("Starting Voice AI Agent with ElevenLabs TTS integration and enhanced barge-in support...")
    
    # Initialize the system before starting the Flask app
    init_system()
    
    # Run the Flask app
    print(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)