#!/usr/bin/env python3
"""
Twilio application with optimized feedback prevention and latency reduction.
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
from telephony.audio_processor import MulawBufferProcessor, AudioProcessor
from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech import ElevenLabsTTS

# Load environment variables
load_dotenv()

# Configure logging with reduced noise
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

async def initialize_system():
    """Initialize all system components with optimized configuration."""
    global twilio_handler, voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with optimized settings...")
    
    # Verify ElevenLabs API key is set
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        logger.warning("ELEVENLABS_API_KEY not set in environment, attempting to proceed without it")
    
    # Define a telephony-optimized prompt
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
    
    # Initialize TTS integration with optimized settings
    from integration.tts_integration import TTSIntegration
    tts = TTSIntegration(
        voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
        enable_caching=True,
        optimize_streaming_latency=3  # Reduced from 4 for better quality/latency balance
    )
    await tts.init()
    
    # Create pipeline with optimized settings
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    # Initialize Twilio handler
    twilio_handler = TwilioHandler(voice_ai_pipeline, base_url)
    await twilio_handler.start()
    
    logger.info("System initialized successfully with optimized settings")

@app.route('/', methods=['GET'])
def index():
    """Simple test endpoint."""
    return "Voice AI Agent is running with optimized settings!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls with optimized TwiML."""
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
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Add a 1 second silence before starting - this helps avoid initial echo problems
        response.pause(length=1)
        
        # Create WebSocket URL for streaming
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream for bi-directional audio
        connect = Connect()
        
        # Create Stream with track parameter for feedback prevention
        stream = Stream(
            name="stream", 
            url=ws_url, 
            track="inbound_track"  # Specify track as inbound for feedback prevention
        )
        
        # Set parameters
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        
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

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with feedback prevention."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Initialize WebSocketHandler with optimized configuration
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Create a new event loop for this WebSocket connection
        loop = asyncio.new_event_loop()
        
        # Create an event to signal termination
        terminate_flag = threading.Event()
        
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
                
                # Run the event loop until terminate flag is set
                while not terminate_flag.is_set():
                    loop.run_until_complete(asyncio.sleep(0.1))
            except Exception as e:
                logger.error(f"Error in WebSocket handler loop: {e}", exc_info=True)
            finally:
                # Close loop
                loop.close()
        
        # Start the thread
        thread = threading.Thread(target=run_ws_handler_loop, daemon=True)
        thread.start()
        
        # Store the thread, loop and terminate flag for cleanup
        call_event_loops[call_sid] = {
            'loop': loop,
            'thread': thread,
            'terminate_flag': terminate_flag
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
            if 'terminate_flag' in info:
                info['terminate_flag'].set()
            
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
    print("Starting Voice AI Agent with optimized settings...")
    
    # Initialize the system before starting the Flask app
    init_system()
    
    # Run the Flask app
    print(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)