#!/usr/bin/env python3
"""
Updated Twilio application using Google Cloud STT v2.25.0+ without deprecated fields.
"""
import os
import sys
import asyncio
import logging
import json
import threading
import numpy as np
from flask import Flask, request, Response
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# Import updated modules
from telephony.websocket.websocket_handler import WebSocketHandler
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech import ElevenLabsTTS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Global variables
base_url = None
voice_ai_pipeline = None
call_event_loops = {}

async def initialize_system():
    """Initialize the Voice AI system with updated components."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI system with updated Google Cloud STT...")
    
    # Get base URL
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set in environment")
        raise ValueError("BASE_URL must be set")
    
    # Verify API keys
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        logger.error("ELEVENLABS_API_KEY not set")
        raise ValueError("ELEVENLABS_API_KEY must be set")
    
    # Check Google Cloud credentials
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Make sure you have authentication configured.")
    
    # Initialize Voice AI Agent with updated configuration
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
        # Updated Google Cloud STT parameters
        language='en-US',
        enhanced_model=True,
        # ElevenLabs parameters
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
        elevenlabs_model_id=os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2')
    )
    await agent.init()
    
    # Initialize TTS integration
    from integration.tts_integration import TTSIntegration
    tts = TTSIntegration(
        voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
        enable_caching=True
    )
    await tts.init()
    
    # Create pipeline with updated components
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,  # Now uses updated Google Cloud STT
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    logger.info("System initialized successfully with updated Google Cloud STT v2.25.0+")

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    return {"status": "Voice AI Agent Running", "version": "updated_v2.25.0"}

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
    logger.info("Received incoming call request")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request form data: {request.form}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Say>System not ready. Please try again.</Say></Response>',
            mimetype='text/xml'
        )
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call from {from_number} to {to_number}, SID: {call_sid}")
    
    try:
        # Create TwiML response
        response = VoiceResponse()
        
        # Add brief pause to avoid initial audio issues
        response.pause(length=1)
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream
        connect = Connect()
        stream = Stream(
            name="speech_stream",
            url=ws_url,
            track="inbound_track"
        )
        
        # Optimized parameters for Google Cloud STT - UPDATED FOR MULAW
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        stream.parameter(name="amd", value="false")  # Disable answering machine detection
        
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"Created TwiML with stream URL: {ws_url}")
        
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Say>An error occurred. Please try again.</Say></Response>',
            mimetype='text/xml'
        )

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # Clean up event loop if call is completed
    if call_status in ['completed', 'failed', 'busy', 'no-answer'] and call_sid in call_event_loops:
        loop_info = call_event_loops[call_sid]
        
        # Signal termination
        if 'loop' in loop_info and loop_info['loop'].is_running():
            loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
        
        # Join thread with timeout
        if 'thread' in loop_info:
            loop_info['thread'].join(timeout=1.0)
        
        # Remove from tracking
        del call_event_loops[call_sid]
        logger.info(f"Cleaned up resources for call {call_sid}")
    
    return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create updated WebSocket handler
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Create event loop for this connection
        loop = asyncio.new_event_loop()
        
        def run_handler_loop():
            asyncio.set_event_loop(loop)
            try:
                # Handle initial connection
                loop.create_task(ws_handler.handle_message(json.dumps({
                    "event": "connected",
                    "protocol": "Call",
                    "version": "1.0.0"
                }), ws))
                
                # Run the loop
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in handler loop: {e}", exc_info=True)
            finally:
                loop.close()
        
        # Start handler thread
        thread = threading.Thread(target=run_handler_loop, daemon=True)
        thread.start()
        
        # Store for cleanup
        call_event_loops[call_sid] = {
            'loop': loop,
            'thread': thread
        }
        
        # Process messages in main thread
        while True:
            try:
                message = ws.receive(timeout=5)
                if message is None:
                    continue
                
                # Send to handler loop
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
        # Cleanup
        if call_sid in call_event_loops:
            info = call_event_loops[call_sid]
            if 'loop' in info and info['loop'].is_running():
                info['loop'].call_soon_threadsafe(info['loop'].stop)
            del call_event_loops[call_sid]
        
        if ws:
            try:
                ws.close()
            except:
                pass
        
        logger.info(f"WebSocket cleanup complete for call {call_sid}")
        return ""

def init_system():
    """Initialize system synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_system())
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting Voice AI Agent with updated Google Cloud STT v2.25.0+...")
    
    # Initialize system
    init_system()
    
    # Start Flask app
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)