#!/usr/bin/env python3
"""
Simplified Twilio application using direct Google Cloud STT v2 integration.
Removes unnecessary abstraction layers for better performance.
"""
import os
import sys
import asyncio
import logging
import json
import time
import threading
from flask import Flask, request, Response, jsonify
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

import simple_websocket

# Import simplified handler
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from telephony.config import HOST, PORT, DEBUG
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech.google_cloud_tts import GoogleCloudTTS

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

# Global instances
voice_ai_pipeline = None
base_url = None

# Track active calls
active_calls = {}

async def initialize_system():
    """Initialize the Voice AI system."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with simplified architecture...")
    
    # Get base URL
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL must be set")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Voice AI Agent
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
    )
    await agent.init()
    
    # Initialize TTS
    from integration.tts_integration import TTSIntegration
    tts = TTSIntegration(
        voice_name="en-US-Neural2-C",
        voice_gender="NEUTRAL",
        language_code="en-US",
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
    
    logger.info("System initialized successfully")

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    return "Voice AI Agent is running (simplified architecture)!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
    logger.info("Received incoming call request")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return Response('''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>System is not initialized. Please try again later.</Say>
</Response>''', mimetype='text/xml')
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    try:
        # Create TwiML response
        response = VoiceResponse()
        
        # Add brief pause
        response.pause(length=1)
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream
        connect = Connect()
        stream = Stream(
            name="stream",
            url=ws_url,
            track="inbound_track"
        )
        
        # Set parameters for better audio quality
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        
        connect.append(stream)
        response.append(connect)
        
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        return Response('''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>An error occurred. Please try again later.</Say>
</Response>''', mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # Clean up completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            del active_calls[call_sid]
            logger.info(f"Cleaned up call {call_sid}")
    
    return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with simplified architecture."""
    logger.info(f"WebSocket connection for call {call_sid}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create simplified handler
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # Create event loop for this WebSocket
        loop = asyncio.new_event_loop()
        
        def run_handler():
            asyncio.set_event_loop(loop)
            try:
                # Handle initial connection
                loop.run_until_complete(
                    handler.handle_message(json.dumps({
                        "event": "connected",
                        "protocol": "Call",
                        "version": "1.0.0"
                    }), ws)
                )
                
                # Keep loop running
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in handler loop: {e}", exc_info=True)
            finally:
                loop.close()
        
        # Start handler thread
        thread = threading.Thread(target=run_handler, daemon=True)
        thread.start()
        
        # Process messages
        while True:
            try:
                message = ws.receive(timeout=5)
                if message is None:
                    continue
                
                # Send message to handler
                asyncio.run_coroutine_threadsafe(
                    handler.handle_message(message, ws),
                    loop
                )
                
            except simple_websocket.ws.ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error establishing WebSocket: {e}", exc_info=True)
    finally:
        # Cleanup
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        if ws:
            try:
                ws.close()
            except:
                pass
        
        logger.info(f"WebSocket cleanup complete for call {call_sid}")
        return ""

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics for active calls."""
    stats = {
        "active_calls": len(active_calls),
        "calls": {}
    }
    
    for call_sid, handler in active_calls.items():
        stats["calls"][call_sid] = handler.get_stats()
    
    return jsonify(stats)

def init_system():
    """Initialize system synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_system())
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting Voice AI Agent with simplified architecture...")
    
    # Initialize system
    init_system()
    
    # Run Flask app
    print(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)