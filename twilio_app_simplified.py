#!/usr/bin/env python3
"""
Fixed Twilio application with proper call handling and WebSocket setup.
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
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Say, Pause

import simple_websocket

# Import fixed handler
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from telephony.config import HOST, PORT, DEBUG
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from integration.tts_integration import TTSIntegration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

# Flask app setup
app = Flask(__name__)

# Global instances
voice_ai_pipeline = None
base_url = None

# Track active calls
active_calls = {}

async def initialize_system():
    """Initialize the Voice AI system with better error handling."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent with fixed architecture...")
    
    # Validate required environment variables
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL environment variable must be set")
    
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_creds:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set, using default credentials")
    elif not os.path.exists(google_creds):
        logger.error(f"Google credentials file not found: {google_creds}")
        raise FileNotFoundError(f"Credentials file not found: {google_creds}")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Voice AI Agent
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
    )
    await agent.init()
    
    # Initialize TTS with fixed configuration
    tts = TTSIntegration(
        voice_name="en-US-Neural2-C",
        voice_gender=None,  # Don't set gender for Neural2 voices
        language_code="en-US",
        enable_caching=True,
        credentials_file=google_creds
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
    return {
        "status": "running",
        "message": "Voice AI Agent is running with fixed architecture",
        "version": "2.0.0"
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check."""
    health_status = {
        "status": "healthy" if voice_ai_pipeline else "initializing",
        "timestamp": time.time(),
        "components": {
            "pipeline": voice_ai_pipeline is not None,
            "active_calls": len(active_calls)
        }
    }
    
    if voice_ai_pipeline:
        health_status["components"].update({
            "stt": hasattr(voice_ai_pipeline, 'speech_recognizer'),
            "tts": hasattr(voice_ai_pipeline, 'tts_integration'),
            "kb": hasattr(voice_ai_pipeline, 'query_engine')
        })
    
    return jsonify(health_status)

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls with proper TwiML response."""
    logger.info("Received incoming call request")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return Response('''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">System is not initialized. Please try again later.</Say>
    <Hangup/>
</Response>''', mimetype='text/xml')
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    try:
        # Validate base_url
        if not base_url:
            raise ValueError("BASE_URL not configured")
        
        # Create TwiML response with proper call flow
        response = VoiceResponse()
        
        # Answer the call with a greeting
        response.say("Hello! Please wait while I connect you to our AI assistant.", voice="alice")
        
        # Add a short pause to ensure the call is answered
        response.pause(length=1)
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Create Connect with Stream
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"  # Only capture inbound audio
        )
        
        connect.append(stream)
        response.append(connect)
        
        # Keep the call alive after streaming ends
        response.say("Thank you for calling. Goodbye!", voice="alice")
        response.hangup()
        
        logger.info(f"Generated TwiML for call {call_sid}")
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        return Response('''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">An error occurred. Please try again later.</Say>
    <Hangup/>
</Response>''', mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks with logging."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    call_duration = request.form.get('CallDuration', '0')
    
    logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration}s")
    
    # Clean up completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            handler = active_calls[call_sid]
            # Trigger cleanup
            try:
                asyncio.create_task(handler._cleanup())
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            del active_calls[call_sid]
            logger.info(f"Cleaned up call {call_sid}")
    
    return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with improved error handling."""
    logger.info(f"WebSocket connection request for call {call_sid}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized for WebSocket connection")
        return ""
    
    ws = None
    handler = None
    
    try:
        # Accept the WebSocket connection
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create handler with improved architecture
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # Process incoming messages
        while True:
            try:
                # Receive message with timeout
                message = ws.receive(timeout=30.0)
                
                if message is None:
                    logger.debug("Received None message, continuing...")
                    continue
                
                # Parse and handle message
                try:
                    data = json.loads(message)
                    event_type = data.get('event')
                    
                    if event_type == 'connected':
                        logger.info(f"WebSocket connected for call {call_sid}")
                        
                    elif event_type == 'start':
                        stream_sid = data.get('streamSid')
                        logger.info(f"Stream started: {stream_sid}")
                        handler.stream_sid = stream_sid
                        
                        # Start STT streaming
                        asyncio.run(handler.stt_client.start_streaming())
                        
                        # Send welcome message
                        asyncio.run(handler._send_response("Hello! How can I help you today?", ws))
                        
                    elif event_type == 'media':
                        # Handle audio data
                        asyncio.run(handler._handle_audio(data, ws))
                        
                    elif event_type == 'stop':
                        logger.info(f"Stream stopped for call {call_sid}")
                        asyncio.run(handler._cleanup())
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                
            except simple_websocket.ws.ConnectionClosed:
                logger.info(f"WebSocket connection closed for call {call_sid}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error establishing WebSocket: {e}", exc_info=True)
        
    finally:
        # Cleanup resources
        if handler:
            try:
                asyncio.run(handler._cleanup())
            except Exception as e:
                logger.error(f"Error during handler cleanup: {e}")
        
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
    """Get comprehensive statistics for monitoring."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "initialized": voice_ai_pipeline is not None,
            "active_calls": len(active_calls),
            "base_url": base_url
        },
        "calls": {}
    }
    
    # Add individual call stats
    for call_sid, handler in active_calls.items():
        try:
            stats["calls"][call_sid] = handler.get_stats()
        except Exception as e:
            logger.error(f"Error getting stats for call {call_sid}: {e}")
            stats["calls"][call_sid] = {"error": str(e)}
    
    return jsonify(stats)

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config = {
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "base_url": base_url,
        "google_credentials": os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        "google_project": os.getenv('GOOGLE_CLOUD_PROJECT')
    }
    return jsonify(config)

def init_system():
    """Initialize system synchronously with proper error handling."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(initialize_system())
        logger.info("System initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        sys.exit(1)
    finally:
        loop.close()

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404

if __name__ == '__main__':
    print("Starting Voice AI Agent with fixed architecture...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"Google Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Initialize system
    init_system()
    
    # Run Flask app
    logger.info(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)