#!/usr/bin/env python3
"""
Updated Twilio application with fixed continuous conversation support.
Handles proper STT session management and WebSocket lifecycle.
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

# Track active calls with session management
active_calls = {}
call_sessions = {}  # Track call session metadata

async def initialize_system():
    """Initialize the Voice AI system with optimized conversation settings."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI Agent for continuous conversation...")
    
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
    
    # Initialize Voice AI Agent with conversation-optimized settings
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
        credentials_file=google_creds  # Pass credentials explicitly
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
    
    # Create pipeline optimized for continuous conversation
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    logger.info("System initialized successfully for continuous conversation")

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Voice AI Agent running with continuous conversation support",
        "version": "2.1.0",
        "active_calls": len(active_calls)
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check with conversation metrics."""
    health_status = {
        "status": "healthy" if voice_ai_pipeline else "initializing",
        "timestamp": time.time(),
        "components": {
            "pipeline": voice_ai_pipeline is not None,
            "active_calls": len(active_calls),
            "active_sessions": len(call_sessions)
        }
    }
    
    if voice_ai_pipeline:
        health_status["components"].update({
            "stt": hasattr(voice_ai_pipeline, 'speech_recognizer'),
            "tts": hasattr(voice_ai_pipeline, 'tts_integration'),
            "kb": hasattr(voice_ai_pipeline, 'query_engine')
        })
    
    # Add session health info
    if call_sessions:
        session_ages = [time.time() - session['start_time'] for session in call_sessions.values()]
        health_status["session_metrics"] = {
            "oldest_session_age": max(session_ages),
            "average_session_age": sum(session_ages) / len(session_ages),
            "sessions_over_5min": len([age for age in session_ages if age > 300])
        }
    
    return jsonify(health_status)

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls with optimized TwiML for continuous conversation."""
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
    
    # Track call session
    call_sessions[call_sid] = {
        "start_time": time.time(),
        "from_number": from_number,
        "to_number": to_number,
        "status": "initiated"
    }
    
    try:
        # Validate base_url
        if not base_url:
            raise ValueError("BASE_URL not configured")
        
        # Create TwiML response optimized for continuous conversation
        response = VoiceResponse()
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Create Connect with Stream - optimized for bidirectional conversation
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"  # Capture inbound audio for processing
        )
        
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"Generated TwiML for continuous conversation - Call {call_sid}")
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "error"
            call_sessions[call_sid]["error"] = str(e)
        
        return Response('''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">An error occurred. Please try again later.</Say>
    <Hangup/>
</Response>''', mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks with session tracking."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    call_duration = request.form.get('CallDuration', '0')
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    
    logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration}s")
    
    # Update session info
    if call_sid in call_sessions:
        call_sessions[call_sid].update({
            "status": call_status,
            "duration": call_duration,
            "end_time": time.time() if call_status in ['completed', 'failed', 'busy', 'no-answer'] else None
        })
    
    # Clean up completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            handler = active_calls[call_sid]
            # Trigger cleanup with session preservation logic
            try:
                asyncio.create_task(handler._cleanup())
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            del active_calls[call_sid]
            logger.info(f"Cleaned up call {call_sid}")
        
        # Keep session info for a while for debugging
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "completed"
            # Clean up old sessions (older than 1 hour)
            cleanup_sessions()
    
    return Response('', status=204)

def cleanup_sessions():
    """Clean up old call sessions to prevent memory leaks."""
    current_time = time.time()
    sessions_to_remove = []
    
    for call_sid, session in call_sessions.items():
        # Remove sessions older than 1 hour
        if current_time - session['start_time'] > 3600:
            sessions_to_remove.append(call_sid)
    
    for call_sid in sessions_to_remove:
        del call_sessions[call_sid]
        logger.debug(f"Cleaned up old session: {call_sid}")

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with continuous conversation support."""
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
        
        # Create handler optimized for continuous conversation
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "connected"
            call_sessions[call_sid]["ws_connected_time"] = time.time()
        
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
                        
                        # Start the conversation properly
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler.start_conversation(ws))
                        finally:
                            loop.close()
                        
                        # Update session
                        if call_sid in call_sessions:
                            call_sessions[call_sid]["stream_started"] = True
                            call_sessions[call_sid]["stream_sid"] = stream_sid
                        
                    elif event_type == 'media':
                        # Handle audio data for continuous conversation
                        # Run async code in a new event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler._handle_audio(data, ws))
                        finally:
                            loop.close()
                        
                    elif event_type == 'stop':
                        logger.info(f"Stream stopped for call {call_sid}")
                        
                        # Run cleanup in event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler._cleanup())
                        finally:
                            loop.close()
                        
                        # Update session
                        if call_sid in call_sessions:
                            call_sessions[call_sid]["stream_stopped"] = True
                        break
                        
                except json.JSONDecodeError as e:
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
                # Run cleanup in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(handler._cleanup())
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error during handler cleanup: {e}")
        
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        # Update session
        if call_sid in call_sessions:
            call_sessions[call_sid]["ws_disconnected_time"] = time.time()
        
        if ws:
            try:
                ws.close()
            except:
                pass
        
        logger.info(f"WebSocket cleanup complete for call {call_sid}")
        return ""

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get comprehensive statistics including conversation metrics."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "initialized": voice_ai_pipeline is not None,
            "active_calls": len(active_calls),
            "total_sessions": len(call_sessions),
            "base_url": base_url
        },
        "calls": {},
        "sessions": {}
    }
    
    # Add individual call stats
    for call_sid, handler in active_calls.items():
        try:
            stats["calls"][call_sid] = handler.get_stats()
        except Exception as e:
            logger.error(f"Error getting stats for call {call_sid}: {e}")
            stats["calls"][call_sid] = {"error": str(e)}
    
    # Add session information
    for call_sid, session in call_sessions.items():
        stats["sessions"][call_sid] = {
            **session,
            "age": time.time() - session['start_time']
        }
    
    # Add conversation metrics
    if active_calls:
        total_transcriptions = sum(handler.transcriptions for handler in active_calls.values())
        total_responses = sum(handler.responses_sent for handler in active_calls.values())
        stats["conversation_metrics"] = {
            "total_transcriptions": total_transcriptions,
            "total_responses": total_responses,
            "average_transcriptions_per_call": total_transcriptions / len(active_calls),
            "average_responses_per_call": total_responses / len(active_calls)
        }
    
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
        "google_project": os.getenv('GOOGLE_CLOUD_PROJECT'),
        "conversation_features": {
            "continuous_streaming": True,
            "session_management": True,
            "auto_reconnection": True
        }
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
    print("Starting Voice AI Agent with continuous conversation support...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"Google Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Initialize system
    init_system()
    
    # Run Flask app
    logger.info(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)