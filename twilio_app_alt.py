#!/usr/bin/env python3
"""
Improved Twilio application using Google Cloud STT v2.32.0 with automatic features.
No hardcoded keywords - relies on API's intelligence.
"""
import os
import sys
import asyncio
import logging
import json
import threading
import signal
from flask import Flask, request, Response
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# Import the improved components
from telephony.websocket.websocket_handler_v2 import WebSocketHandler
from voice_ai_agent_v2 import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from text_to_speech import ElevenLabsTTS

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Global variables
base_url = None
voice_ai_pipeline = None
call_event_loops = {}

async def initialize_system():
    """Initialize the Voice AI system with improved components."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI system with improved Google Cloud STT...")
    
    # Get base URL
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set in environment")
        raise ValueError("BASE_URL must be set")
    
    # Verify required API keys
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        logger.error("ELEVENLABS_API_KEY not set")
        raise ValueError("ELEVENLABS_API_KEY must be set")
    
    # Check Google Cloud credentials
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_creds:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Using default authentication.")
    else:
        logger.info(f"Using Google Cloud credentials from: {google_creds}")
    
    # Initialize Voice AI Agent with improved settings
    try:
        agent = VoiceAIAgent(
            storage_dir='./storage',
            model_name='mistral:7b-instruct-v0.2-q4_0',
            llm_temperature=0.7,
            # Improved Google Cloud STT parameters - no hardcoded keywords
            language='en-US',
            enhanced_model=True,
            # ElevenLabs parameters
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
            elevenlabs_model_id=os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2')
        )
        await agent.init()
        logger.info("Voice AI Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Voice AI Agent: {e}", exc_info=True)
        raise
    
    # Initialize TTS integration
    from integration.tts_integration import TTSIntegration
    try:
        tts = TTSIntegration(
            voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
            enable_caching=True
        )
        await tts.init()
        logger.info("TTS integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}", exc_info=True)
        raise
    
    # Create pipeline with improved components
    try:
        voice_ai_pipeline = VoiceAIAgentPipeline(
            speech_recognizer=agent.speech_recognizer,  # Now uses improved Google Cloud STT
            conversation_manager=agent.conversation_manager,
            query_engine=agent.query_engine,
            tts_integration=tts
        )
        logger.info("Voice AI Pipeline created successfully")
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}", exc_info=True)
        raise
    
    logger.info("System initialization complete with improved Google Cloud STT v2.32.0")
    logger.info("Ready to handle voice calls with automatic speech recognition")

@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    status = {
        "status": "Voice AI Agent Running",
        "version": "improved_v2.32.0",
        "features": [
            "Google Cloud STT v2.32.0 with automatic speech detection",
            "No hardcoded keywords - API handles all speech recognition",
            "ElevenLabs TTS for high-quality voice synthesis",
            "Continuous speech recognition for natural conversations"
        ]
    }
    return status

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "voice_ai_pipeline": voice_ai_pipeline is not None,
            "base_url": base_url is not None,
            "active_calls": len(call_event_loops)
        }
    }

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
    logger.info("Received incoming call request")
    logger.debug(f"Request form data: {request.form}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        response = VoiceResponse()
        response.say("System not ready. Please try again later.")
        return Response(str(response), mimetype='text/xml')
    
    # Get call parameters
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    call_sid = request.form.get('CallSid')
    
    logger.info(f"Incoming call from {from_number} to {to_number}, SID: {call_sid}")
    
    try:
        # Create TwiML response
        response = VoiceResponse()
        
        # Add a brief pause for call establishment
        response.pause(length=1)
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream - optimized for Google Cloud STT
        connect = Connect()
        stream = Stream(
            name="voice_stream",
            url=ws_url,
            track="inbound_track"
        )
        
        # Optimized parameters for Google Cloud STT v2.32.0
        # Using MULAW encoding for better telephony quality
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        stream.parameter(name="amd", value="false")  # Disable answering machine detection
        
        connect.append(stream)
        response.append(connect)
        
        # Optional: Add a brief greeting
        response.say(
            "Hello! I'm connecting you to our AI assistant. Please wait a moment.",
            voice='alice',
            language='en-US'
        )
        
        logger.info(f"Created TwiML with stream URL: {ws_url}")
        
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        response = VoiceResponse()
        response.say("I'm sorry, there was an error connecting your call. Please try again.")
        return Response(str(response), mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status callbacks."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status update: {call_status}")
    
    # Clean up resources for completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
        if call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            
            # Signal termination to the event loop
            if 'loop' in loop_info and loop_info['loop'].is_running():
                loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
            
            # Wait for thread to finish with timeout
            if 'thread' in loop_info and loop_info['thread'].is_alive():
                loop_info['thread'].join(timeout=2.0)
                if loop_info['thread'].is_alive():
                    logger.warning(f"Thread for call {call_sid} did not terminate gracefully")
            
            # Remove from tracking
            del call_event_loops[call_sid]
            logger.info(f"Cleaned up resources for call {call_sid}")
    
    return Response('', status=204)

@app.route('/calls/active', methods=['GET'])
def get_active_calls():
    """Get information about active calls."""
    return {
        "active_calls": len(call_event_loops),
        "call_sids": list(call_event_loops.keys())
    }

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket media stream with improved components."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        # Accept WebSocket connection
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create improved WebSocket handler
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Create dedicated event loop for this connection
        loop = asyncio.new_event_loop()
        
        def run_handler_loop():
            """Run the WebSocket handler in its own thread and event loop."""
            asyncio.set_event_loop(loop)
            try:
                # Send initial connected event
                connected_msg = json.dumps({
                    "event": "connected",
                    "protocol": "Call",
                    "version": "1.0.0"
                })
                loop.create_task(ws_handler.handle_message(connected_msg, ws))
                
                # Run the event loop
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in handler loop for call {call_sid}: {e}", exc_info=True)
            finally:
                # Clean up
                if not loop.is_closed():
                    loop.close()
                logger.info(f"Event loop closed for call {call_sid}")
        
        # Start handler thread
        thread = threading.Thread(target=run_handler_loop, daemon=True)
        thread.start()
        
        # Store thread and loop info for cleanup
        call_event_loops[call_sid] = {
            'loop': loop,
            'thread': thread,
            'handler': ws_handler
        }
        
        # Main message processing loop
        message_count = 0
        try:
            while True:
                # Receive message with timeout
                message = ws.receive(timeout=30)  # 30 second timeout
                
                if message is None:
                    continue
                
                message_count += 1
                logger.debug(f"Received message #{message_count} for call {call_sid}")
                
                # Send message to handler loop for processing
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        ws_handler.handle_message(message, ws),
                        loop
                    )
                else:
                    logger.warning(f"Event loop not running for call {call_sid}, breaking")
                    break
                    
        except Exception as e:
            if "WebSocketError" in str(type(e)) or "ConnectionClosed" in str(type(e)):
                logger.info(f"WebSocket connection closed for call {call_sid}")
            else:
                logger.error(f"Error in message processing for call {call_sid}: {e}", exc_info=True)
    
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection for call {call_sid}: {e}", exc_info=True)
    
    finally:
        # Cleanup resources
        if call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            
            # Stop the event loop gracefully
            if 'loop' in loop_info and loop_info['loop'].is_running():
                loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
            
            # Wait for thread to finish
            if 'thread' in loop_info and loop_info['thread'].is_alive():
                loop_info['thread'].join(timeout=2.0)
            
            # Get final stats from handler
            if 'handler' in loop_info:
                stats = loop_info['handler'].get_session_stats()
                logger.info(f"Final stats for call {call_sid}: {stats}")
            
            # Remove from tracking
            del call_event_loops[call_sid]
        
        # Close WebSocket connection if still open
        if ws:
            try:
                ws.close(reason="Call ended")
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

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Clean up all active calls
    active_calls = list(call_event_loops.keys())
    for call_sid in active_calls:
        logger.info(f"Cleaning up call {call_sid}")
        if call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            
            if 'loop' in loop_info and loop_info['loop'].is_running():
                loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
            
            if 'thread' in loop_info and loop_info['thread'].is_alive():
                loop_info['thread'].join(timeout=1.0)
    
    logger.info("Graceful shutdown complete")
    sys.exit(0)

if __name__ == '__main__':
    print("Starting Voice AI Agent with improved Google Cloud STT v2.32.0...")
    print("Features:")
    print("- No hardcoded keywords - API handles all speech recognition")
    print("- Automatic speech adaptation")
    print("- Enhanced telephony optimization")
    print("- Continuous speech recognition")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        init_system()
        
        # Start Flask app
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print(f"Starting Flask server on {host}:{port}")
        print(f"Debug mode: {debug}")
        print(f"WebSocket endpoint: {os.getenv('BASE_URL', 'Not set')}/ws/stream/<call_sid>")
        
        app.run(host=host, port=port, debug=debug, threaded=True)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)