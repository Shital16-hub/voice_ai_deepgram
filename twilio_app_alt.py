#!/usr/bin/env python3
"""
Twilio application with proper knowledge base integration.
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

# Import your modules - Updated imports
from telephony.audio import MulawBufferProcessor  
from telephony.audio_processor import AudioProcessor  
from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT

# Import new knowledge base components
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.pinecone_manager import PineconeManager

# Import STT integration
from speech_to_text.simple_google_stt import GoogleCloudStreamingSTT
from integration.stt_integration import STTIntegration
from integration.tts_integration import TTSIntegration
from integration.pipeline import VoiceAIAgentPipeline

# Load environment variables
load_dotenv()

# Configure logging
def configure_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # Reduce noise from audio processing
    logging.getLogger('telephony.audio_processor').setLevel(logging.ERROR)
    
    file_handler = logging.FileHandler('full_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

logger = configure_logging()

# Flask app setup
app = Flask(__name__)

# Global instances
twilio_handler = None
voice_ai_pipeline = None
conversation_manager = None
query_engine = None
base_url = None

# Dictionary to store event loops and state for each call
call_event_loops = {}

async def initialize_system():
    """Initialize all system components with proper knowledge base integration."""
    global twilio_handler, voice_ai_pipeline, conversation_manager, query_engine, base_url
    
    logger.info("Initializing Voice AI Agent with OpenAI + Pinecone knowledge base...")
    
    # Get base URL from environment
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set in environment")
        raise ValueError("BASE_URL must be set")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # 1. Initialize Speech Recognition
    logger.info("Initializing Google Cloud Speech Recognition...")
    speech_recognizer = GoogleCloudStreamingSTT(
        language="en-US",
        sample_rate=16000,
        encoding="LINEAR16",
        channels=1,
        interim_results=True,
        enhanced_model=True
    )
    
    # 2. Initialize STT Integration
    stt_integration = STTIntegration(
        speech_recognizer=speech_recognizer,
        language="en-US"
    )
    await stt_integration.init()
    
    # 3. Initialize Pinecone Manager
    logger.info("Initializing Pinecone Manager...")
    pinecone_manager = PineconeManager()
    await pinecone_manager.init()
    
    # 4. Initialize OpenAI Assistant Manager with Pinecone
    logger.info("Initializing OpenAI Assistant Manager...")
    from knowledge_base.openai_assistant_manager import OpenAIAssistantManager
    openai_manager = OpenAIAssistantManager()
    openai_manager.set_pinecone_manager(pinecone_manager)
    await openai_manager.get_or_create_assistant()
    
    # 5. Initialize Conversation Manager
    logger.info("Initializing Conversation Manager...")
    conversation_manager = ConversationManager()
    await conversation_manager.init()
    
    # 6. Initialize Query Engine
    logger.info("Initializing Query Engine...")
    query_engine = QueryEngine()
    await query_engine.init()
    
    # 7. Initialize TTS Integration
    logger.info("Initializing TTS Integration...")
    tts_integration = TTSIntegration(
        voice_id=os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'),
        enable_caching=True
    )
    await tts_integration.init()
    
    # 8. Create Pipeline with all components
    logger.info("Creating Voice AI Pipeline...")
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=stt_integration,  # Pass the STT integration instead of raw speech recognizer
        conversation_manager=conversation_manager,
        query_engine=query_engine,
        tts_integration=tts_integration
    )
    
    # 9. Initialize Twilio handler
    twilio_handler = TwilioHandler(voice_ai_pipeline, base_url)
    await twilio_handler.start()
    
    logger.info("System initialized successfully with knowledge base integration")

# Rest of your Flask routes remain the same...
@app.route('/', methods=['GET'])
def index():
    """Simple test endpoint."""
    return "Voice AI Agent is running with OpenAI + Pinecone knowledge base!"

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls."""
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
        
        # Add initial silence
        response.pause(length=1)
        
        # Create WebSocket URL for streaming
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create Connect with Stream for bi-directional audio
        connect = Connect()
        
        stream = Stream(
            name="stream", 
            url=ws_url, 
            track="inbound_track"
        )
        
        # Set basic parameters
        stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
        
        connect.append(stream)
        response.append(connect)
        
        # Return TwiML
        twiml = str(response)
        return Response(twiml, mimetype='text/xml')
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
    
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    try:
        twilio_handler.handle_status_callback(call_sid, call_status)
        
        if call_status in ['completed', 'failed', 'busy', 'no-answer'] and call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            if 'terminate_flag' in loop_info:
                loop_info['terminate_flag'].set()
                
            if loop_info.get('thread'):
                thread = loop_info['thread']
                thread.join(timeout=1.0)
                
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
    """Handle WebSocket media stream."""
    logger.info(f"WebSocket connection attempt for call {call_sid}")
    
    if not twilio_handler or not voice_ai_pipeline:
        logger.error("System not initialized")
        return ""
    
    ws = None
    try:
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Initialize WebSocketHandler with the properly configured pipeline
        ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=voice_ai_pipeline)
        
        # Create a new event loop for this WebSocket connection
        loop = asyncio.new_event_loop()
        
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
            
            del call_event_loops[call_sid]
        
        if ws:
            try:
                ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        logger.info(f"WebSocket connection cleanup complete for call {call_sid}")
        return ""

def init_system():
    """Run the async initialization in a synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_system())
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting Voice AI Agent with OpenAI + Pinecone knowledge base...")
    
    # Initialize the system before starting the Flask app
    init_system()
    
    # Run the Flask app
    print(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)