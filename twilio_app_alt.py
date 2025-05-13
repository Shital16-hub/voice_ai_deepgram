#!/usr/bin/env python3
"""
Optimized Twilio application with direct MULAW support and improved error handling.
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
import signal
from contextlib import asynccontextmanager
import numpy as np
from flask import Flask, request, Response, jsonify
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

# Import optimized modules
from telephony.audio import MulawBufferProcessor
from telephony.twilio_handler import TwilioHandler
from telephony.websocket_handler import WebSocketHandler
from telephony.config import HOST, PORT, DEBUG, LOG_LEVEL, LOG_FORMAT

# Import knowledge base components
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.pinecone_manager import PineconeManager

# Import optimized STT/TTS integrations
from speech_to_text.simple_google_stt import GoogleCloudStreamingSTT
from integration.stt_integration import STTIntegration
from integration.tts_integration import TTSIntegration
from integration.pipeline import VoiceAIAgentPipeline

# Load environment variables
load_dotenv()

# Set up logging with better configuration
def configure_logging():
    """Configure logging with appropriate levels."""
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set base level
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from verbose modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('simple_websocket').setLevel(logging.WARNING)
    
    # File handler for detailed debugging
    if DEBUG:
        file_handler = logging.FileHandler('voice_ai_debug.log')
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
call_event_loops = {}
initialization_lock = asyncio.Lock()
system_initialized = False

# Graceful shutdown handler
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Close all active connections
    for call_sid, loop_info in call_event_loops.items():
        try:
            if 'loop' in loop_info and loop_info['loop'].is_running():
                loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
        except Exception as e:
            logger.error(f"Error stopping loop for call {call_sid}: {e}")
    
    # Close TTS connections
    if voice_ai_pipeline and hasattr(voice_ai_pipeline, 'tts_integration'):
        try:
            asyncio.run(voice_ai_pipeline.tts_integration.close())
        except Exception as e:
            logger.error(f"Error closing TTS: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def get_system_components():
    """Context manager for system components."""
    global system_initialized
    
    if not system_initialized:
        async with initialization_lock:
            if not system_initialized:  # Double-check locking
                await initialize_system()
                system_initialized = True
    
    yield {
        'twilio_handler': twilio_handler,
        'pipeline': voice_ai_pipeline,
        'conversation_manager': conversation_manager,
        'query_engine': query_engine
    }

async def initialize_system():
    """Initialize all system components with optimizations."""
    global twilio_handler, voice_ai_pipeline, conversation_manager, query_engine, base_url
    
    logger.info("Initializing optimized Voice AI system...")
    
    # Get base URL
    base_url = os.getenv('BASE_URL')
    if not base_url:
        logger.error("BASE_URL not set")
        raise ValueError("BASE_URL must be set")
    
    logger.info(f"Base URL: {base_url}")
    
    try:
        # 1. Initialize Google Cloud STT with MULAW support
        logger.info("Initializing Google Cloud Speech with MULAW...")
        speech_recognizer = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,  # 8kHz for Twilio
            encoding="MULAW",  # Direct MULAW support
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
        
        # Optimize for telephony
        stt_integration.optimize_for_telephony()
        
        # 3. Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pinecone_manager = PineconeManager()
        await pinecone_manager.init()
        
        # Verify Pinecone connection
        stats = await pinecone_manager.get_stats()
        logger.info(f"Pinecone stats: {stats}")
        
        # 4. Initialize OpenAI Assistant
        logger.info("Initializing OpenAI Assistant...")
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
        
        # 7. Initialize TTS with MULAW support
        logger.info("Initializing TTS with MULAW...")
        tts_integration = TTSIntegration(
            voice_id=os.getenv('TTS_VOICE_ID'),
            enable_caching=True
        )
        await tts_integration.init()
        
        # 8. Create optimized pipeline
        logger.info("Creating optimized pipeline...")
        voice_ai_pipeline = VoiceAIAgentPipeline(
            speech_recognizer=stt_integration,
            conversation_manager=conversation_manager,
            query_engine=query_engine,
            tts_integration=tts_integration
        )
        
        # 9. Initialize Twilio handler
        twilio_handler = TwilioHandler(voice_ai_pipeline, base_url)
        await twilio_handler.start()
        
        logger.info("System initialization complete!")
        
        # Test the system with a simple query
        logger.info("Running system test...")
        test_result = await conversation_manager.handle_user_input(
            "system_test", 
            "Hello, this is a test"
        )
        logger.info(f"System test result: {test_result.get('response')[:50] if test_result.get('response') else 'No response'}")
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        raise

# Flask routes
@app.route('/', methods=['GET'])
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "system": "Voice AI Agent with MULAW support",
        "initialized": system_initialized
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check."""
    health_status = {
        "status": "healthy" if system_initialized else "initializing",
        "timestamp": time.time(),
        "components": {}
    }
    
    if system_initialized:
        try:
            # Check pipeline stats
            if voice_ai_pipeline:
                health_status["components"]["pipeline"] = voice_ai_pipeline.get_performance_stats()
            
            # Check TTS stats
            if voice_ai_pipeline and hasattr(voice_ai_pipeline, 'tts_integration'):
                health_status["components"]["tts"] = voice_ai_pipeline.tts_integration.get_stats()
            
            # Check STT stats
            if voice_ai_pipeline and hasattr(voice_ai_pipeline, 'speech_recognizer'):
                if hasattr(voice_ai_pipeline.speech_recognizer, 'get_stats'):
                    health_status["components"]["stt"] = voice_ai_pipeline.speech_recognizer.get_stats()
            
        except Exception as e:
            logger.error(f"Error getting health stats: {e}")
            health_status["error"] = str(e)
    
    return jsonify(health_status)

@app.route('/voice/incoming', methods=['POST'])
async def handle_incoming_call():
    """Handle incoming voice calls with optimizations."""
    logger.info("Processing incoming call...")
    
    async with get_system_components() as components:
        handler = components['twilio_handler']
        
        if not handler:
            logger.error("System not initialized")
            return Response(
                '<?xml version="1.0" encoding="UTF-8"?><Response><Say>System unavailable</Say></Response>',
                mimetype='text/xml'
            )
        
        # Extract call info
        from_number = request.form.get('From')
        to_number = request.form.get('To')
        call_sid = request.form.get('CallSid')
        
        logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
        
        try:
            # Handle the call with optimized TwiML
            response = VoiceResponse()
            
            # Create WebSocket URL for MULAW streaming
            ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
            
            # Create Connect with optimized Stream
            connect = Connect()
            stream = Stream(
                name="audio_stream",
                url=ws_url,
                track="inbound_track"
            )
            
            # Set MULAW parameters for optimal performance
            stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
            
            connect.append(stream)
            response.append(connect)
            
            # Add minimal greeting to start conversation quickly
            response.say("Hello", voice='alice', rate='medium')
            
            return Response(str(response), mimetype='text/xml')
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {e}", exc_info=True)
            return Response(
                '<?xml version="1.0" encoding="UTF-8"?><Response><Say>Service unavailable</Say></Response>',
                mimetype='text/xml'
            )

@app.route('/voice/status', methods=['POST'])
async def handle_status_callback():
    """Handle call status callbacks with cleanup."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    logger.info(f"Status callback - CallSid: {call_sid}, Status: {call_status}")
    
    async with get_system_components() as components:
        handler = components['twilio_handler']
        
        if handler:
            handler.handle_status_callback(call_sid, call_status)
        
        # Clean up event loop resources for completed calls
        if call_status in ['completed', 'failed', 'busy', 'no-answer'] and call_sid in call_event_loops:
            loop_info = call_event_loops[call_sid]
            
            try:
                # Stop the loop
                if 'loop' in loop_info and loop_info['loop'].is_running():
                    loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
                
                # Wait for thread to finish
                if 'thread' in loop_info:
                    loop_info['thread'].join(timeout=1.0)
                
                # Clean up
                del call_event_loops[call_sid]
                logger.info(f"Cleaned up resources for call {call_sid}")
            except Exception as e:
                logger.error(f"Error cleaning up call {call_sid}: {e}")
    
    return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
async def handle_media_stream(call_sid):
    """Handle WebSocket media stream with MULAW optimization."""
    logger.info(f"WebSocket connection for call {call_sid}")
    
    async with get_system_components() as components:
        if not components['pipeline']:
            logger.error("System not initialized")
            return ""
        
        ws = None
        try:
            # Accept WebSocket connection
            ws = Server.accept(request.environ)
            logger.info(f"WebSocket established for call {call_sid}")
            
            # Create WebSocket handler with optimized pipeline
            ws_handler = WebSocketHandler(call_sid=call_sid, pipeline=components['pipeline'])
            
            # Create dedicated event loop
            loop = asyncio.new_event_loop()
            
            def run_ws_handler():
                """Run WebSocket handler in dedicated thread."""
                asyncio.set_event_loop(loop)
                try:
                    # Send initial connected event
                    loop.create_task(ws_handler.handle_message(
                        json.dumps({
                            "event": "connected",
                            "protocol": "Call",
                            "version": "1.0.0"
                        }), ws
                    ))
                    
                    # Run the event loop
                    loop.run_forever()
                except Exception as e:
                    logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
                finally:
                    loop.close()
            
            # Start handler thread
            thread = threading.Thread(target=run_ws_handler, daemon=True)
            thread.start()
            
            # Store for cleanup
            call_event_loops[call_sid] = {
                'loop': loop,
                'thread': thread,
                'start_time': time.time()
            }
            
            # Process incoming messages
            while True:
                try:
                    message = ws.receive(timeout=10)  # 10s timeout
                    if message is None:
                        continue
                    
                    # Forward to handler
                    asyncio.run_coroutine_threadsafe(
                        ws_handler.handle_message(message, ws),
                        loop
                    )
                except Exception as e:
                    if "timeout" not in str(e).lower():
                        logger.error(f"WebSocket message error: {e}")
                    # Exit the loop on error
                    break
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            # Clean up
            if call_sid in call_event_loops:
                loop_info = call_event_loops[call_sid]
                
                try:
                    # Stop the loop gracefully
                    if 'loop' in loop_info and loop_info['loop'].is_running():
                        loop_info['loop'].call_soon_threadsafe(loop_info['loop'].stop)
                    
                    # Wait for thread
                    if 'thread' in loop_info:
                        loop_info['thread'].join(timeout=2.0)
                    
                    # Calculate session duration
                    session_duration = time.time() - loop_info.get('start_time', time.time())
                    logger.info(f"WebSocket session duration: {session_duration:.1f}s")
                    
                    del call_event_loops[call_sid]
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
            
            # Close WebSocket
            if ws:
                try:
                    ws.close()
                except:
                    pass
            
            logger.info(f"WebSocket cleanup complete for call {call_sid}")
    
    return ""

# Test endpoint for system verification
@app.route('/test', methods=['POST'])
async def test_system():
    """Test system components."""
    test_query = request.json.get('query', 'Hello, this is a test')
    
    async with get_system_components() as components:
        if not components['conversation_manager']:
            return jsonify({"error": "System not initialized"}), 503
        
        try:
            start_time = time.time()
            result = await components['conversation_manager'].handle_user_input(
                "test_user",
                test_query
            )
            processing_time = time.time() - start_time
            
            return jsonify({
                "query": test_query,
                "response": result.get('response', 'No response'),
                "processing_time": processing_time,
                "cached": result.get('cached', False),
                "source": result.get('source', 'unknown')
            })
        except Exception as e:
            logger.error(f"Test error: {e}")
            return jsonify({"error": str(e)}), 500

def init_system_sync():
    """Initialize system synchronously for Flask startup."""
    """Run async initialization in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(initialize_system())
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        sys.exit(1)
    finally:
        loop.close()

if __name__ == '__main__':
    print("Starting optimized Voice AI Agent with MULAW support...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    
    # Initialize system before starting Flask
    logger.info("Initializing system...")
    init_system_sync()
    
    # Start Flask app
    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )