"""
Fixed Twilio FastAPI app updated for OpenAI + Pinecone.
CRITICAL FIXES: Fixed session management and connection handling.
"""
#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from telephony.query_engine_api import QueryEngineAPI

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import uvicorn

# Project imports - updated for OpenAI + Pinecone
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from telephony.config import HOST, PORT, DEBUG

# Updated imports for OpenAI + Pinecone
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from integration.tts_integration import TTSIntegration

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging for ultra low latency (less verbose)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs for performance
logging.getLogger('google.cloud').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('uvicorn.access').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Global instances
voice_ai_pipeline = None
base_url = None

# Track active calls (simplified)
active_calls = {}

async def initialize_system():
    """Initialize the Voice AI system with OpenAI + Pinecone and infinite streaming."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing Voice AI with OpenAI + Pinecone and infinite streaming...")
    
    # Validate required environment variables
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL environment variable must be set")
    
    # Check OpenAI and Pinecone API keys
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    if not os.getenv('PINECONE_API_KEY'):
        raise ValueError("PINECONE_API_KEY environment variable must be set")
    
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_creds or not os.path.exists(google_creds):
        raise FileNotFoundError(f"Google credentials file not found: {google_creds}")
    
    logger.info(f"Using BASE_URL: {base_url}")
    
    # Initialize Voice AI Agent with OpenAI + Pinecone and infinite streaming
    agent = VoiceAIAgent(
        storage_dir='./storage',
        openai_model='gpt-4o-mini',  # Fast OpenAI model
        llm_temperature=0.7,  # For more conversational responses
        credentials_file=google_creds,
        # UPDATED: Enable infinite streaming
        use_infinite_streaming=True
    )
    await agent.init()
    
    # Initialize TTS with minimal latency settings
    tts = TTSIntegration(
        voice_name="en-US-Neural2-C",
        voice_gender=None,
        language_code="en-US",
        enable_caching=True,
        credentials_file=google_creds
    )
    await tts.init()
    
    # Create pipeline with OpenAI + Pinecone components and infinite streaming
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    # CRITICAL FIX: Add QueryEngineAPI to pipeline
    from telephony.query_engine_api import QueryEngineAPI
    query_engine_api = QueryEngineAPI(agent.query_engine)
    voice_ai_pipeline.query_engine_api = query_engine_api
    logger.info("QueryEngineAPI attached to pipeline")
    
    logger.info("System initialization completed with infinite streaming for uninterrupted calls")


async def cleanup_system():
    """Cleanup system resources."""
    logger.info("Cleaning up system resources...")
    
    # Close all active calls
    for call_sid, handler in list(active_calls.items()):
        try:
            await handler._cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up call {call_sid}: {e}")
    
    logger.info("System cleanup completed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await initialize_system()
    yield
    # Shutdown
    await cleanup_system()

# Create FastAPI app with minimal middleware for performance
app = FastAPI(
    title="Ultra Low Latency Voice AI with OpenAI + Pinecone",
    description="Voice AI optimized for <2s latency using OpenAI + Pinecone",
    version="3.1.0",
    lifespan=lifespan
)

# Minimal middleware for performance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Ultra Low Latency Voice AI with OpenAI + Pinecone",
        "version": "3.1.0",
        "active_calls": len(active_calls),
        "latency_target": "<2 seconds",
        "knowledge_base": "OpenAI + Pinecone"
    }

@app.post("/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming voice calls with optimized TwiML."""
    logger.info("Incoming call received")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        return Response(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">System not ready. Please try again.</Say>
    <Hangup/>
</Response>''',
            media_type='text/xml'
        )
    
    # Parse form data
    form = await request.form()
    call_sid = form.get('CallSid')
    
    logger.info(f"Processing call: {call_sid}")
    
    try:
        # Create optimized TwiML response
        response = VoiceResponse()
        
        # Create WebSocket URL
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Create Connect with optimized settings for low latency
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        
        # Add optimized parameters for Twilio
        connect.append(stream)
        response.append(connect)
        
        logger.info(f"Generated TwiML for call {call_sid}")
        return Response(content=str(response), media_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        return Response(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">An error occurred. Please try again.</Say>
    <Hangup/>
</Response>''',
            media_type='text/xml'
        )

@app.websocket("/ws/stream/{call_sid}")
async def handle_media_stream(websocket: WebSocket, call_sid: str):
    """Ultra low latency WebSocket handler."""
    logger.info(f"WebSocket connection for call {call_sid}")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        await websocket.close(code=1011)
        return
    
    handler = None
    health_check_task = None
    
    try:
        # Accept connection immediately
        await websocket.accept()
        logger.info(f"WebSocket established: {call_sid}")
        
        # Create handler with ultra low latency settings
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # CRITICAL NEW FIX: Add periodic health check task
        async def periodic_health_check():
            while True:
                try:
                    await handler._ensure_streaming_health()
                except Exception as e:
                    logger.error(f"Error in health check: {e}")
                await asyncio.sleep(10)  # UPDATED: Check more frequently (10 seconds)
                
        health_check_task = asyncio.create_task(periodic_health_check())
        
        # Process messages with minimal overhead
        try:
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    event_type = data.get('event')
                    
                    if event_type == 'connected':
                        logger.info(f"Connected: {call_sid}")
                        
                    elif event_type == 'start':
                        stream_sid = data.get('streamSid')
                        logger.info(f"Stream started: {stream_sid}")
                        handler.stream_sid = stream_sid
                        handler.state.stream_sid = stream_sid
                        
                        # Start conversation immediately
                        await handler.start_conversation(websocket)
                        
                    elif event_type == 'media':
                        # Handle audio with minimal processing
                        await handler._handle_audio(data, websocket)
                        
                    elif event_type == 'stop':
                        logger.info(f"Stream stopped: {call_sid}")
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message[:100]}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {call_sid}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        
    except Exception as e:
        logger.error(f"WebSocket establishment error: {e}")
        
    finally:
        # Clean up health check task
        if health_check_task:
            health_check_task.cancel()
            try:
                await health_check_task
            except asyncio.CancelledError:
                pass
        
        # Clean up handler
        if handler:
            try:
                await handler._cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        logger.info(f"WebSocket cleanup complete: {call_sid}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = {
        "timestamp": time.time(),
        "active_calls": len(active_calls),
        "system_ready": voice_ai_pipeline is not None,
        "knowledge_base": "OpenAI + Pinecone",
        "calls": {}
    }
    
    # Add call-specific stats
    for call_sid, handler in active_calls.items():
        try:
            stats["calls"][call_sid] = handler.get_stats()
        except Exception as e:
            stats["calls"][call_sid] = {"error": str(e)}
    
    return stats

@app.get("/health")
async def health_check():
    """Simple health check."""
    return {
        "status": "healthy" if voice_ai_pipeline else "initializing",
        "timestamp": time.time(),
        "active_calls": len(active_calls),
        "knowledge_base": "OpenAI + Pinecone"
    }

if __name__ == '__main__':
    print("Starting Ultra Low Latency Voice AI with OpenAI + Pinecone...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"OpenAI API Key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"Pinecone API Key: {'Set' if os.getenv('PINECONE_API_KEY') else 'Not set'}")
    print(f"Google Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    
    # Run with optimized settings for latency
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",  # Reduced logging for performance
        access_log=False,     # Disable access logging for performance
        # Optimized for latency
        loop="asyncio",
        http="h11",
        # WebSocket optimizations
        ws_ping_interval=15,  # UPDATED: More frequent pings
        ws_ping_timeout=10,
        # Reduce worker threads for latency
        workers=1
    )