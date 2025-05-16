#!/usr/bin/env python3
"""
Enhanced Twilio application using FastAPI with proper WebSocket support,
Redis session management, and comprehensive monitoring.
"""
import os
import sys
import asyncio
import logging
import json
import time
import redis
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import uvicorn

# Project imports
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from telephony.config import HOST, PORT, DEBUG
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from integration.tts_integration import TTSIntegration

# Monitoring imports
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced logging configuration - FIXED
structlog.configure(
    wrapper_class=structlog.stdlib.BoundLogger,  # Fixed: was LoggerBag
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

# Global instances
voice_ai_pipeline = None
base_url = None
redis_client = None

# Track active calls with enhanced session management
active_calls = {}
call_sessions = {}

# Prometheus metrics
CALL_COUNTER = Counter('voice_ai_calls_total', 'Total number of voice calls', ['status'])
CALL_DURATION = Histogram('voice_ai_call_duration_seconds', 'Call duration in seconds')
ACTIVE_CALLS = Gauge('voice_ai_active_calls', 'Number of active calls')
TRANSCRIPTION_COUNTER = Counter('voice_ai_transcriptions_total', 'Total transcriptions', ['valid'])
RESPONSE_COUNTER = Counter('voice_ai_responses_total', 'Total responses sent')
ERROR_COUNTER = Counter('voice_ai_errors_total', 'Total errors', ['type'])
WEBSOCKET_CONNECTIONS = Gauge('voice_ai_websocket_connections', 'Active WebSocket connections')

def init_redis():
    """Initialize Redis connection."""
    global redis_client
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for session management")
        return True
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. Session management will be limited.")
        redis_client = None
        return False

async def initialize_system():
    """Initialize the Voice AI system with enhanced configuration."""
    global voice_ai_pipeline, base_url
    
    logger.info("Initializing enhanced Voice AI Agent system...")
    
    # Initialize Redis
    init_redis()
    
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
    
    # Initialize Voice AI Agent with enhanced settings
    agent = VoiceAIAgent(
        storage_dir='./storage',
        model_name='mistral:7b-instruct-v0.2-q4_0',
        llm_temperature=0.7,
        credentials_file=google_creds
    )
    await agent.init()
    
    # Initialize enhanced TTS
    tts = TTSIntegration(
        voice_name="en-US-Neural2-C",
        voice_gender=None,
        language_code="en-US",
        enable_caching=True,
        credentials_file=google_creds
    )
    await tts.init()
    
    # Create pipeline with enhanced features
    voice_ai_pipeline = VoiceAIAgentPipeline(
        speech_recognizer=agent.speech_recognizer,
        conversation_manager=agent.conversation_manager,
        query_engine=agent.query_engine,
        tts_integration=tts
    )
    
    logger.info("Enhanced system initialization completed successfully")

async def cleanup_system():
    """Cleanup system resources."""
    logger.info("Cleaning up system resources...")
    
    # Close all active calls
    for call_sid, handler in list(active_calls.items()):
        try:
            await handler._cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up call {call_sid}: {e}")
    
    # Close Redis connection
    if redis_client:
        redis_client.close()
    
    logger.info("System cleanup completed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await initialize_system()
    yield
    # Shutdown
    await cleanup_system()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Voice AI Agent",
    description="Enhanced Voice AI Agent with WebRTC support",
    version="2.1.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def index():
    """Enhanced health check endpoint."""
    return {
        "status": "running",
        "message": "Enhanced Voice AI Agent with WebRTC support",
        "version": "2.1.0",
        "features": {
            "webrtc_echo_cancellation": True,
            "redis_session_management": redis_client is not None,
            "prometheus_metrics": True,
            "circuit_breaker": True
        },
        "active_calls": len(active_calls),
        "total_sessions": len(call_sessions)
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with enhanced metrics."""
    health_status = {
        "status": "healthy" if voice_ai_pipeline else "initializing",
        "timestamp": time.time(),
        "components": {
            "pipeline": voice_ai_pipeline is not None,
            "redis": redis_client is not None,
            "active_calls": len(active_calls),
            "active_sessions": len(call_sessions),
            "websocket_connections": WEBSOCKET_CONNECTIONS._value.get()
        }
    }
    
    if voice_ai_pipeline:
        health_status["components"].update({
            "stt": hasattr(voice_ai_pipeline, 'speech_recognizer'),
            "tts": hasattr(voice_ai_pipeline, 'tts_integration'),
            "kb": hasattr(voice_ai_pipeline, 'query_engine')
        })
    
    # Enhanced session health info
    if call_sessions:
        session_ages = [time.time() - session['start_time'] for session in call_sessions.values()]
        health_status["session_metrics"] = {
            "oldest_session_age": max(session_ages),
            "average_session_age": sum(session_ages) / len(session_ages),
            "sessions_over_5min": len([age for age in session_ages if age > 300])
        }
    
    # Add system health indicators
    health_status["system_health"] = {
        "memory_usage_mb": _get_memory_usage(),
        "cpu_usage_percent": _get_cpu_usage(),
        "disk_usage_percent": _get_disk_usage()
    }
    
    return health_status

@app.post("/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming voice calls with enhanced error handling."""
    CALL_COUNTER.labels(status="incoming").inc()
    logger.info("Received incoming call request")
    
    if not voice_ai_pipeline:
        logger.error("System not initialized")
        ERROR_COUNTER.labels(type="system_not_initialized").inc()
        return Response(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">System is not initialized. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type='text/xml'
        )
    
    # Parse form data
    form = await request.form()
    from_number = form.get('From')
    to_number = form.get('To')
    call_sid = form.get('CallSid')
    
    logger.info(f"Incoming call - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    # Track call session with enhanced metadata
    session_data = {
        "start_time": time.time(),
        "from_number": from_number,
        "to_number": to_number,
        "status": "initiated",
        "features": {
            "webrtc_enabled": True,
            "echo_cancellation": True,
            "redis_session": redis_client is not None
        }
    }
    
    call_sessions[call_sid] = session_data
    
    # Save to Redis if available
    if redis_client:
        try:
            redis_client.setex(f"call_session:{call_sid}", 3600, json.dumps(session_data))
        except Exception as e:
            logger.error(f"Error saving call session to Redis: {e}")
    
    try:
        # Validate base_url
        if not base_url:
            raise ValueError("BASE_URL not configured")
        
        # Create enhanced TwiML response
        response = VoiceResponse()
        
        # Create WebSocket URL with enhanced path
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"WebSocket URL: {ws_url}")
        
        # Create Connect with enhanced Stream settings
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        
        connect.append(stream)
        response.append(connect)
        
        ACTIVE_CALLS.inc()
        logger.info(f"Generated enhanced TwiML for call {call_sid}")
        return Response(content=str(response), media_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}", exc_info=True)
        ERROR_COUNTER.labels(type="incoming_call_error").inc()
        
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "error"
            call_sessions[call_sid]["error"] = str(e)
        
        return Response(
            content='''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">An error occurred. Please try again later.</Say>
    <Hangup/>
</Response>''',
            media_type='text/xml'
        )

@app.post("/voice/status")
async def handle_status_callback(request: Request):
    """Handle call status callbacks with enhanced session tracking."""
    form = await request.form()
    call_sid = form.get('CallSid')
    call_status = form.get('CallStatus')
    call_duration = form.get('CallDuration', '0')
    
    logger.info(f"Call {call_sid} status: {call_status}, duration: {call_duration}s")
    
    # Update metrics
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        CALL_COUNTER.labels(status=call_status).inc()
        if call_duration.isdigit():
            CALL_DURATION.observe(float(call_duration))
            ACTIVE_CALLS.dec()
    
    # Update session info
    if call_sid in call_sessions:
        call_sessions[call_sid].update({
            "status": call_status,
            "duration": call_duration,
            "end_time": time.time() if call_status in ['completed', 'failed', 'busy', 'no-answer'] else None
        })
        
        # Update Redis
        if redis_client:
            try:
                redis_client.setex(f"call_session:{call_sid}", 3600, json.dumps(call_sessions[call_sid]))
            except Exception as e:
                logger.error(f"Error updating call session in Redis: {e}")
    
    # Clean up completed calls
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            handler = active_calls[call_sid]
            try:
                await handler._cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                ERROR_COUNTER.labels(type="cleanup_error").inc()
            del active_calls[call_sid]
            logger.info(f"Cleaned up call {call_sid}")
        
        # Keep session info for debugging
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "completed"
            # Schedule cleanup
            asyncio.create_task(_cleanup_old_sessions())
    
    return Response(status_code=204)

async def _cleanup_old_sessions():
    """Clean up old call sessions to prevent memory leaks."""
    current_time = time.time()
    sessions_to_remove = []
    
    for call_sid, session in call_sessions.items():
        # Remove sessions older than 1 hour
        if current_time - session['start_time'] > 3600:
            sessions_to_remove.append(call_sid)
    
    for call_sid in sessions_to_remove:
        # Remove from memory
        del call_sessions[call_sid]
        # Remove from Redis
        if redis_client:
            try:
                redis_client.delete(f"call_session:{call_sid}")
            except Exception as e:
                logger.error(f"Error removing session from Redis: {e}")
        logger.debug(f"Cleaned up old session: {call_sid}")

@app.websocket("/ws/stream/{call_sid}")
async def handle_media_stream(websocket: WebSocket, call_sid: str):
    """Enhanced WebSocket handler with proper FastAPI WebSocket support."""
    logger.info(f"WebSocket connection request for call {call_sid}")
    WEBSOCKET_CONNECTIONS.inc()
    
    if not voice_ai_pipeline:
        logger.error("System not initialized for WebSocket connection")
        await websocket.close(code=1012)
        return
    
    handler = None
    
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        logger.info(f"WebSocket connection established for call {call_sid}")
        
        # Create enhanced handler
        handler = SimpleWebSocketHandler(call_sid, voice_ai_pipeline)
        active_calls[call_sid] = handler
        
        # Update session status
        if call_sid in call_sessions:
            call_sessions[call_sid]["status"] = "connected"
            call_sessions[call_sid]["ws_connected_time"] = time.time()
        
        # Process incoming messages with enhanced error handling
        try:
            while True:
                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.debug(f"WebSocket timeout for call {call_sid}")
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
                        handler.state.stream_sid = stream_sid
                        
                        # Start the conversation
                        await handler.start_conversation(websocket)
                        
                        # Update session
                        if call_sid in call_sessions:
                            call_sessions[call_sid]["stream_started"] = True
                            call_sessions[call_sid]["stream_sid"] = stream_sid
                        
                    elif event_type == 'media':
                        # Handle audio data
                        await handler._handle_audio(data, websocket)
                        
                    elif event_type == 'stop':
                        logger.info(f"Stream stopped for call {call_sid}")
                        await handler._cleanup()
                        
                        # Update session
                        if call_sid in call_sessions:
                            call_sessions[call_sid]["stream_stopped"] = True
                        break
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {message}")
                    ERROR_COUNTER.labels(type="invalid_json").inc()
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    ERROR_COUNTER.labels(type="message_processing").inc()
                    continue
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for call {call_sid}")
        except Exception as e:
            logger.error(f"Error in WebSocket loop: {e}")
            ERROR_COUNTER.labels(type="websocket_error").inc()
        
    except Exception as e:
        logger.error(f"Error establishing WebSocket: {e}", exc_info=True)
        ERROR_COUNTER.labels(type="websocket_establishment").inc()
        
    finally:
        # Cleanup resources
        WEBSOCKET_CONNECTIONS.dec()
        
        if handler:
            try:
                await handler._cleanup()
            except Exception as e:
                logger.error(f"Error during handler cleanup: {e}")
        
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        # Update session
        if call_sid in call_sessions:
            call_sessions[call_sid]["ws_disconnected_time"] = time.time()
        
        logger.info(f"WebSocket cleanup complete for call {call_sid}")

@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics with enhanced metrics."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "initialized": voice_ai_pipeline is not None,
            "active_calls": len(active_calls),
            "total_sessions": len(call_sessions),
            "base_url": base_url,
            "redis_connected": redis_client is not None,
            "websocket_connections": WEBSOCKET_CONNECTIONS._value.get()
        },
        "calls": {},
        "sessions": {},
        "metrics": {
            "total_calls": CALL_COUNTER._value.sum(),
            "total_transcriptions": TRANSCRIPTION_COUNTER._value.sum(),
            "total_responses": RESPONSE_COUNTER._value.sum(),
            "total_errors": ERROR_COUNTER._value.sum(),
        }
    }
    
    # Add individual call stats
    for call_sid, handler in active_calls.items():
        try:
            call_stats = handler.get_stats()
            stats["calls"][call_sid] = call_stats
            
            # Update Prometheus metrics
            if 'transcriptions' in call_stats:
                TRANSCRIPTION_COUNTER.labels(valid="true").inc(call_stats['transcriptions'])
            if 'invalid_transcriptions' in call_stats:
                TRANSCRIPTION_COUNTER.labels(valid="false").inc(call_stats['invalid_transcriptions'])
            if 'responses_sent' in call_stats:
                RESPONSE_COUNTER.inc(call_stats['responses_sent'])
            
        except Exception as e:
            logger.error(f"Error getting stats for call {call_sid}: {e}")
            stats["calls"][call_sid] = {"error": str(e)}
    
    # Add session information
    for call_sid, session in call_sessions.items():
        stats["sessions"][call_sid] = {
            **session,
            "age": time.time() - session['start_time']
        }
    
    # Add enhanced conversation metrics
    if active_calls:
        total_transcriptions = sum(handler.state.transcriptions for handler in active_calls.values())
        total_responses = sum(handler.state.responses_sent for handler in active_calls.values())
        total_vad_activations = sum(handler.state.vad_activations for handler in active_calls.values())
        total_echo_detections = sum(handler.state.echo_detections for handler in active_calls.values())
        
        stats["conversation_metrics"] = {
            "total_transcriptions": total_transcriptions,
            "total_responses": total_responses,
            "total_vad_activations": total_vad_activations,
            "total_echo_detections": total_echo_detections,
            "average_transcriptions_per_call": total_transcriptions / len(active_calls),
            "average_responses_per_call": total_responses / len(active_calls),
            "vad_efficiency": round(total_vad_activations / max(sum(h.state.audio_received for h in active_calls.values()), 1) * 100, 2),
            "echo_detection_rate": round(total_echo_detections / max(total_transcriptions, 1) * 100, 2)
        }
    
    return stats

@app.get("/metrics")
async def get_prometheus_metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/config")
async def get_config():
    """Get current enhanced configuration."""
    config = {
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "base_url": base_url,
        "google_credentials": os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        "google_project": os.getenv('GOOGLE_CLOUD_PROJECT'),
        "redis_url": os.getenv('REDIS_URL'),
        "features": {
            "webrtc_echo_cancellation": True,
            "voice_activity_detection": True,
            "redis_session_management": redis_client is not None,
            "prometheus_metrics": True,
            "circuit_breaker": True,
            "adaptive_audio_processing": True,
            "enhanced_error_handling": True
        }
    }
    return config

# System monitoring endpoints
@app.get("/system/memory")
async def get_memory_usage():
    """Get system memory usage."""
    return {"memory_usage_mb": _get_memory_usage()}

@app.get("/system/cpu")
async def get_cpu_usage():
    """Get system CPU usage."""
    return {"cpu_usage_percent": _get_cpu_usage()}

@app.get("/system/disk")
async def get_disk_usage():
    """Get system disk usage."""
    return {"disk_usage_percent": _get_disk_usage()}

# Helper functions for system monitoring
def _get_memory_usage() -> float:
    """Get memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / 1024 / 1024, 2)
    except ImportError:
        return 0.0

def _get_cpu_usage() -> float:
    """Get CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0

def _get_disk_usage() -> float:
    """Get disk usage percentage."""
    try:
        import psutil
        return psutil.disk_usage('/').percent
    except ImportError:
        return 0.0

# Error handlers
@app.exception_handler(500)
async def internal_error(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    ERROR_COUNTER.labels(type="internal_server_error").inc()
    return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.exception_handler(404)
async def not_found(request: Request, exc: Exception):
    """Handle 404 errors."""
    return JSONResponse({"error": "Not found"}, status_code=404)

if __name__ == '__main__':
    print("Starting Enhanced Voice AI Agent with WebRTC support...")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"Google Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    print(f"Redis URL: {os.getenv('REDIS_URL', 'Not set')}")
    
    # Run FastAPI app with Uvicorn
    logger.info(f"Starting enhanced server on {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info" if DEBUG else "warning",
        access_log=DEBUG,
        # WebSocket configuration
        ws_ping_interval=20,
        ws_ping_timeout=10,
        # Production settings
        loop="asyncio",
        http="h11"
    )