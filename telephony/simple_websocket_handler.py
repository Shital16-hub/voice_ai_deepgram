"""
Enhanced WebSocket handler with WebRTC-based echo cancellation, improved session management,
Redis-based session persistence, and comprehensive error handling.
"""
import json
import asyncio
import logging
import base64
import time
import os
import redis
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections import deque

# Use the enhanced STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

# Use the enhanced TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Enhanced session state management with Redis persistence."""
    call_sid: str
    stream_sid: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    conversation_active: bool = True
    is_speaking: bool = False
    expecting_speech: bool = True
    call_ended: bool = False
    
    # Enhanced metrics
    audio_received: int = 0
    transcriptions: int = 0
    responses_sent: int = 0
    echo_detections: int = 0
    invalid_transcriptions: int = 0
    vad_activations: int = 0
    
    # Quality metrics
    last_transcription_time: float = field(default_factory=time.time)
    last_audio_time: float = field(default_factory=time.time)
    last_tts_time: Optional[float] = None
    waiting_for_response: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "start_time": self.start_time,
            "conversation_active": self.conversation_active,
            "is_speaking": self.is_speaking,
            "expecting_speech": self.expecting_speech,
            "call_ended": self.call_ended,
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "responses_sent": self.responses_sent,
            "echo_detections": self.echo_detections,
            "invalid_transcriptions": self.invalid_transcriptions,
            "vad_activations": self.vad_activations,
            "last_transcription_time": self.last_transcription_time,
            "last_audio_time": self.last_audio_time,
            "last_tts_time": self.last_tts_time,
            "waiting_for_response": self.waiting_for_response
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary (Redis deserialization)."""
        return cls(**data)

class SimpleWebSocketHandler:
    """
    Enhanced WebSocket handler with WebRTC echo cancellation, Redis session persistence,
    and comprehensive conversation management.
    """
    
    # Enhanced configuration
    MIN_TRANSCRIPTION_LENGTH = 2
    ECHO_DETECTION_WINDOW = 5.0
    SILENCE_TIMEOUT = 10.0
    RESPONSE_TIMEOUT = 8.0
    MAX_RECONNECTION_ATTEMPTS = 3
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with enhanced features and Redis integration."""
        self.call_sid = call_sid
        self.pipeline = pipeline
        
        # Initialize Redis for session persistence
        self._init_redis()
        
        # Get project ID dynamically
        self.project_id = self._get_project_id()
        
        # Initialize enhanced Google Cloud STT v2 with WebRTC features
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file,
            enable_vad=True,  # Enable WebRTC VAD
            enable_echo_suppression=True  # Enable WebRTC echo suppression
        )
        
        # Initialize enhanced Google Cloud TTS with echo callback
        self.tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2",
            echo_suppression_callback=self._tts_echo_callback
        )
        
        # Enhanced session state management
        self.state = SessionState(call_sid=call_sid)
        self._load_session_state()
        
        # Audio processing with enhanced buffering
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        self.min_chunk_size = 160  # 20ms minimum
        
        # WebSocket reference for response sending
        self._ws = None
        
        # Enhanced response tracking
        self.last_response_text = ""
        self.response_queue = asyncio.Queue()
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = 60.0
        self.last_error_time = None
        
        # Reconnection management
        self.reconnection_attempts = 0
        self.last_reconnection_time = None
        
        logger.info(f"Enhanced WebSocket handler initialized - Call: {call_sid}, "
                   f"Project: {self.project_id}, Redis: {'connected' if self.redis_client else 'disconnected'}")
    
    def _init_redis(self):
        """Initialize Redis connection for session persistence."""
        try:
            # Try to connect to Redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis for session persistence")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Session persistence disabled.")
            self.redis_client = None
    
    def _get_project_id(self) -> str:
        """Get project ID with enhanced error handling."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                import json
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Extracted project ID from credentials: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        # Fallback (should be configured properly)
        logger.warning("Using fallback project ID - this should be configured properly")
        return "my-tts-project-458404"
    
    def _tts_echo_callback(self, text: str, audio_data: bytes, fingerprint: Dict[str, Any]):
        """Callback for TTS output to register with echo suppression system."""
        # Register TTS output with STT for echo suppression
        if hasattr(self.stt_client, 'add_tts_fingerprint'):
            self.stt_client.add_tts_fingerprint(text, audio_data)
        
        # Update our own state
        self.state.last_tts_time = time.time()
        self.last_response_text = text
        
        logger.debug(f"Registered TTS output for echo suppression: {text[:30]}...")
    
    def _save_session_state(self):
        """Save session state to Redis."""
        if not self.redis_client:
            return
        
        try:
            session_key = f"voice_session:{self.call_sid}"
            session_data = json.dumps(self.state.to_dict())
            self.redis_client.setex(session_key, 3600, session_data)  # Expire in 1 hour
            logger.debug(f"Saved session state to Redis: {self.call_sid}")
        except Exception as e:
            logger.error(f"Error saving session state: {e}")
    
    def _load_session_state(self):
        """Load session state from Redis."""
        if not self.redis_client:
            return
        
        try:
            session_key = f"voice_session:{self.call_sid}"
            session_data = self.redis_client.get(session_key)
            if session_data:
                data = json.loads(session_data)
                self.state = SessionState.from_dict(data)
                logger.info(f"Loaded session state from Redis: {self.call_sid}")
        except Exception as e:
            logger.error(f"Error loading session state: {e}")
    
    def _clear_session_state(self):
        """Clear session state from Redis."""
        if not self.redis_client:
            return
        
        try:
            session_key = f"voice_session:{self.call_sid}"
            self.redis_client.delete(session_key)
            logger.debug(f"Cleared session state from Redis: {self.call_sid}")
        except Exception as e:
            logger.error(f"Error clearing session state: {e}")
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should activate."""
        if self.error_count < self.circuit_breaker_threshold:
            return False
        
        if self.last_error_time and (time.time() - self.last_error_time) > self.circuit_breaker_reset_time:
            # Reset circuit breaker
            self.error_count = 0
            self.last_error_time = None
            logger.info("Circuit breaker reset")
            return False
        
        return True
    
    def _record_error(self):
        """Record an error for circuit breaker."""
        self.error_count += 1
        self.last_error_time = time.time()
        logger.warning(f"Error recorded, count: {self.error_count}")
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Enhanced audio handling with WebRTC processing and circuit breaker."""
        # Check circuit breaker
        if self._should_circuit_break():
            logger.warning("Circuit breaker active, skipping audio processing")
            return
        
        # Skip audio processing if call has ended
        if self.state.call_ended:
            return
        
        # Skip audio while we're sending a response to prevent echo
        if self.state.waiting_for_response:
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio with error handling
        try:
            audio_data = base64.b64decode(payload)
            self.state.audio_received += 1
            self.state.last_audio_time = time.time()
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            self._record_error()
            return
        
        # Enhanced echo prevention - skip audio if too close to TTS output
        if (self.state.last_tts_time and 
            (time.time() - self.state.last_tts_time) < 1.5):
            logger.debug("Skipping audio - too close to TTS output")
            return
        
        # Start STT if not already started
        if not self.stt_client.is_streaming and self.state.conversation_active:
            logger.info("Starting enhanced STT streaming")
            await self.stt_client.start_streaming()
        
        # Process audio chunks with enhanced error handling
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self._record_error()
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Enhanced transcription handling with WebRTC result processing."""
        if result.is_final and result.text.strip():
            transcription = result.text.strip()
            confidence = result.confidence
            
            logger.info(f"Enhanced transcription (session {result.session_id}): '{transcription}' "
                       f"(confidence: {confidence:.2f}, VAD: {result.vad_detected}, "
                       f"Echo suppressed: {result.echo_suppressed})")
            
            self.state.last_transcription_time = time.time()
            
            # Update VAD statistics
            if result.vad_detected:
                self.state.vad_activations += 1
            
            # Update echo detection statistics
            if result.echo_suppressed:
                self.state.echo_detections += 1
            
            # Enhanced validation
            if self._is_valid_transcription(transcription, confidence, result):
                await self._process_final_transcription(transcription)
            else:
                self.state.invalid_transcriptions += 1
                logger.debug(f"Invalid transcription rejected: '{transcription}' "
                           f"(conf: {confidence:.2f}, VAD: {result.vad_detected})")
            
            # Save state after each transcription
            self._save_session_state()
    
    def _is_valid_transcription(self, transcription: str, confidence: float, 
                               result: StreamingTranscriptionResult) -> bool:
        """Enhanced transcription validation with WebRTC information."""
        # Basic length check
        if len(transcription.split()) < self.MIN_TRANSCRIPTION_LENGTH:
            return False
        
        # Confidence threshold (adaptive based on VAD)
        min_confidence = 0.3 if result.vad_detected else 0.5
        if confidence < min_confidence:
            logger.debug(f"Low confidence transcription: {confidence:.2f}")
            return False
        
        # If echo was suppressed, this should be more stringent
        if result.echo_suppressed:
            logger.debug("Transcription was echo suppressed")
            return False
        
        # Enhanced echo detection using WebRTC features
        if self._is_likely_echo(transcription):
            return False
        
        # Skip common filler words and artifacts
        transcription_lower = transcription.lower().strip()
        skip_patterns = [
            # Common filler words
            'um', 'uh', 'mmm', 'hmm', 'ah', 'er', 'oh',
            # Single words that might be misheard
            'okay', 'ok', 'yes', 'no', 'right', 'sure',
            # Silence indicators
            'silence', 'background', 'noise',
            # System echo patterns
            'ready to help', 'what would you like', 'how can i',
        ]
        
        if transcription_lower in skip_patterns:
            logger.debug(f"Skipping pattern: '{transcription}'")
            return False
        
        # Check for word-for-word matches with recent response
        if self.last_response_text:
            response_words = set(self.last_response_text.lower().split())
            transcription_words = set(transcription_lower.split())
            
            if len(transcription_words) > 0:
                overlap_ratio = len(response_words & transcription_words) / len(transcription_words)
                if overlap_ratio > 0.7:  # 70% overlap indicates echo
                    logger.debug(f"High word overlap with last response: {overlap_ratio:.2f}")
                    return False
        
        return True
    
    def _is_likely_echo(self, transcription: str) -> bool:
        """Enhanced echo detection using multiple heuristics and WebRTC information."""
        # Check timing - if transcription comes too soon after TTS, likely echo
        if (self.state.last_tts_time and 
            (time.time() - self.state.last_tts_time) < 2.0):
            # Check for system phrase patterns
            system_phrases = [
                "i'm ready to help",
                "what would you like to know",
                "how can i help",
                "voice assist",
                "pricing plans",
                "features",
                "basic plan",
                "professional plan",
                "enterprise plan"
            ]
            
            transcription_lower = transcription.lower()
            for phrase in system_phrases:
                if phrase in transcription_lower:
                    logger.debug(f"Echo detected - system phrase: {phrase}")
                    return True
            
            # Check for partial matches with recent TTS
            if self.last_response_text:
                # Split into significant words
                response_words = [w for w in self.last_response_text.lower().split() 
                                if len(w) > 3]
                transcription_words = transcription_lower.split()
                
                # Check for consecutive word matches
                for i in range(len(transcription_words) - 1):
                    consecutive_words = ' '.join(transcription_words[i:i+2])
                    if any(consecutive_words in ' '.join(response_words[j:j+2]) 
                          for j in range(len(response_words) - 1)):
                        logger.debug(f"Echo detected - consecutive words: {consecutive_words}")
                        return True
        
        return False
    
    async def _process_final_transcription(self, transcription: str):
        """Enhanced transcription processing with improved error handling."""
        # Update state
        self.state.transcriptions += 1
        self.state.waiting_for_response = True
        
        logger.info(f"Processing transcription: '{transcription}'")
        
        try:
            # Use asyncio timeout for knowledge base query
            async with asyncio.timeout(self.RESPONSE_TIMEOUT):
                # Query knowledge base through pipeline
                if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                    result = await self.pipeline.query_engine.query(transcription)
                    response_text = result.get("response", "")
                    
                    if response_text:
                        await self._send_response(response_text)
                    else:
                        logger.warning("No response generated from knowledge base")
                        await self._send_response("I'm sorry, I couldn't find an answer to that question.")
                else:
                    logger.error("Pipeline or query engine not available")
                    await self._send_response("I'm sorry, there's an issue with my knowledge base.")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing transcription: {transcription}")
            await self._send_response("I'm sorry, I'm taking longer than expected. Could you please repeat your question?")
            self._record_error()
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error processing your request.")
            self._record_error()
        finally:
            self.state.waiting_for_response = False
            self._save_session_state()
    
    async def _send_response(self, text: str, ws=None):
        """Enhanced response sending with WebRTC optimization and comprehensive error handling."""
        if not text.strip() or self.state.call_ended:
            return
        
        # Use stored WebSocket if not provided
        if ws is None:
            ws = getattr(self, '_ws', None)
            if ws is None:
                logger.error("No WebSocket available for sending response")
                return
        
        try:
            # Set response state
            self.state.is_speaking = True
            self.last_response_text = text
            
            logger.info(f"Sending enhanced response: '{text}'")
            
            # Convert to speech with WebRTC optimization
            try:
                # Use WebRTC-optimized synthesis
                audio_data = await self.tts_client.synthesize_with_optimization(
                    text, 
                    optimization_level=3  # High optimization for real-time
                )
                
                if audio_data:
                    # Send audio in optimized chunks
                    await self._send_audio_chunks_optimized(audio_data, ws)
                    self.state.responses_sent += 1
                    logger.info(f"Successfully sent enhanced response ({len(audio_data)} bytes)")
                else:
                    logger.error("No audio data generated from TTS")
            except Exception as e:
                logger.error(f"Error synthesizing speech: {e}")
                self._record_error()
                # Don't re-raise - just continue without audio response
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
            self._record_error()
        finally:
            # Clear speaking flag
            self.state.is_speaking = False
            
            # Adaptive delay based on response length
            delay = min(0.5 + len(text) / 200, 2.0)
            await asyncio.sleep(delay)
            
            # Ensure STT is still running for continuous conversation
            if (not self.stt_client.is_streaming and 
                self.state.conversation_active and 
                not self.state.call_ended):
                logger.info("Restarting enhanced STT for continuous conversation")
                try:
                    await self.stt_client.start_streaming()
                except Exception as e:
                    logger.error(f"Error restarting STT: {e}")
                    self._record_error()
            
            logger.debug("Ready for next utterance")
            self._save_session_state()
    
    async def _send_audio_chunks_optimized(self, audio_data: bytes, ws):
        """Send audio data with WebRTC-optimized chunking and adaptive pacing."""
        if not self.state.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        # Adaptive chunk size based on connection quality
        base_chunk_size = 320  # 40ms chunks for better real-time performance
        chunk_size = base_chunk_size
        
        # Adjust based on error rate
        if self.error_count > 0:
            chunk_size = max(160, base_chunk_size // 2)  # Smaller chunks if errors
        
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
        successful_chunks = 0
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.state.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                # Send with timeout
                await asyncio.wait_for(
                    asyncio.create_task(self._send_ws_message(ws, message)),
                    timeout=0.1
                )
                successful_chunks += 1
                
                # Adaptive delay based on chunk size and connection quality
                base_delay = chunk_size / 8000  # Match audio duration
                if self.error_count > 0:
                    base_delay *= 1.5  # Slower if errors
                
                await asyncio.sleep(base_delay * 0.8)  # Slightly faster than real-time
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending audio chunk {chunk_num}/{total_chunks}")
                self._record_error()
                # Continue with next chunk
            except Exception as e:
                logger.error(f"Error sending audio chunk {chunk_num}/{total_chunks}: {e}")
                self._record_error()
                break
        
        logger.debug(f"Sent {successful_chunks}/{total_chunks} audio chunks "
                    f"({successful_chunks/total_chunks*100:.1f}% success rate)")
    
    async def _send_ws_message(self, ws, message):
        """Send WebSocket message with error handling."""
        try:
            if hasattr(ws, 'send'):
                # For simple-websocket
                ws.send(json.dumps(message))
            elif hasattr(ws, 'send_text'):
                # For FastAPI WebSocket
                await ws.send_text(json.dumps(message))
            else:
                # For websockets library
                await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise
    
    async def start_conversation(self, ws):
        """Enhanced conversation startup with WebRTC optimization."""
        # Store WebSocket reference
        self._ws = ws
        self.state.stream_sid = getattr(self, 'stream_sid', None)
        
        # Start enhanced STT streaming
        if not self.stt_client.is_streaming:
            logger.info("Starting enhanced STT streaming for conversation")
            await self.stt_client.start_streaming()
        
        # Send welcome message with optimized timing
        await asyncio.sleep(0.2)  # Slight delay for connection stability
        await self._send_response("I'm ready to help. What would you like to know?", ws)
        
        # Save initial state
        self._save_session_state()
    
    async def _cleanup(self):
        """Enhanced cleanup with comprehensive session management."""
        try:
            self.state.call_ended = True
            self.state.conversation_active = False
            
            # Enhanced cleanup decision making
            session_duration = time.time() - self.state.start_time
            
            logger.info("Call ended - stopping enhanced STT session")
            if self.stt_client.is_streaming:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
            
            # Calculate comprehensive statistics
            duration = time.time() - self.state.start_time
            stats = self.get_stats()
            
            # Enhanced logging with conversation quality metrics
            logger.info(f"Enhanced session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio packets: {self.state.audio_received}, "
                       f"Valid transcriptions: {self.state.transcriptions}, "
                       f"Invalid/Echo: {self.state.invalid_transcriptions + self.state.echo_detections}, "
                       f"VAD activations: {self.state.vad_activations}, "
                       f"Responses: {self.state.responses_sent}, "
                       f"Echo detections: {self.state.echo_detections}, "
                       f"Error rate: {stats.get('error_rate', 0):.1f}%")
            
            # Final state save
            self._save_session_state()
            
            # Schedule state cleanup after delay
            asyncio.create_task(self._delayed_state_cleanup())
                       
        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {e}", exc_info=True)
    
    async def _delayed_state_cleanup(self):
        """Clean up session state after a delay to allow for debugging."""
        await asyncio.sleep(300)  # Keep state for 5 minutes
        self._clear_session_state()
        logger.debug(f"Cleared session state for {self.call_sid}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics with enhanced WebRTC metrics."""
        duration = time.time() - self.state.start_time
        
        # Calculate quality metrics
        transcription_rate = self.state.transcriptions / max(duration / 60, 1)  # per minute
        response_rate = self.state.responses_sent / max(duration / 60, 1)       # per minute
        error_rate = self.error_count / max(self.state.audio_received, 1) * 100  # percentage
        
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.state.stream_sid,
            "duration": round(duration, 2),
            "conversation_active": self.state.conversation_active,
            "call_ended": self.state.call_ended,
            
            # Audio processing stats
            "audio_received": self.state.audio_received,
            "transcriptions": self.state.transcriptions,
            "invalid_transcriptions": self.state.invalid_transcriptions,
            "responses_sent": self.state.responses_sent,
            
            # WebRTC-specific metrics
            "vad_activations": self.state.vad_activations,
            "echo_detections": self.state.echo_detections,
            "vad_efficiency": round(self.state.vad_activations / max(self.state.audio_received, 1) * 100, 2),
            
            # Quality metrics
            "transcription_rate": round(transcription_rate, 2),
            "response_rate": round(response_rate, 2),
            "error_rate": round(error_rate, 2),
            "error_count": self.error_count,
            "circuit_breaker_active": self._should_circuit_break(),
            
            # State information
            "is_speaking": self.state.is_speaking,
            "expecting_speech": self.state.expecting_speech,
            "waiting_for_response": self.state.waiting_for_response,
            "project_id": self.project_id,
            "redis_connected": self.redis_client is not None,
            
            # Timing information
            "session_start_time": self.state.start_time,
            "last_transcription_time": self.state.last_transcription_time,
            "last_audio_time": self.state.last_audio_time,
            "last_tts_time": self.state.last_tts_time,
            
            # Performance metrics
            "avg_response_time": round(
                (self.state.last_transcription_time - self.state.start_time) / max(self.state.transcriptions, 1), 
                2
            ),
        }
        
        # Add STT stats if available
        if hasattr(self.stt_client, 'get_stats'):
            stats["stt_stats"] = self.stt_client.get_stats()
        
        # Add TTS stats if available
        if hasattr(self.tts_client, 'get_stats'):
            stats["tts_stats"] = self.tts_client.get_stats()
        
        return stats