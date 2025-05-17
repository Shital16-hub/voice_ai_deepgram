# telephony/simple_websocket_handler.py - CRITICAL FIXES

"""
CRITICAL FIXES for WebSocket handler to resolve speech detection and response issues.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Use the enhanced STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

# Use the enhanced TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Enhanced session state with debugging info."""
    call_sid: str
    stream_sid: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    conversation_active: bool = True
    is_speaking: bool = False
    call_ended: bool = False
    
    # Enhanced metrics with debugging
    audio_received: int = 0
    transcriptions: int = 0
    responses_sent: int = 0
    last_transcription_time: float = field(default_factory=time.time)
    last_tts_time: Optional[float] = None
    
    # Debug metrics
    audio_bytes_total: int = 0
    speech_detected: int = 0
    speech_detected_but_invalid: int = 0
    interim_results_received: int = 0

class SimpleWebSocketHandler:
    """
    FIXED WebSocket handler with critical improvements for speech detection and response.
    """
    
    # CRITICAL FIX: More lenient configuration
    MIN_TRANSCRIPTION_LENGTH = 1  # Accept even single words
    RESPONSE_TIMEOUT = 3.0        # Reduced timeout for faster responses (from 4.0)
    SILENCE_TIMEOUT = 4.0         # CRITICAL: Reduced for faster detection (from 5.0)
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with CRITICAL FIXES for speech detection."""
        self.call_sid = call_sid
        self.pipeline = pipeline
        
        # Get project ID
        self.project_id = self._get_project_id()
        
        # CRITICAL FIX: Initialize Google Cloud STT v2 with FIXED settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,  # CRITICAL: Enable for better debugging
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file,
            enable_vad=True,      # Keep VAD but with proper configuration
            enable_echo_suppression=False
        )
        
        # Initialize Google Cloud TTS
        self.tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        # Enhanced state management
        self.state = SessionState(call_sid=call_sid)
        
        # WebSocket reference for response sending
        self._ws = None
        
        # CRITICAL FIX: Better response tracking
        self.last_response_text = ""
        self.response_in_progress = False
        
        # Error tracking
        self.error_count = 0
        
        # CRITICAL FIX: Enable debug flags
        self.enable_audio_debug = True
        self.stt_debug = True
        
        # CRITICAL FIX: Track conversation state for second call issue
        self.conversation_started = False
        self.first_response_sent = False
        
        # CRITICAL FIX: Health check timer - reduced interval for more frequent checks
        self._last_health_check = time.time()
        self._health_check_interval = 10  # Check health every 10 seconds (reduced from 15)
        
        # CRITICAL NEW: Add buffer for handling partial speech
        self._speech_buffer = ""
        self._last_speech_time = 0
        
        logger.info(f"FIXED WebSocket handler initialized - Call: {call_sid}")
        logger.info(f"CRITICAL FIXES: Interim results enabled, improved VAD, better timeouts")
    
    def _get_project_id(self) -> str:
        """Get project ID with minimal overhead."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Fallback to known project ID
        return "my-tts-project-458404"
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """CRITICAL FIX: Enhanced audio handling with proper echo prevention."""
        # Skip if call ended
        if self.state.call_ended:
            return
        
        # CRITICAL FIX: Add stream renewal based on Google's examples
        now = time.time()
        if (self._last_health_check + self._health_check_interval < now):
            self._last_health_check = now
            await self._ensure_streaming_health()
        
        # CRITICAL FIX: Always ensure STT is running before processing audio
        if not self.stt_client.is_streaming:
            logger.info("STT not streaming - restarting STT session")
            await self.stt_client.start_streaming()
        
        # CRITICAL FIX: More sophisticated echo prevention with safe timeout
        if self.response_in_progress:
            # Allow audio processing during response generation but only wait up to 0.3 seconds
            if (self.state.last_tts_time and 
                (time.time() - self.state.last_tts_time) > 0.2):  # REDUCED from 0.3 to 0.2 for faster responses
                self.response_in_progress = False
                self.state.is_speaking = False
                logger.debug("Cleared response_in_progress flag after echo delay timeout")
            else:
                if self.enable_audio_debug and self.state.audio_received % 100 == 0:
                    logger.debug(f"Skipping audio - Response in progress (last TTS: {time.time() - self.state.last_tts_time:.2f}s ago)")
                return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Enhanced audio processing with debugging
        try:
            audio_data = base64.b64decode(payload)
            self.state.audio_received += 1
            self.state.audio_bytes_total += len(audio_data)
            
            # CRITICAL FIX: Debug audio reception
            if self.enable_audio_debug and self.state.audio_received % 50 == 0:
                logger.info(f"AUDIO DEBUG - Packets: {self.state.audio_received}, "
                          f"Total bytes: {self.state.audio_bytes_total}, "
                          f"Last packet size: {len(audio_data)} bytes")
                
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Process audio with enhanced error handling
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result_enhanced
            )
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.error_count += 1
    
    async def _ensure_streaming_health(self):
        """CRITICAL IMPROVED: Ensure streaming session is healthy with better session management."""
        if not self.stt_client:
            return
            
        # Check if we need to restart streaming
        if not self.stt_client.is_streaming:
            logger.info("STT not streaming, restarting")
            await self.stt_client.start_streaming()
            return
            
        # IMPROVED: More aggressive session rotation to prevent timeouts
        if hasattr(self.stt_client, '_last_streaming_start') and self.stt_client._last_streaming_start:
            streaming_duration = time.time() - self.stt_client._last_streaming_start
            # Rotate session every 2 minutes instead of 4
            if streaming_duration > 120:  # 2 minutes (reduced from 240s)
                logger.info(f"Streaming session active for {streaming_duration}s, preemptively restarting")
                await self.stt_client.stop_streaming()
                await asyncio.sleep(0.1)  # Brief pause to ensure clean restart
                await self.stt_client.start_streaming()
    
    async def _handle_transcription_result_enhanced(self, result: StreamingTranscriptionResult):
        """CRITICAL FIX: Enhanced transcription handling with better validation."""
        if result is None:
            return
            
        # Handle interim results for debugging
        if not result.is_final and result.text.strip():
            self.state.interim_results_received += 1
            self.state.speech_detected += 1
            if self.stt_debug:
                logger.info(f"INTERIM #{self.state.interim_results_received}: '{result.text.strip()}' "
                           f"(confidence: {result.confidence:.2f})")
            
            # NEW: Update speech buffer for partial results
            self._speech_buffer = result.text.strip()
            self._last_speech_time = time.time()
        
        # Process final results
        if result.is_final and result.text.strip():
            transcription = result.text.strip()
            confidence = result.confidence
            
            logger.info(f"FINAL TRANSCRIPTION: '{transcription}' (confidence: {confidence:.2f})")
            
            self.state.last_transcription_time = time.time()
            
            # CRITICAL FIX: Clear any response_in_progress flag to allow new speech processing
            # This ensures we can process a new query immediately
            self.response_in_progress = False
            
            # CRITICAL FIX: IMPROVED validation logic
            if self._is_valid_transcription_enhanced(transcription, confidence):
                await self._process_final_transcription(transcription)
            else:
                self.state.speech_detected_but_invalid += 1
                logger.debug(f"REJECTED transcription: '{transcription}' (confidence: {confidence:.2f})")
    
    def _is_valid_transcription_enhanced(self, transcription: str, confidence: float) -> bool:
        """IMPROVED: Even more lenient validation with better echo prevention."""
        # Basic length check - very lenient
        word_count = len(transcription.split())
        if word_count < self.MIN_TRANSCRIPTION_LENGTH:
            logger.debug(f"REJECT: Too short ({word_count} words)")
            return False
        
        # IMPROVED: Accept ANY confidence score
        if confidence < -0.5:  # Only reject extremely negative confidence
            logger.debug(f"REJECT: Invalid confidence ({confidence:.2f})")
            return False
        
        # IMPROVED: Better echo prevention
        if (self.state.last_tts_time and 
            (time.time() - self.state.last_tts_time) < 0.5):  # Extended from 0.3s to 0.5s
            # Only reject exact or very similar responses
            last_words = set(self.last_response_text.lower().strip().split())
            current_words = set(transcription.lower().strip().split())
            if len(last_words) > 0 and len(current_words) > 0:
                intersection = len(last_words.intersection(current_words))
                similarity = intersection / max(len(last_words), 1)
                if similarity > 0.7:
                    logger.debug(f"REJECT: Echo detected (word similarity: {similarity:.2f})")
                    return False
        
        # Skip only obvious noise patterns
        noise_patterns = ['hmm', 'uh', 'um', 'ah']
        if transcription.lower().strip() in noise_patterns:
            logger.debug(f"REJECT: Noise pattern")
            return False
        
        # IMPROVED: Accept everything else
        logger.info(f"ACCEPT: '{transcription}' (confidence: {confidence:.2f})")
        return True
    
    async def _process_final_transcription(self, transcription: str):
        """CRITICAL FIX: Enhanced transcription processing with better error handling."""
        self.state.transcriptions += 1
        # CRITICAL FIX: Set flag at beginning of processing
        self.response_in_progress = True
        
        logger.info(f"PROCESSING: '{transcription}'")
        
        try:
            # CRITICAL FIX: Use OpenAI + Pinecone with reduced timeout
            response_task = self._get_knowledge_response(transcription)
            response = await asyncio.wait_for(response_task, timeout=self.RESPONSE_TIMEOUT)
            
            if response:
                await self._send_response(response)
            else:
                await self._send_response("I couldn't find an answer to that. Could you please try again?")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing: {transcription}")
            await self._send_response("I'm processing that. Could you repeat your question?")
            self.error_count += 1
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            await self._send_response("I encountered an error. Please try again.")
            self.error_count += 1
        finally:
            # CRITICAL FIX: Small delay before clearing flag to avoid echo
            await asyncio.sleep(0.1)  # Reduced from 0.2 for faster responsiveness
            self.response_in_progress = False
            logger.debug("Cleared response_in_progress flag after processing")
    
    async def _get_knowledge_response(self, transcription: str) -> Optional[str]:
        """CRITICAL FIX: Get response from knowledge base with better error handling."""
        try:
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(transcription)
                response_text = result.get("response", "")
                
                # CRITICAL FIX: Validate response
                if response_text and response_text.strip():
                    return response_text.strip()
                else:
                    logger.warning("Empty response from knowledge base")
                    return None
            else:
                logger.error("No query engine available")
                return "I'm sorry, my knowledge base is not available."
        except Exception as e:
            logger.error(f"Knowledge base error: {e}")
            return None
    
    async def _send_response(self, text: str, ws=None):
        """CRITICAL FIX: Enhanced response sending with better timing."""
        if not text.strip() or self.state.call_ended:
            return
        
        # Use stored WebSocket if not provided
        if ws is None:
            ws = getattr(self, '_ws', None)
            if ws is None:
                logger.error("No WebSocket available")
                return
        
        try:
            self.last_response_text = text
            
            logger.info(f"SENDING RESPONSE: '{text}'")
            
            # TTS synthesis with timeout
            audio_data = await asyncio.wait_for(
                self.tts_client.synthesize(text),
                timeout=2.0  # Reduced timeout from 3.0 to 2.0
            )
            
            if audio_data:
                # CRITICAL FIX: Mark when we start sending audio
                self.state.last_tts_time = time.time()
                self.response_in_progress = True
                
                # Send audio
                await self._send_audio_fast(audio_data, ws)
                self.state.responses_sent += 1
                
                # CRITICAL FIX: Mark first response sent
                if not self.first_response_sent:
                    self.first_response_sent = True
                
                logger.info(f"SENT response ({len(audio_data)} bytes)")
                
                # CRITICAL NEW FIX: Force STT restart after sending response
                try:
                    logger.info("Forcing STT restart after response")
                    if self.stt_client.is_streaming:
                        await self.stt_client.stop_streaming()
                    await asyncio.sleep(0.1)  # Reduced delay from 0.2 to 0.1
                    await self.stt_client.start_streaming()
                    logger.info("STT restarted successfully")
                except Exception as e:
                    logger.error(f"Error restarting STT: {e}")
                
            else:
                logger.error("No audio generated")
                
        except asyncio.TimeoutError:
            logger.error("TTS synthesis timed out")
            self.error_count += 1
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.error_count += 1
        finally:
            # CRITICAL FIX: Ensure STT continues after response - shorter delay
            await asyncio.sleep(0.1)  # Reduced from 0.3 for faster response
            
            # CRITICAL FIX: Clear the response flag
            self.response_in_progress = False
            logger.info("Response flag cleared")
    
    async def _send_audio_fast(self, audio_data: bytes, ws):
        """OPTIMIZED: Send audio with even faster chunking."""
        if not self.state.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        # Optimized chunk size - reduced from 800 to 400 for lower latency
        chunk_size = 400  # 50ms chunks at 8kHz MULAW (reduced from 100ms)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.state.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                # Send immediately
                await self._send_ws_message_fast(ws, message)
                
                # OPTIMIZED: Extremely fast delay 
                delay = chunk_size / 8000 * 0.75  # 75% of real-time for faster playback (reduced from 90%)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break
    
    async def _send_ws_message_fast(self, ws, message):
        """Send WebSocket message with error handling."""
        try:
            if hasattr(ws, 'send_text'):
                # FastAPI WebSocket
                await ws.send_text(json.dumps(message))
            else:
                # Other WebSocket implementations
                await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise
    
    async def start_conversation(self, ws):
        """CRITICAL FIX: Enhanced conversation startup."""
        # Store WebSocket reference
        self._ws = ws
        self.state.stream_sid = getattr(self, 'stream_sid', None)
        
        # CRITICAL FIX: Mark conversation as started
        self.conversation_started = True
        
        # Start STT immediately
        if not self.stt_client.is_streaming:
            logger.info("Starting STT for conversation with FIXED speech detection")
            await self.stt_client.start_streaming()
        
        # Send welcome message with proper delay
        await asyncio.sleep(0.2)  # Reduced from 0.5 to improve initial response time
        await self._send_response("Hello! How can I help you?", ws)
    
    async def _cleanup(self):
        """CRITICAL FIX: Enhanced cleanup with better statistics."""
        try:
            self.state.call_ended = True
            self.state.conversation_active = False
            
            logger.info("Stopping STT session")
            if self.stt_client.is_streaming:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
            
            # Calculate enhanced stats
            duration = time.time() - self.state.start_time
            
            logger.info(f"FINAL SESSION STATS - Duration: {duration:.2f}s")
            logger.info(f"Audio packets: {self.state.audio_received}, "
                       f"Total bytes: {self.state.audio_bytes_total}")
            logger.info(f"Interim results: {self.state.interim_results_received}")
            logger.info(f"Speech detected: {self.state.speech_detected}, "
                       f"Valid transcriptions: {self.state.transcriptions}, "
                       f"Invalid: {self.state.speech_detected_but_invalid}")
            logger.info(f"Responses: {self.state.responses_sent}, "
                       f"Errors: {self.error_count}")
            logger.info(f"First response sent: {self.first_response_sent}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """CRITICAL FIX: Get enhanced stats with more debugging information."""
        duration = time.time() - self.state.start_time
        
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.state.stream_sid,
            "duration": round(duration, 2),
            "conversation_active": self.state.conversation_active,
            "call_ended": self.state.call_ended,
            
            # Enhanced metrics
            "audio_received": self.state.audio_received,
            "audio_bytes_total": self.state.audio_bytes_total,
            "interim_results_received": self.state.interim_results_received,
            "speech_detected": self.state.speech_detected,
            "transcriptions": self.state.transcriptions,
            "speech_detected_but_invalid": self.state.speech_detected_but_invalid,
            "responses_sent": self.state.responses_sent,
            "error_count": self.error_count,
            
            # CRITICAL: Conversation state tracking
            "conversation_started": self.conversation_started,
            "first_response_sent": self.first_response_sent,
            "response_in_progress": self.response_in_progress,
            
            # Performance metrics
            "detection_rate": round(
                (self.state.speech_detected / max(self.state.audio_received, 1)) * 100, 2
            ),
            "transcription_rate": round(
                (self.state.transcriptions / max(self.state.speech_detected, 1)) * 100, 2
            ),
            
            # States
            "is_speaking": self.state.is_speaking,
            "project_id": self.project_id,
            "session_start": self.state.start_time,
            
            # Enhanced features
            "interim_results_enabled": True,
            "critical_fixes_applied": True,
            "debug_enabled": self.stt_debug and self.enable_audio_debug,
            
            # STT client stats
            "stt_stats": self.stt_client.get_stats() if self.stt_client else {},
            
            # CRITICAL NEW: Health check status
            "last_health_check": self._last_health_check,
            "health_check_interval": self._health_check_interval,
            "next_health_check_in": max(0, self._last_health_check + self._health_check_interval - time.time()),
            
            # Timing stats for last activity
            "time_since_last_transcription": round(time.time() - self.state.last_transcription_time, 2),
            "time_since_last_tts": round(time.time() - (self.state.last_tts_time or self.state.start_time), 2)
        }