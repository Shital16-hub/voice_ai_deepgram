# telephony/simple_websocket_handler.py - Enhanced debugging

"""
Enhanced WebSocket handler with improved speech detection and debugging.
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
    Enhanced WebSocket handler with improved speech detection and debugging.
    """
    
    # FIXED: More lenient configuration for speech detection
    MIN_TRANSCRIPTION_LENGTH = 1  # Accept even single words
    RESPONSE_TIMEOUT = 3.0  # Increased timeout
    SILENCE_TIMEOUT = 5.0   # Increased silence timeout
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with enhanced debugging and speech detection."""
        self.call_sid = call_sid
        self.pipeline = pipeline
        
        # Get project ID with minimal processing
        self.project_id = self._get_project_id()
        
        # FIXED: Initialize Google Cloud STT v2 with fixed settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,  # FIXED: Enable interim results for debugging
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file,
            enable_vad=True,  # FIXED: Keep VAD enabled but properly configured
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
        
        # Response tracking
        self.last_response_text = ""
        
        # Error tracking
        self.error_count = 0
        
        # FIXED: Enable debug flags by default for troubleshooting
        self.enable_audio_debug = True
        self.stt_debug = True
        
        logger.info(f"FIXED WebSocket handler initialized - Call: {call_sid}")
        logger.info(f"FIXED: Interim results enabled, improved VAD configuration")
    
    def _get_project_id(self) -> str:
        """Get project ID with minimal overhead."""
        # Try environment variable first (fastest)
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Fallback to known project ID
        return "my-tts-project-458404"
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """FIXED: Enhanced audio handling with detailed debugging."""
        # Skip if call ended
        if self.state.call_ended:
            return
        
        # FIXED: More lenient echo prevention
        if self.state.is_speaking:
            # Allow audio processing sooner after TTS (even more lenient)
            if (self.state.last_tts_time and 
                (time.time() - self.state.last_tts_time) > 0.5):  # Reduced from 1.0s
                self.state.is_speaking = False
            else:
                if self.enable_audio_debug:
                    logger.debug(f"Skipping audio - TTS active (last TTS: {time.time() - self.state.last_tts_time:.2f}s ago)")
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
            
            # FIXED: More frequent audio debugging
            if self.enable_audio_debug and self.state.audio_received % 50 == 0:  # Every 50 packets instead of 100
                logger.info(f"AUDIO DEBUG - Packets: {self.state.audio_received}, "
                          f"Total bytes: {self.state.audio_bytes_total}, "
                          f"Last packet size: {len(audio_data)} bytes")
                
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Start STT if not already started
        if not self.stt_client.is_streaming:
            logger.info("Starting FIXED STT streaming with enhanced speech detection")
            await self.stt_client.start_streaming()
        
        # Process audio with enhanced error handling
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result_enhanced
            )
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.error_count += 1
    
    async def _handle_transcription_result_enhanced(self, result: StreamingTranscriptionResult):
        """FIXED: Enhanced transcription handling with detailed logging."""
        if result is None:
            return
            
        # Handle interim results for debugging
        if not result.is_final and result.text.strip():
            self.state.interim_results_received += 1
            self.state.speech_detected += 1
            if self.stt_debug:
                logger.info(f"INTERIM #{self.state.interim_results_received}: '{result.text.strip()}' "
                           f"(confidence: {result.confidence:.2f})")
        
        # Process final results
        if result.is_final and result.text.strip():
            transcription = result.text.strip()
            confidence = result.confidence
            
            logger.info(f"FINAL TRANSCRIPTION: '{transcription}' (confidence: {confidence:.2f})")
            
            self.state.last_transcription_time = time.time()
            
            # FIXED: More lenient validation
            if self._is_valid_transcription_enhanced(transcription, confidence):
                await self._process_final_transcription(transcription)
            else:
                self.state.speech_detected_but_invalid += 1
                logger.debug(f"REJECTED transcription: '{transcription}' (confidence: {confidence:.2f})")
    
    def _is_valid_transcription_enhanced(self, transcription: str, confidence: float) -> bool:
        """FIXED: Much more lenient validation with detailed logging."""
        # Basic length check - very lenient
        word_count = len(transcription.split())
        if word_count < self.MIN_TRANSCRIPTION_LENGTH:
            logger.debug(f"REJECT: Too short ({word_count} words)")
            return False
        
        # FIXED: Much more lenient confidence threshold
        if confidence < 0.0:  # Only reject clearly invalid results
            logger.debug(f"REJECT: Low confidence ({confidence:.2f})")
            return False
        
        # FIXED: More lenient echo check (reduced timeout)
        if (self.state.last_tts_time and 
            (time.time() - self.state.last_tts_time) < 0.3):  # Reduced from 0.5s
            # Only reject if it's exactly the same
            if transcription.lower() == self.last_response_text.lower():
                logger.debug(f"REJECT: Echo detected")
                return False
        
        # Skip common noise patterns but be more lenient
        noise_patterns = ['hmm', 'uh', 'um', 'ah', 'mm']
        if transcription.lower().strip() in noise_patterns:
            logger.debug(f"REJECT: Noise pattern")
            return False
        
        logger.info(f"ACCEPT: '{transcription}' (confidence: {confidence:.2f})")
        return True
    
    async def _process_final_transcription(self, transcription: str):
        """Enhanced transcription processing with better timeout handling."""
        self.state.transcriptions += 1
        self.state.is_speaking = True  # Prevent echo during processing
        
        logger.info(f"PROCESSING: '{transcription}'")
        
        try:
            # Use asyncio.wait_for with increased timeout
            response_task = self._get_knowledge_response(transcription)
            response = await asyncio.wait_for(response_task, timeout=self.RESPONSE_TIMEOUT)
            
            if response:
                await self._send_response(response)
            else:
                await self._send_response("I couldn't find an answer to that.")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing: {transcription}")
            await self._send_response("I'm processing that. Could you repeat your question?")
            self.error_count += 1
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            await self._send_response("I encountered an error. Please try again.")
            self.error_count += 1
        finally:
            # FIXED: Shorter delay before allowing new speech
            await asyncio.sleep(0.3)  # Reduced from 0.5s
            self.state.is_speaking = False
    
    async def _get_knowledge_response(self, transcription: str) -> Optional[str]:
        """Get response from knowledge base with enhanced error handling."""
        try:
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(transcription)
                return result.get("response", "")
            else:
                return "I'm sorry, my knowledge base is not available."
        except Exception as e:
            logger.error(f"Knowledge base error: {e}")
            return None
    
    async def _send_response(self, text: str, ws=None):
        """Enhanced response sending with better timing."""
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
            
            # TTS synthesis
            audio_data = await self.tts_client.synthesize(text)
            
            if audio_data:
                # Send audio
                await self._send_audio_fast(audio_data, ws)
                self.state.responses_sent += 1
                self.state.last_tts_time = time.time()
                logger.info(f"SENT response ({len(audio_data)} bytes)")
            else:
                logger.error("No audio generated")
                
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.error_count += 1
        finally:
            # Minimal delay before allowing new speech
            await asyncio.sleep(0.3)  # Reduced delay
            
            # Ensure STT continues with better error handling
            if not self.stt_client.is_streaming and self.state.conversation_active:
                try:
                    logger.info("Restarting STT after response")
                    await self.stt_client.start_streaming()
                except Exception as e:
                    logger.error(f"Error restarting STT: {e}")
    
    async def _send_audio_fast(self, audio_data: bytes, ws):
        """Send audio with optimized chunking."""
        if not self.state.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        # Optimized chunk size for Twilio
        chunk_size = 1600  # 200ms chunks
        
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
                
                # Optimized delay
                delay = chunk_size / 8000 * 0.8
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
        """Enhanced conversation startup with better speech detection."""
        # Store WebSocket reference
        self._ws = ws
        self.state.stream_sid = getattr(self, 'stream_sid', None)
        
        # Start STT immediately with enhanced settings
        if not self.stt_client.is_streaming:
            logger.info("Starting STT for conversation with FIXED speech detection")
            await self.stt_client.start_streaming()
        
        # Send welcome message with slight delay for better audio setup
        await asyncio.sleep(0.2)
        await self._send_response("Hello! How can I help you?", ws)
    
    async def _cleanup(self):
        """Enhanced cleanup with detailed statistics."""
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
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced stats with debugging information."""
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
            "fixed_configuration": True,
            "debug_enabled": self.stt_debug and self.enable_audio_debug
        }