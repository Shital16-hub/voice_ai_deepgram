"""
Ultra Low Latency WebSocket handler for Voice AI - Fixed Version
Optimized for <2s latency on Runpod with proper async handling.
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
    """Simplified session state for ultra low latency."""
    call_sid: str
    stream_sid: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    conversation_active: bool = True
    is_speaking: bool = False
    call_ended: bool = False
    
    # Essential metrics only
    audio_received: int = 0
    transcriptions: int = 0
    responses_sent: int = 0
    last_transcription_time: float = field(default_factory=time.time)
    last_tts_time: Optional[float] = None

class SimpleWebSocketHandler:
    """
    Ultra Low Latency WebSocket handler optimized for <2s response time.
    Removed all non-essential features that cause delays.
    """
    
    # Optimized configuration for ultra low latency
    MIN_TRANSCRIPTION_LENGTH = 1  # Accept shorter transcriptions
    RESPONSE_TIMEOUT = 1.5  # Reduced from 8s to 1.5s
    SILENCE_TIMEOUT = 3.0   # Reduced from 10s to 3s
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with minimal overhead for maximum speed."""
        self.call_sid = call_sid
        self.pipeline = pipeline
        
        # Get project ID with minimal processing
        self.project_id = self._get_project_id()
        
        # Initialize Google Cloud STT v2 with ultra low latency settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,  # Only final results for speed
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file,
            enable_vad=True,  # Keep VAD enabled for better detection
            enable_echo_suppression=False  # DISABLED for ultra low latency
        )
        
        # Initialize Google Cloud TTS with minimal latency settings
        self.tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,  # Keep caching for repeated responses
            voice_type="NEURAL2"
            # No echo suppression callback for minimal latency
        )
        
        # Simplified state management
        self.state = SessionState(call_sid=call_sid)
        
        # WebSocket reference for response sending
        self._ws = None
        
        # Ultra simple response tracking
        self.last_response_text = ""
        
        # Minimal error tracking
        self.error_count = 0
        
        logger.info(f"Ultra Low Latency WebSocket handler initialized - Call: {call_sid}")
    
    def _get_project_id(self) -> str:
        """Get project ID with minimal overhead."""
        # Try environment variable first (fastest)
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Fallback to known project ID
        return "my-tts-project-458404"
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Ultra fast audio handling with minimal processing."""
        # Skip if call ended
        if self.state.call_ended:
            return
        
        # Skip if speaking (simple echo prevention)
        if self.state.is_speaking:
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Minimal audio processing
        try:
            audio_data = base64.b64decode(payload)
            self.state.audio_received += 1
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Start STT if not already started
        if not self.stt_client.is_streaming:
            logger.info("Starting ultra low latency STT streaming")
            await self.stt_client.start_streaming()
        
        # Process audio with minimal error handling
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.error_count += 1
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Ultra fast transcription handling."""
        if result.is_final and result.text.strip():
            transcription = result.text.strip()
            confidence = result.confidence
            
            logger.info(f"Transcription: '{transcription}' (confidence: {confidence:.2f})")
            
            self.state.last_transcription_time = time.time()
            
            # Minimal validation for ultra low latency
            if self._is_valid_transcription_fast(transcription, confidence):
                await self._process_final_transcription(transcription)
            else:
                logger.debug(f"Invalid transcription: '{transcription}'")
    
    def _is_valid_transcription_fast(self, transcription: str, confidence: float) -> bool:
        """Ultra fast validation with minimal checks."""
        # Basic length check
        if len(transcription.split()) < self.MIN_TRANSCRIPTION_LENGTH:
            return False
        
        # Reduced confidence threshold for speed
        if confidence < 0.2:
            return False
        
        # Minimal echo check - just timing
        if (self.state.last_tts_time and 
            (time.time() - self.state.last_tts_time) < 0.8):  # Reduced from 2.0s
            # Very simple echo check - if response contains exactly same text
            if transcription.lower() == self.last_response_text.lower():
                return False
        
        return True
    
    async def _process_final_transcription(self, transcription: str):
        """Ultra fast transcription processing with timeout fix."""
        self.state.transcriptions += 1
        self.state.is_speaking = True  # Prevent echo during processing
        
        logger.info(f"Processing: '{transcription}'")
        
        try:
            # FIXED: Use asyncio.wait_for instead of asyncio.timeout (Python 3.10 compatible)
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
            self.state.is_speaking = False
    
    async def _get_knowledge_response(self, transcription: str) -> Optional[str]:
        """Get response from knowledge base with minimal overhead."""
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
        """Ultra fast response sending with minimal processing."""
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
            
            logger.info(f"Sending response: '{text}'")
            
            # Ultra fast TTS synthesis
            audio_data = await self.tts_client.synthesize(text)
            
            if audio_data:
                # Send audio in large chunks for minimal latency
                await self._send_audio_fast(audio_data, ws)
                self.state.responses_sent += 1
                self.state.last_tts_time = time.time()
                logger.info(f"Sent response ({len(audio_data)} bytes)")
            else:
                logger.error("No audio generated")
                
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.error_count += 1
        finally:
            # Minimal delay
            await asyncio.sleep(0.1)
            
            # Ensure STT continues
            if not self.stt_client.is_streaming and self.state.conversation_active:
                try:
                    await self.stt_client.start_streaming()
                except Exception as e:
                    logger.error(f"Error restarting STT: {e}")
    
    async def _send_audio_fast(self, audio_data: bytes, ws):
        """Send audio with maximum speed - larger chunks, minimal delays."""
        if not self.state.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        # Use larger chunks for minimal latency
        chunk_size = 1600  # 200ms chunks (larger than before)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.state.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                # Send immediately with no timeout
                await self._send_ws_message_fast(ws, message)
                
                # Minimal delay - process at 80% of realtime for stability
                delay = chunk_size / 8000 * 0.8
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break
    
    async def _send_ws_message_fast(self, ws, message):
        """Send WebSocket message as fast as possible."""
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
        """Ultra fast conversation startup."""
        # Store WebSocket reference
        self._ws = ws
        self.state.stream_sid = getattr(self, 'stream_sid', None)
        
        # Start STT immediately
        if not self.stt_client.is_streaming:
            logger.info("Starting STT for conversation")
            await self.stt_client.start_streaming()
        
        # Send welcome message with minimal delay
        await asyncio.sleep(0.1)
        await self._send_response("Hello! How can I help you?", ws)
    
    async def _cleanup(self):
        """Fast cleanup with minimal processing."""
        try:
            self.state.call_ended = True
            self.state.conversation_active = False
            
            logger.info("Stopping STT session")
            if self.stt_client.is_streaming:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
            
            # Calculate stats
            duration = time.time() - self.state.start_time
            
            logger.info(f"Session cleanup - Duration: {duration:.2f}s, "
                       f"Audio packets: {self.state.audio_received}, "
                       f"Transcriptions: {self.state.transcriptions}, "
                       f"Responses: {self.state.responses_sent}, "
                       f"Errors: {self.error_count}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple stats for monitoring."""
        duration = time.time() - self.state.start_time
        
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.state.stream_sid,
            "duration": round(duration, 2),
            "conversation_active": self.state.conversation_active,
            "call_ended": self.state.call_ended,
            
            # Core metrics
            "audio_received": self.state.audio_received,
            "transcriptions": self.state.transcriptions,
            "responses_sent": self.state.responses_sent,
            "error_count": self.error_count,
            
            # Performance metrics
            "avg_response_time": round(
                (self.state.last_transcription_time - self.state.start_time) / max(self.state.transcriptions, 1), 
                2
            ),
            
            # States
            "is_speaking": self.state.is_speaking,
            "project_id": self.project_id,
            "session_start": self.state.start_time,
        }