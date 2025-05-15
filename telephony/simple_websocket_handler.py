"""
Corrected WebSocket handler for continuous conversation.
Ensures proper stream restart and audio processing.
"""
import json
import asyncio
import logging
import base64
import time
import os
import threading
from typing import Dict, Any, Optional

# Use the STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    Corrected WebSocket handler with proper continuous conversation support.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with proper configuration."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically
        self.project_id = self._get_project_id()
        
        # Initialize STT and TTS
        self._init_clients()
        
        # State management
        self.conversation_active = True
        self.is_speaking = False
        self.stt_active = False
        
        # Audio processing
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        
        # Timing
        self.last_audio_time = time.time()
        self.last_transcription_time = time.time()
        
        # Stats
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        self.start_time = time.time()
        
        # Restart timer
        self.restart_timer = None
        
        logger.info(f"WebSocket handler initialized for call {call_sid}")
    
    def _get_project_id(self) -> str:
        """Get project ID from environment or credentials file."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
        
        return "my-tts-project-458404"
    
    def _init_clients(self):
        """Initialize STT and TTS clients."""
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Initialize STT
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file
        )
        
        # Initialize TTS
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
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], ws):
        """Handle different WebSocket events."""
        try:
            if event_type == 'connected':
                logger.info("WebSocket connected")
                
            elif event_type == 'start':
                self.stream_sid = data.get('streamSid')
                logger.info(f"Stream started: {self.stream_sid}")
                
                # Start STT and send greeting
                await self._start_stt_stream()
                await self._send_response("Hello! How can I help you today?", ws)
                
            elif event_type == 'media':
                await self._handle_audio(data, ws)
                
            elif event_type == 'stop':
                logger.info("Stream stopped")
                await self._cleanup()
                
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}", exc_info=True)
    
    async def _start_stt_stream(self):
        """Start STT streaming with automatic restart capability."""
        try:
            if self.stt_active:
                await self.stt_client.stop_streaming()
                await asyncio.sleep(0.1)
            
            await self.stt_client.start_streaming()
            self.stt_active = True
            logger.info("STT stream started")
            
            # Schedule automatic restart in 45 seconds (before timeout)
            if self.restart_timer:
                self.restart_timer.cancel()
            
            loop = asyncio.get_event_loop()
            self.restart_timer = loop.call_later(45.0, self._schedule_stt_restart)
            
        except Exception as e:
            logger.error(f"Error starting STT stream: {e}")
            self.stt_active = False
    
    def _schedule_stt_restart(self):
        """Schedule STT stream restart in the main event loop."""
        if self.conversation_active and not self.is_speaking:
            logger.info("Scheduling STT stream restart")
            asyncio.create_task(self._start_stt_stream())
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Handle incoming audio data."""
        # Skip audio while speaking
        if self.is_speaking:
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio
        try:
            audio_data = base64.b64decode(payload)
            self.audio_received += 1
            self.last_audio_time = time.time()
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Ensure STT is active
        if not self.stt_active:
            await self._start_stt_stream()
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process chunks
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = bytes(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            try:
                result = await self.stt_client.process_audio_chunk(
                    chunk, 
                    callback=self._handle_transcription_result
                )
                
                if result and result.is_final and result.text.strip():
                    await self._process_transcription(result.text, ws)
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                # Restart STT on error
                await self._start_stt_stream()
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle transcription results."""
        if result.is_final and result.text.strip():
            logger.info(f"Final transcription: '{result.text}' (conf: {result.confidence:.2f})")
        elif result.text.strip():
            logger.debug(f"Interim: '{result.text}'")
    
    async def _process_transcription(self, transcription: str, ws):
        """Process transcription and generate response."""
        # Skip short or filler words
        if (not transcription or 
            len(transcription.strip()) < 3 or
            transcription.lower().strip() in ['um', 'uh', 'mmm', 'hmm', 'oh']):
            return
        
        self.transcriptions += 1
        self.last_transcription_time = time.time()
        logger.info(f"Processing transcription #{self.transcriptions}: '{transcription}'")
        
        try:
            # Stop STT while processing/speaking
            if self.stt_active:
                await self.stt_client.stop_streaming()
                self.stt_active = False
                if self.restart_timer:
                    self.restart_timer.cancel()
            
            # Query knowledge base
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(transcription)
                response_text = result.get("response", "")
                
                if response_text:
                    await self._send_response(response_text, ws)
                else:
                    await self._send_response("I'm sorry, I couldn't find an answer to that question.", ws)
            else:
                await self._send_response("I'm sorry, there's an issue with my knowledge base.", ws)
                
            # Restart STT after speaking is done
            await asyncio.sleep(0.5)  # Give TTS time to complete
            await self._start_stt_stream()
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error.", ws)
            await self._start_stt_stream()
    
    async def _send_response(self, text: str, ws):
        """Send TTS response."""
        if not text.strip():
            return
        
        try:
            self.is_speaking = True
            logger.info(f"Sending response: '{text[:50]}...'")
            
            # Convert to speech
            audio_data = await self.tts_client.synthesize(text)
            
            if audio_data:
                await self._send_audio_chunks(audio_data, ws)
                self.responses_sent += 1
                logger.info(f"Response sent #{self.responses_sent} ({len(audio_data)} bytes)")
            else:
                logger.error("No audio data generated")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            self.is_speaking = False
    
    async def _send_audio_chunks(self, audio_data: bytes, ws):
        """Send audio in chunks to Twilio."""
        if not self.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        chunk_size = 400  # 50ms chunks
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                ws.send(json.dumps(message))
                await asyncio.sleep(0.02)  # 50ms chunks
                
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            self.conversation_active = False
            
            # Cancel restart timer
            if self.restart_timer:
                self.restart_timer.cancel()
            
            # Stop STT
            if self.stt_active:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
                self.stt_active = False
            
            duration = time.time() - self.start_time
            logger.info(f"Session cleanup completed: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio: {self.audio_received}, "
                       f"Transcriptions: {self.transcriptions}, "
                       f"Responses: {self.responses_sent}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def cleanup_sync(self):
        """Synchronous cleanup for call from non-async context."""
        try:
            self.conversation_active = False
            
            # Cancel timer
            if self.restart_timer:
                self.restart_timer.cancel()
            
            # Note: Cannot await in sync context, but we set flags
            # The async cleanup will handle the rest when possible
            logger.info("Sync cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self.start_time
        
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "responses_sent": self.responses_sent,
            "is_speaking": self.is_speaking,
            "stt_active": self.stt_active,
            "conversation_active": self.conversation_active,
            "project_id": self.project_id,
            "last_audio_ago": round(time.time() - self.last_audio_time, 2),
            "last_transcription_ago": round(time.time() - self.last_transcription_time, 2)
        }