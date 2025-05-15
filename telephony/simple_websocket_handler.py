"""
Fixed WebSocket handler with synchronous processing and proper call flow.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional

# Use the simplified STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

# Use the fixed TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    Fixed WebSocket handler with proper Google Cloud STT v2 streaming and TTS.
    Handles call flow synchronously to prevent race conditions.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with proper configuration."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically (no hardcoding)
        self.project_id = self._get_project_id()
        
        # Initialize Google Cloud STT v2 with optimal telephony settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,  # Only final results for better accuracy
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file
        )
        
        # Initialize Google Cloud TTS with fixed configuration
        self.tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",  # Specific voice name
            voice_gender=None,  # Let Google handle gender for Neural2 voices
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        # State
        self.conversation_active = True
        self.is_speaking = False
        self.stt_started = False
        
        # Audio processing
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        
        # Stats
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        self.start_time = time.time()
        
        logger.info(f"SimpleWebSocketHandler initialized for call {call_sid}, project: {self.project_id}")
    
    def _get_project_id(self) -> str:
        """Get project ID from environment or credentials file."""
        # Check environment variable first
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
        
        # Fallback (should be removed in production)
        logger.warning("Using fallback project ID - this should be configured properly")
        return "my-tts-project-458404"
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Handle incoming audio data with proper buffering."""
        if self.is_speaking:
            return  # Skip audio while speaking to avoid feedback
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio
        try:
            audio_data = base64.b64decode(payload)
            self.audio_received += 1
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Start STT if not already started
        if not self.stt_started:
            await self.stt_client.start_streaming()
            self.stt_started = True
            logger.info("Started STT streaming")
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process chunks when buffer is large enough
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = bytes(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Process chunk with STT
            try:
                result = await self.stt_client.process_audio_chunk(
                    chunk, 
                    callback=self._handle_transcription_result
                )
                
                # Handle final results immediately
                if result and result.is_final and result.text.strip():
                    await self._process_final_transcription(result.text, ws)
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle streaming transcription results."""
        if result.is_final and result.text.strip():
            logger.info(f"Final transcription: '{result.text}' (confidence: {result.confidence:.2f})")
        elif result.text.strip():
            logger.debug(f"Interim: '{result.text}'")
    
    async def _process_final_transcription(self, transcription: str, ws):
        """Process final transcription and generate response."""
        # Skip very short transcriptions
        if not transcription or len(transcription.strip()) < 2:
            return
        
        # Skip common non-speech sounds
        if transcription.lower().strip() in ['um', 'uh', 'mmm', 'hmm']:
            return
        
        self.transcriptions += 1
        logger.info(f"Processing transcription: '{transcription}'")
        
        try:
            # Query knowledge base through pipeline
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(transcription)
                response_text = result.get("response", "")
                
                if response_text:
                    await self._send_response(response_text, ws)
                else:
                    logger.warning("No response generated from knowledge base")
                    await self._send_response("I'm sorry, I couldn't find an answer to that question.", ws)
            else:
                logger.error("Pipeline or query engine not available")
                await self._send_response("I'm sorry, there's an issue with my knowledge base.", ws)
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error processing your request.", ws)
    
    async def _send_response(self, text: str, ws):
        """Send TTS response with proper error handling."""
        if not text.strip():
            return
        
        try:
            self.is_speaking = True
            logger.info(f"Sending response: '{text}'")
            
            # Convert to speech
            audio_data = await self.tts_client.synthesize(text)
            
            if audio_data:
                # Send audio in chunks
                await self._send_audio_chunks(audio_data, ws)
                self.responses_sent += 1
                logger.info(f"Successfully sent response ({len(audio_data)} bytes)")
            else:
                logger.error("No audio data generated from TTS")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            self.is_speaking = False
            # Add a small delay after speaking to avoid cutting off audio
            await asyncio.sleep(0.5)
    
    async def _send_audio_chunks(self, audio_data: bytes, ws):
        """Send audio data as chunks to Twilio."""
        if not self.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        chunk_size = 400  # 50ms chunks for smooth playback
        
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
                
                # Small delay to maintain real-time playback
                await asyncio.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Error sending audio chunk {i}: {e}")
                break
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            self.conversation_active = False
            
            # Stop STT streaming
            if self.stt_started:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
                self.stt_started = False
            
            duration = time.time() - self.start_time
            
            logger.info(f"Session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio: {self.audio_received}, "
                       f"Transcriptions: {self.transcriptions}, "
                       f"Responses: {self.responses_sent}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self.start_time
        
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "responses_sent": self.responses_sent,
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "stt_started": self.stt_started,
            "project_id": self.project_id
        }
        
        # Add STT stats if available
        if hasattr(self.stt_client, 'get_stats'):
            stats["stt_stats"] = self.stt_client.get_stats()
        
        # Add TTS stats if available
        if hasattr(self.tts_client, 'get_stats'):
            stats["tts_stats"] = self.tts_client.get_stats()
        
        return stats