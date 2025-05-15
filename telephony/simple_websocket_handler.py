"""
Fixed WebSocket handler with proper STT session management for continuous conversation.
This handles automatic STT reconnection and maintains conversation state properly.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional

# Use the fixed STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

# Use the fixed TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    WebSocket handler optimized for continuous conversation with proper session management.
    Automatically handles STT reconnection for seamless multi-turn interactions.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with continuous conversation support."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically
        self.project_id = self._get_project_id()
        
        # Initialize Google Cloud STT v2 with continuous streaming settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,  # Final results only for better conversation flow
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file
        )
        
        # Initialize Google Cloud TTS with fixed configuration
        self.tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,  # Don't set gender for Neural2 voices
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        # Conversation state management
        self.conversation_active = True
        self.is_speaking = False
        self.expecting_speech = True  # Always expect speech for continuous conversation
        
        # Audio processing
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        self.min_chunk_size = 160  # 20ms minimum
        
        # Session management for continuous conversation
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.last_audio_time = time.time()
        
        # Track conversation state
        self.waiting_for_response = False
        self.last_response_time = time.time()
        
        # Stats
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        
        logger.info(f"WebSocket handler initialized for continuous conversation - Call: {call_sid}, Project: {self.project_id}")
    
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
        """Handle incoming audio data with continuous conversation support."""
        # Don't skip audio while speaking - we want to be ready for interruptions
        if self.waiting_for_response:
            return  # But skip if we're still processing a response
        
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
        
        # Start STT if not already started (for the initial welcome message)
        if not self.stt_client.is_streaming:
            logger.info("Starting STT streaming for continuous conversation")
            await self.stt_client.start_streaming()
        
        # Process audio chunks immediately for real-time response
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle streaming transcription results with session tracking."""
        if result.is_final and result.text.strip():
            logger.info(f"Final transcription (session {result.session_id}): '{result.text}' (confidence: {result.confidence:.2f})")
            self.last_transcription_time = time.time()
            
            # Process the transcription immediately for continuous conversation
            await self._process_final_transcription(result.text, None)
    
    async def _process_final_transcription(self, transcription: str, ws=None):
        """Process final transcription and generate response."""
        # Clean up transcription
        cleaned_text = transcription.strip()
        
        # Skip very short or meaningless transcriptions
        if not cleaned_text or len(cleaned_text) < 2:
            logger.debug(f"Skipping short transcription: '{cleaned_text}'")
            return
        
        # Skip common non-speech sounds and responses that echo our output
        skip_words = {'um', 'uh', 'mmm', 'hmm', 'ah', 'er', 'hello', 'hi', 'ready', 'help'}
        words = cleaned_text.lower().split()
        
        # Skip if it's just filler words or echoes our greeting
        if len(words) <= 3 and all(word in skip_words for word in words):
            logger.debug(f"Skipping filler/echo: '{cleaned_text}'")
            return
        
        # Check if this seems like an echo of our previous response
        if self.responses_sent > 0:
            # Simple check for common echo patterns
            echo_phrases = ["ready to help", "what would you like", "how can i help"]
            if any(phrase in cleaned_text.lower() for phrase in echo_phrases):
                logger.debug(f"Skipping potential echo: '{cleaned_text}'")
                return
        
        self.transcriptions += 1
        self.waiting_for_response = True
        logger.info(f"Processing transcription: '{cleaned_text}'")
        
        try:
            # Query knowledge base through pipeline
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(cleaned_text)
                response_text = result.get("response", "")
                
                if response_text:
                    await self._send_response(response_text)
                else:
                    logger.warning("No response generated from knowledge base")
                    await self._send_response("I'm sorry, I couldn't find an answer to that question.")
            else:
                logger.error("Pipeline or query engine not available")
                await self._send_response("I'm sorry, there's an issue with my knowledge base.")
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error processing your request.")
        finally:
            self.waiting_for_response = False
    
    async def _send_response(self, text: str, ws=None):
        """Send TTS response with proper conversation flow."""
        if not text.strip():
            return
        
        # Use the stored WebSocket if not provided
        if ws is None:
            ws = getattr(self, '_ws', None)
            if ws is None:
                logger.error("No WebSocket available for sending response")
                return
        
        try:
            # Set speaking flag to indicate we're responding
            self.is_speaking = True
            self.last_response_time = time.time()
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
            # Clear speaking flag and resume listening
            self.is_speaking = False
            
            # Small delay to ensure audio playback completes
            await asyncio.sleep(0.5)
            
            # Make sure STT is still running for continuous conversation
            if not self.stt_client.is_streaming and self.conversation_active:
                logger.info("Restarting STT for continuous conversation")
                await self.stt_client.start_streaming()
            
            logger.debug("Ready for next utterance")
    
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
    
    async def start_conversation(self, ws):
        """Start the conversation with proper STT initialization."""
        # Store WebSocket reference for later use
        self._ws = ws
        
        # Start STT streaming
        if not self.stt_client.is_streaming:
            logger.info("Starting STT streaming for conversation")
            await self.stt_client.start_streaming()
        
        # Send welcome message
        await self._send_response("I'm ready to help. What would you like to know?", ws)
    
    async def _cleanup(self):
        """Clean up resources while maintaining session if needed."""
        try:
            self.conversation_active = False
            
            # Check if we should keep the session alive for potential continuation
            session_duration = time.time() - self.session_start_time
            time_since_last_transcription = time.time() - self.last_transcription_time
            
            # Keep session alive if it's recent and active
            if session_duration < 300 and time_since_last_transcription < 60:  # 5 min total, 1 min idle
                logger.info("Keeping STT session alive for potential continuation")
            else:
                logger.info("Stopping STT session due to inactivity")
                if self.stt_client.is_streaming:
                    await self.stt_client.stop_streaming()
                    await self.stt_client.cleanup()
            
            duration = time.time() - self.session_start_time
            
            logger.info(f"Session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio: {self.audio_received}, "
                       f"Transcriptions: {self.transcriptions}, "
                       f"Responses: {self.responses_sent}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics including STT session info."""
        duration = time.time() - self.session_start_time
        
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "responses_sent": self.responses_sent,
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "expecting_speech": self.expecting_speech,
            "project_id": self.project_id,
            "session_start_time": self.session_start_time,
            "last_transcription_time": self.last_transcription_time,
            "waiting_for_response": self.waiting_for_response
        }
        
        # Add STT stats if available
        if hasattr(self.stt_client, 'get_stats'):
            stats["stt_stats"] = self.stt_client.get_stats()
        
        # Add TTS stats if available
        if hasattr(self.tts_client, 'get_stats'):
            stats["tts_stats"] = self.tts_client.get_stats()
        
        return stats