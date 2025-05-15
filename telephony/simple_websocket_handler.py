"""
Fixed WebSocket handler with proper continuous conversation support.
Handles session persistence and automatic reconnection for multi-turn conversations.
"""
import json
import asyncio
import logging
import base64
import time
import os
from typing import Dict, Any, Optional

# Use the updated STT implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult

# Use the fixed TTS implementation
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    WebSocket handler optimized for continuous conversation with proper session management.
    Maintains STT session across multiple utterances for seamless multi-turn interactions.
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
        
        # Silence detection for better conversation flow
        self.silence_threshold = 0.01
        self.max_silence_chunks = 30  # ~3 seconds of silence before processing
        self.silence_count = 0
        
        # Session management
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.consecutive_silence = 0
        
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
        if not self.stt_client.is_streaming:
            logger.info("Starting STT streaming for continuous conversation")
            await self.stt_client.start_streaming()
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process chunks when buffer is large enough
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = bytes(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Check for silence to optimize processing
            if self._is_silence(chunk):
                self.silence_count += 1
                if self.silence_count < self.max_silence_chunks:
                    continue  # Skip processing until we have enough silence
            else:
                self.silence_count = 0  # Reset silence counter
            
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
    
    def _is_silence(self, audio_chunk: bytes) -> bool:
        """Simple silence detection based on audio amplitude."""
        if len(audio_chunk) == 0:
            return True
        
        # Calculate RMS amplitude for μ-law audio
        import audioop
        try:
            # Convert μ-law to linear PCM for analysis
            linear_audio = audioop.ulaw2lin(audio_chunk, 2)
            rms = audioop.rms(linear_audio, 2)
            # Normalize RMS (32767 is max for 16-bit)
            normalized_rms = rms / 32767.0
            return normalized_rms < self.silence_threshold
        except Exception:
            # Fallback to simple byte analysis
            avg_amplitude = sum(audio_chunk) / len(audio_chunk)
            return avg_amplitude < (self.silence_threshold * 255)
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle streaming transcription results with session tracking."""
        if result.is_final and result.text.strip():
            logger.info(f"Final transcription (session {result.session_id}): '{result.text}' (confidence: {result.confidence:.2f})")
            self.last_transcription_time = time.time()
        elif result.text.strip():
            logger.debug(f"Interim: '{result.text}'")
    
    async def _process_final_transcription(self, transcription: str, ws):
        """Process final transcription and generate response."""
        # Clean up transcription
        cleaned_text = transcription.strip()
        
        # Skip very short or meaningless transcriptions
        if not cleaned_text or len(cleaned_text) < 2:
            logger.debug(f"Skipping short transcription: '{cleaned_text}'")
            return
        
        # Skip common non-speech sounds
        skip_words = {'um', 'uh', 'mmm', 'hmm', 'ah', 'er'}
        if cleaned_text.lower() in skip_words:
            logger.debug(f"Skipping filler word: '{cleaned_text}'")
            return
        
        self.transcriptions += 1
        logger.info(f"Processing transcription: '{cleaned_text}'")
        
        try:
            # Query knowledge base through pipeline
            if hasattr(self.pipeline, 'query_engine') and self.pipeline.query_engine:
                result = await self.pipeline.query_engine.query(cleaned_text)
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
        """Send TTS response with proper conversation flow."""
        if not text.strip():
            return
        
        try:
            # Set speaking flag to pause audio processing
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
            # Clear speaking flag and resume listening
            self.is_speaking = False
            # Small delay to ensure audio playback completes
            await asyncio.sleep(0.5)
            
            # Reset buffers for next utterance
            self.audio_buffer.clear()
            self.silence_count = 0
            
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
    
    async def _cleanup(self):
        """Clean up resources while maintaining session if possible."""
        try:
            self.conversation_active = False
            
            # Only stop STT if we're truly done (not just between utterances)
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
            "last_transcription_time": self.last_transcription_time
        }
        
        # Add STT stats if available
        if hasattr(self.stt_client, 'get_stats'):
            stats["stt_stats"] = self.stt_client.get_stats()
        
        # Add TTS stats if available
        if hasattr(self.tts_client, 'get_stats'):
            stats["tts_stats"] = self.tts_client.get_stats()
        
        return stats