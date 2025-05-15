"""
Enhanced WebSocket handler with proper session management, echo prevention,
and robust error handling for continuous conversation.
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
    Enhanced WebSocket handler with robust session management and echo prevention.
    Optimized for continuous conversation with proper error handling and cleanup.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with enhanced conversation support and echo prevention."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Get project ID dynamically with better error handling
        self.project_id = self._get_project_id()
        
        # Initialize Google Cloud STT v2 with enhanced settings
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,  # Only final results to avoid noise
            project_id=self.project_id,
            location="global",
            credentials_file=credentials_file
        )
        
        # Initialize Google Cloud TTS with enhanced settings
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
        
        # Enhanced conversation state management
        self.conversation_active = True
        self.is_speaking = False
        self.expecting_speech = True
        self.call_ended = False
        
        # Audio processing with flow control
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        self.min_chunk_size = 160  # 20ms minimum
        
        # Enhanced session management
        self.session_start_time = time.time()
        self.last_transcription_time = time.time()
        self.last_audio_time = time.time()
        self.last_tts_time = None
        
        # Response tracking for echo prevention
        self.waiting_for_response = False
        self.last_response_time = time.time()
        self.last_response_text = ""
        
        # Enhanced stats tracking
        self.audio_received = 0
        self.transcriptions = 0
        self.responses_sent = 0
        self.echo_detections = 0
        self.invalid_transcriptions = 0
        
        # WebSocket reference for response sending
        self._ws = None
        
        logger.info(f"Enhanced WebSocket handler initialized - Call: {call_sid}, Project: {self.project_id}")
    
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
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Handle audio with enhanced flow control and echo prevention."""
        # Skip audio processing if call has ended
        if self.call_ended:
            return
        
        # Skip audio while we're sending a response to prevent echo
        if self.waiting_for_response:
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio with error handling
        try:
            audio_data = base64.b64decode(payload)
            self.audio_received += 1
            self.last_audio_time = time.time()
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Enhanced check: Skip audio if we just sent TTS to prevent immediate echo
        if self.last_tts_time and (time.time() - self.last_tts_time) < 2.0:
            logger.debug("Skipping audio - too close to TTS output")
            return
        
        # Start STT if not already started
        if not self.stt_client.is_streaming and self.conversation_active:
            logger.info("Starting STT streaming for conversation")
            await self.stt_client.start_streaming()
        
        # Process audio chunks with enhanced error handling
        try:
            await self.stt_client.process_audio_chunk(
                audio_data, 
                callback=self._handle_transcription_result
            )
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle transcription results with enhanced echo detection and validation."""
        if result.is_final and result.text.strip():
            transcription = result.text.strip()
            confidence = result.confidence
            
            logger.info(f"Final transcription (session {result.session_id}): '{transcription}' (confidence: {confidence:.2f})")
            self.last_transcription_time = time.time()
            
            # Enhanced validation and echo detection
            if self._is_valid_transcription(transcription, confidence):
                await self._process_final_transcription(transcription, None)
            else:
                self.invalid_transcriptions += 1
                logger.debug(f"Invalid transcription rejected: '{transcription}' (conf: {confidence:.2f})")
    
    def _is_valid_transcription(self, transcription: str, confidence: float) -> bool:
        """Enhanced transcription validation with echo detection."""
        # Basic length check
        if len(transcription) < 2:
            return False
        
        # Confidence threshold (lower for telephony)
        if confidence < 0.3:
            logger.debug(f"Low confidence transcription: {confidence:.2f}")
            return False
        
        # Enhanced echo detection
        if self._is_likely_echo(transcription):
            self.echo_detections += 1
            logger.debug(f"Echo detected: '{transcription}'")
            return False
        
        # Skip common filler words and short responses
        transcription_lower = transcription.lower().strip()
        skip_patterns = [
            # Common filler words
            'um', 'uh', 'mmm', 'hmm', 'ah', 'er', 'oh',
            # Single words that might be misheard
            'only', 'series', 'okay', 'ok', 'yes', 'no',
            # Partial echo patterns (from our TTS responses)
            'ready to help', 'what would you like', 'how can i',
            'voice assist', 'features', 'pricing', 'plan'
        ]
        
        # Check if transcription is just a skip pattern
        if transcription_lower in skip_patterns:
            logger.debug(f"Skipping pattern: '{transcription}'")
            return False
        
        # Check for word-for-word matches with recent response
        if self.last_response_text:
            # Simple word overlap check
            response_words = set(self.last_response_text.lower().split())
            transcription_words = set(transcription_lower.split())
            
            if len(transcription_words) > 0:
                overlap_ratio = len(response_words & transcription_words) / len(transcription_words)
                if overlap_ratio > 0.8:  # 80% overlap indicates echo
                    logger.debug(f"High word overlap with last response: {overlap_ratio:.2f}")
                    return False
        
        return True
    
    def _is_likely_echo(self, transcription: str) -> bool:
        """Enhanced echo detection using multiple heuristics."""
        # Check timing - if transcription comes too soon after TTS, likely echo
        if self.last_tts_time and (time.time() - self.last_tts_time) < 3.0:
            # Check for substring matches with recent TTS output
            if hasattr(self.stt_client, 'last_spoken_texts'):
                for spoken_text, timestamp in self.stt_client.last_spoken_texts:
                    if time.time() - timestamp < 5.0:  # Within 5 seconds
                        if (transcription.lower() in spoken_text.lower() or 
                            spoken_text.lower() in transcription.lower()):
                            return True
        
        # Check against specific system phrases
        system_phrases = [
            "i'm ready to help",
            "what would you like to know",
            "voice assist offers",
            "voice assist features",
            "pricing plans"
        ]
        
        for phrase in system_phrases:
            if phrase in transcription.lower():
                return True
        
        return False
    
    async def _process_final_transcription(self, transcription: str, ws=None):
        """Process transcription with enhanced error handling and response management."""
        # Update state
        self.transcriptions += 1
        self.waiting_for_response = True
        
        logger.info(f"Processing transcription: '{transcription}'")
        
        try:
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
                
        except Exception as e:
            logger.error(f"Error processing transcription: {e}", exc_info=True)
            await self._send_response("I'm sorry, I encountered an error processing your request.")
        finally:
            self.waiting_for_response = False
    
    async def _send_response(self, text: str, ws=None):
        """Send TTS response with enhanced error handling and echo prevention."""
        if not text.strip() or self.call_ended:
            return
        
        # Use stored WebSocket if not provided
        if ws is None:
            ws = getattr(self, '_ws', None)
            if ws is None:
                logger.error("No WebSocket available for sending response")
                return
        
        try:
            # Set response state
            self.is_speaking = True
            self.last_response_time = time.time()
            self.last_response_text = text
            
            logger.info(f"Sending response: '{text}'")
            
            # Inform STT client about TTS output for echo detection
            if hasattr(self.stt_client, 'add_tts_text'):
                self.stt_client.add_tts_text(text)
            
            # Convert to speech with error handling
            try:
                audio_data = await self.tts_client.synthesize(text)
                self.last_tts_time = time.time()
                
                if audio_data:
                    # Send audio in chunks with proper pacing
                    await self._send_audio_chunks(audio_data, ws)
                    self.responses_sent += 1
                    logger.info(f"Successfully sent response ({len(audio_data)} bytes)")
                else:
                    logger.error("No audio data generated from TTS")
            except Exception as e:
                logger.error(f"Error synthesizing speech: {e}")
                # Don't re-raise - just continue without audio response
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
        finally:
            # Clear speaking flag
            self.is_speaking = False
            
            # Small delay to ensure audio playback completes
            await asyncio.sleep(0.8)
            
            # Ensure STT is still running for continuous conversation
            if not self.stt_client.is_streaming and self.conversation_active and not self.call_ended:
                logger.info("Restarting STT for continuous conversation")
                try:
                    await self.stt_client.start_streaming()
                except Exception as e:
                    logger.error(f"Error restarting STT: {e}")
            
            logger.debug("Ready for next utterance")
    
    async def _send_audio_chunks(self, audio_data: bytes, ws):
        """Send audio data with proper chunking and error handling."""
        if not self.stream_sid or not ws:
            logger.warning("Cannot send audio: missing stream_sid or websocket")
            return
        
        chunk_size = 400  # 50ms chunks for smooth playback
        total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                ws.send(json.dumps(message))
                
                # Dynamic delay based on chunk size
                await asyncio.sleep(0.025)  # 25ms delay
                
            except Exception as e:
                logger.error(f"Error sending audio chunk {chunk_num}/{total_chunks}: {e}")
                break
        
        logger.debug(f"Sent {total_chunks} audio chunks")
    
    async def start_conversation(self, ws):
        """Start conversation with enhanced initialization."""
        # Store WebSocket reference
        self._ws = ws
        
        # Start STT streaming
        if not self.stt_client.is_streaming:
            logger.info("Starting STT streaming for conversation")
            await self.stt_client.start_streaming()
        
        # Send welcome message with delay to ensure connection is stable
        await asyncio.sleep(0.1)
        await self._send_response("I'm ready to help. What would you like to know?", ws)
    
    async def _cleanup(self):
        """Enhanced cleanup with proper session management."""
        try:
            self.call_ended = True
            self.conversation_active = False
            
            # Check if we should keep the session alive
            session_duration = time.time() - self.session_start_time
            time_since_last_transcription = time.time() - self.last_transcription_time
            
            # More aggressive cleanup since call ended
            logger.info("Call ended - stopping STT session")
            if self.stt_client.is_streaming:
                await self.stt_client.stop_streaming()
                await self.stt_client.cleanup()
            
            # Calculate final statistics
            duration = time.time() - self.session_start_time
            
            # Enhanced logging with conversation metrics
            logger.info(f"Session cleanup completed. Stats: "
                       f"Duration: {duration:.2f}s, "
                       f"Audio packets: {self.audio_received}, "
                       f"Valid transcriptions: {self.transcriptions}, "
                       f"Invalid/Echo: {self.invalid_transcriptions + self.echo_detections}, "
                       f"Responses: {self.responses_sent}, "
                       f"Echo detections: {self.echo_detections}")
                       
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        duration = time.time() - self.session_start_time
        
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration": round(duration, 2),
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "invalid_transcriptions": self.invalid_transcriptions,
            "echo_detections": self.echo_detections,
            "responses_sent": self.responses_sent,
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "call_ended": self.call_ended,
            "expecting_speech": self.expecting_speech,
            "project_id": self.project_id,
            "session_start_time": self.session_start_time,
            "last_transcription_time": self.last_transcription_time,
            "last_audio_time": self.last_audio_time,
            "waiting_for_response": self.waiting_for_response,
            # Add quality metrics
            "transcription_rate": round(self.transcriptions / max(duration / 60, 1), 2),  # per minute
            "response_rate": round(self.responses_sent / max(duration / 60, 1), 2),      # per minute
            "echo_rate": round(self.echo_detections / max(self.audio_received, 1) * 100, 2),  # percentage
        }
        
        # Add STT stats if available
        if hasattr(self.stt_client, 'get_stats'):
            stats["stt_stats"] = self.stt_client.get_stats()
        
        # Add TTS stats if available
        if hasattr(self.tts_client, 'get_stats'):
            stats["tts_stats"] = self.tts_client.get_stats()
        
        return stats