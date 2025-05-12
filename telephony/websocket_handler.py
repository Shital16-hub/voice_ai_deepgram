"""
Main WebSocket handler for Twilio media streams - Refactored for better organization.
"""
import json
import base64
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from telephony.audio.audio_stream_processor import AudioStreamProcessor
from telephony.speech.speech_recognition_manager import SpeechRecognitionManager
from telephony.response.knowledge_base_processor import KnowledgeBaseProcessor
from telephony.response.echo_detection import EchoDetection
from telephony.utils.connection_manager import ConnectionManager
from telephony.config import AUDIO_BUFFER_SIZE, MAX_BUFFER_SIZE

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Main WebSocket handler for Twilio media streams.
    Coordinates between different components for audio processing, speech recognition,
    and response generation.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Initialize components
        self.audio_processor = AudioStreamProcessor()
        self.speech_manager = SpeechRecognitionManager()
        self.kb_processor = KnowledgeBaseProcessor(pipeline)
        self.echo_detection = EchoDetection()
        self.connection_manager = ConnectionManager(call_sid)
        
        # State tracking
        self.is_speaking = False
        self.speech_interrupted = False
        self.current_audio_chunks = []
        self.is_processing = False
        self.conversation_active = True
        self.sequence_number = 0
        
        # Connection state
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Processing locks and buffers
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Timing and throttling
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0
        
        # Conversation flow management
        self.pause_after_response = 0.3
        self.min_words_for_valid_query = 1
        
        logger.info(f"WebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        if not message:
            logger.warning("Received empty message")
            return
            
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            # Route to appropriate handler
            handlers = {
                'connected': self._handle_connected,
                'start': self._handle_start,
                'media': self._handle_media,
                'stop': self._handle_stop,
                'mark': self._handle_mark
            }
            
            handler = handlers.get(event_type)
            if handler:
                await handler(data, ws)
            else:
                logger.warning(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle connected event."""
        logger.info(f"WebSocket connected for call {self.call_sid}")
        
        # Update connection state
        self.connected = True
        self.connection_active.set()
        
        # Start keep-alive
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        
        # Reset all state for new stream
        await self._reset_stream_state()
        
        # Initialize speech recognition
        await self.speech_manager.start_session()
        
        # Initialize TTS if needed
        await self.kb_processor.init_tts()
        
        # Send welcome message
        await self.send_text_response("I'm listening. How can I help you today?", ws)
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data."""
        if not self.conversation_active:
            return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        try:
            # Decode and process audio
            audio_data = base64.b64decode(payload)
            
            # Skip if system is speaking
            if self.is_speaking:
                return
            
            # Process through audio component
            processed_data = self.audio_processor.process_incoming_audio(audio_data)
            
            if processed_data is None:
                return  # Still buffering
            
            # Check timing since last response
            if self._in_pause_period():
                return
            
            # Process audio buffer when ready
            if self.audio_processor.should_process_buffer() and not self.is_processing:
                async with self.processing_lock:
                    if not self.is_processing:
                        self.is_processing = True
                        try:
                            await self._process_audio_buffer(ws)
                        finally:
                            self.is_processing = False
            
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        
        # Cleanup
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        
        await self.speech_manager.stop_session()
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event for audio playback tracking."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    async def _process_audio_buffer(self, ws) -> None:
        """Process accumulated audio data."""
        try:
            # Get audio from processor
            audio_data = self.audio_processor.get_audio_for_processing()
            
            # Fix: Properly check numpy array
            if audio_data is None or len(audio_data) == 0:
                return
            
            # Process through speech recognition
            transcription = await self.speech_manager.process_audio(audio_data)
            
            # Check if this is an echo
            if self.echo_detection.is_echo(transcription):
                logger.info("Detected echo of system speech, ignoring")
                return
            
            # Validate transcription
            if not self.speech_manager.is_valid_transcription(transcription):
                logger.debug(f"Invalid transcription: '{transcription}'")
                return
            
            # Avoid duplicates
            if transcription == self.last_transcription:
                logger.info("Duplicate transcription, not processing again")
                return
            
            logger.info(f"Complete transcription: {transcription}")
            
            # Process through knowledge base
            response = await self.kb_processor.generate_response(transcription, self.call_sid)
            
            if response:
                # Track for echo detection
                self.echo_detection.add_system_response(response)
                
                # Convert to speech and send
                await self._send_audio_response(response, ws)
                
                # Update state
                self.last_transcription = transcription
                self.last_response_time = time.time()
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            # Send fallback response
            await self._send_fallback_response(ws)

    async def _handle_stop(self, data: Dict[str, Any], ws=None) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        
        # Cleanup
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        await self.speech_manager.stop_session()


    async def _handle_mark(self, data: Dict[str, Any], ws=None) -> None:
        """Handle mark event for audio playback tracking."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
        
        async def _send_audio_response(self, response: str, ws) -> None:
            """Send audio response to Twilio."""
            try:
                # Generate speech
                speech_audio = await self.kb_processor.generate_speech(response)
                
                # Update state
                self.is_speaking = True
                
                # Send audio
                await self._send_audio(speech_audio, ws)
                
                # Update state
                self.is_speaking = False
                
            except Exception as e:
                logger.error(f"Error sending audio response: {e}")
                await self._send_fallback_response(ws)
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to Twilio."""
        if not self.connected:
            logger.warning("WebSocket connection is closed, cannot send audio")
            return
        
        try:
            # Convert and split audio if needed
            processed_audio = self.audio_processor.prepare_outgoing_audio(audio_data)
            chunks = self.audio_processor.split_audio_for_streaming(processed_audio)
            
            logger.debug(f"Sending {len(chunks)} audio chunks ({len(processed_audio)} bytes total)")
            
            # Send chunks
            for i, chunk in enumerate(chunks):
                try:
                    audio_base64 = base64.b64encode(chunk).decode('utf-8')
                    message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": audio_base64}
                    }
                    ws.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        break
        except Exception as e:
            logger.error(f"Error preparing audio for sending: {e}")

    
    async def send_text_response(self, text: str, ws) -> None:
        """Send a text response by converting to speech."""
        try:
            # Track for echo detection
            self.echo_detection.add_system_response(text)
            
            # Generate and send speech
            await self._send_audio_response(text, ws)
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
    
    async def _send_fallback_response(self, ws) -> None:
        """Send a fallback response when errors occur."""
        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
        try:
            # Try to generate and send fallback speech directly
            speech_audio = await self.kb_processor.generate_speech(fallback_message)
            if speech_audio:
                self.is_speaking = True
                await self._send_audio(speech_audio, ws)
                self.is_speaking = False
            else:
                logger.error("Could not generate fallback speech")
        except Exception as e:
            logger.error(f"Failed to send fallback response: {e}")
    
    async def _reset_stream_state(self) -> None:
        """Reset all state for a new stream."""
        self.audio_processor.reset()
        self.speech_manager.reset()
        self.echo_detection.reset()
        
        self.is_speaking = False
        self.is_processing = False
        self.speech_interrupted = False
        self.silence_start_time = None
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0
        self.conversation_active = True
        self.sequence_number = 0
    
    def _in_pause_period(self) -> bool:
        """Check if we're in a pause period after last response."""
        time_since_last_response = time.time() - self.last_response_time
        return time_since_last_response < self.pause_after_response
    
    async def _keep_alive_loop(self, ws) -> None:
        """Send periodic keep-alive messages."""
        try:
            while self.conversation_active:
                await asyncio.sleep(10)
                
                if not self.stream_sid or not self.connected:
                    continue
                    
                try:
                    message = {
                        "event": "ping",
                        "streamSid": self.stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        self.conversation_active = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")

    async def _send_audio_response(self, response: str, ws) -> None:
        """Send audio response to Twilio."""
        try:
            # Generate speech
            speech_audio = await self.kb_processor.generate_speech(response)
            
            if speech_audio:
                # Update state
                self.is_speaking = True
                
                # Send audio
                await self._send_audio(speech_audio, ws)
                
                # Update state
                self.is_speaking = False
                
                logger.info(f"Sent audio response for: '{response[:50]}...'")
            else:
                logger.error("No speech audio generated")
                await self._send_fallback_response(ws)
                
        except Exception as e:
            logger.error(f"Error sending audio response: {e}")
            await self._send_fallback_response(ws)

    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to Twilio."""
        if not self.connected:
            logger.warning("WebSocket connection is closed, cannot send audio")
            return
        
        # Convert and split audio if needed
        processed_audio = self.audio_processor.prepare_outgoing_audio(audio_data)
        chunks = self.audio_processor.split_audio_for_streaming(processed_audio)
        
        # Send chunks
        for i, chunk in enumerate(chunks):
            try:
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                if "Connection closed" in str(e):
                    self.connected = False
                    self.connection_active.clear()
                    break
    
    async def _send_fallback_response(self, ws) -> None:
        """Send a fallback response when errors occur."""
        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
        try:
            await self.send_text_response(fallback_message, ws)
        except Exception as e:
            logger.error(f"Failed to send fallback response: {e}")