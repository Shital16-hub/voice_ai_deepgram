"""
Enhanced message router with corrected async lock usage.
"""
import json
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MessageRouter:
    """Message router with proper async lock handling."""
    
    def __init__(self, ws_handler):
        """Initialize message router."""
        self.ws_handler = ws_handler
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        # Statistics
        self.messages_received = 0
        self.media_events_processed = 0
        self.transcriptions_generated = 0
        self.responses_sent = 0
        self.errors_encountered = 0
        
        logger.info("MessageRouter initialized")
    
    async def route_message(self, message: str, ws) -> None:
        """Route incoming WebSocket message."""
        self.messages_received += 1
        
        if not message:
            logger.warning(f"Message #{self.messages_received}: Empty message received")
            return
        
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Message #{self.messages_received}: {event_type}")
            
            # Route to appropriate handler
            if event_type == 'connected':
                await self.ws_handler.connection_manager.handle_connected(data, ws)
            elif event_type == 'start':
                await self._handle_start(data, ws)
            elif event_type == 'media':
                await self._handle_media(data, ws)
            elif event_type == 'stop':
                await self._handle_stop(data)
            elif event_type == 'mark':
                await self._handle_mark(data)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message #{self.messages_received}: {str(e)}")
        except Exception as e:
            logger.error(f"Error routing message #{self.messages_received}: {e}", exc_info=True)
            self.errors_encountered += 1
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        logger.info("Handling stream start event")
        
        # Extract stream information
        self.ws_handler.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.ws_handler.stream_sid}")
        
        # Delegate to connection manager
        await self.ws_handler.connection_manager.handle_start(data, ws)
        
        # Reset components
        self.ws_handler.audio_manager.clear_buffer()
        self.ws_handler.audio_manager.update_response_time()
        self.is_processing = False
        
        # Send welcome message (faster response)
        await asyncio.sleep(0.1)
        await self.ws_handler.send_text_response("Hello! How can I help you today?", ws)
        logger.info("Sent welcome message")
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with corrected async lock."""
        self.media_events_processed += 1
        
        if not self.ws_handler.conversation_active:
            return
        
        # Process audio through audio manager
        audio_data = await self.ws_handler.audio_manager.process_media(data)
        
        if audio_data and not self.is_processing:
            # Use async context manager properly
            async with self.processing_lock:
                if not self.is_processing:
                    self.is_processing = True
                    try:
                        await self._process_audio_optimized(audio_data, ws)
                    finally:
                        self.is_processing = False
    
    async def _process_audio_optimized(self, audio_data: bytes, ws) -> None:
        """Process audio with optimized pipeline."""
        try:
            logger.debug(f"Processing audio: {len(audio_data)} bytes")
            
            # Process through speech recognition
            transcription = await self.ws_handler.speech_processor.process_audio(audio_data)
            
            if not transcription:
                return
            
            logger.info(f"Transcription: '{transcription}'")
            
            # Minimal validation
            if not self.ws_handler.is_valid_transcription(transcription):
                logger.info(f"Invalid transcription: '{transcription}'")
                return
            
            # Check for echo
            if self.ws_handler.speech_processor.is_echo_of_system_speech(transcription):
                logger.info(f"Echo detected: '{transcription}'")
                return
            
            self.transcriptions_generated += 1
            logger.info(f"Valid transcription #{self.transcriptions_generated}: '{transcription}'")
            
            # Clear buffer
            self.ws_handler.audio_manager.clear_buffer()
            
            # Generate and send response
            logger.info("Generating response...")
            response = await self.ws_handler.response_generator.generate_response(transcription)
            
            if response:
                logger.info(f"Response: '{response}'")
                
                # Add to echo history
                self.ws_handler.speech_processor.add_to_echo_history(response)
                
                # Send response
                self.ws_handler.audio_manager.set_speaking_state(True)
                await self.ws_handler.response_generator.send_text_response(response, ws)
                self.responses_sent += 1
                
                # Update state
                self.ws_handler.audio_manager.set_speaking_state(False)
                self.ws_handler.audio_manager.update_response_time()
            else:
                logger.warning("No response generated")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.errors_encountered += 1
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info("Handling stream stop event")
        await self.ws_handler.connection_manager.handle_stop(data)
        self.ws_handler.conversation_active = False
        
        # Stop speech session
        await self.ws_handler.speech_processor.stop_speech_session()
        
        # Log final stats
        self._log_final_stats()
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        logger.info("=== Final Processing Statistics ===")
        logger.info(f"Messages received: {self.messages_received}")
        logger.info(f"Media events processed: {self.media_events_processed}")
        logger.info(f"Transcriptions generated: {self.transcriptions_generated}")
        logger.info(f"Responses sent: {self.responses_sent}")
        logger.info(f"Errors encountered: {self.errors_encountered}")
        
        # Calculate success rates
        if self.media_events_processed > 0:
            transcription_rate = (self.transcriptions_generated / self.media_events_processed) * 100
            logger.info(f"Transcription success rate: {transcription_rate:.1f}%")
        
        if self.transcriptions_generated > 0:
            response_rate = (self.responses_sent / self.transcriptions_generated) * 100
            logger.info(f"Response success rate: {response_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message router statistics."""
        return {
            "messages_received": self.messages_received,
            "media_events_processed": self.media_events_processed,
            "transcriptions_generated": self.transcriptions_generated,
            "responses_sent": self.responses_sent,
            "errors_encountered": self.errors_encountered,
            "is_processing": self.is_processing
        }