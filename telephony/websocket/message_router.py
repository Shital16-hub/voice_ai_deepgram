"""
WebSocket message routing and event handling.
"""
import json
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MessageRouter:
    """Routes WebSocket messages to appropriate handlers."""
    
    def __init__(self, ws_handler):
        """Initialize message router."""
        self.ws_handler = ws_handler
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
    
    async def route_message(self, message: str, ws) -> None:
        """
        Route incoming WebSocket message to appropriate handler.
        
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
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error routing message: {e}", exc_info=True)
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        # Get stream_sid from the start event
        self.ws_handler.stream_sid = data.get('streamSid')
        logger.info(f"Set stream_sid: {self.ws_handler.stream_sid}")
        
        # Delegate to connection manager
        await self.ws_handler.connection_manager.handle_start(data, ws)
        
        # Reset all components
        self.ws_handler.audio_manager.clear_buffer()
        self.ws_handler.audio_manager.update_response_time()
        self.is_processing = False
        
        # Send welcome message after a short delay
        await asyncio.sleep(0.5)  # Small delay to ensure stream is ready
        await self.ws_handler.send_text_response("I'm listening. How can I help you today?", ws)
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data."""
        if not self.ws_handler.conversation_active:
            return
        
        # Process audio through audio manager
        audio_data = await self.ws_handler.audio_manager.process_media(data)
        
        if audio_data and not self.is_processing:
            async with self.processing_lock:
                if not self.is_processing:
                    self.is_processing = True
                    try:
                        await self._process_audio(audio_data, ws)
                    finally:
                        self.is_processing = False
    
    async def _process_audio(self, audio_data: bytes, ws) -> None:
        """Process audio through the complete pipeline."""
        try:
            # Process through speech recognition
            transcription = await self.ws_handler.speech_processor.process_audio(audio_data)
            
            if not transcription:
                return
            
            # Clean and validate transcription
            transcription = self.ws_handler.cleanup_transcription(transcription)
            
            if not self.ws_handler.is_valid_transcription(transcription):
                return
            
            # Check for echo
            if self.ws_handler.speech_processor.is_echo_of_system_speech(transcription):
                logger.info("Detected echo, ignoring")
                return
            
            logger.info(f"Complete transcription: {transcription}")
            
            # Clear buffer after successful transcription
            self.ws_handler.audio_manager.clear_buffer()
            
            # Generate response
            response = await self.ws_handler.response_generator.generate_response(transcription)
            
            if response:
                # Add to echo history
                self.ws_handler.speech_processor.add_to_echo_history(response)
                
                # Set speaking state
                self.ws_handler.audio_manager.set_speaking_state(True)
                
                # Convert to speech and send
                await self.ws_handler.response_generator.send_text_response(response, ws)
                
                # Update state
                self.ws_handler.audio_manager.set_speaking_state(False)
                self.ws_handler.audio_manager.update_response_time()
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        await self.ws_handler.connection_manager.handle_stop(data)
        self.ws_handler.conversation_active = False
        
        # Stop speech session
        await self.ws_handler.speech_processor.stop_speech_session()
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")