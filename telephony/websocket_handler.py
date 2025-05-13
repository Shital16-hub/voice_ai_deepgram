"""
Refactored WebSocket handler using modular components.

This file now orchestrates the various specialized handlers for cleaner code organization.
"""
import json
import logging
import time
from typing import Dict, Any, Optional

from .websocket.audio_processor import AudioHandler
from .websocket.speech_recognizer import SpeechRecognitionHandler
from .websocket.response_generator import ResponseGenerator
from .websocket.message_handler import MessageHandler
from .websocket.base import ConnectionState

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Main WebSocket handler that orchestrates all components.
    
    This class coordinates audio processing, speech recognition, response generation,
    and message handling in a modular way.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler with modular components.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.pipeline = pipeline
        
        # Initialize all component handlers
        self.audio_handler = AudioHandler()
        self.speech_handler = SpeechRecognitionHandler()
        self.response_generator = ResponseGenerator(pipeline)
        self.message_handler = MessageHandler(call_sid)
        
        # Wire up components
        self.message_handler.set_components(
            self.audio_handler,
            self.speech_handler,
            self.response_generator
        )
        
        # Conversation flow management
        self.pause_after_response = 0.3
        self.min_transcription_length = 3
        
        logger.info(f"WebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Main entry point for handling WebSocket messages.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        # Delegate to message handler
        await self.message_handler.handle_message(message, ws)
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send a text response by converting to speech.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Track for echo detection
            self.speech_handler.add_system_response(text)
            
            # Convert to speech
            speech_audio, error = await self.response_generator.text_to_speech(text)
            
            if speech_audio and not error:
                # Send through message handler
                await self.message_handler._send_audio_response(speech_audio, ws)
                logger.info(f"Sent text response: '{text}'")
                
                # Update last response time
                self.message_handler.last_response_time = time.time()
            else:
                logger.error(f"Failed to generate speech for text: {error}")
        except Exception as e:
            logger.error(f"Error sending text response: {e}", exc_info=True)
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.message_handler.call_info.state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]
    
    @property
    def is_streaming(self) -> bool:
        """Check if actively streaming."""
        return self.message_handler.call_info.state == ConnectionState.STREAMING
    
    @property
    def stream_sid(self) -> Optional[str]:
        """Get the current stream SID."""
        return self.message_handler.call_info.stream_sid
    
    async def cleanup(self):
        """Clean up all resources."""
        try:
            # Stop speech recognition
            await self.speech_handler.stop_speech_session()
            
            # Stop keep-alive
            await self.message_handler.keep_alive_manager.stop()
            
            # Clear audio buffers
            self.audio_handler.clear_buffer()
            
            logger.info(f"WebSocket handler cleanup complete for call {self.call_sid}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")