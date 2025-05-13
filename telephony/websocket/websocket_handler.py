# telephony/websocket/websocket_handler.py

"""
Updated WebSocket handler using simplified Google Cloud STT v2.25.0+.
Removes overcomplicated abstractions and focuses on working implementation.
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional

# Import all the required components
from telephony.websocket.connection_manager import ConnectionManager
from telephony.websocket.audio_manager import AudioManager
from telephony.websocket.speech_processor import SpeechProcessor
from telephony.websocket.response_generator import ResponseGenerator
from telephony.websocket.message_router import MessageRouter

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Updated WebSocket handler with simplified Google Cloud STT v2.25.0+.
    Maintains compatibility with existing code while using updated components.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler with updated components.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        logger.info(f"Initializing WebSocketHandler for call {call_sid}")
        
        # Initialize components using the updated implementations
        self.connection_manager = ConnectionManager(call_sid)
        self.audio_manager = AudioManager()
        self.speech_processor = SpeechProcessor(pipeline)  # Uses updated STT
        self.response_generator = ResponseGenerator(pipeline, self)
        self.message_router = MessageRouter(self)
        
        # State tracking
        self.conversation_active = True
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        logger.info("WebSocketHandler initialized with updated components")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        # Use the message router to handle the message
        await self.message_router.route_message(message, ws)
    
    # Delegate specific operations to component managers
    async def send_text_response(self, text: str, ws) -> None:
        """Send text response through response generator."""
        await self.response_generator.send_text_response(text, ws)
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription through speech processor."""
        return self.speech_processor.cleanup_transcription(text)
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription through speech processor."""
        return self.speech_processor.is_valid_transcription(text)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "conversation_active": self.conversation_active,
            "audio_stats": self.audio_manager.get_stats(),
            "speech_stats": self.speech_processor.get_stats(),
            "message_router_stats": self.message_router.get_stats() if hasattr(self.message_router, 'get_stats') else {}
        }