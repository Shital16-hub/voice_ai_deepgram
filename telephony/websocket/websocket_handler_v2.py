# telephony/websocket/websocket_handler_v2.py

"""
Updated WebSocket handler using improved audio manager and speech processor.
Better handling of continuous speech recognition.
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional

# Import the improved components
from telephony.websocket.connection_manager import ConnectionManager
from telephony.websocket.audio_manager_v2 import AudioManager
from telephony.websocket.speech_processor_v3 import SpeechProcessor
from telephony.websocket.response_generator import ResponseGenerator
from telephony.websocket.message_router_v2 import MessageRouter

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Improved WebSocket handler for better continuous speech recognition.
    Uses enhanced components that rely on Google Cloud STT's automatic features.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize WebSocket handler with improved components.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        logger.info(f"Initializing WebSocketHandler v2 for call {call_sid}")
        
        # Initialize improved components
        self.connection_manager = ConnectionManager(call_sid)
        self.audio_manager = AudioManager()  # Using AudioManager v2
        self.speech_processor = SpeechProcessor(pipeline)  # Using SpeechProcessor v3
        self.response_generator = ResponseGenerator(pipeline, self)
        self.message_router = MessageRouter(self)  # Using MessageRouter v2
        
        # Enhanced state tracking
        self.conversation_active = True
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        # Utterance management
        self.current_utterance_parts = []
        self.utterance_timeout = 2.0  # Seconds of silence before finalizing utterance
        self.last_speech_time = 0
        
        logger.info("WebSocketHandler v2 initialized with improved speech recognition")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message with improved processing.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        # Route to the improved message router
        await self.message_router.route_message(message, ws)
    
    async def send_text_response(self, text: str, ws) -> None:
        """Send text response through response generator."""
        # Add to echo history before sending
        self.speech_processor.add_to_echo_history(text)
        await self.response_generator.send_text_response(text, ws)
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription - now minimal thanks to API features."""
        return self.speech_processor.cleanup_transcription(text)
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription - simplified thanks to API confidence."""
        return self.speech_processor.is_valid_transcription(text)
    
    async def handle_partial_utterance(self) -> None:
        """Handle partial utterance completion due to timeout."""
        # Check if we should finalize current utterance
        partial_result = await self.speech_processor.handle_utterance_timeout()
        
        if partial_result and not self.is_processing:
            logger.info(f"Processing partial utterance: '{partial_result}'")
            
            # Process the accumulated partial transcription
            async with self.processing_lock:
                self.is_processing = True
                try:
                    await self._process_transcription(partial_result)
                finally:
                    self.is_processing = False
    
    async def _process_transcription(self, transcription: str) -> None:
        """
        Process a complete transcription.
        
        Args:
            transcription: Complete transcription text
        """
        # Clean and validate
        cleaned_transcription = self.cleanup_transcription(transcription)
        
        if not self.is_valid_transcription(cleaned_transcription):
            logger.info(f"Invalid transcription: '{cleaned_transcription}'")
            return
        
        # Check for echo
        if self.speech_processor.is_echo_of_system_speech(cleaned_transcription):
            logger.info(f"Detected echo, ignoring: '{cleaned_transcription}'")
            return
        
        logger.info(f"Processing valid transcription: '{cleaned_transcription}'")
        
        # Generate response
        response = await self.response_generator.generate_response(cleaned_transcription)
        
        if response:
            # Send response
            await self.send_text_response(response, None)  # ws will be handled by response_generator
        else:
            logger.warning("No response generated")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "conversation_active": self.conversation_active,
            "is_processing": self.is_processing,
            "audio_stats": self.audio_manager.get_stats(),
            "speech_stats": self.speech_processor.get_stats(),
            "message_router_stats": getattr(self.message_router, 'get_stats', lambda: {})()
        }