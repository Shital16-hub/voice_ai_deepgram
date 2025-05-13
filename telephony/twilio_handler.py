"""
Optimized Twilio handler for voice calls with better audio quality.
"""
import logging
from typing import Optional, Dict, Any
from twilio.twiml.voice_response import VoiceResponse, Connect, Start, Stream
from twilio.rest import Client

from telephony.config import (
    TWILIO_ACCOUNT_SID, 
    TWILIO_AUTH_TOKEN, 
    TWILIO_PHONE_NUMBER,
    MAX_CALL_DURATION
)
from telephony.call_manager import CallManager
from telephony.websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)

class TwilioHandler:
    """
    Optimized Twilio handler for better audio quality and lower latency.
    """
    
    def __init__(self, pipeline, base_url: str):
        """
        Initialize Twilio handler.
        
        Args:
            pipeline: Voice AI pipeline instance
            base_url: Base URL for webhooks
        """
        self.pipeline = pipeline
        self.base_url = base_url.rstrip('/')
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.call_manager = CallManager()
    
    async def start(self):
        """Start the Twilio handler."""
        await self.call_manager.start()
        logger.info("Twilio handler started")
    
    async def stop(self):
        """Stop the Twilio handler."""
        await self.call_manager.stop()
        logger.info("Twilio handler stopped")
    
    def handle_incoming_call(self, from_number: str, to_number: str, call_sid: str) -> str:
        """
        Handle incoming voice call with optimized audio settings.
        
        Args:
            from_number: Caller phone number
            to_number: Called phone number
            call_sid: Twilio call SID
            
        Returns:
            TwiML response as string
        """
        logger.info(f"Incoming call from {from_number} to {to_number} (SID: {call_sid})")
        
        # Add call to manager
        self.call_manager.add_call(call_sid, from_number, to_number)
        
        # Create TwiML response with optimized settings
        response = VoiceResponse()
        
        # WebSocket URL
        ws_url = f'{self.base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        logger.info(f"Setting up WebSocket stream at: {ws_url}")
        
        try:
            # Use Connect and Stream for bidirectional audio streaming
            connect = Connect()
            
            # Optimized stream parameters for better audio quality
            stream = Stream(
                url=ws_url,
                track="inbound_track"  # Changed to track="inbound_track" for clarity
            )
            
            # Optimized parameters for telephony
            stream.parameter(name="mediaEncoding", value="audio/x-mulaw;rate=8000")
            stream.parameter(name="amd", value="false")  # Disable answering machine detection
            
            connect.append(stream)
            response.append(connect)
            
            # Brief welcome message
            response.say("Hello! How can I help you today?", voice='alice')
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error setting up streaming: {e}", exc_info=True)
            
            # Fallback to basic TwiML
            response = VoiceResponse()
            response.say("I'm sorry, there was a connection error. Please try again.", voice='alice')
            return str(response)
    
    def handle_status_callback(self, call_sid: str, call_status: str) -> None:
        """
        Handle call status callback.
        
        Args:
            call_sid: Twilio call SID
            call_status: Call status
        """
        logger.info(f"Call {call_sid} status: {call_status}")
        
        # Update call status
        self.call_manager.update_call_status(call_sid, call_status)
        
        # Remove call if completed or failed
        if call_status in ['completed', 'failed', 'busy', 'no-answer']:
            self.call_manager.remove_call(call_sid)