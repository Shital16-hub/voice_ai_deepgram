"""
Message handling for WebSocket events.
"""
import json
import logging
import asyncio
from typing import Dict, Any, Optional
import numpy as np

from .base import WebSocketEventHandler, KeepAliveManager, ConnectionState

logger = logging.getLogger(__name__)

class MessageHandler(WebSocketEventHandler):
    """Handles WebSocket message parsing and dispatch."""
    
    def __init__(self, call_sid: str):
        """Initialize the message handler."""
        super().__init__(call_sid)
        self.keep_alive_manager = KeepAliveManager()
        
        # Component references - to be set by WebSocketHandler
        self.audio_handler = None
        self.speech_handler = None
        self.response_generator = None
    
    def set_components(self, audio_handler, speech_handler, response_generator):
        """Set component references."""
        self.audio_handler = audio_handler
        self.speech_handler = speech_handler
        self.response_generator = response_generator
    
    async def handle_message(self, message: str, ws) -> None:
        """Handle incoming WebSocket message."""
        if not message:
            logger.warning("Received empty message")
            return
        
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            # Handle the event
            await self.handle_event(event_type, data, ws)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle WebSocket connected event."""
        logger.info(f"WebSocket connected for call {self.call_sid}")
        logger.info(f"Connected data: {data}")
        
        # Set connection state
        self.call_info.state = ConnectionState.CONNECTED
        self.connection_active.set()
        
        # Start keep-alive
        if self.call_info.stream_sid:
            await self.keep_alive_manager.start(ws, self.call_info.stream_sid)
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        self.call_info.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.call_info.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
        
        # Reset state for new stream
        if self.audio_handler:
            self.audio_handler.clear_buffer()
        
        self.call_info.state = ConnectionState.STREAMING
        self.call_info.start_time = asyncio.get_event_loop().time()
        
        # Initialize speech recognition
        if self.speech_handler:
            await self.speech_handler.start_speech_session()
        
        # Start keep-alive
        await self.keep_alive_manager.start(ws, self.call_info.stream_sid)
        
        # Send welcome message
        if self.response_generator:
            await self.response_generator.send_welcome_message(ws, self.call_info.stream_sid)
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data."""
        if self.call_info.state != ConnectionState.STREAMING:
            logger.debug("Not streaming, ignoring media")
            return
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        # Process audio through audio handler
        if self.audio_handler:
            processed_data = self.audio_handler.process_media_payload(payload)
            
            if processed_data and self.audio_handler.should_process_audio():
                # Get audio for processing
                pcm_audio, audio_size = self.audio_handler.get_audio_for_processing()
                
                if audio_size > 0:
                    # Process through speech recognition (if not already processing)
                    async with self.processing_lock:
                        if not self.is_processing:
                            self.is_processing = True
                            try:
                                await self._process_audio_chunk(pcm_audio, ws)
                            finally:
                                self.is_processing = False
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.call_info.stream_sid}")
        self.call_info.state = ConnectionState.DISCONNECTED
        self.connection_active.clear()
        
        # Stop keep-alive
        await self.keep_alive_manager.stop()
        
        # Stop speech recognition
        if self.speech_handler:
            await self.speech_handler.stop_speech_session()
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event for audio playback tracking."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    async def _process_audio_chunk(self, pcm_audio, ws):
        """Process an audio chunk through the entire pipeline."""
        try:
            # Convert to bytes for speech recognition
            audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
            
            # Process through speech recognition
            if self.speech_handler:
                transcription_results = await self.speech_handler.process_audio_chunk(audio_bytes)
                
                # Process transcription results
                if transcription_results:
                    for result in transcription_results:
                        transcription = result.get('text', '')
                        
                        # Clean up transcription
                        cleaned_transcription = self.speech_handler.cleanup_transcription(transcription)
                        
                        # Check for echo and validity
                        if cleaned_transcription and not self.speech_handler.is_echo_of_system_speech(cleaned_transcription):
                            if self.speech_handler.is_valid_transcription(cleaned_transcription):
                                # Clear buffer since we have a valid transcription
                                self.audio_handler.clear_buffer()
                                
                                # Generate response
                                await self._generate_and_send_response(cleaned_transcription, ws)
                                return
                else:
                    # No results yet, reduce buffer to prevent it from growing too large
                    self.audio_handler.reduce_buffer(factor=0.8)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            # Reduce buffer on error
            self.audio_handler.reduce_buffer()
    
    async def _generate_and_send_response(self, transcription: str, ws):
        """Generate response and send it back as audio."""
        try:
            # Generate response
            if self.response_generator:
                response_data = await self.response_generator.generate_response(transcription)
                
                if response_data.get("success", False):
                    response_text = response_data["text"]
                    
                    # Track this response for echo detection
                    if self.speech_handler:
                        self.speech_handler.add_system_response(response_text)
                    
                    # Convert to speech
                    speech_audio, error = await self.response_generator.text_to_speech(response_text)
                    
                    if speech_audio and not error:
                        # Send audio back
                        await self._send_audio_response(speech_audio, ws)
                        
                        # Update last response time
                        self.last_response_time = asyncio.get_event_loop().time()
                    else:
                        logger.error(f"TTS error: {error}")
                        # Send fallback response
                        fallback_audio = await self.response_generator.generate_fallback_response()
                        if fallback_audio:
                            await self._send_audio_response(fallback_audio, ws)
                else:
                    # Send fallback response
                    fallback_audio = await self.response_generator.generate_fallback_response()
                    if fallback_audio:
                        await self._send_audio_response(fallback_audio, ws)
        except Exception as e:
            logger.error(f"Error generating and sending response: {e}", exc_info=True)
            # Try to send fallback response
            if self.response_generator:
                fallback_audio = await self.response_generator.generate_fallback_response()
                if fallback_audio:
                    await self._send_audio_response(fallback_audio, ws)
    
    async def _send_audio_response(self, audio_data: bytes, ws):
        """Send audio response through WebSocket."""
        try:
            import base64
            
            # Check connection
            if self.call_info.state != ConnectionState.STREAMING:
                logger.warning("Connection not active, cannot send audio")
                return
            
            # Split audio into chunks
            chunks = self._split_audio_into_chunks(audio_data)
            
            logger.debug(f"Sending {len(chunks)} audio chunks ({len(audio_data)} bytes total)")
            
            for i, chunk in enumerate(chunks):
                try:
                    # Encode to base64
                    audio_base64 = base64.b64encode(chunk).decode('utf-8')
                    
                    # Create message
                    message = {
                        "event": "media",
                        "streamSid": self.call_info.stream_sid,
                        "media": {
                            "payload": audio_base64
                        }
                    }
                    
                    # Send message
                    ws.send(json.dumps(message))
                except Exception as e:
                    if "Connection closed" in str(e):
                        logger.warning(f"WebSocket connection closed while sending chunk {i+1}/{len(chunks)}")
                        self.call_info.state = ConnectionState.DISCONNECTED
                        return
                    else:
                        logger.error(f"Error sending audio chunk {i+1}/{len(chunks)}: {e}")
                        return
            
            logger.debug(f"Successfully sent all {len(chunks)} audio chunks")
            
        except Exception as e:
            logger.error(f"Error sending audio response: {e}", exc_info=True)
            if "Connection closed" in str(e):
                self.call_info.state = ConnectionState.DISCONNECTED
    
    def _split_audio_into_chunks(self, audio_data: bytes, chunk_size: int = 800) -> list:
        """Split audio into chunks for transmission."""
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        return chunks