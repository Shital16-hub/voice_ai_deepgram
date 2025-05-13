# telephony/websocket/message_router_v2.py

"""
Improved message router for better continuous speech handling.
"""
import json
import logging
import asyncio
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MessageRouter:
    """Improved message router with better speech continuity handling."""
    
    def __init__(self, ws_handler):
        """Initialize improved message router."""
        self.ws_handler = ws_handler
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        # Enhanced transcription management
        self.transcription_buffer = []
        self.last_final_time = 0
        self.utterance_gap_threshold = 1.0  # 1 second gap to separate utterances
        
        # Statistics
        self.messages_received = 0
        self.media_events_processed = 0
        self.interim_results_received = 0
        self.final_results_received = 0
        self.utterances_processed = 0
        self.responses_sent = 0
        
        logger.info("MessageRouter v2 initialized with improved speech handling")
    
    async def route_message(self, message: str, ws) -> None:
        """Route incoming WebSocket message with improved handling."""
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
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start with initialization."""
        logger.info("Handling stream start event")
        
        # Store stream info
        self.ws_handler.stream_sid = data.get('streamSid')
        start_data = data.get('start', {})
        media_format = start_data.get('mediaFormat', {})
        
        logger.info(f"Stream started - SID: {self.ws_handler.stream_sid}")
        logger.info(f"Media format: {media_format}")
        
        # Reset components for new conversation
        self.ws_handler.audio_manager.clear_buffer()
        self.ws_handler.audio_manager.update_response_time()
        
        # Reset router state
        self.transcription_buffer.clear()
        self.last_final_time = 0
        self.is_processing = False
        
        # Delegate to connection manager
        await self.ws_handler.connection_manager.handle_start(data, ws)
        
        # Send welcome message
        await asyncio.sleep(0.5)
        welcome_msg = "Hello! I'm ready to help you. What would you like to know?"
        await self.ws_handler.send_text_response(welcome_msg, ws)
        logger.info("Sent welcome message")
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media with improved continuous speech processing."""
        self.media_events_processed += 1
        
        if not self.ws_handler.conversation_active:
            logger.debug("Conversation not active, skipping media")
            return
        
        # Process audio through improved audio manager
        audio_data = await self.ws_handler.audio_manager.process_media(data)
        
        if audio_data and not self.is_processing:
            logger.debug(f"Got audio data for processing: {len(audio_data)} bytes")
            
            # Process without waiting (non-blocking)
            asyncio.create_task(self._process_audio_async(audio_data, ws))
    
    async def _process_audio_async(self, audio_data: bytes, ws) -> None:
        """Process audio asynchronously for better responsiveness."""
        try:
            # Process through improved speech processor
            async def result_callback(result):
                await self._handle_transcription_result(result, ws)
            
            # Process audio with callback
            result = await self.ws_handler.speech_processor.process_audio(
                audio_data=audio_data,
                callback=result_callback
            )
            
            # Handle final result if returned directly
            if result:
                await self._handle_final_transcription(result, ws)
                
        except Exception as e:
            logger.error(f"Error in async audio processing: {e}", exc_info=True)
    
    async def _handle_transcription_result(self, result, ws) -> None:
        """Handle transcription result from speech processor."""
        if result.is_final:
            self.final_results_received += 1
            await self._handle_final_transcription(result.text, ws)
        else:
            self.interim_results_received += 1
            await self._handle_interim_transcription(result.text)
    
    async def _handle_interim_transcription(self, text: str) -> None:
        """Handle interim transcription results."""
        if text:
            logger.debug(f"Interim: '{text}'")
            # Could be used for real-time display or early processing
    
    async def _handle_final_transcription(self, text: str, ws) -> None:
        """Handle final transcription with utterance management."""
        if not text:
            return
        
        current_time = time.time()
        
        # Clean and validate
        cleaned_text = self.ws_handler.cleanup_transcription(text)
        
        if not self.ws_handler.is_valid_transcription(cleaned_text):
            logger.debug(f"Invalid transcription ignored: '{cleaned_text}'")
            return
        
        # Check for echo
        if self.ws_handler.speech_processor.is_echo_of_system_speech(cleaned_text):
            logger.info(f"Echo detected, ignoring: '{cleaned_text}'")
            return
        
        logger.info(f"Final transcription: '{cleaned_text}'")
        
        # Check if this is part of a continuous utterance
        time_gap = current_time - self.last_final_time
        
        if time_gap < self.utterance_gap_threshold and self.transcription_buffer:
            # Continue existing utterance
            self.transcription_buffer.append(cleaned_text)
            logger.debug(f"Added to current utterance (gap: {time_gap:.2f}s)")
        else:
            # Process any previous utterance
            if self.transcription_buffer:
                await self._process_complete_utterance(ws)
            
            # Start new utterance
            self.transcription_buffer = [cleaned_text]
            logger.debug("Started new utterance")
        
        self.last_final_time = current_time
        
        # Set a timer to process utterance if no more speech comes
        asyncio.create_task(self._utterance_timeout_check(current_time, ws))
    
    async def _utterance_timeout_check(self, start_time: float, ws) -> None:
        """Check if utterance should be processed due to timeout."""
        await asyncio.sleep(self.utterance_gap_threshold)
        
        # Only process if no new speech has come since this check started
        if self.last_final_time <= start_time and self.transcription_buffer:
            logger.info("Processing utterance due to timeout")
            await self._process_complete_utterance(ws)
    
    async def _process_complete_utterance(self, ws) -> None:
        """Process a complete utterance."""
        if not self.transcription_buffer:
            return
        
        async with self.processing_lock:
            if self.is_processing:
                return
            
            self.is_processing = True
            
            try:
                # Combine utterance parts
                complete_utterance = " ".join(self.transcription_buffer).strip()
                self.transcription_buffer.clear()
                
                if not complete_utterance:
                    return
                
                self.utterances_processed += 1
                logger.info(f"Processing complete utterance #{self.utterances_processed}: '{complete_utterance}'")
                
                # Clear buffer after successful recognition
                self.ws_handler.audio_manager.clear_buffer()
                
                # Generate response
                response = await self.ws_handler.response_generator.generate_response(complete_utterance)
                
                if response:
                    # Set speaking state
                    self.ws_handler.audio_manager.set_speaking_state(True)
                    
                    # Send response
                    await self.ws_handler.send_text_response(response, ws)
                    self.responses_sent += 1
                    
                    logger.info(f"Sent response #{self.responses_sent}")
                    
                    # Reset speaking state
                    self.ws_handler.audio_manager.set_speaking_state(False)
                else:
                    logger.warning("No response generated")
                    
            finally:
                self.is_processing = False
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info("Handling stream stop event")
        
        # Process any remaining utterance
        if self.transcription_buffer:
            logger.info("Processing final utterance before stop")
            await self._process_complete_utterance(None)
        
        # Stop speech session
        await self.ws_handler.speech_processor.stop_speech_session()
        
        # Update state
        self.ws_handler.conversation_active = False
        await self.ws_handler.connection_manager.handle_stop(data)
        
        # Log final statistics
        self._log_session_stats()
    
    async def _handle_mark(self, data: Dict[str, Any]) -> None:
        """Handle mark event."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")
    
    def _log_session_stats(self) -> None:
        """Log final session statistics."""
        logger.info("=== Session Statistics ===")
        logger.info(f"Messages received: {self.messages_received}")
        logger.info(f"Media events processed: {self.media_events_processed}")
        logger.info(f"Interim results: {self.interim_results_received}")
        logger.info(f"Final results: {self.final_results_received}")
        logger.info(f"Utterances processed: {self.utterances_processed}")
        logger.info(f"Responses sent: {self.responses_sent}")
        
        # Get component stats
        audio_stats = self.ws_handler.audio_manager.get_stats()
        speech_stats = self.ws_handler.speech_processor.get_stats()
        
        logger.info(f"Audio success rate: {audio_stats.get('success_rate', 0):.1f}%")
        logger.info(f"Speech success rate: {speech_stats.get('success_rate', 0):.1f}%")
        logger.info("========================")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        return {
            "messages_received": self.messages_received,
            "media_events_processed": self.media_events_processed,
            "interim_results_received": self.interim_results_received,
            "final_results_received": self.final_results_received,
            "utterances_processed": self.utterances_processed,
            "responses_sent": self.responses_sent,
            "is_processing": self.is_processing,
            "transcription_buffer_size": len(self.transcription_buffer),
            "time_since_last_final": time.time() - self.last_final_time
        }