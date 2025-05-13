# telephony/websocket/optimized_websocket_handler.py

"""
Optimized WebSocket handler that integrates all generalized, high-performance components.
No domain-specific hardcoding - suitable for any application.
"""
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from telephony.websocket.connection_manager import ConnectionManager
from telephony.websocket.intelligent_audio_manager import IntelligentAudioManager
from telephony.websocket.generalized_speech_processor import GeneralizedSpeechProcessor
from telephony.websocket.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

class OptimizedWebSocketHandler:
    """
    Optimized WebSocket handler with focus on performance and quality,
    without domain-specific assumptions.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """
        Initialize optimized WebSocket handler.
        
        Args:
            call_sid: Twilio call SID
            pipeline: Voice AI pipeline instance
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        logger.info(f"Initializing OptimizedWebSocketHandler for call {call_sid}")
        
        # Initialize optimized components
        self.connection_manager = ConnectionManager(call_sid)
        self.audio_manager = IntelligentAudioManager()
        self.speech_processor = GeneralizedSpeechProcessor(pipeline)
        self.response_generator = ResponseGenerator(pipeline, self)
        
        # Create optimized message router
        self.message_router = OptimizedMessageRouter(self)
        
        # State tracking
        self.conversation_active = True
        self.session_start_time = time.time()
        
        # Performance monitoring
        self.total_messages = 0
        self.total_processing_time = 0.0
        self.peak_processing_time = 0.0
        
        logger.info("OptimizedWebSocketHandler initialized with generalized components")
    
    async def handle_message(self, message: str, ws) -> None:
        """
        Handle incoming WebSocket message with performance optimization.
        
        Args:
            message: JSON message from Twilio
            ws: WebSocket connection
        """
        start_time = time.time()
        self.total_messages += 1
        
        await self.message_router.route_message(message, ws)
        
        # Track performance
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.peak_processing_time = max(self.peak_processing_time, processing_time)
        
        if processing_time > 0.1:  # Log slow processing
            logger.warning(f"Slow message processing: {processing_time:.3f}s")
    
    # Delegate methods
    async def send_text_response(self, text: str, ws) -> None:
        """Send text response efficiently."""
        await self.response_generator.send_text_response(text, ws)
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription using generalized rules."""
        return self.speech_processor._cleanup_transcription(text)
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription using general criteria."""
        return self.speech_processor.is_valid_transcription(text)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "session_duration": round(session_duration, 2),
            "conversation_active": self.conversation_active,
            "performance": {
                "total_messages": self.total_messages,
                "avg_processing_time": round(self.total_processing_time / max(self.total_messages, 1), 3),
                "peak_processing_time": round(self.peak_processing_time, 3),
                "messages_per_second": round(self.total_messages / max(session_duration, 1), 2)
            },
            "audio_stats": self.audio_manager.get_stats(),
            "speech_stats": self.speech_processor.get_stats(),
            "message_router_stats": self.message_router.get_stats()
        }


class OptimizedMessageRouter:
    """Optimized message router with performance focus and error resilience."""
    
    def __init__(self, ws_handler):
        """Initialize optimized message router."""
        self.ws_handler = ws_handler
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        # Performance tracking
        self.messages_received = 0
        self.media_events_processed = 0
        self.successful_transcriptions = 0
        self.responses_sent = 0
        self.errors_encountered = 0
        
        # Error resilience
        self.consecutive_failures = 0
        self.max_retries = 3
        self.backoff_base = 0.1
        
        # Adaptive processing
        self.processing_times = []
        self.max_processing_time_samples = 10
        
        logger.info("OptimizedMessageRouter initialized")
    
    async def route_message(self, message: str, ws) -> None:
        """Route incoming WebSocket message with optimized handling."""
        self.messages_received += 1
        
        if not message:
            logger.debug(f"Empty message #{self.messages_received}")
            return
        
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            # Route to appropriate handler
            if event_type == 'connected':
                await self.ws_handler.connection_manager.handle_connected(data, ws)
            elif event_type == 'start':
                await self._handle_start(data, ws)
            elif event_type == 'media':
                await self._handle_media_optimized(data, ws)
            elif event_type == 'stop':
                await self._handle_stop(data)
            else:
                logger.debug(f"Unhandled event type: {event_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message #{self.messages_received}: {e}")
            self.errors_encountered += 1
        except Exception as e:
            logger.error(f"Error routing message #{self.messages_received}: {e}")
            self.errors_encountered += 1
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start with optimization setup."""
        self.ws_handler.stream_sid = data.get('streamSid')
        
        # Delegate to connection manager
        await self.ws_handler.connection_manager.handle_start(data, ws)
        
        # Reset adaptive parameters
        self.consecutive_failures = 0
        self.processing_times.clear()
        
        # Clear any buffered audio
        self.ws_handler.audio_manager.update_response_time()
        
        logger.info(f"Stream optimized and ready: {self.ws_handler.stream_sid}")
    
    async def _handle_media_optimized(self, data: Dict[str, Any], ws) -> None:
        """Handle media with optimized processing pipeline."""
        self.media_events_processed += 1
        
        if not self.ws_handler.conversation_active:
            return
        
        # Process audio through intelligent manager
        audio_data = await self.ws_handler.audio_manager.process_media(data)
        
        if audio_data and not self.is_processing:
            # Acquire lock with timeout to prevent deadlocks
            try:
                await asyncio.wait_for(self.processing_lock.acquire(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Audio processing lock timeout")
                return
            
            try:
                if not self.is_processing:  # Double-check inside lock
                    self.is_processing = True
                    await self._process_audio_optimized(audio_data, ws)
            finally:
                self.is_processing = False
                self.processing_lock.release()
    
    async def _process_audio_optimized(self, audio_data: bytes, ws) -> None:
        """Process audio with optimized error handling and retry logic."""
        processing_start = time.time()
        
        try:
            # Process through generalized speech processor
            transcription = await self.ws_handler.speech_processor.process_audio(audio_data)
            
            if not transcription:
                return
            
            # Validate using general criteria
            if not self.ws_handler.is_valid_transcription(transcription):
                logger.debug(f"Invalid transcription: '{transcription}'")
                return
            
            # Check for echo using general algorithm
            if self.ws_handler.speech_processor.is_echo_of_system_speech(transcription):
                logger.debug(f"Echo detected: '{transcription}'")
                return
            
            self.successful_transcriptions += 1
            logger.info(f"Valid transcription #{self.successful_transcriptions}: '{transcription}'")
            
            # Generate response
            response = await self.ws_handler.response_generator.generate_response(transcription)
            
            if response:
                # Add to echo history
                self.ws_handler.speech_processor.add_to_echo_history(response)
                
                # Send response
                self.ws_handler.audio_manager.set_speaking_state(True)
                await self.ws_handler.response_generator.send_text_response(response, ws)
                self.responses_sent += 1
                
                # Update state
                self.ws_handler.audio_manager.set_speaking_state(False)
                self.ws_handler.audio_manager.update_response_time()
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            self.errors_encountered += 1
            self.consecutive_failures += 1
            
            # Apply backoff for consecutive failures
            if self.consecutive_failures > 1:
                backoff_time = min(
                    self.backoff_base * (2 ** (self.consecutive_failures - 1)),
                    2.0  # Max 2 second backoff
                )
                logger.info(f"Applying backoff: {backoff_time:.2f}s")
                await asyncio.sleep(backoff_time)
        
        finally:
            # Track processing time
            processing_time = time.time() - processing_start
            self.processing_times.append(processing_time)
            
            if len(self.processing_times) > self.max_processing_time_samples:
                self.processing_times.pop(0)
    
    async def _handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop with cleanup."""
        logger.info("Handling optimized stream stop")
        
        # Stop speech processing
        await self.ws_handler.speech_processor.stop_session()
        
        # Update state
        self.ws_handler.conversation_active = False
        await self.ws_handler.connection_manager.handle_stop(data)
        
        # Force flush any remaining audio
        await self.ws_handler.audio_manager.force_flush()
        
        # Log final statistics
        self._log_session_summary()
    
    def _log_session_summary(self) -> None:
        """Log comprehensive session summary."""
        logger.info("=== Optimized Session Summary ===")
        logger.info(f"Messages received: {self.messages_received}")
        logger.info(f"Media events processed: {self.media_events_processed}")
        logger.info(f"Successful transcriptions: {self.successful_transcriptions}")
        logger.info(f"Responses sent: {self.responses_sent}")
        logger.info(f"Errors encountered: {self.errors_encountered}")
        
        # Calculate rates
        transcription_rate = (self.successful_transcriptions / max(self.media_events_processed, 1)) * 100
        response_rate = (self.responses_sent / max(self.successful_transcriptions, 1)) * 100
        
        logger.info(f"Transcription rate: {transcription_rate:.1f}%")
        logger.info(f"Response rate: {response_rate:.1f}%")
        
        # Processing time statistics
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            logger.info(f"Avg processing time: {avg_time:.3f}s")
            logger.info(f"Max processing time: {max_time:.3f}s")
        
        logger.info("================================")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive message router statistics."""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        return {
            "messages_received": self.messages_received,
            "media_events_processed": self.media_events_processed,
            "successful_transcriptions": self.successful_transcriptions,
            "responses_sent": self.responses_sent,
            "errors_encountered": self.errors_encountered,
            "consecutive_failures": self.consecutive_failures,
            "is_processing": self.is_processing,
            "performance": {
                "avg_processing_time": round(avg_processing_time, 3),
                "max_processing_time": max(self.processing_times) if self.processing_times else 0,
                "transcription_rate": round((self.successful_transcriptions / max(self.media_events_processed, 1)) * 100, 2),
                "response_rate": round((self.responses_sent / max(self.successful_transcriptions, 1)) * 100, 2)
            }
        }