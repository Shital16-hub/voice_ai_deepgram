"""
Optimized WebSocket handler with direct MULAW support and reduced latency.
"""
import json
import base64
import asyncio
import logging
import time
from typing import Dict, Any, Optional
import numpy as np

from telephony.audio.audio_stream_processor import AudioStreamProcessor
from telephony.speech.speech_recognition_manager import SpeechRecognitionManager
from telephony.response.knowledge_base_processor import KnowledgeBaseProcessor
from telephony.response.echo_detection import EchoDetection
from telephony.utils.connection_manager import ConnectionManager
from telephony.config import AUDIO_BUFFER_SIZE, MAX_BUFFER_SIZE

logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    Optimized WebSocket handler with MULAW support and minimal latency.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize WebSocket handler with optimizations."""
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
        self.is_processing = False
        self.conversation_active = True
        self.sequence_number = 0
        
        # Connection state
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Processing optimization
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        self.keep_alive_interval = 5  # Reduced interval
        
        # Timing optimization - reduced for faster response
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0
        self.pause_after_response = 0.1  # Reduced pause
        self.min_words_for_valid_query = 1
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_history = 10
        self.total_processed_chunks = 0
        self.error_count = 0
        
        logger.info(f"WebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """Handle incoming WebSocket message with optimization."""
        if not message:
            logger.warning("Received empty message")
            return
            
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            # Route to handlers
            handlers = {
                'connected': self._handle_connected,
                'start': self._handle_start,
                'media': self._handle_media,
                'stop': self._handle_stop,
                'mark': self._handle_mark,
                'ping': self._handle_ping  # Added ping handler
            }
            
            handler = handlers.get(event_type)
            if handler:
                await handler(data, ws)
            else:
                logger.debug(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            self.error_count += 1
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle connected event with immediate response."""
        logger.info(f"WebSocket connected for call {self.call_sid}")
        
        # Set connection state immediately
        self.connected = True
        self.connection_active.set()
        self.connection_manager.set_connected(True)
        
        # Start keep-alive immediately
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
        
        # Send immediate acknowledgment
        try:
            ack_message = {
                "event": "ack",
                "streamSid": None
            }
            ws.send(json.dumps(ack_message))
        except Exception as e:
            logger.warning(f"Could not send ack: {e}")
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start with faster initialization."""
        self.stream_sid = data.get('streamSid')
        media_format = data.get('media', {})
        
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Media format: {media_format}")
        
        # Reset stream state
        await self._reset_stream_state()
        
        # Initialize components with MULAW support
        try:
            # Initialize STT with MULAW support
            stt_integration = getattr(self.pipeline, 'speech_recognizer', None)
            if stt_integration:
                # Configure for MULAW if possible
                if hasattr(stt_integration, 'speech_recognizer'):
                    stt_client = stt_integration.speech_recognizer
                    if hasattr(stt_client, 'encoding'):
                        stt_client.encoding = "MULAW"
                        stt_client.sample_rate = 8000
            
            await self.speech_manager.start_session()
            logger.info("Speech recognition initialized with MULAW support")
        except Exception as e:
            logger.error(f"Error starting speech recognition: {e}")
        
        # Initialize TTS
        await self.kb_processor.init_tts()
        
        # Send immediate greeting with optimized timing
        greeting = "Hello! How can I help you today?"
        # Don't wait for TTS, send immediately
        asyncio.create_task(self.send_text_response(greeting, ws))
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media with optimized MULAW processing."""
        if not self.conversation_active:
            return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        try:
            # Decode audio data (already in MULAW format from Twilio)
            audio_data = base64.b64decode(payload)
            
            # Skip processing if system is speaking
            if self.is_speaking:
                return
            
            # Process through optimized audio component
            processed_data = self.audio_processor.process_incoming_audio(audio_data)
            
            if processed_data is None:
                return  # Still buffering
            
            # Check timing constraints
            if self._in_pause_period():
                return
            
            # Process audio buffer with optimized threshold
            if self.audio_processor.should_process_buffer() and not self.is_processing:
                # Use create_task for immediate scheduling
                asyncio.create_task(self._process_audio_buffer_optimized(ws))
            
        except Exception as e:
            logger.error(f"Error processing media: {e}")
            self.error_count += 1
    
    async def _process_audio_buffer_optimized(self, ws) -> None:
        """Optimized audio buffer processing with parallel operations."""
        async with self.processing_lock:
            if self.is_processing:
                return
                
            self.is_processing = True
            process_start = time.time()
            
            try:
                # Get audio data
                audio_data = self.audio_processor.get_audio_for_processing()
                
                if audio_data is None or len(audio_data) == 0:
                    return
                
                # Process STT, KB query, and TTS preparation in parallel where possible
                tasks = []
                
                # Start STT processing
                stt_task = asyncio.create_task(self.speech_manager.process_audio(audio_data))
                tasks.append(("stt", stt_task))
                
                # Wait for STT with shorter timeout
                try:
                    transcription = await asyncio.wait_for(stt_task, timeout=1.5)
                except asyncio.TimeoutError:
                    logger.warning("STT timeout")
                    return
                
                # Check transcription quality
                if not transcription or not self.speech_manager.is_valid_transcription(transcription):
                    logger.debug(f"Invalid transcription: '{transcription}'")
                    return
                
                # Check for echo
                if self.echo_detection.is_echo(transcription):
                    logger.info("Echo detected, ignoring")
                    return
                
                # Avoid duplicate processing
                if transcription == self.last_transcription:
                    logger.debug("Duplicate transcription ignored")
                    return
                
                logger.info(f"Processing: {transcription}")
                
                # Start KB processing and TTS preparation in parallel
                kb_task = asyncio.create_task(
                    self.kb_processor.generate_response(transcription, self.call_sid)
                )
                
                # Prepare TTS while waiting for KB response
                tts_prep_task = asyncio.create_task(self.kb_processor.init_tts())
                
                try:
                    # Wait for KB response with timeout
                    response, _ = await asyncio.gather(
                        asyncio.wait_for(kb_task, timeout=2.0),
                        tts_prep_task,
                        return_exceptions=True
                    )
                except asyncio.TimeoutError:
                    logger.warning("KB timeout, using fallback")
                    response = self._get_fallback_response(transcription)
                
                if response and response.strip():
                    # Track for echo detection
                    self.echo_detection.add_system_response(response)
                    
                    # Generate and send speech
                    await self._send_audio_response_optimized(response, ws)
                    
                    # Update state
                    self.last_transcription = transcription
                    self.last_response_time = time.time()
                
                # Track performance
                total_time = time.time() - process_start
                self.processing_times.append(total_time)
                if len(self.processing_times) > self.max_processing_history:
                    self.processing_times.pop(0)
                
                self.total_processed_chunks += 1
                
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logger.info(f"Processing completed in {total_time:.2f}s (avg: {avg_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error in audio processing: {e}", exc_info=True)
                # Send fallback response
                await self._send_fallback_response(ws)
                self.error_count += 1
            finally:
                self.is_processing = False
    
    async def _send_audio_response_optimized(self, response: str, ws) -> None:
        """Optimized audio response with minimal latency."""
        try:
            # Set speaking state immediately
            self.is_speaking = True
            
            # Generate speech with reduced timeout
            speech_start = time.time()
            try:
                speech_task = asyncio.create_task(
                    self.kb_processor.generate_speech(response)
                )
                speech_audio = await asyncio.wait_for(speech_task, timeout=1.5)
            except asyncio.TimeoutError:
                logger.warning("TTS timeout")
                await self._send_fallback_response(ws)
                return
            
            if speech_audio:
                # Send audio immediately
                await self._send_audio_optimized(speech_audio, ws)
                
                speech_time = time.time() - speech_start
                logger.info(f"Sent audio ({speech_time:.2f}s): '{response[:50]}...'")
            else:
                logger.error("No speech audio generated")
                await self._send_fallback_response(ws)
                
        except Exception as e:
            logger.error(f"Error sending audio response: {e}")
            await self._send_fallback_response(ws)
        finally:
            # Reset speaking state with minimal delay
            await asyncio.sleep(0.1)
            self.is_speaking = False
    
    async def _send_audio_optimized(self, audio_data: bytes, ws) -> None:
        """Send audio with optimized chunking for MULAW."""
        if not self.connected:
            logger.warning("WebSocket not connected")
            return
        
        try:
            # The audio should already be in MULAW format from ElevenLabs
            # Split into optimized chunks for Twilio
            chunks = self.audio_processor.split_audio_for_streaming(audio_data)
            
            logger.debug(f"Sending {len(chunks)} audio chunks")
            
            # Send chunks with minimal delay
            for i, chunk in enumerate(chunks):
                try:
                    audio_base64 = base64.b64encode(chunk).decode('utf-8')
                    message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": audio_base64
                        }
                    }
                    
                    ws.send(json.dumps(message))
                    
                    # Small delay between chunks for Twilio processing
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.02)  # 20ms between chunks
                        
                except Exception as e:
                    logger.error(f"Error sending chunk {i+1}: {e}")
                    if "closed" in str(e).lower():
                        self.connected = False
                        self.connection_active.clear()
                        break
            
        except Exception as e:
            logger.error(f"Error preparing audio: {e}")
    
    async def _keep_alive_loop(self, ws) -> None:
        """Optimized keep-alive with health checking."""
        try:
            while self.conversation_active and self.connected:
                await asyncio.sleep(self.keep_alive_interval)
                
                if not self.stream_sid:
                    continue
                    
                try:
                    # Send ping with timestamp
                    ping_message = {
                        "event": "ping",
                        "streamSid": self.stream_sid,
                        "timestamp": time.time()
                    }
                    ws.send(json.dumps(ping_message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Keep-alive error: {e}")
                    if "closed" in str(e).lower():
                        self.connected = False
                        self.connection_active.clear()
                        self.conversation_active = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")
    
    def _in_pause_period(self) -> bool:
        """Check if we're in pause period with optimized timing."""
        time_since_last_response = time.time() - self.last_response_time
        return time_since_last_response < self.pause_after_response
    
    def _get_fallback_response(self, transcription: str) -> str:
        """Get contextual fallback response."""
        transcription_lower = transcription.lower()
        
        # Quick pattern matching for common queries
        if any(word in transcription_lower for word in ["price", "cost", "plan"]):
            return "I can help with pricing. Our plans start at $499 per month."
        elif any(word in transcription_lower for word in ["feature", "what do"]):
            return "We offer voice recognition and natural language processing. What specific feature interests you?"
        elif any(word in transcription_lower for word in ["help", "support"]):
            return "I'm here to help! What can I assist you with today?"
        else:
            return "I understand you have a question. Could you please rephrase it?"
    
    async def _reset_stream_state(self) -> None:
        """Reset state for new stream."""
        self.audio_processor.reset()
        self.speech_manager.reset()
        self.echo_detection.reset()
        
        self.is_speaking = False
        self.is_processing = False
        self.speech_interrupted = False
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.conversation_active = True
        self.sequence_number = 0
        self.processing_times = []
        self.total_processed_chunks = 0
        self.error_count = 0
    
    async def send_text_response(self, text: str, ws) -> None:
        """Send text response as speech."""
        try:
            self.echo_detection.add_system_response(text)
            await self._send_audio_response_optimized(text, ws)
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
    
    async def _send_fallback_response(self, ws) -> None:
        """Send optimized fallback response."""
        fallback_message = "I'm sorry, could you try again?"
        try:
            speech_audio = await self.kb_processor.generate_speech(fallback_message)
            if speech_audio:
                self.is_speaking = True
                await self._send_audio_optimized(speech_audio, ws)
                await asyncio.sleep(0.1)
                self.is_speaking = False
        except Exception as e:
            logger.error(f"Error sending fallback: {e}")
    
    async def _handle_stop(self, data: Dict[str, Any], ws=None) -> None:
        """Handle stream stop with cleanup."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        
        # Cancel keep-alive
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Stop speech recognition
        await self.speech_manager.stop_session()
        
        # Log performance stats
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            logger.info(f"Session stats - Avg processing: {avg_time:.2f}s, "
                       f"Chunks: {self.total_processed_chunks}, Errors: {self.error_count}")
    
    async def _handle_mark(self, data: Dict[str, Any], ws=None) -> None:
        """Handle mark event for audio sync."""
        mark = data.get('mark', {})
        name = mark.get('name')
        if name:
            logger.debug(f"Mark received: {name}")
    
    async def _handle_ping(self, data: Dict[str, Any], ws=None) -> None:
        """Handle ping response from Twilio."""
        logger.debug("Received ping response from Twilio")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "connected": self.connected,
            "total_chunks_processed": self.total_processed_chunks,
            "error_count": self.error_count,
            "is_processing": self.is_processing,
            "is_speaking": self.is_speaking,
            "average_processing_time": (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0
            ),
            "audio_processor_stats": self.audio_processor.get_stats()
        }