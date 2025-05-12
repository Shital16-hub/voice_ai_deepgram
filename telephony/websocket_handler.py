"""
Fixed WebSocket handler with improved error handling and latency optimization.
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
    Optimized WebSocket handler with better error handling and reduced latency.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize WebSocket handler."""
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
        
        # Processing locks and buffers
        self.processing_lock = asyncio.Lock()
        self.keep_alive_task = None
        
        # Timing and throttling
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0
        
        # Conversation flow management
        self.pause_after_response = 0.2  # Reduced from 0.3
        self.min_words_for_valid_query = 1
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_history = 10
        
        logger.info(f"WebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """Handle incoming WebSocket message with better error handling."""
        if not message:
            logger.warning("Received empty message")
            return
            
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            logger.debug(f"Received WebSocket event: {event_type}")
            
            # Route to appropriate handler
            handlers = {
                'connected': self._handle_connected,
                'start': self._handle_start,
                'media': self._handle_media,
                'stop': self._handle_stop,
                'mark': self._handle_mark
            }
            
            handler = handlers.get(event_type)
            if handler:
                await handler(data, ws)
            else:
                logger.warning(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle connected event."""
        logger.info(f"WebSocket connected for call {self.call_sid}")
        
        # Update connection state
        self.connected = True
        self.connection_active.set()
        
        # Start keep-alive
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        
        # Reset all state for new stream
        await self._reset_stream_state()
        
        # Initialize speech recognition
        await self.speech_manager.start_session()
        
        # Initialize TTS
        await self.kb_processor.init_tts()
        
        # Send welcome message quickly
        await self.send_text_response("Welcome! How can I help you today?", ws)
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event with audio data - optimized for latency."""
        if not self.conversation_active:
            return
            
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return
        
        try:
            # Decode and process audio
            audio_data = base64.b64decode(payload)
            
            # Skip if system is speaking to prevent echo
            if self.is_speaking:
                return
            
            # Process through audio component
            processed_data = self.audio_processor.process_incoming_audio(audio_data)
            
            if processed_data is None:
                return  # Still buffering
            
            # Check timing since last response
            if self._in_pause_period():
                return
            
            # Process audio buffer when ready (optimized)
            if self.audio_processor.should_process_buffer() and not self.is_processing:
                # Use asyncio.create_task for better concurrency
                asyncio.create_task(self._process_audio_buffer_optimized(ws))
            
        except Exception as e:
            logger.error(f"Error processing media payload: {e}", exc_info=True)
    
    async def _process_audio_buffer_optimized(self, ws) -> None:
        """Optimized audio buffer processing with timeout and better error handling."""
        async with self.processing_lock:
            if self.is_processing:
                return
                
            self.is_processing = True
            process_start = time.time()
            
            try:
                # Get audio from processor with timeout
                audio_task = asyncio.create_task(self._get_audio_for_processing())
                try:
                    audio_data = await asyncio.wait_for(audio_task, timeout=0.5)
                except asyncio.TimeoutError:
                    logger.warning("Audio processing timeout")
                    return
                
                if audio_data is None or len(audio_data) == 0:
                    return
                
                # Process through speech recognition with timeout
                stt_start = time.time()
                try:
                    transcription_task = asyncio.create_task(
                        self.speech_manager.process_audio(audio_data)
                    )
                    transcription = await asyncio.wait_for(transcription_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("STT timeout")
                    return
                
                stt_time = time.time() - stt_start
                
                # Check if this is an echo
                if self.echo_detection.is_echo(transcription):
                    logger.info("Detected echo of system speech, ignoring")
                    return
                
                # Validate transcription
                if not self.speech_manager.is_valid_transcription(transcription):
                    logger.debug(f"Invalid transcription: '{transcription}'")
                    return
                
                # Avoid duplicates
                if transcription == self.last_transcription:
                    logger.info("Duplicate transcription, not processing again")
                    return
                
                logger.info(f"Complete transcription: {transcription}")
                
                # Process through knowledge base with timeout
                kb_start = time.time()
                try:
                    response_task = asyncio.create_task(
                        self.kb_processor.generate_response(transcription, self.call_sid)
                    )
                    response = await asyncio.wait_for(response_task, timeout=3.0)  # Increased to 3s
                except asyncio.TimeoutError:
                    logger.warning("Knowledge base timeout, using fallback")
                    response = self._get_fallback_response(transcription)
                
                kb_time = time.time() - kb_start
                
                if response:
                    # Track for echo detection
                    self.echo_detection.add_system_response(response)
                    
                    # Convert to speech and send
                    await self._send_audio_response_optimized(response, ws)
                    
                    # Update state
                    self.last_transcription = transcription
                    self.last_response_time = time.time()
                
                # Track performance
                total_time = time.time() - process_start
                self.processing_times.append(total_time)
                if len(self.processing_times) > self.max_processing_history:
                    self.processing_times.pop(0)
                
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logger.info(f"Processing completed in {total_time:.2f}s (avg: {avg_time:.2f}s, STT: {stt_time:.2f}s, KB: {kb_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
                # Send fallback response
                await self._send_fallback_response(ws)
            finally:
                self.is_processing = False
    
    async def _get_audio_for_processing(self) -> Optional[np.ndarray]:
        """Get audio data for processing (async)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.audio_processor.get_audio_for_processing
        )
    
    async def _send_audio_response_optimized(self, response: str, ws) -> None:
        """Optimized audio response generation and sending."""
        try:
            # Set speaking state immediately to prevent processing more audio
            self.is_speaking = True
            
            # Generate speech with timeout
            speech_start = time.time()
            try:
                speech_task = asyncio.create_task(
                    self.kb_processor.generate_speech(response)
                )
                speech_audio = await asyncio.wait_for(speech_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("TTS timeout")
                await self._send_fallback_response(ws)
                return
            
            speech_time = time.time() - speech_start
            
            if speech_audio:
                # Send audio
                await self._send_audio(speech_audio, ws)
                logger.info(f"Sent audio response ({speech_time:.2f}s) for: '{response[:50]}...'")
            else:
                logger.error("No speech audio generated")
                await self._send_fallback_response(ws)
                
        except Exception as e:
            logger.error(f"Error sending audio response: {e}")
            await self._send_fallback_response(ws)
        finally:
            # Reset speaking state
            self.is_speaking = False
    
    def _get_fallback_response(self, transcription: str) -> str:
        """Get appropriate fallback response based on transcription."""
        transcription_lower = transcription.lower()
        
        if any(word in transcription_lower for word in ["price", "pricing", "cost", "plan"]):
            return "I understand you're asking about pricing. Let me help you with that information."
        elif any(word in transcription_lower for word in ["feature", "features", "capability"]):
            return "I can tell you about our features. What specific functionality would you like to know about?"
        elif any(word in transcription_lower for word in ["help", "support"]):
            return "I'm here to help! What specific question do you have?"
        else:
            return "I understand you have a question. Could you please rephrase it?"
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to Twilio with optimizations."""
        if not self.connected:
            logger.warning("WebSocket connection is closed, cannot send audio")
            return
        
        try:
            # Convert and split audio if needed
            processed_audio = self.audio_processor.prepare_outgoing_audio(audio_data)
            chunks = self.audio_processor.split_audio_for_streaming(processed_audio)
            
            logger.debug(f"Sending {len(chunks)} audio chunks ({len(processed_audio)} bytes total)")
            
            # Send chunks in parallel for better performance
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                # Create task for sending chunk
                chunk_task = asyncio.create_task(
                    self._send_chunk(ws, json.dumps(message), i+1, len(chunks))
                )
                chunk_tasks.append(chunk_task)
            
            # Wait for all chunks to be sent
            await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error preparing audio for sending: {e}")
    
    async def _send_chunk(self, ws, message: str, chunk_num: int, total_chunks: int) -> None:
        """Send a single audio chunk."""
        try:
            ws.send(message)
        except Exception as e:
            logger.error(f"Error sending audio chunk {chunk_num}/{total_chunks}: {e}")
            if "Connection closed" in str(e):
                self.connected = False
                self.connection_active.clear()
    
    async def _keep_alive_loop(self, ws) -> None:
        """Optimized keep-alive loop."""
        try:
            while self.conversation_active:
                await asyncio.sleep(5)  # Reduced from 10 for more responsive connection monitoring
                
                if not self.stream_sid or not self.connected:
                    continue
                    
                try:
                    message = {
                        "event": "ping",
                        "streamSid": self.stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        self.conversation_active = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")
    
    def _in_pause_period(self) -> bool:
        """Check if we're in a pause period after last response."""
        time_since_last_response = time.time() - self.last_response_time
        return time_since_last_response < self.pause_after_response
    
    async def _reset_stream_state(self) -> None:
        """Reset all state for a new stream."""
        self.audio_processor.reset()
        self.speech_manager.reset()
        self.echo_detection.reset()
        
        self.is_speaking = False
        self.is_processing = False
        self.speech_interrupted = False
        self.last_transcription = ""
        self.last_response_time = time.time()
        self.last_audio_output_time = 0
        self.conversation_active = True
        self.sequence_number = 0
        self.processing_times = []
    
    async def send_text_response(self, text: str, ws) -> None:
        """Send a text response by converting to speech."""
        try:
            # Track for echo detection
            self.echo_detection.add_system_response(text)
            
            # Generate and send speech
            await self._send_audio_response_optimized(text, ws)
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
    
    async def _send_fallback_response(self, ws) -> None:
        """Send a fallback response when errors occur."""
        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
        try:
            # Try to generate and send fallback speech directly
            speech_audio = await self.kb_processor.generate_speech(fallback_message)
            if speech_audio:
                self.is_speaking = True
                await self._send_audio(speech_audio, ws)
                self.is_speaking = False
            else:
                logger.error("Could not generate fallback speech")
        except Exception as e:
            logger.error(f"Failed to send fallback response: {e}")
    
    async def _handle_stop(self, data: Dict[str, Any], ws=None) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.conversation_active = False
        self.connected = False
        self.connection_active.clear()
        
        # Cleanup
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
        
        await self.speech_manager.stop_session()
    
    async def _handle_mark(self, data: Dict[str, Any], ws=None) -> None:
        """Handle mark event for audio playback tracking."""
        mark = data.get('mark', {})
        name = mark.get('name')
        logger.debug(f"Mark received: {name}")