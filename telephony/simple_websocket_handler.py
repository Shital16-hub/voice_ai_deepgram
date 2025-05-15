"""
Fixed simplified WebSocket handler that properly handles Google Cloud STT v2 streaming.
"""
import json
import asyncio
import logging
import base64
import time
from typing import Dict, Any, Optional
import queue
import threading

# Google Cloud Speech v2
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# TTS for responses
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

class SimpleWebSocketHandler:
    """
    Fixed WebSocket handler with proper Google Cloud STT v2 streaming.
    """
    
    def __init__(self, call_sid: str, pipeline):
        """Initialize with direct STT integration."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.pipeline = pipeline
        
        # Google Cloud STT v2 setup
        self.project_id = "my-tts-project-458404"  # Use your project ID
        self.client = SpeechClient()
        
        # STT configuration for telephony
        self.recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                sample_rate_hertz=8000,
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW,
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model="telephony",
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=False,  # Only final results
            ),
        )
        
        # TTS setup
        self.tts_client = GoogleCloudTTS(
            voice_name="en-US-Neural2-C",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True
        )
        
        # State
        self.conversation_active = True
        self.is_speaking = False
        
        # Audio streaming
        self.audio_queue = queue.Queue()
        self.audio_buffer = bytearray()
        self.chunk_size = 800  # 100ms at 8kHz
        
        # STT streaming - FIXED
        self.stt_stream_active = False
        self.stt_thread = None
        self.stop_stt = threading.Event()
        
        # Stats
        self.audio_received = 0
        self.transcriptions = 0
        
        logger.info(f"SimpleWebSocketHandler initialized for call {call_sid}")
    
    async def handle_message(self, message: str, ws) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            event_type = data.get('event')
            
            if event_type == 'connected':
                logger.info("WebSocket connected")
                
            elif event_type == 'start':
                self.stream_sid = data.get('streamSid')
                logger.info(f"Stream started: {self.stream_sid}")
                
                # Start STT streaming properly
                self._start_stt_streaming()
                
                # Send welcome message
                await self._send_response("Hello! How can I help you today?", ws)
                
            elif event_type == 'media':
                await self._handle_audio(data, ws)
                
            elif event_type == 'stop':
                logger.info("Stream stopped")
                self.conversation_active = False
                self._stop_stt_streaming()
                
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    def _start_stt_streaming(self):
        """Start Google Cloud STT streaming in a separate thread."""
        self.stt_stream_active = True
        self.stop_stt.clear()
        
        # Start STT thread
        self.stt_thread = threading.Thread(target=self._run_stt_streaming, daemon=True)
        self.stt_thread.start()
        
        logger.info("Started STT streaming thread")
    
    def _run_stt_streaming(self):
        """Run STT streaming in separate thread to handle blocking calls."""
        try:
            recognizer = f"projects/{self.project_id}/locations/global/recognizers/_"
            
            # Create initial config request
            config_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=self.streaming_config,
            )
            
            # Create request generator
            def request_generator():
                # Send initial config
                yield config_request
                
                # Send audio chunks
                while not self.stop_stt.is_set():
                    try:
                        # Get audio chunk with timeout
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        if audio_chunk is None:  # End signal
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
                        self.audio_queue.task_done()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in request generator: {e}")
                        break
            
            # Start streaming recognize
            responses = self.client.streaming_recognize(request_generator())
            
            # Process responses
            for response in responses:
                if self.stop_stt.is_set():
                    break
                
                for result in response.results:
                    if result.alternatives and result.is_final:
                        transcript = result.alternatives[0].transcript.strip()
                        
                        if transcript and len(transcript) > 2:
                            self.transcriptions += 1
                            logger.info(f"Transcription: '{transcript}'")
                            
                            # Handle transcription in main thread
                            asyncio.run_coroutine_threadsafe(
                                self._handle_transcription(transcript),
                                asyncio.get_event_loop()
                            )
                            
        except Exception as e:
            logger.error(f"Error in STT streaming: {e}", exc_info=True)
        finally:
            self.stt_stream_active = False
            logger.info("STT streaming thread ended")
    
    async def _handle_audio(self, data: Dict[str, Any], ws):
        """Handle incoming audio data."""
        if self.is_speaking:
            return  # Skip audio while speaking
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        # Decode audio
        audio_data = base64.b64decode(payload)
        self.audio_received += 1
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Send chunks when buffer is large enough
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = bytes(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Add to STT queue (non-blocking)
            try:
                self.audio_queue.put(chunk, block=False)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
    
    async def _handle_transcription(self, transcription: str):
        """Handle transcription and generate response."""
        try:
            # Query knowledge base
            if hasattr(self.pipeline, 'query_engine'):
                result = await self.pipeline.query_engine.query(transcription)
                response_text = result.get("response", "")
                
                if response_text:
                    await self._send_response(response_text, None)
                    
        except Exception as e:
            logger.error(f"Error handling transcription: {e}")
    
    async def _send_response(self, text: str, ws):
        """Send TTS response."""
        try:
            self.is_speaking = True
            
            # Convert to speech
            audio_data = await self.tts_client.synthesize(text)
            
            # Send audio in chunks
            await self._send_audio_chunks(audio_data, ws)
            
            self.is_speaking = False
            logger.info(f"Sent response: '{text}'")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.is_speaking = False
    
    async def _send_audio_chunks(self, audio_data: bytes, ws):
        """Send audio data as chunks."""
        if not self.stream_sid or not ws:
            return
        
        chunk_size = 400  # 50ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            audio_base64 = base64.b64encode(chunk).decode('utf-8')
            
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": audio_base64}
            }
            
            try:
                ws.send(json.dumps(message))
                await asyncio.sleep(0.02)  # Small delay between chunks
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break
    
    def _stop_stt_streaming(self):
        """Stop STT streaming."""
        self.conversation_active = False
        self.stop_stt.set()
        
        # Signal end of stream
        try:
            self.audio_queue.put(None, block=False)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self.stt_thread and self.stt_thread.is_alive():
            self.stt_thread.join(timeout=2.0)
        
        logger.info(f"Session ended. Audio: {self.audio_received}, Transcriptions: {self.transcriptions}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "audio_received": self.audio_received,
            "transcriptions": self.transcriptions,
            "is_speaking": self.is_speaking,
            "conversation_active": self.conversation_active,
            "stt_stream_active": self.stt_stream_active
        }