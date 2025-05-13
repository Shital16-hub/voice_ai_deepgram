"""
Response generation and TTS handling.
"""
import logging
import asyncio
import base64
import json
from typing import Optional

from text_to_speech import ElevenLabsTTS
from telephony.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles response generation and text-to-speech conversion."""
    
    def __init__(self, pipeline, ws_handler):
        """Initialize response generator."""
        self.pipeline = pipeline
        self.ws_handler = ws_handler  # Reference to get stream_sid
        self.elevenlabs_tts = None
        self.audio_processor = AudioProcessor()
        self._initialize_tts()
    
    def _initialize_tts(self) -> None:
        """Initialize ElevenLabs TTS client."""
        try:
            import os
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            voice_id = os.environ.get("TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
            model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")
            
            self.elevenlabs_tts = ElevenLabsTTS(
                api_key=api_key,
                voice_id=voice_id,
                model_id=model_id,
                container_format="mulaw",
                sample_rate=8000,
                optimize_streaming_latency=4
            )
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
    
    async def generate_response(self, transcription: str) -> Optional[str]:
        """
        Generate response from knowledge base.
        
        Args:
            transcription: User transcription
            
        Returns:
            Generated response text
        """
        try:
            if hasattr(self.pipeline, 'query_engine'):
                query_result = await self.pipeline.query_engine.query(transcription)
                return query_result.get("response", "")
            return None
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def convert_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as bytes
        """
        try:
            if self.elevenlabs_tts:
                return await self.elevenlabs_tts.synthesize(text)
            else:
                # Fallback to pipeline TTS
                return await self.pipeline.tts_integration.text_to_speech(text)
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            raise
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send text response by converting to speech and sending to WebSocket.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # Convert to speech
            speech_audio = await self.convert_to_speech(text)
            
            # Convert to mulaw if needed
            if not self.elevenlabs_tts or self.elevenlabs_tts.container_format != "mulaw":
                mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
            else:
                mulaw_audio = speech_audio
            
            # Send audio
            await self._send_audio(mulaw_audio, ws)
            
            logger.info(f"Sent text response: '{text}'")
            
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to WebSocket."""
        try:
            # Get stream_sid from the handler
            stream_sid = self.ws_handler.stream_sid
            
            if not stream_sid:
                logger.error("No stream_sid available for sending audio")
                return
            
            # Split into chunks
            chunks = self._split_audio_into_chunks(audio_data)
            
            logger.debug(f"Sending {len(chunks)} audio chunks with stream_sid: {stream_sid}")
            
            for i, chunk in enumerate(chunks):
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                try:
                    ws.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending chunk {i}: {e}")
                    if "Connection closed" in str(e):
                        logger.warning("WebSocket connection closed during audio send")
                        return
            
            logger.debug(f"Successfully sent {len(chunks)} audio chunks")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            if "Connection closed" in str(e):
                logger.warning("WebSocket connection closed during audio send")
    
    def _split_audio_into_chunks(self, audio_data: bytes) -> list:
        """Split audio into smaller chunks."""
        chunk_size = 800  # 100ms at 8kHz
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
            
        return chunks