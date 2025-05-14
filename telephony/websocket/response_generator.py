"""
Enhanced response generation with temporal echo tracking integration.
"""
import logging
import asyncio
import base64
import json
from typing import Optional

from text_to_speech import GoogleCloudTTS
from telephony.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Enhanced response generator with temporal echo tracking integration."""
    
    def __init__(self, pipeline, ws_handler):
        """Initialize response generator with echo tracking."""
        self.pipeline = pipeline
        self.ws_handler = ws_handler
        self.audio_processor = AudioProcessor()
        self.tts_client = None
        self._initialize_tts()
    
    def _initialize_tts(self) -> None:
        """Initialize Google Cloud TTS client with optimized settings."""
        try:
            import os
            
            # Get voice configuration from environment or use defaults
            voice_type = os.getenv("TTS_VOICE_TYPE", "NEURAL2")
            voice_name = os.getenv("TTS_VOICE_NAME", None)
            voice_gender = os.getenv("TTS_VOICE_GENDER", "NEUTRAL")
            language_code = os.getenv("TTS_LANGUAGE_CODE", "en-US")
            
            # Auto-select voice if not specified
            if not voice_name:
                from text_to_speech.config import get_recommended_voice
                voice_name = get_recommended_voice(language_code, voice_type, voice_gender)
            
            # Initialize Google Cloud TTS with telephony optimization
            self.tts_client = GoogleCloudTTS(
                voice_name=voice_name,
                voice_gender=voice_gender,
                language_code=language_code,
                voice_type=voice_type,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,          # Match Twilio's rate
                enable_caching=True
            )
            
            logger.info(f"Initialized Google Cloud TTS with {voice_type} voice: {voice_name}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            # Will fall back to pipeline TTS if available
    
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
        Convert text to speech audio using Google Cloud TTS.
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as bytes (mulaw format)
        """
        try:
            if self.tts_client:
                # Use Google Cloud TTS (already returns mulaw format)
                return await self.tts_client.synthesize(text)
            else:
                # Fallback to pipeline TTS if Google Cloud TTS is not available
                if hasattr(self.pipeline, 'tts_integration'):
                    speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                    # Convert to mulaw for Twilio if needed
                    return self.audio_processor.convert_to_mulaw_direct(speech_audio)
                else:
                    raise Exception("No TTS client available")
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            raise
    
    async def send_text_response(self, text: str, ws) -> None:
        """
        Send text response with echo tracking integration.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        try:
            # IMPORTANT: Register TTS output for echo detection BEFORE sending
            if hasattr(self.ws_handler, 'speech_processor'):
                # Register the text that will be spoken for temporal echo detection
                self.ws_handler.speech_processor.register_tts_output(text)
                # Also add to traditional echo history
                self.ws_handler.speech_processor.add_to_echo_history(text)
            
            # Set speaking state
            self.ws_handler.audio_manager.set_speaking_state(True)
            
            # Convert to speech (already in mulaw format)
            mulaw_audio = await self.convert_to_speech(text)
            
            # Send audio
            await self._send_audio(mulaw_audio, ws)
            
            logger.info(f"Sent response with echo tracking: '{text}'")
            
            # Update state
            self.ws_handler.audio_manager.set_speaking_state(False)
            self.ws_handler.audio_manager.update_response_time()
            
        except Exception as e:
            logger.error(f"Error sending text response: {e}")
            # Ensure speaking state is reset
            self.ws_handler.audio_manager.set_speaking_state(False)
    
    async def _send_audio(self, audio_data: bytes, ws) -> None:
        """Send audio data to WebSocket with optimized chunking."""
        try:
            stream_sid = self.ws_handler.stream_sid
            
            if not stream_sid:
                logger.error("No stream_sid available for sending audio")
                return
            
            # Split into optimal chunks for streaming
            chunks = self._split_audio_into_chunks(audio_data)
            
            logger.debug(f"Sending {len(chunks)} audio chunks")
            
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
                    # Optimal delay between chunks for smooth playback
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.02)
                except Exception as e:
                    logger.error(f"Error sending chunk {i}: {e}")
                    if "Connection closed" in str(e):
                        logger.warning("WebSocket connection closed during audio send")
                        return
            
            logger.debug(f"Successfully sent {len(chunks)} audio chunks")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
    
    def _split_audio_into_chunks(self, audio_data: bytes) -> list:
        """Split audio into optimal chunks for Twilio streaming."""
        # Optimal chunk size for 8kHz mulaw (100ms)
        chunk_size = 800
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
            
        return chunks
    
    def get_tts_info(self) -> dict:
        """Get information about the current TTS configuration."""
        if self.tts_client:
            return {
                "provider": "Google Cloud TTS",
                "voice_type": getattr(self.tts_client, 'voice_type', 'Unknown'),
                "voice_name": getattr(self.tts_client, 'voice_name', 'Unknown'),
                "language_code": getattr(self.tts_client, 'language_code', 'Unknown'),
                "sample_rate": 8000,
                "format": "mulaw"
            }
        else:
            return {
                "provider": "Pipeline TTS (fallback)",
                "status": "Google Cloud TTS not available"
            }