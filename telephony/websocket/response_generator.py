"""
Response generation and TTS handling.
"""
import logging
import os
from typing import Optional, Dict, Any, List, Callable, Awaitable

from text_to_speech import ElevenLabsTTS
from telephony.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles knowledge base queries and TTS generation."""
    
    def __init__(self, pipeline):
        """Initialize the response generator."""
        self.pipeline = pipeline
        self.audio_processor = AudioProcessor()
        
        # Initialize ElevenLabs TTS
        self.elevenlabs_tts = None
        self._init_elevenlabs_tts()
    
    def _init_elevenlabs_tts(self):
        """Initialize ElevenLabs TTS client."""
        try:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            voice_id = os.environ.get("TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
            model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")
            
            self.elevenlabs_tts = ElevenLabsTTS(
                api_key=api_key,
                voice_id=voice_id,
                model_id=model_id,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,  # For Twilio compatibility
                optimize_streaming_latency=4  # Maximum optimization
            )
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}, model ID: {model_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            # Will fall back to pipeline TTS integration
    
    async def generate_response(self, transcription: str) -> Dict[str, Any]:
        """Generate response from knowledge base."""
        try:
            if hasattr(self.pipeline, 'query_engine'):
                query_result = await self.pipeline.query_engine.query(transcription)
                response = query_result.get("response", "")
                
                logger.info(f"Generated response: {response}")
                return {
                    "text": response,
                    "sources": query_result.get("sources", []),
                    "success": True
                }
            else:
                logger.error("Query engine not available in pipeline")
                return {
                    "text": "I'm sorry, I'm having trouble processing your request.",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "text": "I'm sorry, I'm having trouble understanding. Could you try again?",
                "success": False,
                "error": str(e)
            }
    
    async def text_to_speech(self, text: str) -> tuple[bytes, Optional[str]]:
        """Convert text to speech."""
        try:
            # Try using direct ElevenLabs TTS first
            if self.elevenlabs_tts:
                try:
                    speech_audio = await self.elevenlabs_tts.synthesize(text)
                    logger.info(f"Generated speech with ElevenLabs TTS: {len(speech_audio)} bytes")
                    
                    # Convert to mulaw for Twilio if needed
                    if self.elevenlabs_tts.container_format != "mulaw":
                        mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                    else:
                        mulaw_audio = speech_audio
                    
                    return mulaw_audio, None
                except Exception as e:
                    logger.error(f"Error with ElevenLabs TTS, falling back to pipeline TTS: {e}")
            
            # Fall back to pipeline's TTS integration
            if hasattr(self.pipeline, 'tts_integration'):
                speech_audio = await self.pipeline.tts_integration.text_to_speech(text)
                
                # Convert to mulaw for Twilio
                mulaw_audio = self.audio_processor.convert_to_mulaw(speech_audio)
                
                logger.info(f"Generated speech with pipeline TTS: {len(mulaw_audio)} bytes")
                return mulaw_audio, None
            else:
                logger.error("TTS integration not available")
                return b'', "TTS not available"
                
        except Exception as e:
            logger.error(f"Error in text-to-speech synthesis: {e}")
            return b'', str(e)
    
    async def generate_fallback_response(self) -> bytes:
        """Generate a fallback response when main processing fails."""
        fallback_message = "I'm sorry, I'm having trouble understanding. Could you try again?"
        
        try:
            # Try using ElevenLabs TTS
            if self.elevenlabs_tts:
                fallback_audio = await self.elevenlabs_tts.synthesize(fallback_message)
                
                # Convert to mulaw if needed
                if self.elevenlabs_tts.container_format != "mulaw":
                    return self.audio_processor.convert_to_mulaw(fallback_audio)
                else:
                    return fallback_audio
            
            # Fall back to pipeline TTS
            elif hasattr(self.pipeline, 'tts_integration'):
                fallback_audio = await self.pipeline.tts_integration.text_to_speech(fallback_message)
                return self.audio_processor.convert_to_mulaw(fallback_audio)
            
            else:
                logger.error("No TTS available for fallback response")
                return b''
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return b''
    
    async def send_welcome_message(self, ws, stream_sid: str) -> None:
        """Send a welcome message when the call starts."""
        welcome_text = "I'm listening. How can I help you today?"
        
        try:
            # Generate welcome audio
            welcome_audio, error = await self.text_to_speech(welcome_text)
            
            if welcome_audio and not error:
                # Send welcome message through WebSocket
                await self._send_audio_through_ws(welcome_audio, ws, stream_sid)
            else:
                logger.error(f"Failed to generate welcome message: {error}")
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
    
    async def _send_audio_through_ws(self, audio_data: bytes, ws, stream_sid: str):
        """Send audio data through WebSocket."""
        import json
        import base64
        
        # Split audio into chunks
        chunks = self._split_audio_into_chunks(audio_data)
        
        for chunk in chunks:
            try:
                # Encode audio to base64
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                # Create media message
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                # Send message
                ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break
    
    def _split_audio_into_chunks(self, audio_data: bytes, chunk_size: int = 800) -> List[bytes]:
        """Split audio into smaller chunks for streaming."""
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        return chunks