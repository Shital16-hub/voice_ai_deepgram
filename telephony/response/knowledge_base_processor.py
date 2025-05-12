"""
Knowledge base processing and TTS generation.
"""
import logging
import os
from typing import Optional
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    """Handles knowledge base queries and TTS generation."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.elevenlabs_tts = None
    
    async def init_tts(self) -> None:
        """Initialize ElevenLabs TTS if not already initialized."""
        if self.elevenlabs_tts is None:
            try:
                # Get API key from environment
                api_key = os.environ.get("ELEVENLABS_API_KEY")
                voice_id = os.environ.get("TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
                model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")
                
                # Create ElevenLabs TTS client
                self.elevenlabs_tts = ElevenLabsTTS(
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=model_id,
                    container_format="mulaw",
                    sample_rate=8000,
                    optimize_streaming_latency=4
                )
                
                logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}, model ID: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing ElevenLabs TTS: {e}")
    
    async def generate_response(self, transcription: str, user_id: str) -> Optional[str]:
        """
        Generate response from knowledge base.
        
        Args:
            transcription: User's transcription
            user_id: User identifier
            
        Returns:
            Generated response text
        """
        try:
            # Use conversation manager if available
            if hasattr(self.pipeline, 'conversation_manager'):
                response_result = await self.pipeline.conversation_manager.handle_user_input(
                    user_id=user_id,
                    message=transcription
                )
                response = response_result.get("response", "")
            # Fallback to query engine
            elif hasattr(self.pipeline, 'query_engine'):
                query_result = await self.pipeline.query_engine.query(transcription)
                response = query_result.get("response", "")
            else:
                logger.error("No conversation manager or query engine available")
                return None
            
            logger.info(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def generate_speech(self, response: str) -> Optional[bytes]:
        """
        Generate speech audio from text response.
        
        Args:
            response: Text response to convert
            
        Returns:
            Audio bytes or None if error
        """
        try:
            # Try using direct ElevenLabs TTS first
            if self.elevenlabs_tts:
                speech_audio = await self.elevenlabs_tts.synthesize(response)
                logger.info(f"Generated speech with ElevenLabs TTS: {len(speech_audio)} bytes")
                return speech_audio
            
            # Fall back to pipeline's TTS integration
            elif hasattr(self.pipeline, 'tts_integration'):
                speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                
                # Convert to mulaw for Twilio if needed
                from telephony.audio_processor import AudioProcessor
                audio_processor = AudioProcessor()
                mulaw_audio = audio_processor.convert_to_mulaw(speech_audio)
                
                logger.info(f"Generated speech with pipeline TTS: {len(mulaw_audio)} bytes")
                return mulaw_audio
            
            else:
                logger.error("No TTS system available")
                return None
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None