# voice_ai_agent.py

"""
Voice AI Agent main class optimized for Twilio telephony with Google Cloud STT v2.32.0+
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

# Import the updated Google Cloud STT
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

# ElevenLabs TTS imports
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class optimized for Twilio telephony."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent optimized for telephony.
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        
        # STT Parameters - optimized for Twilio
        self.language = kwargs.get('language', 'en-US')
        self.enhanced_model = kwargs.get('enhanced_model', True)
        
        # TTS Parameters for ElevenLabs
        self.elevenlabs_api_key = kwargs.get('elevenlabs_api_key', os.getenv('ELEVENLABS_API_KEY'))
        self.elevenlabs_voice_id = kwargs.get('elevenlabs_voice_id', os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'))
        self.elevenlabs_model_id = kwargs.get('elevenlabs_model_id', os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2'))
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
    
    # Remove the noise floor and audio processing methods - we don't want to modify audio
    
    async def init(self):
        """Initialize all components optimized for telephony."""
        logger.info("Initializing Voice AI Agent for telephony with Google Cloud STT v2.32.0+...")
        
        # Initialize speech recognizer with optimal telephony settings
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language=self.language,
            sample_rate=8000,  # Match Twilio's 8kHz
            encoding="MULAW",   # Match Twilio's mulaw format
            channels=1,
            interim_results=False,  # Disable for lower latency
            speech_context_phrases=None,  # Don't hardcode - let the model decide
            enhanced_model=self.enhanced_model
        )
        
        # Initialize STT integration 
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.language[:2]
        )
        
        # Initialize document store and index manager
        doc_store = DocumentStore()
        index_manager = IndexManager(storage_dir=self.storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            index_manager=index_manager, 
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature
        )
        await self.query_engine.init()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            skip_greeting=True  # Skip greeting for better phone experience
        )
        await self.conversation_manager.init()
        
        # Initialize ElevenLabs TTS
        try:
            if not self.elevenlabs_api_key:
                raise ValueError("ElevenLabs API key is required")
                
            self.tts_client = ElevenLabsTTS(
                api_key=self.elevenlabs_api_key,
                voice_id=self.elevenlabs_voice_id,
                model_id=self.elevenlabs_model_id,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,  # For Twilio compatibility
                optimize_streaming_latency=3
            )
            
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {self.elevenlabs_voice_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            raise
        
        logger.info("Voice AI Agent initialization complete")
        
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data without any preprocessing - let Google handle it.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Don't preprocess audio - pass it directly to STT
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            # Generate speech using ElevenLabs TTS
            if response and response.get("response"):
                try:
                    speech_audio = await self.tts_client.synthesize(response["response"])
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "speech_audio": speech_audio,
                        "status": "success"
                    }
                except Exception as e:
                    logger.error(f"Error synthesizing speech: {e}")
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "error": f"Speech synthesis error: {str(e)}",
                        "status": "tts_error"
                    }
            else:
                return {
                    "transcription": transcription,
                    "response": response.get("response", ""),
                    "status": "success"
                }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected"
            }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None and
                self.tts_client is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close Google Cloud streaming session if active
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
            await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()