"""
Voice AI Agent main class updated to pass credentials file to STT component.
This ensures both STT and TTS use the same credentials handling approach.
"""
import os
import logging
import asyncio
import time
import json
from typing import Optional, Dict, Any, Union, Callable, Awaitable

# Google Cloud STT imports
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

# Google Cloud TTS imports
from text_to_speech.google_cloud_tts import GoogleCloudTTS

# Ensure project ID is set
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = "my-tts-project-458404"

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class optimized for telephony with consistent credentials handling."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        llm_temperature: float = 0.7,
        credentials_file: Optional[str] = None,  # Added credentials_file parameter
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with Google Cloud STT v2 and TTS.
        Both components now use the same credentials handling approach.
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        
        # Credentials handling - find the file if not provided
        self.credentials_file = credentials_file
        if not self.credentials_file:
            # Try common locations
            possible_paths = [
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                '/workspace/credentials/my-tts-project-458404-8ab56bac7265.json',
                './credentials/my-tts-project-458404-8ab56bac7265.json',
            ]
            for path in possible_paths:
                if path and os.path.exists(path):
                    self.credentials_file = path
                    logger.info(f"Found credentials file: {path}")
                    break
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # TTS Parameters for Google Cloud TTS
        self.tts_voice_name = kwargs.get('tts_voice_name', os.getenv('TTS_VOICE_NAME', 'en-US-Neural2-C'))
        self.tts_voice_gender = kwargs.get('tts_voice_gender', os.getenv('TTS_VOICE_GENDER', 'NEUTRAL'))
        self.tts_language_code = kwargs.get('tts_language_code', os.getenv('TTS_LANGUAGE_CODE', 'en-US'))
        
        # Get project ID from environment or credentials file
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not in environment, try to extract from credentials file
        if not self.project_id and self.credentials_file and os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    self.project_id = creds_data.get('project_id')
                    # Set environment variable for consistency
                    if self.project_id:
                        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
                        logger.info(f"Extracted and set project ID: {self.project_id}")
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
        
        # Set the environment variable for Google Cloud clients
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {self.credentials_file}")
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        logger.info("VoiceAIAgent initialized for telephony with consistent credentials handling")
        
    async def init(self):
        """Initialize all components with optimal telephony settings and consistent credentials."""
        logger.info("Initializing Voice AI Agent components with Google Cloud STT v2 and TTS...")
        
        # Initialize speech recognizer with Google Cloud v2 and explicit credentials
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language=self.stt_language,
            sample_rate=8000,  # Match Twilio exactly
            encoding="MULAW",  # Match Twilio exactly
            channels=1,
            interim_results=False,  # Only final results for better accuracy
            project_id=self.project_id,
            location="global",
            credentials_file=self.credentials_file  # Pass credentials file explicitly
        )
        
        # Initialize STT integration with zero preprocessing
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
        )
        await self.stt_integration.init(project_id=self.project_id)
        
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
        
        # Initialize conversation manager (skip greeting for telephony)
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            skip_greeting=True  # Better for telephony
        )
        await self.conversation_manager.init()
        
        # Initialize Google Cloud TTS with telephony optimization and explicit credentials
        try:
            # Initialize with telephony-optimized settings and explicit credentials
            self.tts_client = GoogleCloudTTS(
                credentials_file=self.credentials_file,  # Pass credentials file explicitly
                voice_name=self.tts_voice_name,
                voice_gender=self.tts_voice_gender,
                language_code=self.tts_language_code,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,  # For Twilio compatibility
                enable_caching=True,
                voice_type="NEURAL2"  # Use Neural2 for best quality
            )
            
            logger.info(f"Initialized Google Cloud TTS with voice: {self.tts_voice_name}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            raise
        
        # Mark as initialized
        self._initialized = True
        logger.info("Voice AI Agent initialization complete (optimized for telephony)")
        
    async def process_audio(
        self,
        audio_data: Union[bytes],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with ZERO preprocessing - pass directly to Google Cloud STT v2.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Pass audio directly to STT with no modifications
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(transcription)
            
            # Generate speech using Google Cloud TTS
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
                    logger.error(f"Error synthesizing speech with Google Cloud TTS: {e}")
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
        return getattr(self, '_initialized', False)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close Google Cloud streaming session if active
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
            await self.speech_recognizer.stop_streaming()
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Mark as not initialized
        self._initialized = False