"""
Voice AI Agent main class updated for OpenAI + Pinecone integration.
Replaces Ollama + Chroma with OpenAI + Pinecone for better telephony performance.
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

# OpenAI + Pinecone knowledge base imports
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.document_store import DocumentStore
from knowledge_base.index_manager import IndexManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.openai_embeddings import OpenAIEmbeddings
from knowledge_base.openai_llm import OpenAILLM

# Google Cloud TTS imports
from text_to_speech.google_cloud_tts import GoogleCloudTTS

# Ensure project ID is set
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = "my-tts-project-458404"

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class using OpenAI + Pinecone for ultra-low latency telephony."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        openai_model: str = 'gpt-4o-mini',  # Fast OpenAI model
        llm_temperature: float = 0.3,  # Lower for faster, more consistent responses
        credentials_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with OpenAI + Pinecone.
        """
        self.storage_dir = storage_dir
        self.openai_model = openai_model
        self.llm_temperature = llm_temperature
        
        # Credentials handling
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
        
        # Verify required API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
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
        
        # OpenAI + Pinecone components
        self.embeddings = None
        self.index_manager = None
        self.llm = None
        
        logger.info("VoiceAIAgent initialized for telephony with OpenAI + Pinecone")
        
    async def init(self):
        """Initialize all components with OpenAI + Pinecone integration."""
        logger.info("Initializing Voice AI Agent with OpenAI + Pinecone...")
        
        # Initialize speech recognizer with Google Cloud v2
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language=self.stt_language,
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=self.project_id,
            location="global",
            credentials_file=self.credentials_file
        )
        
        # Initialize STT integration
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
        )
        await self.stt_integration.init(project_id=self.project_id)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone index manager
        self.index_manager = IndexManager(embedding_model=self.embeddings)
        await self.index_manager.init()
        
        # Initialize OpenAI LLM
        from knowledge_base.config import get_openai_config
        openai_config = get_openai_config()
        openai_config["model"] = self.openai_model
        openai_config["temperature"] = self.llm_temperature
        
        self.llm = OpenAILLM(config=openai_config)
        
        # Initialize query engine with OpenAI + Pinecone
        self.query_engine = QueryEngine(
            index_manager=self.index_manager,
            llm=self.llm
        )
        await self.query_engine.init()
        
        # Initialize conversation manager (skip greeting for telephony)
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            skip_greeting=True,  # Better for telephony
            max_history_turns=4,  # Shorter for telephony
            context_window_tokens=2048  # Optimized for telephony
        )
        await self.conversation_manager.init()
        
        # Initialize Google Cloud TTS
        try:
            self.tts_client = GoogleCloudTTS(
                credentials_file=self.credentials_file,
                voice_name=self.tts_voice_name,
                voice_gender=self.tts_voice_gender,
                language_code=self.tts_language_code,
                container_format="mulaw",
                sample_rate=8000,
                enable_caching=True,
                voice_type="NEURAL2"
            )
            
            logger.info(f"Initialized Google Cloud TTS with voice: {self.tts_voice_name}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS: {e}")
            raise
        
        # Mark as initialized
        self._initialized = True
        logger.info("Voice AI Agent initialization complete with OpenAI + Pinecone")
        
    async def process_audio(
        self,
        audio_data: Union[bytes],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with zero preprocessing - optimized for OpenAI + Pinecone.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Pass audio directly to STT with no modifications
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager (uses OpenAI + Pinecone)
            response = await self.conversation_manager.handle_user_input(transcription)
            
            # Generate speech using Google Cloud TTS
            if response and response.get("response"):
                try:
                    speech_audio = await self.tts_client.synthesize(response["response"])
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "speech_audio": speech_audio,
                        "status": "success",
                        "engine": "openai_pinecone"
                    }
                except Exception as e:
                    logger.error(f"Error synthesizing speech: {e}")
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "error": f"Speech synthesis error: {str(e)}",
                        "status": "tts_error",
                        "engine": "openai_pinecone"
                    }
            else:
                return {
                    "transcription": transcription,
                    "response": response.get("response", ""),
                    "status": "success",
                    "engine": "openai_pinecone"
                }
        else:
            logger.info("Invalid or empty transcription")
            return {
                "status": "invalid_transcription",
                "transcription": result.get("transcription", ""),
                "error": "No valid speech detected",
                "engine": "openai_pinecone"
            }
    
    async def add_documents(self, documents_directory: str):
        """
        Add documents to the knowledge base using OpenAI + Pinecone.
        
        Args:
            documents_directory: Directory containing documents to index
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Load documents
        doc_store = DocumentStore()
        documents = doc_store.load_documents_from_directory(documents_directory)
        
        # Add to Pinecone index
        doc_ids = await self.index_manager.add_documents(documents)
        
        logger.info(f"Added {len(doc_ids)} documents to OpenAI + Pinecone knowledge base")
        return doc_ids
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        stats = await self.query_engine.get_stats()
        stats["engine_type"] = "openai_pinecone"
        return stats
    
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
        
        logger.info("Voice AI Agent shutdown complete")