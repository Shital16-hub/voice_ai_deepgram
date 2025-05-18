"""
Voice AI Agent main class updated for OpenAI + Pinecone integration.
CRITICAL FIXES: Proper initialization order, retry logic, and error handling.
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

class VoiceAIAgentPipeline:
    """FIXED Voice AI Agent class with proper initialization and error handling."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        openai_model: str = 'gpt-4o-mini',
        llm_temperature: float = 0.7,
        credentials_file: Optional[str] = None,
        use_infinite_streaming: bool = True,  # UPDATED: Add infinite streaming option
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with infinite streaming support.
        """
        self.storage_dir = storage_dir
        self.openai_model = openai_model
        self.llm_temperature = llm_temperature
        self.use_infinite_streaming = use_infinite_streaming  # UPDATED: Store option
        
        # Credentials handling (same as before)
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
        
        # CRITICAL FIX: Verify required API keys with retry
        self._verify_api_keys()
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        
        # TTS Parameters for Google Cloud TTS
        self.tts_voice_name = kwargs.get('tts_voice_name', os.getenv('TTS_VOICE_NAME', 'en-US-Neural2-C'))
        self.tts_voice_gender = kwargs.get('tts_voice_gender', os.getenv('TTS_VOICE_GENDER', 'NEUTRAL'))
        self.tts_language_code = kwargs.get('tts_language_code', os.getenv('TTS_LANGUAGE_CODE', 'en-US'))
        
        # Get project ID from environment or credentials file
        self.project_id = self._get_project_id()
        
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
        
        # CRITICAL FIX: Initialization state tracking
        self._initialized = False
        self._initialization_error = None
        
        logger.info(f"VoiceAIAgent initialized with {'infinite streaming' if use_infinite_streaming else 'standard STT'}")
    
    def _verify_api_keys(self):
        """CRITICAL FIX: Verify API keys with proper error messages."""
        # Check OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        # Check Pinecone API key
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        logger.info("✅ All required API keys verified")
    
    def _get_project_id(self) -> str:
        """CRITICAL FIX: Get project ID with better error handling."""
        # Try environment variable first
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            return project_id
        
        # Try to extract from credentials file
        if self.credentials_file and os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        # Set environment variable for consistency
                        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                        logger.info(f"Extracted and set project ID: {project_id}")
                        return project_id
            except Exception as e:
                logger.error(f"Error reading credentials file: {e}")
        
        raise ValueError(
            "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
            "or ensure your credentials file contains a project_id field."
        )
    
    async def init(self):
        """UPDATED: Initialize with infinite streaming support."""
        if self._initialized:
            return
        
        logger.info(f"Initializing Voice AI Agent with {'infinite streaming' if self.use_infinite_streaming else 'standard STT'}...")
        
        try:
            # Set the environment variable for Google Cloud clients
            if self.credentials_file:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
                logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {self.credentials_file}")
            
            # UPDATED: Initialize speech recognizer based on mode
            logger.info("Step 1: Initializing Google Cloud STT...")
            
            if self.use_infinite_streaming:
                # Use infinite streaming implementation
                from speech_to_text.infinite_streaming_stt import InfiniteStreamingSTT
                self.speech_recognizer = InfiniteStreamingSTT(
                    project_id=self.project_id,
                    language=self.stt_language,
                    sample_rate=8000,
                    encoding="MULAW",
                    channels=1,
                    interim_results=True,
                    location="global",
                    credentials_file=self.credentials_file,
                    session_max_duration=240,    # 4 minutes per session
                    session_overlap_seconds=30   # 30 second overlap
                )
                logger.info("✅ Infinite streaming STT initialized")
            else:
                # Use standard implementation
                self.speech_recognizer = GoogleCloudStreamingSTT(
                    language=self.stt_language,
                    sample_rate=8000,
                    encoding="MULAW",
                    channels=1,
                    interim_results=True,  # CRITICAL: Enable for debugging
                    project_id=self.project_id,
                    location="global",
                    credentials_file=self.credentials_file
                )
                logger.info("✅ Standard Google Cloud STT initialized")
            
            # Initialize STT integration
            self.stt_integration = STTIntegration(
                speech_recognizer=self.speech_recognizer,
                language=self.stt_language
            )
            await self.stt_integration.init(project_id=self.project_id)
            logger.info("✅ STT integration initialized")
            
            # CRITICAL FIX: 2. Initialize embeddings with retry
            logger.info("Step 2: Initializing OpenAI embeddings...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.embeddings = OpenAIEmbeddings()
                    # Test embeddings with a simple query
                    test_embedding = await asyncio.wait_for(
                        self.embeddings.embed_text("test"),
                        timeout=10.0
                    )
                    logger.info("✅ OpenAI embeddings initialized and tested")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to initialize OpenAI embeddings after {max_retries} attempts: {e}")
                    logger.warning(f"Embeddings init attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # CRITICAL FIX: 3. Initialize Pinecone with retry logic
            logger.info("Step 3: Initializing Pinecone index...")
            for attempt in range(max_retries):
                try:
                    self.index_manager = IndexManager(embedding_model=self.embeddings)
                    await asyncio.wait_for(self.index_manager.init(), timeout=30.0)
                    logger.info("✅ Pinecone index initialized")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to initialize Pinecone after {max_retries} attempts: {e}")
                    logger.warning(f"Pinecone init attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # CRITICAL FIX: 4. Initialize OpenAI LLM with timeout and proper config
            logger.info("Step 4: Initializing OpenAI LLM...")
            from knowledge_base.config import get_openai_config
            openai_config = get_openai_config()
            openai_config["model"] = self.openai_model
            openai_config["temperature"] = self.llm_temperature
            openai_config["max_tokens"] = 100  # UPDATED: Increased for longer responses
            
            self.llm = OpenAILLM(config=openai_config)
            
            # Test LLM with a simple query
            test_response = await asyncio.wait_for(
                self.llm.generate_response("Hello"),
                timeout=15.0
            )
            logger.info(f"✅ OpenAI LLM initialized and tested: {test_response[:50]}...")
            
            # CRITICAL FIX: 5. Initialize query engine with timeouts
            logger.info("Step 5: Initializing Query Engine...")
            self.query_engine = QueryEngine(
                index_manager=self.index_manager,
                llm=self.llm
            )
            await self.query_engine.init()
            logger.info("✅ Query Engine initialized")
            
            # CRITICAL FIX: 6. Initialize conversation manager (optimized for telephony)
            logger.info("Step 6: Initializing Conversation Manager...")
            self.conversation_manager = ConversationManager(
                query_engine=self.query_engine,
                skip_greeting=True,      # Better for telephony
                max_history_turns=3,     # UPDATED: Include more context
                context_window_tokens=2048  # UPDATED: Increased for better context
            )
            await self.conversation_manager.init()
            logger.info("✅ Conversation Manager initialized")
            
            # CRITICAL FIX: 7. Initialize Google Cloud TTS last
            logger.info("Step 7: Initializing Google Cloud TTS...")
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
                
                # Test TTS with a simple phrase
                test_audio = await asyncio.wait_for(
                    self.tts_client.synthesize("Hello"),
                    timeout=10.0
                )
                logger.info(f"✅ Google Cloud TTS initialized and tested ({len(test_audio)} bytes)")
                
            except Exception as e:
                logger.error(f"Error initializing Google Cloud TTS: {e}")
                raise
            
            # Mark as initialized
            self._initialized = True
            logger.info(f"🎉 Voice AI Agent initialization complete with {'infinite streaming' if self.use_infinite_streaming else 'standard STT'}")
            
            # CRITICAL FIX: Log final status
            await self._log_initialization_status()
            
        except Exception as e:
            self._initialization_error = e
            logger.error(f"❌ CRITICAL ERROR during initialization: {e}")
            raise RuntimeError(f"Failed to initialize Voice AI Agent: {e}")
    
    async def _log_initialization_status(self):
        """Log final initialization status for debugging."""
        try:
            # Get knowledge base stats
            kb_stats = await self.get_knowledge_base_stats()
            doc_count = kb_stats.get("index_stats", {}).get("vector_store", {}).get("total_vectors", 0)
            
            logger.info(f"📊 Final Status:")
            logger.info(f"  - Documents in knowledge base: {doc_count}")
            logger.info(f"  - OpenAI model: {self.openai_model}")
            logger.info(f"  - TTS voice: {self.tts_voice_name}")
            logger.info(f"  - Streaming mode: {'Infinite' if self.use_infinite_streaming else 'Standard'}")
            logger.info(f"  - Ready for telephony calls")
            
        except Exception as e:
            logger.warning(f"Could not log final status: {e}")
    
    async def process_audio(
        self,
        audio_data: Union[bytes],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Enhanced process audio with support for infinite streaming."""
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized. Call init() first.")
        
        try:
            # Check if we're using infinite streaming
            is_infinite_streaming = hasattr(self.speech_recognizer, 'process_audio_chunk') and \
                                 not hasattr(self.speech_recognizer, 'stopped') and \
                                 hasattr(self.speech_recognizer, 'is_streaming')
            
            # If using infinite streaming, ensure it's started
            if is_infinite_streaming and not self.speech_recognizer.is_streaming:
                logger.info("Starting infinite streaming session")
                await self.speech_recognizer.start_streaming()
                
            # Pass audio to STT integration
            result = await asyncio.wait_for(
                self.stt_integration.transcribe_audio_data(audio_data, callback=callback),
                timeout=30.0
            )
            
            # Only process valid transcriptions
            if result.get("is_valid", False) and result.get("transcription"):
                transcription = result["transcription"]
                logger.info(f"Valid transcription: {transcription}")
                
                # Process through conversation manager
                response = await asyncio.wait_for(
                    self.conversation_manager.handle_user_input(transcription),
                    timeout=25.0
                )
                
                # Generate speech using TTS
                if response and response.get("response"):
                    try:
                        speech_audio = await asyncio.wait_for(
                            self.tts_client.synthesize(response["response"]),
                            timeout=10.0
                        )
                        return {
                            "transcription": transcription,
                            "response": response.get("response", ""),
                            "speech_audio": speech_audio,
                            "status": "success",
                            "engine": "openai_pinecone",
                            "streaming_mode": "infinite" if is_infinite_streaming else "standard"
                        }
                    except Exception as e:
                        logger.error(f"Error synthesizing speech: {e}")
                        return {
                            "transcription": transcription,
                            "response": response.get("response", ""),
                            "error": f"Speech synthesis error: {str(e)}",
                            "status": "tts_error",
                            "engine": "openai_pinecone",
                            "streaming_mode": "infinite" if is_infinite_streaming else "standard"
                        }
                else:
                    return {
                        "transcription": transcription,
                        "response": response.get("response", ""),
                        "status": "success",
                        "engine": "openai_pinecone",
                        "streaming_mode": "infinite" if is_infinite_streaming else "standard"
                    }
            else:
                logger.info("Invalid or empty transcription")
                return {
                    "status": "invalid_transcription",
                    "transcription": result.get("transcription", ""),
                    "error": "No valid speech detected",
                    "engine": "openai_pinecone",
                    "streaming_mode": "infinite" if is_infinite_streaming else "standard"
                }
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "status": "error",
                "error": str(e),
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
        
        # Add to Pinecone index with timeout
        doc_ids = await asyncio.wait_for(
            self.index_manager.add_documents(documents),
            timeout=120.0  # 2 minutes for document indexing
        )
        
        logger.info(f"Added {len(doc_ids)} documents to OpenAI + Pinecone knowledge base")
        return doc_ids
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics with error handling."""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            stats = await asyncio.wait_for(
                self.query_engine.get_stats(),
                timeout=10.0
            )
            stats["engine_type"] = "openai_pinecone"
            stats["initialization_status"] = "completed"
            stats["streaming_mode"] = "infinite" if self.use_infinite_streaming else "standard"
            return stats
        except Exception as e:
            logger.error(f"Error getting KB stats: {e}")
            return {
                "error": str(e),
                "engine_type": "openai_pinecone",
                "initialization_status": "error",
                "streaming_mode": "infinite" if self.use_infinite_streaming else "standard"
            }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return self._initialized and not self._initialization_error
    
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