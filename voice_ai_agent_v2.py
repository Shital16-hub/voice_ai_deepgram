# voice_ai_agent_v2.py

"""
Improved Voice AI Agent using Google Cloud STT v2.32.0 with automatic features.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

# Import the improved Google Cloud STT
from speech_to_text.google_cloud_stt_v2 import GoogleCloudStreamingSTT_V2
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

# ElevenLabs TTS imports
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Improved Voice AI Agent class using Google Cloud STT v2.32.0 automatic features."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with improved Google Cloud STT.
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        
        # STT Parameters for improved Google Cloud STT
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
        
    async def init(self):
        """Initialize all components with improved Google Cloud STT."""
        logger.info("Initializing Voice AI Agent components with improved Google Cloud STT v2.32.0...")
        
        # Initialize speech recognizer with improved Google Cloud STT
        self.speech_recognizer = GoogleCloudStreamingSTT_V2(
            language=self.language,
            sample_rate=8000,  # Use 8kHz for Twilio compatibility
            encoding="MULAW",   # Use MULAW for Twilio
            channels=1,
            interim_results=True,  # Enable for better responsiveness
            enhanced_model=self.enhanced_model,
            timeout=60.0  # Longer timeout for stability
        )
        
        # Initialize STT integration with the improved client
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.language[:2]  # Extract language code (e.g., 'en' from 'en-US')
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
        
        # Initialize conversation manager with optimized parameters
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            # Skip greeting for better telephone experience
            skip_greeting=True
        )
        await self.conversation_manager.init()
        
        # Initialize ElevenLabs TTS client with optimized parameters
        try:
            if not self.elevenlabs_api_key:
                raise ValueError("ElevenLabs API key is required. Please set ELEVENLABS_API_KEY in environment variables.")
                
            self.tts_client = ElevenLabsTTS(
                api_key=self.elevenlabs_api_key,
                voice_id=self.elevenlabs_voice_id,
                model_id=self.elevenlabs_model_id,
                container_format="mulaw",  # For Twilio compatibility
                sample_rate=8000,  # For Twilio compatibility
                optimize_streaming_latency=3  # Balanced optimization
            )
            
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {self.elevenlabs_voice_id}, model ID: {self.elevenlabs_model_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            raise
        
        logger.info("Voice AI Agent initialization complete with improved Google Cloud STT v2.32.0")
        
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with improved Google Cloud STT.
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Use STT integration for processing with improved Google Cloud STT
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
                    logger.error(f"Error synthesizing speech with ElevenLabs: {e}")
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
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Process streaming audio with improved real-time response."""
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Start streaming session
        await self.speech_recognizer.start_streaming()
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process through improved Google Cloud STT
                async def process_result(result):
                    # Only handle final results for complete utterances
                    if result.is_final:
                        # Clean up transcription
                        transcription = self.stt_integration.cleanup_transcription(result.text)
                        
                        # Process if valid
                        if transcription and self.stt_integration.is_valid_transcription(transcription):
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(transcription)
                            
                            # Generate speech with ElevenLabs TTS
                            speech_audio = None
                            tts_error = None
                            
                            if response and response.get("response"):
                                try:
                                    speech_audio = await self.tts_client.synthesize(response["response"])
                                except Exception as e:
                                    logger.error(f"Error synthesizing speech with ElevenLabs: {e}")
                                    tts_error = str(e)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "speech_audio": speech_audio,
                                "tts_error": tts_error,
                                "confidence": result.confidence,
                                "is_final": True
                            }
                            
                            nonlocal results_count
                            results_count += 1
                            
                            # Call callback if provided
                            if result_callback:
                                await result_callback(result_data)
                
                # Process chunk
                await self.speech_recognizer.process_audio_chunk(chunk, process_result)
                
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            # Return stats
            return {
                "status": "complete",
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            
            # Stop streaming session
            await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
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