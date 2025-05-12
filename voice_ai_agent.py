"""
Voice AI Agent main class - Updated for OpenAI + Pinecone integration.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np

# Speech processing imports (unchanged)
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration

# New knowledge base imports
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.pinecone_manager import PineconeManager

# ElevenLabs TTS imports (unchanged)
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class - Updated for OpenAI Assistants + Pinecone."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with OpenAI + Pinecone.
        
        Args:
            storage_dir: Directory for persistent storage
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        self.stt_keywords = kwargs.get('keywords', ['price', 'plan', 'cost', 'subscription', 'service'])
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
        self.document_processor = None
        self.pinecone_manager = None
        
        # Audio processing
        self.noise_floor = 0.005
        self.noise_samples = []
        self.max_noise_samples = 20
    
    async def init(self):
        """Initialize all components with OpenAI + Pinecone."""
        logger.info("Initializing Voice AI Agent with OpenAI + Pinecone...")
        
        # Initialize speech recognizer with Google Cloud
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language=self.stt_language,
            sample_rate=16000,
            encoding="LINEAR16",
            channels=1,
            interim_results=True,
            speech_context_phrases=self.stt_keywords,
            enhanced_model=self.enhanced_model
        )
        
        # Initialize STT integration 
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
        )
        
        # Initialize new knowledge base components
        self.conversation_manager = ConversationManager()
        await self.conversation_manager.init()
        
        self.query_engine = QueryEngine()
        await self.query_engine.init()
        
        self.document_processor = DocumentProcessor()
        
        self.pinecone_manager = PineconeManager()
        await self.pinecone_manager.init()
        
        # Initialize ElevenLabs TTS client
        try:
            if not self.elevenlabs_api_key:
                raise ValueError("ElevenLabs API key is required")
                
            self.tts_client = ElevenLabsTTS(
                api_key=self.elevenlabs_api_key,
                voice_id=self.elevenlabs_voice_id,
                model_id=self.elevenlabs_model_id,
                container_format="mulaw",
                sample_rate=8000,
                optimize_streaming_latency=4
            )
            
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {self.elevenlabs_voice_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            raise
        
        logger.info("Voice AI Agent initialization complete")
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        user_id: Optional[str] = None,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with Google Cloud STT and OpenAI response.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            user_id: User identifier for conversation tracking
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            audio_data = self._process_audio(audio_data)
        
        # Use STT integration for processing
        result = await self.stt_integration.transcribe_audio_data(audio_data, callback=callback)
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Process through conversation manager
            response = await self.conversation_manager.handle_user_input(
                user_id=user_id or "default_user",
                message=transcription
            )
            
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
    
    async def process_streaming_audio(
        self,
        audio_stream,
        user_id: Optional[str] = None,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response.
        
        Args:
            audio_stream: Async iterator of audio chunks
            user_id: User identifier
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
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
                
                # Process audio for better recognition
                if isinstance(chunk, np.ndarray):
                    chunk = self._process_audio(chunk)
                
                # Process through Google Cloud STT
                async def process_result(result):
                    # Only handle final results
                    if result.is_final:
                        transcription = self.stt_integration.cleanup_transcription(result.text)
                        
                        # Process if valid
                        if transcription and self.stt_integration.is_valid_transcription(transcription):
                            # Get response from conversation manager
                            response = await self.conversation_manager.handle_user_input(
                                user_id=user_id or "default_user",
                                message=transcription
                            )
                            
                            # Generate speech with ElevenLabs TTS
                            speech_audio = None
                            tts_error = None
                            
                            if response and response.get("response"):
                                try:
                                    speech_audio = await self.tts_client.synthesize(response["response"])
                                except Exception as e:
                                    logger.error(f"Error synthesizing speech: {e}")
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
            await self.speech_recognizer.stop_streaming()
            
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": chunks_processed,
                "results_count": results_count,
                "total_time": time.time() - start_time
            }
    
    async def add_knowledge_from_text(self, text: str, source: str = "manual_input") -> Dict[str, Any]:
        """
        Add knowledge from text to the knowledge base.
        
        Args:
            text: Text content to add
            source: Source identifier
            
        Returns:
            Result of the operation
        """
        try:
            # Process text into documents
            documents = self.document_processor.process_text(text, source)
            
            # Add to Pinecone
            upserted_count = await self.pinecone_manager.upsert_documents(documents)
            
            logger.info(f"Added {upserted_count} documents from text")
            return {
                "success": True,
                "documents_added": upserted_count,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error adding knowledge from text: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_knowledge_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Add knowledge from a file to the knowledge base.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Result of the operation
        """
        try:
            # Process file into documents
            documents = self.document_processor.process_file(file_path)
            
            # Add to Pinecone
            upserted_count = await self.pinecone_manager.upsert_documents(documents)
            
            logger.info(f"Added {upserted_count} documents from file {file_path}")
            return {
                "success": True,
                "documents_added": upserted_count,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error adding knowledge from file: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return await self.pinecone_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {"error": str(e)}
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio for better speech recognition."""
        try:
            from scipy import signal
            
            # Update noise floor from quiet sections
            self._update_noise_floor(audio)
            
            # Apply high-pass filter
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply band-pass filter for telephony
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply pre-emphasis
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Apply noise gate
            threshold = self.noise_floor * 3.0
            audio = np.where(np.abs(audio) < threshold, 0, audio)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (0.9 / max_val)
                
            return audio
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio
    
    def _update_noise_floor(self, audio: np.ndarray) -> None:
        """Update noise floor estimate from quiet sections."""
        frame_size = min(len(audio), int(0.02 * 16000))
        if frame_size <= 1:
            return
            
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        frame_energies = [np.mean(np.square(frame)) for frame in frames]
        
        if len(frame_energies) > 0:
            sorted_energies = sorted(frame_energies)
            quiet_count = max(1, len(sorted_energies) // 10)
            quiet_energies = sorted_energies[:quiet_count]
            
            self.noise_samples.extend(quiet_energies)
            
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples = self.noise_samples[-self.max_noise_samples:]
            
            if self.noise_samples:
                self.noise_floor = max(
                    0.001,
                    min(0.02, np.percentile(self.noise_samples, 90) * 1.5)
                )
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.speech_recognizer is not None and 
                self.conversation_manager is not None and 
                self.query_engine is not None and
                self.tts_client is not None and
                self.pinecone_manager is not None)
    
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Close Google Cloud streaming session
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
            await self.speech_recognizer.stop_streaming()
        
        # No specific cleanup needed for OpenAI/Pinecone components
        logger.info("Voice AI Agent shutdown complete")