# voice_ai_agent.py

"""
Voice AI Agent main class that coordinates all components with Google Cloud STT integration
and ElevenLabs TTS.
Generic version that works with any knowledge base.
"""
import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np
from scipy import signal

# Update the import to use the enhanced GoogleCloudStreamingSTT
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine

# ElevenLabs TTS imports
from text_to_speech import ElevenLabsTTS

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components with Google Cloud STT and ElevenLabs TTS."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent with Google Cloud STT and ElevenLabs TTS.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            llm_temperature: LLM temperature for response generation
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        self.stt_keywords = kwargs.get('keywords', ['price', 'plan', 'cost', 'subscription', 'service'])
        
        # Whether to use enhanced model for telephony
        self.enhanced_model = kwargs.get('enhanced_model', True)
        
        # Barge-in detection parameters
        self.enable_barge_in = kwargs.get('enable_barge_in', True)
        self.barge_in_threshold = kwargs.get('barge_in_threshold', 0.02)
        
        # TTS Parameters for ElevenLabs
        self.elevenlabs_api_key = kwargs.get('elevenlabs_api_key', os.getenv('ELEVENLABS_API_KEY'))
        # Updated to use environment variables with better defaults
        self.elevenlabs_voice_id = kwargs.get('elevenlabs_voice_id', os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL'))  # Bella voice
        self.elevenlabs_model_id = kwargs.get('elevenlabs_model_id', os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2'))  # Latest model
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Agent state tracking
        self.agent_is_speaking = False
        self.current_response_interrupted = False
        
    async def init(self):
        """Initialize all components with Google Cloud STT and ElevenLabs TTS."""
        logger.info("Initializing Voice AI Agent components with Google Cloud STT and ElevenLabs TTS...")
        
        # Initialize speech recognizer with enhanced Google Cloud STT
        self.speech_recognizer = GoogleCloudStreamingSTT(
            language=self.stt_language,
            sample_rate=16000,
            encoding="LINEAR16",
            channels=1,
            interim_results=True,
            speech_context_phrases=self.stt_keywords,
            enhanced_model=self.enhanced_model,
            vad_enabled=True,  # Enable voice activity detection
            barge_in_threshold=self.barge_in_threshold  # Set barge-in sensitivity
        )
        
        # Initialize STT integration 
        self.stt_integration = STTIntegration(
            speech_recognizer=self.speech_recognizer,
            language=self.stt_language
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
                optimize_streaming_latency=4  # Maximum optimization for real-time performance
            )
            
            logger.info(f"Initialized ElevenLabs TTS with voice ID: {self.elevenlabs_voice_id}, model ID: {self.elevenlabs_model_id}")
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs TTS: {e}")
            raise
        
        logger.info("Voice AI Agent initialization complete with Google Cloud STT and ElevenLabs TTS")
    
    def set_agent_speaking(self, speaking: bool) -> None:
        """
        Set whether the agent is currently speaking, for barge-in detection.
        
        Args:
            speaking: True if agent is speaking, False otherwise
        """
        # Update state
        self.agent_is_speaking = speaking
        
        # Also update the speech recognizer for barge-in detection
        if self.speech_recognizer:
            self.speech_recognizer.set_agent_speaking(speaking)
        
        # And update the STT integration if available
        if self.stt_integration:
            self.stt_integration.set_agent_speaking(speaking)
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with Google Cloud STT.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Create a wrapper callback that handles barge-in detection
        async def barge_in_callback(result):
            # Check if this is a barge-in during agent speech
            if hasattr(result, 'barge_in_detected') and result.barge_in_detected and self.agent_is_speaking:
                logger.info("Barge-in detected during agent speech!")
                # Set flag for interruption
                self.current_response_interrupted = True
                
                # If additional callback provided, forward the event
                if callback:
                    # Create a specialized event for the barge-in
                    barge_in_event = {
                        "type": "barge_in",
                        "timestamp": time.time(),
                        "energy_level": result.energy_level if hasattr(result, 'energy_level') else 0.0,
                        "transcription": result.text if hasattr(result, 'text') else ""
                    }
                    await callback(barge_in_event)
            # Forward the result to the original callback
            elif callback:
                await callback(result)
        
        # Use STT integration for processing with Google Cloud
        result = await self.stt_integration.transcribe_audio_data(
            audio_data, 
            callback=barge_in_callback
        )
        
        # Add barge-in information to the result
        if "barge_in_detected" not in result and hasattr(self, 'current_response_interrupted'):
            result["barge_in_detected"] = self.current_response_interrupted
        
        # Only process valid transcriptions
        if result.get("is_valid", False) and result.get("transcription"):
            transcription = result["transcription"]
            logger.info(f"Valid transcription: {transcription}")
            
            # Check if this is part of a barge-in
            if result.get("barge_in_detected", False):
                logger.info("Processing barge-in transcription")
                
                # Generate response with priority for barge-in
                response = await self.conversation_manager.handle_user_input(transcription)
                
                # Reset interruption flag
                self.current_response_interrupted = False
                
                # Return with special flag for barge-in
                if response and response.get("response"):
                    try:
                        self.set_agent_speaking(True)
                        speech_audio = await self.tts_client.synthesize(response["response"])
                        self.set_agent_speaking(False)
                        
                        return {
                            "transcription": transcription,
                            "response": response.get("response", ""),
                            "speech_audio": speech_audio,
                            "status": "success",
                            "was_barge_in": True
                        }
                    except Exception as e:
                        logger.error(f"Error synthesizing speech with ElevenLabs: {e}")
                        return {
                            "transcription": transcription,
                            "response": response.get("response", ""),
                            "error": f"Speech synthesis error: {str(e)}",
                            "status": "tts_error",
                            "was_barge_in": True
                        }
            else:
                # Standard processing for non-barge-in
                response = await self.conversation_manager.handle_user_input(transcription)
                
                # Generate speech using ElevenLabs TTS
                if response and response.get("response"):
                    try:
                        self.set_agent_speaking(True)
                        speech_audio = await self.tts_client.synthesize(response["response"])
                        self.set_agent_speaking(False)
                        
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
                "error": "No valid speech detected",
                "barge_in_detected": result.get("barge_in_detected", False)
            }
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response.
        
        Args:
            audio_stream: Async iterator of audio chunks
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
        barge_ins_detected = 0
        
        # Start streaming session
        await self.speech_recognizer.start_streaming()
        
        try:
            # Process each audio chunk
            async for chunk in audio_stream:
                chunks_processed += 1
                
                # Process through Google Cloud STT
                async def process_result(result):
                    # Handle barge-in detection
                    if hasattr(result, 'barge_in_detected') and result.barge_in_detected and self.agent_is_speaking:
                        logger.info("Barge-in detected during agent speech!")
                        # Set flag for interruption
                        self.current_response_interrupted = True
                        
                        nonlocal barge_ins_detected
                        barge_ins_detected += 1
                        
                        # Forward barge-in event if callback provided
                        if result_callback:
                            barge_in_event = {
                                "type": "barge_in",
                                "timestamp": time.time(),
                                "energy_level": result.energy_level if hasattr(result, 'energy_level') else 0.0,
                                "transcription": result.text if hasattr(result, 'text') else "",
                                "is_final": False,
                                "barge_in_detected": True
                            }
                            await result_callback(barge_in_event)
                    
                    # Only handle final results for response generation
                    if hasattr(result, 'is_final') and result.is_final and result.text:
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
                                    self.set_agent_speaking(True)
                                    speech_audio = await self.tts_client.synthesize(response["response"])
                                except Exception as e:
                                    logger.error(f"Error synthesizing speech with ElevenLabs: {e}")
                                    tts_error = str(e)
                                finally:
                                    self.set_agent_speaking(False)
                            
                            # Format result
                            result_data = {
                                "transcription": transcription,
                                "response": response.get("response", ""),
                                "speech_audio": speech_audio,
                                "tts_error": tts_error,
                                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                                "is_final": True,
                                "was_barge_in": self.current_response_interrupted
                            }
                            
                            # Reset interruption flag
                            self.current_response_interrupted = False
                            
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
                "barge_ins_detected": barge_ins_detected,
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
                "barge_ins_detected": barge_ins_detected,
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