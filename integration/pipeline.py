# integration/pipeline.py

"""
End-to-end pipeline orchestration for Voice AI Agent.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.google_cloud_stt_v2 import GoogleCloudStreamingSTT_V2
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.query_engine import QueryEngine
from integration.tts_integration import TTSIntegration

# Minimum word count for a valid user query
MIN_VALID_WORDS = 2

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudStreamingSTT_V2, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """Initialize the pipeline with existing components."""
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT
        self.using_google_cloud = isinstance(speech_recognizer, GoogleCloudStreamingSTT_V2)
        logger.info(f"Pipeline initialized with {'Google Cloud' if self.using_google_cloud else 'Other'} STT and ElevenLabs TTS")
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            speech_output_path: Path to save speech output
            
        Returns:
            Results dictionary
        """
        logger.info(f"Starting pipeline with audio data: {type(audio_data)}")
        
        # Track timing
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Convert audio data to proper format if needed
        if isinstance(audio_data, bytes):
            # MULAW audio is already in bytes
            audio = audio_data
        elif isinstance(audio_data, np.ndarray):
            # Convert numpy array to bytes properly
            if audio_data.dtype == np.float32:
                # For MULAW, convert to bytes (this is where the issue was!)
                import audioop
                # First convert to int16 PCM
                linear_audio = (audio_data * 32767).astype(np.int16).tobytes()
                # Then convert to mulaw
                audio = audioop.lin2ulaw(linear_audio, 2)
            else:
                audio = audio_data.tobytes()
        else:
            audio = audio_data
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Log audio info
        logger.info(f"Audio size: {len(audio)} bytes")
        
        # Process for transcription - THIS IS THE KEY FIX
        try:
            # Use STT integration for processing
            result = await self.stt_helper.transcribe_audio_data(audio, is_short_audio=True)
            
            if not result.get("transcription"):
                logger.warning("No transcription obtained")
                return {"error": "No transcription detected", "transcription": ""}
            
            transcription = result["transcription"]
            
            # Validate transcription
            if not self.stt_helper.is_valid_transcription(transcription):
                logger.warning(f"Transcription not valid for processing: '{transcription}'")
                return {"error": "Invalid transcription", "transcription": transcription}
                
            timings = {"stt": time.time() - stt_start}
            logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
            return {"error": f"Transcription error: {str(e)}"}
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Query the knowledge base
            query_result = await self.query_engine.query(transcription)
            response = query_result.get("response", "")
            
            if not response:
                logger.warning("No response generated from knowledge base")
                return {
                    "transcription": transcription,
                    "error": "No response generated",
                    "timings": timings
                }
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {
                "error": f"Knowledge base error: {str(e)}",
                "transcription": transcription,
                "timings": timings
            }
        
        # STAGE 3: Text-to-Speech with ElevenLabs
        logger.info("STAGE 3: Text-to-Speech with ElevenLabs")
        tts_start = time.time()
        
        try:
            # Convert response to speech using ElevenLabs
            speech_audio = await self.tts_integration.text_to_speech(response)
            
            # Save speech audio if output file specified
            if speech_output_path:
                os.makedirs(os.path.dirname(os.path.abspath(speech_output_path)), exist_ok=True)
                with open(speech_output_path, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {speech_output_path}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Compile results
            return {
                "transcription": transcription,
                "response": response,
                "speech_audio_size": len(speech_audio),
                "speech_audio": speech_audio,
                "timings": timings,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response,
                "timings": timings
            }