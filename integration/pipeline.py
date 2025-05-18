"""
End-to-end pipeline orchestration for Voice AI Agent.

This module provides high-level functions for running the complete
STT -> Knowledge Base -> TTS pipeline with Google Cloud STT integration
and Google Cloud TTS.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
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
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline with Google Cloud STT v2
    and Google Cloud TTS.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudStreamingSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with existing components.
        
        Args:
            speech_recognizer: Initialized STT component (Google Cloud v2)
            conversation_manager: Initialized conversation manager
            query_engine: Initialized query engine
            tts_integration: Initialized TTS integration
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT v2
        self.using_google_cloud = isinstance(speech_recognizer, GoogleCloudStreamingSTT)
        logger.info(f"Pipeline initialized with {'Google Cloud v2' if self.using_google_cloud else 'Other'} STT and Google Cloud TTS")
    
    async def _is_valid_transcription(self, transcription: str) -> bool:
        """
        Check if a transcription is valid and should be processed.
        
        Args:
            transcription: The transcription text
            
        Returns:
            True if the transcription is valid
        """
        # First clean up the transcription
        cleaned_text = self.stt_helper.cleanup_transcription(transcription)
        
        # If it's empty after cleaning, it's not valid
        if not cleaned_text:
            return False
            
        # Check if it has enough words
        words = cleaned_text.split()
        if len(words) < MIN_VALID_WORDS:
            return False
            
        return True
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to the input audio file
            output_speech_file: Path to save the output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting end-to-end pipeline with audio: {audio_file_path}")
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Log audio file info
        import os
        logger.info(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=8000)  # Changed to 8000 for Twilio
            logger.info(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}", exc_info=True)
            return {"error": f"Error loading audio file: {e}"}
        
        # Process for transcription
        logger.info("Transcribing audio...")
        transcription, duration = await self._transcribe_audio(audio)
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Retrieve context and generate response
            retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
            query_result = await self.query_engine.query(transcription)
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech with Google Cloud TTS
        logger.info("STAGE 3: Text-to-Speech with Google Cloud TTS")
        tts_start = time.time()
        
        try:
            # Convert response to speech using Google Cloud TTS
            speech_audio = await self.tts_integration.text_to_speech(response)
            
            # Save speech audio if output file specified
            if output_speech_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_speech_file)), exist_ok=True)
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
        except Exception as e:
            logger.error(f"Error in TTS stage: {e}")
            return {
                "error": f"TTS error: {str(e)}",
                "transcription": transcription,
                "response": response
            }
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"End-to-end pipeline completed in {total_time:.2f}s")
        
        # Compile results
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": None if output_speech_file else speech_audio,
            "timings": timings,
            "total_time": total_time
        }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        Process audio data with streaming response directly to speech.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time for tracking
        start_time = time.time()
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        try:
            # Ensure audio is in the right format
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array if needed
                audio = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Transcribe audio
            transcription, duration = await self._transcribe_audio(audio)
            
            # Validate transcription
            is_valid = await self._is_valid_transcription(transcription)
            if not is_valid:
                logger.warning(f"Transcription not valid for processing: '{transcription}'")
                return {"error": "No valid transcription detected", "transcription": transcription}
                
            logger.info(f"Transcription: {transcription}")
            transcription_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
        
        # Stream the response with Google Cloud TTS
        try:
            # Stream the response directly to TTS
            total_chunks = 0
            total_audio_bytes = 0
            response_start_time = time.time()
            full_response = ""
            
            # Use the query engine's streaming method
            async for chunk in self.query_engine.query_with_streaming(transcription):
                chunk_text = chunk.get("chunk", "")
                
                if chunk_text:
                    # Add to full response
                    full_response += chunk_text
                    
                    # Convert to speech with Google Cloud TTS and send to callback
                    audio_data = await self.tts_integration.text_to_speech(chunk_text)
                    await audio_callback(audio_data)
                    
                    # Update stats
                    total_chunks += 1
                    total_audio_bytes += len(audio_data)
            
            # Calculate stats
            response_time = time.time() - response_start_time
            total_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "transcription_time": transcription_time,
                "response_time": response_time,
                "total_time": total_time,
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "full_response": full_response
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "transcription": transcription,
                "transcription_time": transcription_time
            }
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio data using Google Cloud STT v2.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        logger.info(f"Transcribing audio: {len(audio)} samples")
        
        # Check if we're using Google Cloud STT
        if self.using_google_cloud:
            return await self._transcribe_audio_google_cloud(audio)
        else:
            # Fallback to a more generic approach for other STT systems
            return await self._transcribe_audio_generic(audio)
    
    async def _transcribe_audio_google_cloud(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Google Cloud STT v2.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        try:
            # Convert to mulaw for Twilio compatibility
            if audio.dtype == np.float32:
                import audioop
                pcm_data = (audio * 32767).astype(np.int16).tobytes()
                audio_bytes = audioop.lin2ulaw(pcm_data, 2)
            else:
                audio_bytes = audio.tobytes()
            
            # Start a streaming session
            await self.speech_recognizer.start_streaming()
            
            # Track final results
            final_results = []
            
            # Process callback to collect results
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # Process audio in chunks
            chunk_size = 4096  # ~0.5s at 8kHz
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                result = await self.speech_recognizer.process_audio_chunk(chunk, collect_result)
                
                # Add final results directly
                if result and result.is_final:
                    final_results.append(result)
            
            # Stop streaming
            transcription, duration = await self.speech_recognizer.stop_streaming()
            
            # If we didn't get a transcription from stop_streaming but have final results
            if not transcription and final_results:
                # Get best final result based on confidence
                best_result = max(final_results, key=lambda r: r.confidence)
                transcription = best_result.text
                # Calculate duration
                duration = len(audio) / 8000  # 8kHz for Twilio
            
            # Clean up the transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in Google Cloud transcription: {e}", exc_info=True)
            return "", len(audio) / 8000
    
    async def _transcribe_audio_generic(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Generic transcription method for any STT system.
        """
        try:
            # Start streaming
            if hasattr(self.speech_recognizer, 'start_streaming'):
                await self.speech_recognizer.start_streaming()
            
            # Process audio chunk
            if hasattr(self.speech_recognizer, 'process_audio_chunk'):
                await self.speech_recognizer.process_audio_chunk(audio)
            
            # Get final transcription
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                transcription, duration = await self.speech_recognizer.stop_streaming()
            else:
                # If stop_streaming not available, use a default approach
                transcription = ""
                duration = len(audio) / 8000
            
            # Clean up transcription if helper available
            if hasattr(self.stt_helper, 'cleanup_transcription'):
                transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in generic transcription: {e}", exc_info=True)
            return "", len(audio) / 8000