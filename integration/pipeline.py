"""
End-to-end pipeline orchestration - Updated for OpenAI + Pinecone.
"""
import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable

import numpy as np

# Updated imports for new knowledge base
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from integration.tts_integration import TTSIntegration

# Minimum word count for a valid user query
MIN_VALID_WORDS = 2

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration - Updated for OpenAI + Pinecone.
    """
    
    def __init__(
        self,
        speech_recognizer: Union[GoogleCloudStreamingSTT, Any],
        conversation_manager: ConversationManager,
        query_engine: QueryEngine,
        tts_integration: TTSIntegration
    ):
        """
        Initialize the pipeline with updated components.
        
        Args:
            speech_recognizer: Initialized STT component  
            conversation_manager: OpenAI conversation manager
            query_engine: OpenAI + Pinecone query engine
            tts_integration: TTS integration with ElevenLabs
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Create a helper for filtering out non-speech transcriptions
        self.stt_helper = STTIntegration(speech_recognizer)
        
        # Determine if we're using Google Cloud STT
        self.using_google_cloud = isinstance(speech_recognizer, GoogleCloudStreamingSTT)
        logger.info(f"Pipeline initialized with {'Google Cloud' if self.using_google_cloud else 'Other'} STT and OpenAI + Pinecone")
    
    async def _is_valid_transcription(self, transcription: str) -> bool:
        """Check if a transcription is valid and should be processed."""
        cleaned_text = self.stt_helper.cleanup_transcription(transcription)
        
        if not cleaned_text:
            return False
            
        words = cleaned_text.split()
        if len(words) < MIN_VALID_WORDS:
            return False
            
        return True
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        user_id: Optional[str] = None,
        output_speech_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to the input audio file
            user_id: User identifier for conversation tracking
            output_speech_file: Path to save the output speech file (optional)
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info(f"Starting pipeline with audio: {audio_file_path}")
        
        # Track timing for each stage
        timings = {}
        start_time = time.time()
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Log audio file info
        logger.info(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")
        
        from speech_to_text.utils.audio_utils import load_audio_file
        
        # Load audio file
        try:
            audio, sample_rate = load_audio_file(audio_file_path, target_sr=16000)
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
            logger.warning(f"Transcription not valid: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query with OpenAI
        logger.info("STAGE 2: Knowledge Base Query with OpenAI + Pinecone")
        kb_start = time.time()
        
        try:
            # Use conversation manager for response generation
            response_result = await self.conversation_manager.handle_user_input(
                user_id=user_id or "default_user",
                message=transcription
            )
            response = response_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech with ElevenLabs
        logger.info("STAGE 3: Text-to-Speech with ElevenLabs")
        tts_start = time.time()
        
        try:
            # Convert response to speech
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
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
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
        user_id: Optional[str] = None,
        audio_callback: Callable[[bytes], Awaitable[None]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data with streaming response.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            user_id: User identifier
            audio_callback: Callback to handle audio data
            
        Returns:
            Dictionary with stats about the process
        """
        logger.info(f"Starting streaming pipeline with audio: {type(audio_data)}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Ensure audio is in the right format
            if isinstance(audio_data, bytes):
                audio = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio = audio_data
            
            # Transcribe audio
            transcription, duration = await self._transcribe_audio(audio)
            
            # Validate transcription
            is_valid = await self._is_valid_transcription(transcription)
            if not is_valid:
                logger.warning(f"Transcription not valid: '{transcription}'")
                return {"error": "No valid transcription detected", "transcription": transcription}
                
            logger.info(f"Transcription: {transcription}")
            transcription_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
        
        # Stream the response
        try:
            total_chunks = 0
            total_audio_bytes = 0
            response_start_time = time.time()
            full_response = ""
            
            # Use conversation manager's streaming method
            async for chunk in self.conversation_manager.handle_user_input_streaming(
                user_id=user_id or "default_user",
                message=transcription
            ):
                chunk_text = chunk.get("chunk", "")
                
                if chunk_text:
                    # Add to full response
                    full_response += chunk_text
                    
                    # Convert to speech and send to callback
                    audio_data = await self.tts_integration.text_to_speech(chunk_text)
                    if audio_callback:
                        await audio_callback(audio_data)
                    
                    # Update stats
                    total_chunks += 1
                    total_audio_bytes += len(audio_data)
                
                if chunk.get("done", False):
                    if chunk.get("full_response"):
                        full_response = chunk["full_response"]
                    break
            
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
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        user_id: Optional[str] = None,
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio data through the complete pipeline.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            user_id: User identifier
            speech_output_path: Path to save speech output
            
        Returns:
            Results dictionary
        """
        logger.info(f"Starting pipeline with audio data: {type(audio_data)}")
        
        # Track timing
        start_time = time.time()
        
        # Convert audio data if needed
        if isinstance(audio_data, bytes):
            audio = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio = audio_data
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Process for transcription
        transcription, duration = await self._transcribe_audio(audio)
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings = {"stt": time.time() - stt_start}
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query with OpenAI + Pinecone")
        kb_start = time.time()
        
        try:
            # Use conversation manager for response generation
            response_result = await self.conversation_manager.handle_user_input(
                user_id=user_id or "default_user",
                message=transcription
            )
            response = response_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech
        logger.info("STAGE 3: Text-to-Speech with ElevenLabs")
        tts_start = time.time()
        
        try:
            # Convert response to speech
            speech_audio = await self.tts_integration.text_to_speech(response)
            
            # Save speech audio if output file specified
            if speech_output_path:
                os.makedirs(os.path.dirname(os.path.abspath(speech_output_path)), exist_ok=True)
                with open(speech_output_path, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {speech_output_path}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s")
            
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
                "response": response
            }
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe audio data using Google Cloud STT."""
        logger.info(f"Transcribing audio: {len(audio)} samples")
        
        if self.using_google_cloud:
            return await self._transcribe_audio_google_cloud(audio)
        else:
            return await self._transcribe_audio_generic(audio)
    
    async def _transcribe_audio_google_cloud(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe audio using Google Cloud STT."""
        try:
            # Convert to 16-bit PCM bytes
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Start a streaming session
            await self.speech_recognizer.start_streaming()
            
            # Track final results
            final_results = []
            
            # Process callback to collect results
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # Process audio in chunks
            chunk_size = 4096  # ~128ms at 16kHz
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
                duration = best_result.end_time - best_result.start_time if best_result.end_time > 0 else len(audio) / 16000
            
            # Clean up the transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in Google Cloud transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def _transcribe_audio_generic(self, audio: np.ndarray) -> tuple[str, float]:
        """Generic transcription method for any STT system."""
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
                transcription = ""
                duration = len(audio) / 16000
            
            # Clean up transcription
            if hasattr(self.stt_helper, 'cleanup_transcription'):
                transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in generic transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def process_realtime_stream(
        self,
        audio_chunk_generator: AsyncIterator[np.ndarray],
        user_id: Optional[str] = None,
        audio_output_callback: Callable[[bytes], Awaitable[None]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a real-time audio stream.
        
        Args:
            audio_chunk_generator: Async generator producing audio chunks
            user_id: User identifier
            audio_output_callback: Callback to handle output audio data
            
        Yields:
            Status updates and results
        """
        logger.info("Starting real-time audio stream processing")
        
        # Track state
        is_speaking = False
        processing = False
        last_transcription = ""
        silence_frames = 0
        max_silence_frames = 5
        
        # Create audio buffer for processing
        audio_buffer = []
        
        # Timing stats
        start_time = time.time()
        
        try:
            # Initialize the speech recognizer
            if hasattr(self.speech_recognizer, 'start_streaming'):
                await self.speech_recognizer.start_streaming()
            
            # Track results
            results = []
            
            # Define result callback
            async def result_callback(result):
                results.append(result)
                logger.debug(f"Received transcription result: {result.text if hasattr(result, 'text') else str(result)}")
            
            # Process incoming audio chunks
            async for audio_chunk in audio_chunk_generator:
                # Convert if needed
                if isinstance(audio_chunk, bytes):
                    audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Check for silence
                is_speech = np.mean(np.abs(audio_chunk)) > 0.01  # Simple energy-based detector
                
                if not is_speech:
                    silence_frames += 1
                else:
                    silence_frames = 0
                    
                # Add to buffer
                audio_buffer.append(audio_chunk)
                
                # Process the audio chunk
                if hasattr(self.speech_recognizer, 'process_audio_chunk'):
                    result = await self.speech_recognizer.process_audio_chunk(
                        audio_chunk=audio_chunk,
                        callback=result_callback
                    )
                    
                    # Check for final result
                    if result and hasattr(result, 'is_final') and result.is_final:
                        # Clean up transcription
                        transcription = self.stt_helper.cleanup_transcription(result.text)
                        
                        # Validate transcription
                        if transcription and await self._is_valid_transcription(transcription) and transcription != last_transcription:
                            # Yield status update
                            yield {
                                "status": "transcribed",
                                "transcription": transcription
                            }
                            
                            # Generate response
                            if not is_speaking and not processing:
                                processing = True
                                try:
                                    # Use conversation manager
                                    response_result = await self.conversation_manager.handle_user_input(
                                        user_id=user_id or "default_user",
                                        message=transcription
                                    )
                                    response = response_result.get("response", "")
                                    
                                    if response:
                                        # Mark agent as speaking
                                        is_speaking = True
                                        
                                        # Convert to speech
                                        speech_audio = await self.tts_integration.text_to_speech(response)
                                        
                                        # Send through callback
                                        if audio_output_callback:
                                            await audio_output_callback(speech_audio)
                                        
                                        # Agent is done speaking
                                        is_speaking = False
                                        
                                        # Yield response
                                        yield {
                                            "status": "response",
                                            "transcription": transcription,
                                            "response": response,
                                            "audio_size": len(speech_audio) if speech_audio else 0
                                        }
                                        
                                        # Update last transcription
                                        last_transcription = transcription
                                finally:
                                    processing = False
            
            # Process any final audio
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                final_transcription, _ = await self.speech_recognizer.stop_streaming()
                final_transcription = self.stt_helper.cleanup_transcription(final_transcription)
                
                if final_transcription and await self._is_valid_transcription(final_transcription) and final_transcription != last_transcription:
                    # Generate final response
                    response_result = await self.conversation_manager.handle_user_input(
                        user_id=user_id or "default_user",
                        message=final_transcription
                    )
                    final_response = response_result.get("response", "")
                    
                    if final_response:
                        # Mark agent as speaking
                        is_speaking = True
                        
                        # Convert to speech
                        final_speech = await self.tts_integration.text_to_speech(final_response)
                        
                        # Send through callback
                        if audio_output_callback:
                            await audio_output_callback(final_speech)
                        
                        # Agent is done speaking
                        is_speaking = False
                        
                        # Yield final response
                        yield {
                            "status": "final",
                            "transcription": final_transcription,
                            "response": final_response,
                            "audio_size": len(final_speech) if final_speech else 0,
                            "total_time": time.time() - start_time
                        }
            
            # Yield completion
            yield {
                "status": "complete",
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in real-time stream processing: {e}", exc_info=True)
            yield {
                "status": "error",
                "error": str(e),
                "total_time": time.time() - start_time
            }