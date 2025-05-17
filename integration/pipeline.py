"""
End-to-end pipeline orchestration for Voice AI Agent.
Updated to use OpenAI + Pinecone instead of LlamaIndex with CRITICAL LATENCY FIXES.

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
from knowledge_base.query_engine import QueryEngine

from integration.tts_integration import TTSIntegration

# Minimum word count for a valid user query
MIN_VALID_WORDS = 1  # REDUCED from 2 for better recognition

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    End-to-end pipeline orchestration for Voice AI Agent.
    
    Provides a high-level interface for running the complete
    STT -> Knowledge Base -> TTS pipeline with Google Cloud STT v2
    and Google Cloud TTS, using OpenAI + Pinecone for knowledge base.
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
            query_engine: Initialized query engine (OpenAI + Pinecone)
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
        logger.info(f"Pipeline initialized with {'Google Cloud v2' if self.using_google_cloud else 'Other'} STT and OpenAI + Pinecone")
        
        # IMPROVED: Add latency tracking
        self.total_request_count = 0
        self.total_latency = 0
        self.recent_latencies = []
    
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
            
        # IMPROVED: More lenient check - accept even single words
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
        
        # STAGE 2: Knowledge Base Query (OpenAI + Pinecone)
        logger.info("STAGE 2: Knowledge Base Query (OpenAI + Pinecone)")
        kb_start = time.time()
        
        try:
            # IMPROVED: Use faster async timeout with better error handling
            query_task = self.query_engine.query(transcription)
            query_result = await asyncio.wait_for(query_task, timeout=5.0)  # Reduced timeout from 10s to 5s
            response = query_result.get("response", "")
            
            if not response:
                return {"error": "No response generated from knowledge base"}
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated: {response[:50]}...")
            
        except asyncio.TimeoutError:
            logger.error(f"Knowledge base query timed out for: '{transcription}'")
            return {"error": "Knowledge base query timed out. Please try again."}
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech with Google Cloud TTS
        logger.info("STAGE 3: Text-to-Speech with Google Cloud TTS")
        tts_start = time.time()
        
        try:
            # IMPROVED: Faster TTS with timeout
            tts_task = self.tts_integration.text_to_speech(response)
            speech_audio = await asyncio.wait_for(tts_task, timeout=3.0)  # Reduced from 5s to 3s
            
            # Save speech audio if output file specified
            if output_speech_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_speech_file)), exist_ok=True)
                with open(output_speech_file, "wb") as f:
                    f.write(speech_audio)
                logger.info(f"Saved speech audio to {output_speech_file}")
            
            timings["tts"] = time.time() - tts_start
            logger.info(f"TTS completed in {timings['tts']:.2f}s, generated {len(speech_audio)} bytes")
            
        except asyncio.TimeoutError:
            logger.error("TTS synthesis timed out")
            return {
                "error": "TTS synthesis timed out",
                "transcription": transcription,
                "response": response
            }
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
        
        # IMPROVED: Track latency
        self.total_request_count += 1
        self.total_latency += total_time
        self.recent_latencies.append(total_time)
        if len(self.recent_latencies) > 10:
            self.recent_latencies.pop(0)
        
        # Compile results
        return {
            "transcription": transcription,
            "response": response,
            "speech_audio_size": len(speech_audio),
            "speech_audio": None if output_speech_file else speech_audio,
            "timings": timings,
            "total_time": total_time,
            "engine": "openai_pinecone"
        }
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]]
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Process audio with parallel execution for lower latency.
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
        
        # IMPROVED: Stream the response with parallel processing 
        try:
            # Start generating the response immediately
            response_results = []
            response_start_time = time.time()
            
            # Use conversation manager's streaming for faster processing
            response_task = self.conversation_manager.generate_streaming_response(transcription)
            
            # Process chunks as they come in
            total_chunks = 0
            total_audio_bytes = 0
            full_response = ""
            
            # Create TTS tasks in parallel with minimal buffer
            current_chunk = ""
            buffered_chunks = []
            
            async def process_chunk_to_speech(chunk_text):
                """Process a text chunk to speech in parallel."""
                if not chunk_text:
                    return None
                    
                # Convert to speech with 3s timeout
                try:
                    audio_data = await asyncio.wait_for(
                        self.tts_integration.text_to_speech(chunk_text), 
                        timeout=3.0
                    )
                    return audio_data
                except Exception as e:
                    logger.error(f"Error synthesizing chunk: {e}")
                    return None
            
            async for chunk in response_task:
                chunk_text = chunk.get("chunk", "")
                
                if chunk_text:
                    # Add to full response
                    full_response += chunk_text
                    current_chunk += chunk_text
                    
                    # Only process chunks that end with sentence terminators or are long enough
                    if (any(c in current_chunk for c in ['.', '!', '?']) or 
                        len(current_chunk.split()) > 5):
                        
                        # Create speech task in parallel
                        tts_task = asyncio.create_task(process_chunk_to_speech(current_chunk))
                        buffered_chunks.append((current_chunk, tts_task))
                        current_chunk = ""
                
                # Process any completed chunks 
                completed_chunks = []
                for i, (text, task) in enumerate(buffered_chunks):
                    if task.done():
                        audio_data = task.result()
                        if audio_data:
                            await audio_callback(audio_data)
                            total_chunks += 1
                            total_audio_bytes += len(audio_data)
                        completed_chunks.append(i)
                
                # Remove processed chunks
                for i in sorted(completed_chunks, reverse=True):
                    buffered_chunks.pop(i)
                
                # Handle completion
                if chunk.get("done", False):
                    if chunk.get("full_response"):
                        full_response = chunk.get("full_response")
                    
                    # Process any remaining chunk
                    if current_chunk:
                        audio_data = await process_chunk_to_speech(current_chunk)
                        if audio_data:
                            await audio_callback(audio_data)
                            total_chunks += 1
                            total_audio_bytes += len(audio_data)
                    
                    # Process any remaining buffered chunks
                    for _, task in buffered_chunks:
                        if not task.done():
                            continue
                        audio_data = task.result()
                        if audio_data:
                            await audio_callback(audio_data)
                            total_chunks += 1
                            total_audio_bytes += len(audio_data)
                    
                    break
            
            # Calculate stats
            response_time = time.time() - response_start_time
            total_time = time.time() - start_time
            
            # Update latency tracking
            self.total_request_count += 1
            self.total_latency += total_time
            self.recent_latencies.append(total_time)
            if len(self.recent_latencies) > 10:
                self.recent_latencies.pop(0)
            
            return {
                "transcription": transcription,
                "transcription_time": transcription_time,
                "response_time": response_time,
                "total_time": total_time,
                "total_chunks": total_chunks,
                "total_audio_bytes": total_audio_bytes,
                "full_response": full_response,
                "engine": "openai_pinecone",
                "avg_latency": sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return {
                "error": f"Streaming error: {str(e)}",
                "transcription": transcription,
                "transcription_time": transcription_time,
                "engine": "openai_pinecone"
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
        IMPROVED: Transcribe audio using Google Cloud STT v2 with better error handling.
        
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
            
            # IMPROVED: Ensure clean session
            if hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming:
                await self.speech_recognizer.stop_streaming()
                
            # Start a streaming session
            await self.speech_recognizer.start_streaming()
            
            # Track final results
            final_results = []
            
            # Process callback to collect results
            async def collect_result(result):
                if result.is_final:
                    final_results.append(result)
            
            # IMPROVED: Process audio in smaller chunks for faster processing
            chunk_size = 2048  # ~0.25s at 8kHz (reduced from 4096)
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
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics for monitoring."""
        return {
            "request_count": self.total_request_count,
            "avg_latency": self.total_latency / max(1, self.total_request_count),
            "recent_latency": sum(self.recent_latencies) / max(1, len(self.recent_latencies)),
            "min_latency": min(self.recent_latencies) if self.recent_latencies else 0,
            "max_latency": max(self.recent_latencies) if self.recent_latencies else 0,
            "target_latency": 2.0
        }