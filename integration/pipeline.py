"""
End-to-end pipeline orchestration for Voice AI Agent with optimized latency 
and response length control.
"""
import os
import asyncio
import logging
import time
import re
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable
import numpy as np
from scipy import signal

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
    End-to-end pipeline orchestration for Voice AI Agent with optimized
    latency and response length control.
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
            speech_recognizer: Initialized STT component
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
        
        # Determine if we're using Google Cloud STT
        self.using_google_cloud = isinstance(speech_recognizer, GoogleCloudStreamingSTT)
        logger.info(f"Pipeline initialized with {'Google Cloud' if self.using_google_cloud else 'Other'} STT")
        
        # System prompt that encourages brevity for telephony
        self.system_prompt = """You are a voice AI assistant for phone conversations. Your responses MUST be brief and concise:
1. Keep answers under 30 words when possible
2. Use simple language and short sentences 
3. Avoid complex explanations or lists
4. Provide only the most essential information first
5. Speak conversationally with natural pauses
6. Focus on one key point per response

The caller cannot see any visual information, so be direct and clear. Always prioritize brevity over comprehensiveness."""
    
    def _optimize_response_for_telephony(self, text: str, max_words: int = 50) -> str:
        """
        Optimize a response for telephony by making it shorter and more direct.
        
        Args:
            text: Original response
            max_words: Maximum words for optimized response
            
        Returns:
            Optimized response
        """
        # Count words
        words = text.split()
        
        # If already short enough, return as is
        if len(words) <= max_words:
            return text
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Always include the first sentence
        optimized = sentences[0]
        
        # If first sentence is very short, add more sentences up to the word limit
        current_words = len(optimized.split())
        i = 1
        
        while current_words < max_words and i < len(sentences):
            sentence_words = len(sentences[i].split())
            
            # Add next sentence if it doesn't exceed limit
            if current_words + sentence_words <= max_words:
                optimized += " " + sentences[i]
                current_words += sentence_words
            else:
                break
                
            i += 1
            
        # If we're still under the limit but couldn't add a full sentence,
        # add a truncated version of the next sentence to reach closer to the max
        if current_words < max_words * 0.7 and i < len(sentences):
            remaining_words = max_words - current_words
            next_sentence_words = sentences[i].split()
            
            if len(next_sentence_words) > remaining_words + 3:  # Only truncate if significantly longer
                truncated = " ".join(next_sentence_words[:remaining_words])
                
                # Add sentence-ending punctuation if missing
                if not truncated.endswith(('.', '!', '?')):
                    truncated += "."
                    
                optimized += " " + truncated
        
        logger.info(f"Optimized response from {len(words)} to {len(optimized.split())} words")
        return optimized
    
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
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings["stt"] = time.time() - stt_start
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query with improved prompt and length control
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Create query with modified system prompt for brevity
            retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
            
            # Use custom system prompt with brevity instructions
            query_result = await self.query_engine.query(
                query_text=transcription,
                system_prompt=self.system_prompt
            )
            
            raw_response = query_result.get("response", "")
            
            # Optimize response length for telephony
            if raw_response:
                response = self._optimize_response_for_telephony(raw_response)
                
                # Update the query result with optimized response
                query_result["response"] = response
                
                timings["kb"] = time.time() - kb_start
                logger.info(f"Response generated and optimized: {response[:50]}...")
            else:
                return {"error": "No response generated from knowledge base"}
                
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
        # STAGE 3: Text-to-Speech with optimized settings
        logger.info("STAGE 3: Text-to-Speech")
        tts_start = time.time()
        
        try:
            # Convert response to speech using optimized TTS
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
    
    async def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio data using Google Cloud STT.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
        logger.info(f"Transcribing audio: {len(audio)} samples")
        
        # Apply audio preprocessing for noise reduction
        processed_audio = self._preprocess_audio(audio)
        
        # Check if we're using Google Cloud STT
        if self.using_google_cloud:
            return await self._transcribe_audio_google_cloud(processed_audio)
        else:
            # Fallback to a more generic approach for other STT systems
            return await self._transcribe_audio_generic(processed_audio)
            
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data to reduce noise and improve speech recognition.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Preprocessed audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply band-pass filter for telephony frequency range (300-3400 Hz)
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply pre-emphasis to boost high frequencies (improves clarity)
            pre_emphasized = np.append(filtered_audio[0], filtered_audio[1:] - 0.97 * filtered_audio[:-1])
            
            # 4. Simple noise gate (suppress very low amplitudes)
            noise_gate_threshold = 0.015
            noise_gate = np.where(np.abs(pre_emphasized) < noise_gate_threshold, 0, pre_emphasized)
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
                
            # Log stats about the audio
            orig_energy = np.mean(np.abs(audio_data))
            proc_energy = np.mean(np.abs(normalized))
            logger.debug(f"Audio preprocessing: original energy={orig_energy:.4f}, processed energy={proc_energy:.4f}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}", exc_info=True)
            return audio_data  # Return original audio if preprocessing fails
    
    async def _transcribe_audio_google_cloud(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Google Cloud STT.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, duration)
        """
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
                # Calculate duration
                duration = best_result.end_time - best_result.start_time if best_result.end_time > 0 else len(audio) / 16000
            
            # Clean up the transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in Google Cloud transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def _transcribe_audio_generic(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Generic transcription method for any STT system.
        """
        try:
            # Save original VAD setting if available
            original_vad = getattr(self.speech_recognizer, 'vad_enabled', None)
            
            # Start streaming
            if hasattr(self.speech_recognizer, 'start_streaming'):
                self.speech_recognizer.start_streaming()
            
            # Process audio chunk
            if hasattr(self.speech_recognizer, 'process_audio_chunk'):
                await self.speech_recognizer.process_audio_chunk(audio)
            
            # Get final transcription
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                transcription, duration = await self.speech_recognizer.stop_streaming()
            else:
                # If stop_streaming not available, use a default approach
                transcription = ""
                duration = len(audio) / 16000
            
            # Clean up transcription if helper available
            if hasattr(self.stt_helper, 'cleanup_transcription'):
                transcription = self.stt_helper.cleanup_transcription(transcription)
            
            # Restore original VAD setting if applicable
            if original_vad is not None and hasattr(self.speech_recognizer, 'vad_enabled'):
                self.speech_recognizer.vad_enabled = original_vad
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Error in generic transcription: {e}", exc_info=True)
            return "", len(audio) / 16000
    
    async def process_audio_streaming(
        self,
        audio_data: Union[bytes, np.ndarray],
        audio_callback: Callable[[bytes], Awaitable[None]],
        use_streaming_response: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio data with streaming response directly to speech.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            audio_callback: Callback to handle audio data
            use_streaming_response: Whether to use streaming response generation
            
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
            
            # Preprocess audio for better recognition
            audio = self._preprocess_audio(audio)
            
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
        
        # Stream the response with improved latency
        try:
            # Stream the response directly to TTS
            total_chunks = 0
            total_audio_bytes = 0
            response_start_time = time.time()
            full_response = ""
            
            # Determine whether to use streaming or optimized non-streaming
            if use_streaming_response:
                # Use the query engine's streaming method for word-by-word response
                async for chunk in self.query_engine.query_with_streaming(transcription):
                    chunk_text = chunk.get("chunk", "")
                    
                    if chunk_text:
                        # Add to full response
                        full_response += chunk_text
                        
                        # For very small chunks (1-2 words), accumulate before sending
                        word_count = len(chunk_text.split())
                        if word_count <= 2 and not any(c in chunk_text for c in ['.', '!', '?', ',']):
                            # Skip immediate TTS for minor chunks to avoid excessive latency overhead
                            continue
                        
                        # Convert to speech with ElevenLabs and send to callback
                        audio_data = await self.tts_integration.text_to_speech(chunk_text)
                        await audio_callback(audio_data)
                        
                        # Update stats
                        total_chunks += 1
                        total_audio_bytes += len(audio_data)
            else:
                # Use optimized non-streaming approach for lower latency
                # Get response with telephony-optimized system prompt
                query_result = await self.query_engine.query(
                    query_text=transcription,
                    system_prompt=self.system_prompt
                )
                
                response = query_result.get("response", "")
                
                # Optimize response length for telephony
                if response:
                    # Apply optimization for telephony
                    optimized_response = self._optimize_response_for_telephony(response)
                    
                    # Set as full response
                    full_response = optimized_response
                    
                    # Convert to speech with optimized settings
                    audio_data = await self.tts_integration.text_to_speech(optimized_response)
                    
                    # Send to callback
                    await audio_callback(audio_data)
                    
                    # Update stats
                    total_chunks = 1
                    total_audio_bytes = len(audio_data)
            
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
        
        # Convert audio data to numpy array if needed
        if isinstance(audio_data, bytes):
            audio = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio = audio_data
        
        # Preprocess audio for better recognition
        audio = self._preprocess_audio(audio)
        
        # STAGE 1: Speech-to-Text
        logger.info("STAGE 1: Speech-to-Text")
        stt_start = time.time()
        
        # Process for transcription
        transcription, duration = await self._transcribe_audio(audio)
        
        # Validate transcription
        is_valid = await self._is_valid_transcription(transcription)
        if not is_valid:
            logger.warning(f"Transcription not valid for processing: '{transcription}'")
            return {"error": "No valid transcription detected", "transcription": transcription}
            
        timings = {"stt": time.time() - stt_start}
        logger.info(f"Transcription completed in {timings['stt']:.2f}s: {transcription}")
        
        # STAGE 2: Knowledge Base Query
        logger.info("STAGE 2: Knowledge Base Query")
        kb_start = time.time()
        
        try:
            # Retrieve context and generate response with optimized prompt
            retrieval_results = await self.query_engine.retrieve_with_sources(transcription)
            query_result = await self.query_engine.query(
                query_text=transcription,
                system_prompt=self.system_prompt
            )
            
            raw_response = query_result.get("response", "")
            
            if not raw_response:
                return {"error": "No response generated from knowledge base"}
            
            # Optimize response length for telephony
            response = self._optimize_response_for_telephony(raw_response)
                
            timings["kb"] = time.time() - kb_start
            logger.info(f"Response generated and optimized: {response[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in KB stage: {e}")
            return {"error": f"Knowledge base error: {str(e)}"}
        
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
                "response": response
            }
    
    async def process_realtime_stream(
        self,
        audio_chunk_generator: AsyncIterator[np.ndarray],
        audio_output_callback: Callable[[bytes], Awaitable[None]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a real-time audio stream without barge-in support.
        
        Args:
            audio_chunk_generator: Async generator producing audio chunks
            audio_output_callback: Callback to handle output audio data
            
        Yields:
            Status updates and results
        """
        logger.info("Starting real-time audio stream processing")
        
        # Reset conversation state
        if self.conversation_manager:
            self.conversation_manager.reset()
        
        # Track state
        is_speaking = False
        processing = False
        last_transcription = ""
        silence_frames = 0
        max_silence_frames = 5  # Number of silent chunks before processing
        
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
                
                # Preprocess audio for better recognition
                audio_chunk = self._preprocess_audio(audio_chunk)
                
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
                            
                            # Generate response - only if we're not already speaking
                            if not is_speaking and not processing:
                                processing = True
                                try:
                                    # Query knowledge base with optimized prompt
                                    query_result = await self.query_engine.query(
                                        query_text=transcription,
                                        system_prompt=self.system_prompt
                                    )
                                    raw_response = query_result.get("response", "")
                                    
                                    # Optimize response for telephony
                                    response = self._optimize_response_for_telephony(raw_response)
                                    
                                    if response:
                                        # Mark agent as speaking
                                        is_speaking = True
                                        
                                        # Convert to speech
                                        speech_audio = await self.tts_integration.text_to_speech(response)
                                        
                                        # Send through callback
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
                
                # If we have enough silence frames and it's time to process
                if silence_frames >= max_silence_frames and not processing and not is_speaking:
                    # Get final transcription if available
                    if hasattr(self.speech_recognizer, 'stop_streaming'):
                        transcription, _ = await self.speech_recognizer.stop_streaming()
                        
                        # Clean up and validate
                        transcription = self.stt_helper.cleanup_transcription(transcription)
                        
                        if transcription and await self._is_valid_transcription(transcription) and transcription != last_transcription:
                            # Process response
                            processing = True
                            try:
                                # Query knowledge base with optimized prompt
                                query_result = await self.query_engine.query(
                                    query_text=transcription,
                                    system_prompt=self.system_prompt
                                )
                                raw_response = query_result.get("response", "")
                                
                                # Optimize response for telephony
                                response = self._optimize_response_for_telephony(raw_response)
                                
                                if response:
                                    # Mark agent as speaking
                                    is_speaking = True
                                    
                                    # Convert to speech
                                    speech_audio = await self.tts_integration.text_to_speech(response)
                                    
                                    # Send through callback
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
                        
                        # Reset for next utterance
                        if hasattr(self.speech_recognizer, 'start_streaming'):
                            await self.speech_recognizer.start_streaming()
                    
                    # Reset silence counter
                    silence_frames = 0
            
            # Process any final audio
            if hasattr(self.speech_recognizer, 'stop_streaming'):
                final_transcription, _ = await self.speech_recognizer.stop_streaming()
                final_transcription = self.stt_helper.cleanup_transcription(final_transcription)
                
                if final_transcription and await self._is_valid_transcription(final_transcription) and final_transcription != last_transcription:
                    # Generate final response with optimized prompt
                    query_result = await self.query_engine.query(
                        query_text=final_transcription,
                        system_prompt=self.system_prompt
                    )
                    raw_response = query_result.get("response", "")
                    
                    # Optimize response for telephony
                    final_response = self._optimize_response_for_telephony(raw_response)
                    
                    if final_response:
                        # Mark agent as speaking
                        is_speaking = True
                        
                        # Convert to speech
                        final_speech = await self.tts_integration.text_to_speech(final_response)
                        
                        # Send through callback
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