"""
Optimized pipeline for reduced latency with proper error handling and streaming.
"""
import os
import asyncio
import logging
import time
import concurrent.futures
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable
import numpy as np

logger = logging.getLogger(__name__)

class VoiceAIAgentPipeline:
    """
    Optimized pipeline for low-latency voice conversations.
    """
    
    def __init__(
        self,
        speech_recognizer: Any,
        conversation_manager: Any,
        query_engine: Any,
        tts_integration: Any
    ):
        """
        Initialize the optimized pipeline.
        
        Args:
            speech_recognizer: STT component  
            conversation_manager: OpenAI conversation manager
            query_engine: OpenAI + Pinecone query engine
            tts_integration: TTS integration with ElevenLabs
        """
        self.speech_recognizer = speech_recognizer
        self.conversation_manager = conversation_manager
        self.query_engine = query_engine
        self.tts_integration = tts_integration
        
        # Performance optimizations
        self.parallel_processing = True
        self.use_cache = True
        self.response_cache = {}
        self.max_cache_size = 100
        
        # Timing optimizations - reduced for faster response
        self.min_audio_duration = 0.3  # Reduced from 0.5
        self.max_response_time = 2.0   # Reduced from 3.0
        
        # Pre-compute responses for common queries
        self._common_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "bye": "Thank you for calling. Have a great day!",
            "goodbye": "Goodbye! Feel free to call us again if you need help."
        }
        
        # Executor for parallel processing with reduced workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Performance monitoring
        self.processing_times = []
        self.error_count = 0
        self.success_count = 0
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        user_id: Optional[str] = None,
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio with optimized latency and error handling.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            user_id: User identifier
            speech_output_path: Path to save speech output
            
        Returns:
            Results dictionary
        """
        start_time = time.time()
        
        try:
            # Convert and validate audio
            if isinstance(audio_data, bytes):
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = audio_data
            
            # Check audio duration
            duration = len(audio) / 16000
            if duration < self.min_audio_duration:
                logger.debug(f"Audio too short: {duration:.2f}s")
                return {"error": "Audio too short", "duration": duration}
            
            # Step 1: Speech-to-Text with optimized timeout
            transcription_start = time.time()
            
            try:
                transcription_task = asyncio.create_task(self._transcribe_audio_optimized(audio))
                transcription, stt_duration = await asyncio.wait_for(
                    transcription_task, 
                    timeout=1.5  # Reduced from 2.0
                )
            except asyncio.TimeoutError:
                logger.warning("STT timeout exceeded")
                return {"error": "Speech recognition timeout"}
            
            transcription_time = time.time() - transcription_start
            
            if not transcription or len(transcription.split()) < 1:
                logger.info("No valid transcription")
                return {"error": "No speech detected", "transcription_time": transcription_time}
            
            logger.info(f"Transcription ({transcription_time:.2f}s): {transcription}")
            
            # Check for cached responses first
            if self.use_cache:
                cached_response = self._get_cached_response(transcription)
                if cached_response:
                    logger.info("Using cached response")
                    speech_audio = await self._generate_speech_parallel(cached_response)
                    self.success_count += 1
                    return {
                        "transcription": transcription,
                        "response": cached_response,
                        "speech_audio": speech_audio,
                        "total_time": time.time() - start_time,
                        "cached": True
                    }
            
            # Step 2: Knowledge Base Query with parallel optimization
            kb_start = time.time()
            
            # Use conversation manager with reduced timeout
            try:
                response_task = asyncio.create_task(
                    self.conversation_manager.handle_user_input(
                        user_id=user_id or "default_user",
                        message=transcription
                    )
                )
                response_result = await asyncio.wait_for(
                    response_task,
                    timeout=self.max_response_time
                )
            except asyncio.TimeoutError:
                logger.warning("Knowledge base timeout, using direct search")
                # Fallback to direct search
                search_result = await self._direct_search_fallback(transcription)
                if search_result:
                    response_result = {"response": search_result}
                else:
                    response_result = {"response": self._get_fallback_response(transcription)}
            
            response = response_result.get("response", "")
            kb_time = time.time() - kb_start
            
            if not response:
                response = self._get_fallback_response(transcription)
                logger.warning("Empty response, using fallback")
            
            logger.info(f"Response generated ({kb_time:.2f}s): {response[:50]}...")
            
            # Cache the response
            if self.use_cache:
                self._cache_response(transcription, response)
            
            # Step 3: Text-to-Speech with optimization
            tts_start = time.time()
            
            try:
                speech_audio = await asyncio.wait_for(
                    self._generate_speech_optimized(response),
                    timeout=1.5  # Reduced from 2.0
                )
            except asyncio.TimeoutError:
                logger.warning("TTS timeout")
                self.error_count += 1
                return {
                    "transcription": transcription,
                    "response": response,
                    "error": "TTS timeout",
                    "total_time": time.time() - start_time
                }
            
            tts_time = time.time() - tts_start
            logger.info(f"TTS completed ({tts_time:.2f}s): {len(speech_audio) if speech_audio else 0} bytes")
            
            # Save audio if requested
            if speech_output_path and speech_audio:
                os.makedirs(os.path.dirname(os.path.abspath(speech_output_path)), exist_ok=True)
                with open(speech_output_path, "wb") as f:
                    f.write(speech_audio)
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.2f}s")
            
            # Track performance
            self.processing_times.append(total_time)
            if len(self.processing_times) > 50:
                self.processing_times.pop(0)
            self.success_count += 1
            
            return {
                "transcription": transcription,
                "response": response,
                "speech_audio": speech_audio,
                "total_time": total_time,
                "timings": {
                    "stt": transcription_time,
                    "kb": kb_time,
                    "tts": tts_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            self.error_count += 1
            return {
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def _transcribe_audio_optimized(self, audio: np.ndarray) -> tuple[str, float]:
        """Optimized audio transcription with proper integration."""
        try:
            # If we have an STT integration object, use it
            if hasattr(self.speech_recognizer, 'transcribe_audio_data'):
                # This is an STTIntegration object
                result = await self.speech_recognizer.transcribe_audio_data(
                    audio_data=audio,
                    is_short_audio=True  # Changed to True for faster processing
                )
                
                transcription = result.get('transcription', '')
                duration = result.get('duration', len(audio) / 16000)
                
                # Clean transcription
                if transcription:
                    transcription = self._clean_transcription(transcription)
                
                return transcription, duration
            else:
                # Direct Google Cloud STT handling
                await self.speech_recognizer.start_streaming()
                
                # Convert to bytes format
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                
                # Track results
                final_results = []
                
                async def collect_result(result):
                    if result.is_final:
                        final_results.append(result)
                
                # Process audio
                await self.speech_recognizer.process_audio_chunk(audio_bytes, collect_result)
                
                # Stop streaming and get result
                transcription, duration = await self.speech_recognizer.stop_streaming()
                
                if transcription:
                    transcription = self._clean_transcription(transcription)
                
                return transcription, duration
                
        except Exception as e:
            logger.error(f"Error in optimized transcription: {e}")
            return "", len(audio) / 16000
    
    async def _direct_search_fallback(self, query: str) -> Optional[str]:
        """Direct Pinecone search as fallback when OpenAI Assistant fails."""
        try:
            # Check if we can access Pinecone directly through the conversation manager
            if hasattr(self.conversation_manager, 'pinecone_manager'):
                results = await self.conversation_manager.pinecone_manager.query(
                    query_text=query,
                    top_k=3,
                    include_metadata=True
                )
                
                if results:
                    # Format results into a response
                    context = ""
                    for result in results:
                        if result.get("metadata") and result["metadata"].get("text"):
                            text = result["metadata"]["text"]
                            context += f"{text}\n\n"
                    
                    if context:
                        # Create a simple response based on the context
                        if any(word in query.lower() for word in ["price", "pricing", "cost", "plan"]):
                            return f"Based on our information: {context[:200]}..."
                        elif any(word in query.lower() for word in ["feature", "features", "capability"]):
                            return f"Our features include: {context[:200]}..."
                        else:
                            return f"Here's what I found: {context[:200]}..."
                
            return None
        except Exception as e:
            logger.error(f"Error in direct search fallback: {e}")
            return None
    
    async def _generate_speech_optimized(self, text: str) -> Optional[bytes]:
        """Generate speech with optimizations and error handling."""
        try:
            if not text:
                return None
            
            # Use TTS integration with retry logic
            for attempt in range(2):  # Reduced from 3 attempts
                try:
                    return await self.tts_integration.text_to_speech(text)
                except Exception as e:
                    logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
                    if attempt < 1:  # Last attempt
                        await asyncio.sleep(0.1)  # Brief retry delay
                    else:
                        logger.error(f"All TTS attempts failed for: {text[:50]}...")
                        return None
            
            return None
                
        except Exception as e:
            logger.error(f"Error in optimized speech generation: {e}")
            return None
    
    async def _generate_speech_parallel(self, text: str) -> Optional[bytes]:
        """Generate speech using parallel processing."""
        try:
            # Use asyncio.create_task for better concurrency
            return await asyncio.create_task(self._generate_speech_optimized(text))
        except Exception as e:
            logger.error(f"Error in parallel speech generation: {e}")
            return None
    
    def _get_cached_response(self, transcription: str) -> Optional[str]:
        """Get cached response for transcription with fuzzy matching."""
        if not self.use_cache:
            return None
        
        # Normalize transcription for cache lookup
        normalized = transcription.lower().strip()
        
        # Check exact match first
        if normalized in self.response_cache:
            return self.response_cache[normalized]
        
        # Check common responses
        for key, response in self._common_responses.items():
            if key in normalized:
                return response
        
        # Check partial matches with better threshold
        words = normalized.split()
        for cached_key in self.response_cache:
            cached_words = cached_key.split()
            if len(cached_words) >= 2:  # Only check meaningful phrases
                # Simple similarity check
                common_words = set(words) & set(cached_words)
                if len(common_words) >= min(2, len(words) // 2):
                    return self.response_cache[cached_key]
        
        return None
    
    def _cache_response(self, transcription: str, response: str):
        """Cache response with expiration."""
        if not self.use_cache:
            return
        
        normalized = transcription.lower().strip()
        self.response_cache[normalized] = response
        
        # Limit cache size
        if len(self.response_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.response_cache.keys())[:10]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _get_fallback_response(self, transcription: str) -> str:
        """Get contextual fallback response."""
        transcription_lower = transcription.lower()
        
        # Question words
        if any(word in transcription_lower for word in ["what", "how", "when", "where", "why", "which"]):
            return "I'd be happy to help answer that. Could you provide a bit more context?"
        
        # Help requests
        elif any(word in transcription_lower for word in ["help", "support", "assist", "need"]):
            return "I'm here to help! What specific information are you looking for?"
        
        # Information requests
        elif any(word in transcription_lower for word in ["tell", "explain", "show", "describe", "information", "info"]):
            return "I can provide information about that. Could you be more specific?"
        
        # Pricing/product queries
        elif any(word in transcription_lower for word in ["price", "cost", "plan", "pricing", "feature"]):
            return "I can help you with pricing and features. Let me search for that information."
        
        # Default
        else:
            return "I understand you have a question. Could you please rephrase it so I can better assist you?"
    
    def _clean_transcription(self, text: str) -> str:
        """Clean transcription for better processing."""
        if not text:
            return ""
        
        # Remove noise indicators
        import re
        cleaned = re.sub(r'\[.*?\]', '', text)  # Remove [noise], [music], etc.
        cleaned = re.sub(r'\(.*?\)', '', text)  # Remove (background noise), etc.
        cleaned = re.sub(r'<.*?>', '', text)    # Remove <unclear>, etc.
        
        # Clean up spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {"error": "No processing times recorded"}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        min_time = min(self.processing_times)
        max_time = max(self.processing_times)
        
        return {
            "average_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "cache_size": len(self.response_cache) if self.use_cache else 0,
            "recent_processing_times": self.processing_times[-10:]
        }