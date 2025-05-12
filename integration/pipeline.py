"""
Optimized pipeline for reduced latency - Updated for OpenAI + Pinecone.
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
        
        # Timing optimizations
        self.min_audio_duration = 0.5  # Minimum audio duration to process
        self.max_response_time = 3.0   # Maximum time for response generation
        
        # Pre-compute responses for common queries
        self._common_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "bye": "Thank you for calling. Have a great day!",
            "goodbye": "Goodbye! Feel free to call us again if you need help."
        }
        
        # Executor for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    async def process_audio_data(
        self,
        audio_data: Union[bytes, np.ndarray],
        user_id: Optional[str] = None,
        speech_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio with optimized latency.
        
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
            
            # Step 1: Speech-to-Text (async with timeout)
            transcription_start = time.time()
            
            try:
                transcription_task = asyncio.create_task(self._transcribe_audio_optimized(audio))
                transcription, stt_duration = await asyncio.wait_for(
                    transcription_task, 
                    timeout=2.0  # Maximum 2 seconds for STT
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
                    return {
                        "transcription": transcription,
                        "response": cached_response,
                        "speech_audio": speech_audio,
                        "total_time": time.time() - start_time,
                        "cached": True
                    }
            
            # Step 2: Knowledge Base Query (parallel with speech preparation)
            kb_start = time.time()
            
            # Start TTS preparation in parallel
            if self.parallel_processing:
                tts_init_task = asyncio.create_task(self._prepare_tts())
            
            try:
                # Use conversation manager with timeout
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
                logger.warning("Knowledge base timeout, using fallback")
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
            
            # Step 3: Text-to-Speech (optimized)
            tts_start = time.time()
            
            # Wait for TTS preparation if running in parallel
            if self.parallel_processing and 'tts_init_task' in locals():
                await tts_init_task
            
            try:
                speech_audio = await asyncio.wait_for(
                    self._generate_speech_optimized(response),
                    timeout=2.0  # Maximum 2 seconds for TTS
                )
            except asyncio.TimeoutError:
                logger.warning("TTS timeout")
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
            return {
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def _transcribe_audio_optimized(self, audio: np.ndarray) -> tuple[str, float]:
        """Optimized audio transcription."""
        try:
            # If we have an STT integration object, use it
            if hasattr(self.speech_recognizer, 'transcribe_audio_data'):
                # This is an STTIntegration object
                result = await self.speech_recognizer.transcribe_audio_data(
                    audio_data=audio,
                    is_short_audio=False
                )
                
                transcription = result.get('transcription', '')
                duration = result.get('duration', len(audio) / 16000)
                
                # Clean transcription
                if hasattr(self, '_clean_transcription'):
                    transcription = self._clean_transcription(transcription)
                
                return transcription, duration
            else:
                # This is a raw GoogleCloudStreamingSTT object
                # Start streaming session if not started
                if not (hasattr(self.speech_recognizer, 'is_streaming') and self.speech_recognizer.is_streaming):
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
                
                # Get final result
                if final_results:
                    best_result = max(final_results, key=lambda r: r.confidence)
                    transcription = best_result.text
                    duration = best_result.end_time - best_result.start_time if best_result.end_time > 0 else len(audio) / 16000
                else:
                    # Fallback: stop and restart streaming to get result
                    transcription, duration = await self.speech_recognizer.stop_streaming()
                    if transcription:
                        await self.speech_recognizer.start_streaming()  # Restart for next use
                
                # Clean transcription
                if hasattr(self, '_clean_transcription'):
                    transcription = self._clean_transcription(transcription)
                
                return transcription, duration
                
        except Exception as e:
            logger.error(f"Error in optimized transcription: {e}")
            return "", len(audio) / 16000
    
    async def _prepare_tts(self):
        """Prepare TTS client in parallel."""
        try:
            if hasattr(self.tts_integration, 'init'):
                await self.tts_integration.init()
        except Exception as e:
            logger.error(f"Error preparing TTS: {e}")
    
    async def _generate_speech_optimized(self, text: str) -> Optional[bytes]:
        """Generate speech with optimizations."""
        try:
            if not text:
                return None
            
            # Use TTS integration
            if hasattr(self.tts_integration, 'text_to_speech'):
                return await self.tts_integration.text_to_speech(text)
            else:
                logger.error("TTS integration missing text_to_speech method")
                return None
                
        except Exception as e:
            logger.error(f"Error in optimized speech generation: {e}")
            return None
    
    async def _generate_speech_parallel(self, text: str) -> Optional[bytes]:
        """Generate speech using parallel processing."""
        try:
            # Run TTS in executor for better performance
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                lambda: asyncio.run(self._generate_speech_optimized(text))
            )
        except Exception as e:
            logger.error(f"Error in parallel speech generation: {e}")
            return None
    
    def _get_cached_response(self, transcription: str) -> Optional[str]:
        """Get cached response for transcription."""
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
        
        # Check partial matches
        for cached_key in self.response_cache:
            if len(cached_key) > 3 and cached_key in normalized:
                return self.response_cache[cached_key]
        
        return None
    
    def _cache_response(self, transcription: str, response: str):
        """Cache response for future use."""
        if not self.use_cache:
            return
        
        normalized = transcription.lower().strip()
        self.response_cache[normalized] = response
        
        # Limit cache size
        if len(self.response_cache) > self.max_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.response_cache.keys())[:10]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _get_fallback_response(self, transcription: str) -> str:
        """Get fallback response based on transcription - works for any domain."""
        transcription_lower = transcription.lower()
        
        # Question words - suggest clarification
        if any(word in transcription_lower for word in ["what", "how", "when", "where", "why", "which"]):
            return "I'd be happy to help answer that. Could you provide a bit more context or detail?"
        
        # Help requests - offer assistance
        elif any(word in transcription_lower for word in ["help", "support", "assist", "need"]):
            return "I'm here to help! What specific information are you looking for?"
        
        # Information requests - ask for specifics
        elif any(word in transcription_lower for word in ["tell", "explain", "show", "describe", "information", "info"]):
            return "I can provide information about that. Could you be more specific about what you'd like to know?"
        
        # Comparison requests - ask for clarification
        elif any(word in transcription_lower for word in ["compare", "difference", "better", "versus", "vs"]):
            return "I can help you compare different options. What specifically would you like me to compare?"
        
        # Yes/No clarifications
        elif transcription_lower in ["yes", "yeah", "yep", "no", "nope"]:
            return "Could you please provide more context about what you'd like to know?"
        
        # Very short responses
        elif len(transcription_lower.split()) < 3:
            return "I didn't quite catch that. Could you tell me more about what you're looking for?"
        
        # Default - generic response that works for any domain
        else:
            return "I understand you have a question. Could you please rephrase it or provide more details so I can better assist you?"
    
    def _clean_transcription(self, text: str) -> str:
        """Clean transcription for better processing."""
        if not text:
            return ""
        
        # Remove noise indicators
        import re
        cleaned = re.sub(r'\[.*?\]', '', text)
        cleaned = re.sub(r'\(.*?\)', '', text)
        cleaned = re.sub(r'<.*?>', '', text)
        
        # Clean up spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned