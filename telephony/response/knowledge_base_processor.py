"""
Fixed Knowledge base processing and TTS generation.
"""
import logging
import os
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    """Handles knowledge base queries and TTS generation - Fixed version."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Initialize components properly
        if hasattr(pipeline, 'conversation_manager'):
            self.conversation_manager = pipeline.conversation_manager
        if hasattr(pipeline, 'query_engine'):
            self.query_engine = pipeline.query_engine
            
        # Get TTS from pipeline or create it
        if hasattr(pipeline, 'tts_integration'):
            self.tts_integration = pipeline.tts_integration
        else:
            self.tts_integration = None
    
    async def init_tts(self) -> None:
        """Initialize TTS components if not already initialized."""
        if self.tts_integration and not self.tts_client:
            self.tts_client = self.tts_integration.tts_client
            
        if not self.tts_client:
            try:
                # Import and initialize ElevenLabs TTS directly
                from text_to_speech import ElevenLabsTTS
                
                # Get API key and settings from environment
                api_key = os.environ.get("ELEVENLABS_API_KEY")
                voice_id = os.environ.get("TTS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
                model_id = os.environ.get("TTS_MODEL_ID", "eleven_turbo_v2")
                
                if not api_key:
                    raise ValueError("ELEVENLABS_API_KEY not found in environment")
                
                # Create ElevenLabs TTS client optimized for Twilio
                self.tts_client = ElevenLabsTTS(
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=model_id,
                    container_format="mulaw",  # Use mulaw for Twilio compatibility
                    sample_rate=8000,  # 8kHz for Twilio
                    optimize_streaming_latency=2,  # Balanced optimization for quality
                    enable_caching=True
                )
                
                logger.info(f"Initialized ElevenLabs TTS with voice ID: {voice_id}, model: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing TTS: {e}")
                raise
    
    async def generate_response(self, transcription: str, user_id: str) -> Optional[str]:
        """
        Generate response from knowledge base with improved error handling.
        
        Args:
            transcription: User's transcription
            user_id: User identifier
            
        Returns:
            Generated response text
        """
        if not transcription or not transcription.strip():
            logger.warning("Empty transcription provided")
            return "I didn't catch that. Could you please repeat?"
        
        try:
            response = None
            
            # Try conversation manager first
            if self.conversation_manager:
                logger.info(f"Processing with conversation manager: '{transcription}'")
                response_result = await self.conversation_manager.handle_user_input(
                    user_id=user_id,
                    message=transcription
                )
                
                if response_result and "response" in response_result:
                    response = response_result["response"]
                    
                    # Check if response is empty or generic error message
                    if not response or response.strip() == "I'm sorry, I couldn't generate a response.":
                        logger.warning("Conversation manager returned empty/error response")
                        response = None
                    else:
                        logger.info(f"Conversation manager generated response: {response[:100]}...")
                        return response
            
            # Fallback to query engine if conversation manager failed
            if not response and self.query_engine:
                logger.info(f"Fallback to query engine: '{transcription}'")
                query_result = await self.query_engine.query(transcription, user_id=user_id)
                
                if query_result and "response" in query_result:
                    response = query_result["response"]
                    if response and response.strip():
                        logger.info(f"Query engine generated response: {response[:100]}...")
                        return response
            
            # Final fallback - contextual error message
            if not response:
                logger.warning(f"No response generated for: '{transcription}'")
                # Provide a helpful fallback based on common queries
                if any(word in transcription.lower() for word in ["price", "pricing", "cost", "plan"]):
                    return "I understand you're asking about pricing. We offer several plans to meet different needs. Would you like me to explain our pricing structure?"
                elif any(word in transcription.lower() for word in ["feature", "features", "capability"]):
                    return "I can tell you about our features. What specific functionality are you interested in learning about?"
                else:
                    return "I'm not sure I understood your question correctly. Could you please rephrase it or ask something specific?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I'm experiencing some technical difficulties. Can you please try asking your question again?"
    
    async def generate_speech(self, response: str) -> Optional[bytes]:
        """
        Generate speech audio from text response with retry logic.
        
        Args:
            response: Text response to convert
            
        Returns:
            Audio bytes or None if error
        """
        if not response or not response.strip():
            logger.warning("Empty response provided for speech generation")
            return None
        
        try:
            # Ensure TTS is initialized
            await self.init_tts()
            
            if not self.tts_client:
                logger.error("TTS client not available")
                return None
            
            # Generate speech with retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # Clean up text for better speech synthesis
                    cleaned_text = self._clean_text_for_speech(response)
                    
                    # Log attempt
                    logger.info(f"Generating speech (attempt {attempt + 1}): '{cleaned_text[:50]}...'")
                    
                    # Generate speech
                    speech_audio = await self.tts_client.synthesize(cleaned_text)
                    
                    if speech_audio and len(speech_audio) > 0:
                        logger.info(f"Successfully generated speech: {len(speech_audio)} bytes")
                        return speech_audio
                    else:
                        logger.warning(f"TTS returned empty audio on attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.error(f"TTS attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(0.5)  # Brief delay before retry
                    else:
                        raise
            
            logger.error("All TTS attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}", exc_info=True)
            return None
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean and prepare text for better speech synthesis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove markdown syntax
        import re
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)    # Remove italic
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)      # Remove code blocks
        
        # Clean up excessive punctuation
        cleaned = re.sub(r'([.!?]){2,}', r'\1', cleaned)  # Multiple punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)            # Multiple spaces
        
        # Ensure proper sentence ending
        cleaned = cleaned.strip()
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        return cleaned