"""
Knowledge base processing and TTS generation with proper integration.
"""
import logging
import os
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    """Handles knowledge base queries and TTS generation with proper integration."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.conversation_manager = None
        self.query_engine = None
        self.tts_integration = None
        
        # Get components from pipeline
        if hasattr(pipeline, 'conversation_manager'):
            self.conversation_manager = pipeline.conversation_manager
        if hasattr(pipeline, 'query_engine'):
            self.query_engine = pipeline.query_engine
        if hasattr(pipeline, 'tts_integration'):
            self.tts_integration = pipeline.tts_integration
            
        logger.info("Knowledge base processor initialized with pipeline components")
    
    async def init_tts(self) -> None:
        """Initialize TTS components if not already initialized."""
        if self.tts_integration and self.tts_integration.initialized:
            logger.info("TTS already initialized")
            return
            
        if not self.tts_integration:
            logger.error("TTS integration not found in pipeline")
            raise ValueError("TTS integration not available")
    
    async def generate_response(self, transcription: str, user_id: str) -> Optional[str]:
        """
        Generate response from knowledge base.
        
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
                    
                    if response and response.strip():
                        logger.info(f"Conversation manager generated response: {response[:100]}...")
                        return response
                    else:
                        logger.warning("Conversation manager returned empty response")
            
            # Fallback to query engine
            if not response and self.query_engine:
                logger.info(f"Fallback to query engine: '{transcription}'")
                query_result = await self.query_engine.query(transcription, user_id=user_id)
                
                if query_result and "response" in query_result:
                    response = query_result["response"]
                    if response and response.strip():
                        logger.info(f"Query engine generated response: {response[:100]}...")
                        return response
            
            # Final fallback
            if not response:
                logger.warning(f"No response generated for: '{transcription}'")
                return self._get_fallback_response(transcription)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I'm experiencing some technical difficulties. Can you please try asking your question again?"
    
    async def generate_speech(self, response: str) -> Optional[bytes]:
        """
        Generate speech audio from text response.
        
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
            
            if not self.tts_integration:
                logger.error("TTS integration not available")
                return None
            
            # Generate speech with retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # Clean up text for better speech synthesis
                    cleaned_text = self._clean_text_for_speech(response)
                    
                    # Log attempt
                    logger.info(f"Generating speech (attempt {attempt + 1}): '{cleaned_text[:50]}...'")
                    
                    # Generate speech through TTS integration
                    speech_audio = await self.tts_integration.text_to_speech(cleaned_text)
                    
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
    
    def _get_fallback_response(self, transcription: str) -> str:
        """Get contextual fallback response - works for any domain."""
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
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean and prepare text for better speech synthesis."""
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