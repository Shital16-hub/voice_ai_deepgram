# knowledge_base/openai_llm.py - FIXED VERSION

"""
OpenAI-powered LLM integration optimized for telephony conversations.
CRITICAL FIXES: Added timeouts, better error handling, and ultra-fast responses.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator
from openai import AsyncOpenAI
import json

from knowledge_base.config import get_openai_config

logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI LLM client optimized for telephony conversations with CRITICAL fixes."""
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize OpenAI LLM client with timeout protection."""
        self.config = config or get_openai_config()
        
        if not self.config["api_key"]:
            raise ValueError("OpenAI API key is required")
        
        # CRITICAL FIX: Initialize with timeout settings
        self.client = AsyncOpenAI(
            api_key=self.config["api_key"],
            timeout=5.0  # Increased default timeout slightly for better reliability
        )
        
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.system_prompt = self.config["system_prompt"]
        
        # CRITICAL FIX: Extract extra performance settings
        self.top_p = self.config.get("top_p", 0.3)  # Increased from 0.1 for better variety
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.1)  # Added small penalty
        self.request_timeout = self.config.get("timeout", 3.0)  # 3s timeout for request
        
        logger.info(f"Initialized OpenAI LLM with model: {self.model} (timeout: {self.request_timeout}s)")
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using OpenAI with timeout protection.
        
        Args:
            query: User query
            context: Retrieved context from knowledge base
            chat_history: Previous conversation history
            
        Returns:
            Generated response
        """
        messages = self._create_messages(query, context, chat_history)
        
        try:
            # CRITICAL FIX: Add ultra-fast performance settings and timeout
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # CRITICAL FIX: Optimized settings
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    response_format={"type": "text"},
                    stream=False,
                    timeout=self.request_timeout
                ),
                timeout=self.request_timeout + 1.0  # Add 1s buffer for async timeout
            )
            
            result = response.choices[0].message.content.strip()
            
            # CRITICAL FIX: Ensure response is appropriate length for telephony
            # If too long, keep just first 2 sentences
            sentences = result.split('.')
            if len(sentences) > 2:
                result = '.'.join(sentences[:2]).strip() + '.'
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out")
            return "I'm processing that. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble right now. Please try again."
    
    async def generate_streaming_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response for real-time telephony with timeout protection.
        
        Args:
            query: User query
            context: Retrieved context from knowledge base
            chat_history: Previous conversation history
            
        Yields:
            Response chunks
        """
        messages = self._create_messages(query, context, chat_history)
        
        try:
            # CRITICAL FIX: Add ultra-fast streaming
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # CRITICAL FIX: Optimized settings for better responses
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=True,
                    timeout=self.request_timeout
                ),
                timeout=2.0  # Quick timeout for stream creation
            )
            
            word_count = 0
            sentence_count = 0
            current_sentence = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    current_sentence += content
                    
                    # Check if we've completed a sentence
                    if '.' in content or '!' in content or '?' in content:
                        sentence_count += 1
                    
                    # CRITICAL FIX: Limit response length for telephony
                    word_count += len(content.split())
                    if word_count > 30 or sentence_count >= 2:  # Increased from 12 to 30 words, max 2 sentences
                        yield content
                        break
                    
                    yield content
                    
        except asyncio.TimeoutError:
            logger.error("OpenAI streaming timed out")
            yield "I'm having trouble processing that right now."
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield "I'm sorry, could you please try again?"
    
    def _create_messages(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Create messages for OpenAI chat completion with telephony optimization."""
        messages = []
        
        # Add system prompt with context
        system_content = self.system_prompt
        if context:
            # CRITICAL FIX: Include context but keep it brief
            system_content += f"\n\nRelevant information:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add chat history (limit to most recent exchange for telephony)
        if chat_history:
            # CRITICAL FIX: Include two exchanges for better context
            for message in chat_history[-4:]:  # Last two exchanges (four messages)
                messages.append(message)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_length": len(self.system_prompt),
            "timeout_configured": self.request_timeout,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

def create_telephony_optimized_messages(
    user_message: str,
    context: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Create optimized messages for telephony use case with CRITICAL fixes.
    
    Args:
        user_message: Current user message
        context: Retrieved context from knowledge base
        chat_history: Recent conversation history
        
    Returns:
        Formatted messages for OpenAI
    """
    messages = []
    
    # CRITICAL FIX: Improved system prompt for telephony
    system_prompt = """You are a helpful voice assistant. Keep responses brief but conversational (under 2-3 sentences). Be friendly and engaging."""
    
    # Add context to system prompt if available
    if context:
        # CRITICAL FIX: Extremely limit context length
        if len(context) > 200:  # Super short!
            context = context[:197] + "..."
        system_prompt += f"\n\nRelevant Information:\n{context}"
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # CRITICAL FIX: Add minimal chat history for telephony
    if chat_history:
        # Include last two messages for better context
        recent_messages = chat_history[-2:]
        for message in recent_messages:
            messages.append(message)
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages