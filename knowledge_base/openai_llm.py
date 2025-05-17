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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI LLM client with timeout protection."""
        self.config = config or get_openai_config()
        
        if not self.config["api_key"]:
            raise ValueError("OpenAI API key is required")
        
        # CRITICAL FIX: Initialize with timeout settings
        self.client = AsyncOpenAI(
            api_key=self.config["api_key"],
            timeout=30.0  # CRITICAL: Add timeout to prevent hanging
        )
        
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.system_prompt = self.config["system_prompt"]
        
        logger.info(f"Initialized OpenAI LLM with model: {self.model} (timeout: 30s)")
    
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
            # CRITICAL FIX: Add timeout to OpenAI call
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # Use latest features for better performance
                    response_format={"type": "text"},
                    stream=False,
                    timeout=15.0  # CRITICAL: Specific timeout for generation
                ),
                timeout=20.0  # CRITICAL: Overall timeout
            )
            
            result = response.choices[0].message.content.strip()
            
            # CRITICAL FIX: Ensure response is short for telephony
            if len(result.split()) > 15:
                # Truncate to first sentence if too long
                sentences = result.split('.')
                if sentences:
                    result = sentences[0] + '.'
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out")
            return "I'm processing that request. Could you please try again?"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble processing that right now. Could you rephrase your question?"
    
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
            # CRITICAL FIX: Add timeout to streaming
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=15.0
                ),
                timeout=5.0  # Quick timeout for stream creation
            )
            
            word_count = 0
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    # CRITICAL FIX: Limit response length for telephony
                    if word_count >= 15:
                        break
                    
                    # Count words
                    word_count += len(content.split())
                    
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
            # CRITICAL FIX: Truncate context for telephony
            if len(context) > 500:
                context = context[:497] + "..."
            system_content += f"\n\nRelevant information:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add chat history (limit to recent exchanges for telephony)
        if chat_history:
            # CRITICAL FIX: Only last 2 exchanges for speed
            for message in chat_history[-4:]:  # Last 2 exchanges
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
            "timeout_configured": True
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
    
    # CRITICAL FIX: Ultra-optimized system prompt for telephony
    system_prompt = """You are a helpful voice assistant for customer support calls.

CRITICAL RULES:
- Keep ALL responses under 15 words
- Never use lists, bullets, or complex formatting
- Speak naturally like a human
- Be direct and helpful
- If you don't know something, say "I'm not sure about that"
- No introductory phrases
- Get straight to the point

Context will be provided to help answer questions. Use it naturally in conversation."""
    
    # Add context to system prompt if available
    if context:
        # CRITICAL FIX: Limit context length
        if len(context) > 300:
            context = context[:297] + "..."
        system_prompt += f"\n\nRelevant Information:\n{context}"
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # CRITICAL FIX: Add minimal chat history for telephony
    if chat_history:
        # Only include last exchange (2 messages max)
        recent_history = chat_history[-2:]
        for msg in recent_history:
            messages.append(msg)
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages