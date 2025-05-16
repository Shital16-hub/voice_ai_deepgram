"""
OpenAI-powered LLM integration optimized for telephony conversations.
Uses latest OpenAI models with telephony-specific optimizations.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator
from openai import AsyncOpenAI
import json

from knowledge_base.config import get_openai_config

logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI LLM client optimized for telephony conversations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI LLM client."""
        self.config = config or get_openai_config()
        
        if not self.config["api_key"]:
            raise ValueError("OpenAI API key is required")
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.system_prompt = self.config["system_prompt"]
        
        logger.info(f"Initialized OpenAI LLM with model: {self.model}")
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using OpenAI with telephony optimization.
        
        Args:
            query: User query
            context: Retrieved context from knowledge base
            chat_history: Previous conversation history
            
        Returns:
            Generated response
        """
        messages = self._create_messages(query, context, chat_history)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # Use latest features for better performance
                response_format={"type": "text"},
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
    
    async def generate_streaming_response(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response for real-time telephony.
        
        Args:
            query: User query
            context: Retrieved context from knowledge base
            chat_history: Previous conversation history
            
        Yields:
            Response chunks
        """
        messages = self._create_messages(query, context, chat_history)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield "I'm sorry, I'm having trouble processing that right now."
    
    def _create_messages(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Create messages for OpenAI chat completion."""
        messages = []
        
        # Add system prompt with context
        system_content = self.system_prompt
        if context:
            system_content += f"\n\nRelevant information:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add chat history (limit to recent exchanges for telephony)
        if chat_history:
            for message in chat_history[-6:]:  # Last 3 exchanges
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
            "system_prompt_length": len(self.system_prompt)
        }

def create_telephony_optimized_messages(
    user_message: str,
    context: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Create optimized messages for telephony use case.
    
    Args:
        user_message: Current user message
        context: Retrieved context from knowledge base
        chat_history: Recent conversation history
        
    Returns:
        Formatted messages for OpenAI
    """
    messages = []
    
    # Telephony-optimized system prompt
    system_prompt = """You are a helpful voice assistant for customer support calls.

CRITICAL RULES:
- Keep responses under 30 words when possible
- Speak naturally like a human
- Use simple, clear language
- Never use lists, bullets, or complex formatting
- Sound friendly and professional
- If unsure, say so briefly and offer alternative help
- Stay focused on the customer's immediate need

Context will be provided to help answer questions. Use it naturally in conversation."""
    
    # Add context to system prompt if available
    if context:
        system_prompt += f"\n\nRelevant Information:\n{context}"
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Add recent chat history (keep it short for telephony)
    if chat_history:
        # Only include last 2 exchanges (4 messages max)
        recent_history = chat_history[-4:]
        for msg in recent_history:
            messages.append(msg)
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages