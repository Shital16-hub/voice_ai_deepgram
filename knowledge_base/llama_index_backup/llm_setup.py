"""
Ultra Low Latency LLM Configuration for Sub-2s Response
Optimized Ollama settings for fastest possible response generation.
"""
import logging
import os
from typing import Dict, Any, Optional, List

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

# Ultra low latency model configuration
DEFAULT_MODEL = "mistral:7b-instruct-v0.2-q4_0"  # Keep this model but optimize parameters
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 3  # Reduced from 60s to 3s for ultra low latency

# Ultra aggressive LLM parameters for speed
DEFAULT_TEMPERATURE = 0.1      # Much lower for faster, more deterministic responses
DEFAULT_MAX_TOKENS = 150       # Reduced from 1024 to 150 for shorter, faster responses
DEFAULT_CONTEXT_WINDOW = 1024  # Reduced from 4096 to 1024 for speed

def get_ollama_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Ollama:
    """
    Get ultra fast Ollama LLM optimized for <2s response time.
    """
    model = model_name or DEFAULT_MODEL
    base_url = kwargs.pop("base_url", OLLAMA_BASE_URL)
    request_timeout = kwargs.pop("request_timeout", OLLAMA_TIMEOUT)

    # Ultra aggressive parameters for speed
    params = {
        "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        # Ultra fast inference parameters
        "top_k": 1,           # Only consider top token for speed
        "top_p": 0.1,         # Very low top_p for deterministic, fast responses
        "repeat_penalty": 1.0,  # No repeat penalty for speed
        "num_ctx": DEFAULT_CONTEXT_WINDOW,  # Smaller context for speed
        **kwargs
    }
    
    try:
        # Initialize with ultra low latency settings
        ollama_llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
            **params
        )
        
        # Set additional ultra fast parameters
        ollama_llm.additional_kwargs = {
            "num_predict": max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            "num_ctx": DEFAULT_CONTEXT_WINDOW,
            "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
            "top_k": 1,
            "top_p": 0.1,
        }
        
        logger.info(f"Initialized ultra fast Ollama LLM: {model} (timeout: {request_timeout}s)")
        return ollama_llm
        
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {e}")
        raise

def format_system_prompt(
    base_prompt: str,
    retrieved_context: Optional[str] = None
) -> str:
    """
    Create ultra concise system prompt for fastest processing.
    """
    # Ultra concise base prompt
    concise_prompt = """You are a customer support AI. Answer briefly and directly using the provided context. Keep responses under 30 words when possible."""
    
    if retrieved_context:
        # Truncate context for speed
        max_context_length = 500  # Much shorter context
        if len(retrieved_context) > max_context_length:
            retrieved_context = retrieved_context[:max_context_length] + "..."
        
        return f"{concise_prompt}\n\nContext: {retrieved_context}"
    
    return concise_prompt

def create_chat_messages(
    system_prompt: str,
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[ChatMessage]:
    """
    Create minimal chat messages for fastest processing.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    ]
    
    # Include only the last exchange for speed
    if chat_history and len(chat_history) > 0:
        # Only include the very last exchange
        last_messages = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        
        for message in last_messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                messages.append(ChatMessage(role=MessageRole.USER, content=content))
            elif role == "assistant":
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
    
    # Add current user message
    messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
    
    return messages

# Ultra fast query processing
def optimize_query_for_speed(query: str) -> str:
    """Optimize query for fastest processing."""
    # Keep query short and direct
    words = query.split()
    if len(words) > 20:
        # Truncate very long queries
        query = " ".join(words[:20]) + "?"
    
    return query

# Optimized knowledge base configuration
KB_CONFIG = {
    "chunk_size": 256,           # Smaller chunks for faster retrieval
    "chunk_overlap": 20,         # Minimal overlap
    "top_k": 2,                  # Retrieve only 2 most relevant chunks
    "min_score": 0.3,            # Lower threshold for speed
}

# Ultra fast conversation manager settings
CONVERSATION_CONFIG = {
    "max_history_turns": 3,      # Keep only last 3 turns
    "context_window_tokens": 512, # Much smaller context window
    "use_langgraph": False,      # Disabled for speed
    "skip_greeting": True,       # Skip greeting for telephony
}