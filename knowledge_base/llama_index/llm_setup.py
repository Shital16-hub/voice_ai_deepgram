"""
LLM setup for LlamaIndex integration with OpenAI.
"""
import logging
import os
from typing import Dict, Any, Optional, List

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

from knowledge_base.openai_pinecone_config import get_openai_config

logger = logging.getLogger(__name__)

def check_openai_availability(api_key: Optional[str] = None) -> bool:
    """
    Check if OpenAI API is accessible and properly configured.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        True if OpenAI API is available
    """
    import openai
    
    try:
        # Set up API key
        local_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not local_api_key:
            logger.error("OpenAI API key not provided")
            return False
        
        # Create a simple client to test API access
        client = openai.OpenAI(api_key=local_api_key)
        models = client.models.list()
        
        return len(models.data) > 0
    except Exception as e:
        logger.error(f"OpenAI not available: {e}")
        return False

def get_openai_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> OpenAI:
    """
    Get OpenAI LLM for LlamaIndex integration.
    
    Args:
        model_name: Model name to use (defaults to environment or gpt-3.5-turbo)
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters to pass to OpenAI
        
    Returns:
        Configured OpenAI LLM
    """
    # Get OpenAI config
    openai_config = get_openai_config()
    
    # Get parameters with fallbacks to config values
    model = model_name or openai_config["model"]
    api_key = kwargs.pop("api_key", openai_config["api_key"])
    api_base = kwargs.pop("api_base", openai_config["api_base"])
    api_type = kwargs.pop("api_type", openai_config["api_type"])
    api_version = kwargs.pop("api_version", openai_config["api_version"])

    # Check if OpenAI is available
    if not check_openai_availability(api_key):
        raise ValueError(f"OpenAI API not available. Please check your API key and connection.")
    
    # Build parameters
    params = {
        "temperature": temperature if temperature is not None else openai_config["temperature"],
        "max_tokens": max_tokens if max_tokens is not None else openai_config["max_tokens"],
        **kwargs
    }
    
    try:
        # Initialize OpenAI LLM
        openai_llm = OpenAI(
            model=model,
            api_key=api_key,
            api_base=api_base if api_base else None,
            api_type=api_type,
            api_version=api_version,
            **params
        )
        
        logger.info(f"Initialized OpenAI LLM with model: {model}")
        return openai_llm
        
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLM: {e}")
        raise

def setup_global_llm(model_name: Optional[str] = None, **kwargs) -> OpenAI:
    """
    Set up LlamaIndex to use OpenAI LLM globally.
    
    Args:
        model_name: Model name to use
        **kwargs: Additional parameters to pass to OpenAI
        
    Returns:
        The configured OpenAI LLM instance
    """
    # Get OpenAI LLM
    openai_llm = get_openai_llm(model_name, **kwargs)
    
    # Set as global LLM
    Settings.llm = openai_llm
    
    logger.info(f"Set OpenAI LLM as global LlamaIndex LLM with model: {openai_llm.model}")
    return openai_llm

def format_system_prompt(
    base_prompt: str,
    retrieved_context: Optional[str] = None
) -> str:
    """
    Format system prompt with optional retrieved context.
    
    Args:
        base_prompt: Base system prompt
        retrieved_context: Optional context from retrieval
        
    Returns:
        Formatted system prompt
    """
    if not retrieved_context:
        return base_prompt
    
    # Add the retrieved context to the system prompt
    return f"{base_prompt}\n\nRelevant information:\n{retrieved_context}"

def create_chat_messages(
    system_prompt: str,
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[ChatMessage]:
    """
    Create formatted chat messages for LLM.
    
    Args:
        system_prompt: System prompt
        user_message: User message
        chat_history: Optional chat history
        
    Returns:
        List of ChatMessage objects
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    ]
    
    # Add chat history if provided
    if chat_history:
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                messages.append(ChatMessage(role=MessageRole.USER, content=content))
            elif role == "assistant":
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
    
    # Add user message
    messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
    
    return messages

# LangGraph preparation - these functions will be enhanced later
def get_llm_node_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the input needed for the LLM node from the state.
    
    Args:
        state: The current state object
        
    Returns:
        The input for the LLM node
    """
    # For now, just return a simple extraction, will be enhanced for LangGraph
    return {
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "chat_history": state.get("history", [])
    }

def prepare_for_langgraph():
    """
    Placeholder function to prepare for LangGraph integration.
    Will be implemented when migrating to LangGraph.
    """
    logger.info("LangGraph integration will be implemented in a future update")
    pass