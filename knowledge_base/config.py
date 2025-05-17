"""
Configuration settings for OpenAI + Pinecone knowledge base.
CRITICAL FIXES: Ultra-low latency telephony optimizations.
"""
import os
from typing import Dict, Any, List, Optional

# Load environment variables at module level
from dotenv import load_dotenv
load_dotenv()

# CRITICAL: OpenAI Configuration - Ultra-optimized for telephony speed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # FASTEST model
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # UPDATED: Increased for variety
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "100"))  # UPDATED: Increased for longer responses

# CRITICAL: Timeout settings for ultra-fast performance
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "4.0"))
PINECONE_TIMEOUT = float(os.getenv("PINECONE_TIMEOUT", "4.0"))
EMBEDDINGS_TIMEOUT = float(os.getenv("EMBEDDINGS_TIMEOUT", "4.0"))

# Pinecone Configuration - Get from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# CRITICAL: Retrieval settings optimized for ULTRA-FAST responses
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "2"))  # UPDATED: Increased to 2 docs
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.5"))  # UPDATED: Reduced threshold

# CRITICAL: Conversation context settings for telephony
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "2"))  # UPDATED: Increased to 2
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "1024"))

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    ".txt", ".md", ".csv", ".json",
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".xml",
]

# CRITICAL: Updated system prompt for more conversational and longer responses
TELEPHONY_SYSTEM_PROMPT = """You are a helpful voice assistant engaging in phone conversations. Be friendly, personable, and helpful. Keep responses under 3 sentences but make them sound natural and conversational. Maintain a friendly, helpful tone. If you don't know an answer, be honest but helpful."""

# OpenAI embedding dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration optimized for telephony speed."""
    # Re-load environment to get latest values
    load_dotenv(override=True)
    
    # Get API key from multiple sources
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    # If still not found, try reading .env file directly
    if not api_key:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        except FileNotFoundError:
            pass
    
    return {
        "api_key": api_key,
        "model": OPENAI_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "system_prompt": TELEPHONY_SYSTEM_PROMPT,
        "embedding_dimensions": EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536),
        # CRITICAL: Add timeout settings
        "timeout": OPENAI_TIMEOUT,
        "embeddings_timeout": EMBEDDINGS_TIMEOUT,
        # CRITICAL: Updated generation settings
        "top_p": 0.9,             # UPDATED: Higher for more variety
        "frequency_penalty": 0.3, # UPDATED: Added to reduce repetition 
        "presence_penalty": 0.3,  # UPDATED: Added to encourage diverse topics
        "max_retries": 1,
    }

# Rest of the file remains the same

def get_pinecone_config() -> Dict[str, Any]:
    """Get Pinecone configuration with timeout settings."""
    # Re-load environment to get latest values
    load_dotenv(override=True)
    
    # Get API key from multiple sources
    api_key = PINECONE_API_KEY or os.getenv("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY")
    
    # If still not found, try reading .env file directly
    if not api_key:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('PINECONE_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        except FileNotFoundError:
            pass
    
    return {
        "api_key": api_key,
        "environment": PINECONE_ENVIRONMENT,
        "index_name": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "embedding_dimensions": EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536),
        # CRITICAL: Reduced timeout setting
        "timeout": PINECONE_TIMEOUT
    }

def get_document_processor_config() -> Dict[str, Any]:
    """Get document processor configuration optimized for speed."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "supported_types": SUPPORTED_DOCUMENT_TYPES
    }

def get_retriever_config() -> Dict[str, Any]:
    """Get retriever configuration optimized for speed."""
    return {
        "top_k": DEFAULT_RETRIEVE_COUNT,
        "min_score": MINIMUM_RELEVANCE_SCORE,
        "include_metadata": True,
        # CRITICAL: Add timeout for retrieval
        "timeout": PINECONE_TIMEOUT
    }

# CRITICAL: Ultra-fast performance settings for telephony
PERFORMANCE_CONFIG = {
    # Response generation settings
    "max_response_words": 30,         # Increased from 10 for better responses
    "target_response_time": 1.0,       # Target 1 second total
    "streaming_chunk_size": 5,         # Small chunks for TTS
    
    # Retrieval optimization
    "max_context_words": 50,           # CRITICAL: Ultra-limited context
    "retrieval_top_k": 1,              # Only 1 most relevant doc
    "retrieval_min_score": 0.65,       # Optimized balance for speed and relevance
    
    # Model optimization
    "temperature": 0.3,                # Increased from 0 for better quality
    "top_p": 0.3,                      # Increased from 0.1 for more natural responses
    "frequency_penalty": 0.0,          # No penalties for speed
    "presence_penalty": 0.1,           # Small penalty to reduce repetition
    
    # Timeout configuration
    "openai_timeout": 3.0,             # 3 seconds max (reduced from 4)
    "pinecone_timeout": 3.0,           # 3 seconds max (reduced from 4)
    "total_response_timeout": 5.0,     # 5 seconds total (reduced from 10)
}

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration for ultra-fast telephony."""
    return PERFORMANCE_CONFIG