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
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))  # CRITICAL: Zero for maximum speed
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "30"))       # CRITICAL: Very short responses

# CRITICAL: Timeout settings for ultra-fast performance
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "4.0"))           # CRITICAL: Reduced to 4s
PINECONE_TIMEOUT = float(os.getenv("PINECONE_TIMEOUT", "4.0"))       # CRITICAL: Reduced to 4s
EMBEDDINGS_TIMEOUT = float(os.getenv("EMBEDDINGS_TIMEOUT", "4.0"))   # CRITICAL: Reduced to 4s

# Pinecone Configuration - Get from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))      # CRITICAL: Smaller chunks
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20")) # CRITICAL: Less overlap
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# CRITICAL: Retrieval settings optimized for ULTRA-FAST responses
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "1"))  # CRITICAL: Just one doc!
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.7"))  # Higher threshold

# CRITICAL: Conversation context settings for telephony
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "1"))  # CRITICAL: Minimal history
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "512"))         # CRITICAL: Tiny context

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    ".txt", ".md", ".csv", ".json",
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".xml",
]

# CRITICAL: Ultra-short system prompt for tiny responses
TELEPHONY_SYSTEM_PROMPT = """You are a telephony assistant. Keep ALL responses under 10 words. Be clear, direct and helpful."""

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
        # CRITICAL: Add ultra-fast settings
        "top_p": 0.1,             # Very focused output 
        "frequency_penalty": 0.0, # No penalties for speed
        "presence_penalty": 0.0,  # No penalties for speed
        "max_retries": 1,         # Minimal retries for speed
    }

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
    "max_response_words": 10,          # CRITICAL: Extremely short responses
    "target_response_time": 1.0,       # Target 1 second total
    "streaming_chunk_size": 5,         # Small chunks for TTS
    
    # Retrieval optimization
    "max_context_words": 50,           # CRITICAL: Ultra-limited context
    "retrieval_top_k": 1,              # Only 1 most relevant doc
    "retrieval_min_score": 0.7,        # Higher threshold for speed
    
    # Model optimization
    "temperature": 0.0,                # Zero for maximum determinism
    "top_p": 0.1,                      # Very focused generation
    "frequency_penalty": 0.0,          # No penalties for speed
    "presence_penalty": 0.0,           # No penalties for speed
    
    # Timeout configuration
    "openai_timeout": 4.0,             # 4 seconds max
    "pinecone_timeout": 4.0,           # 4 seconds max
    "total_response_timeout": 10.0,    # 10 seconds total
}

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration for ultra-fast telephony."""
    return PERFORMANCE_CONFIG