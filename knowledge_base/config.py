"""
Configuration settings for OpenAI + Pinecone knowledge base.
Optimized for ultra-low latency telephony applications.
CLEAN VERSION - Fixed import issues
"""
import os
from typing import Dict, Any, List, Optional

# Load environment variables at module level
from dotenv import load_dotenv
load_dotenv()

# OpenAI Configuration - Get from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

# Pinecone Configuration - Get from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# Retrieval settings
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "3"))
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.7"))

# Conversation context settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "3"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "2048"))

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    ".txt", ".md", ".csv", ".json",
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".xml",
]

# Telephony-optimized prompts
TELEPHONY_SYSTEM_PROMPT = """You are a helpful voice assistant for customer support calls. 

CRITICAL INSTRUCTIONS:
- Keep responses under 30 words when possible
- Speak naturally and conversationally 
- Use simple, clear language
- Avoid lists, bullet points, or complex formatting
- Sound human and friendly
- If you don't know something, say so briefly and offer to help differently
- Stay focused on the customer's question
"""

# OpenAI embedding dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration optimized for telephony."""
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
        "embedding_dimensions": EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536)
    }

def get_pinecone_config() -> Dict[str, Any]:
    """Get Pinecone configuration."""
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
        "embedding_dimensions": EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536)
    }

def get_document_processor_config() -> Dict[str, Any]:
    """Get document processor configuration."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "supported_types": SUPPORTED_DOCUMENT_TYPES
    }

def get_retriever_config() -> Dict[str, Any]:
    """Get retriever configuration."""
    return {
        "top_k": DEFAULT_RETRIEVE_COUNT,
        "min_score": MINIMUM_RELEVANCE_SCORE,
        "include_metadata": True
    }