"""
Configuration settings for OpenAI + Pinecone knowledge base.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-knowledge")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_DIMENSION = 1536  # For text-embedding-3-small

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# Retrieval Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MINIMUM_SIMILARITY_SCORE = float(os.getenv("MINIMUM_SIMILARITY_SCORE", "0.75"))

# Cache Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

# Rate Limiting
MAX_TOKENS_PER_DAY = int(os.getenv("MAX_TOKENS_PER_DAY", "1000000"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    ".txt", ".md", ".pdf", ".docx", ".doc", 
    ".csv", ".json", ".html", ".htm"
]

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration."""
    return {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "max_tokens": OPENAI_MAX_TOKENS,
        "temperature": OPENAI_TEMPERATURE
    }

def get_pinecone_config() -> Dict[str, Any]:
    """Get Pinecone configuration."""
    return {
        "api_key": PINECONE_API_KEY,
        "index_name": PINECONE_INDEX_NAME,
        "environment": PINECONE_ENVIRONMENT,
        "dimension": PINECONE_DIMENSION
    }

def get_processing_config() -> Dict[str, Any]:
    """Get document processing configuration."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "supported_types": SUPPORTED_DOCUMENT_TYPES
    }

def get_retrieval_config() -> Dict[str, Any]:
    """Get retrieval configuration."""
    return {
        "top_k": DEFAULT_TOP_K,
        "min_similarity": MINIMUM_SIMILARITY_SCORE
    }