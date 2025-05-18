"""
Configuration settings for the knowledge base component.
"""
import os
from typing import Dict, Any, List, Optional

# Vector database settings - Updated for Pinecone
VECTOR_DIMENSION = 1536  # For OpenAI embeddings (text-embedding-ada-002)

# Embedding model settings - Updated for OpenAI
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DEVICE = "cpu"  # OpenAI is cloud-based, but keeping this for compatibility
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))

# Retrieval settings
DEFAULT_RETRIEVE_COUNT = int(os.getenv("DEFAULT_RETRIEVE_COUNT", "3"))
MINIMUM_RELEVANCE_SCORE = float(os.getenv("MINIMUM_RELEVANCE_SCORE", "0.6"))
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "False").lower() == "true"

# Conversation context settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "4096"))

# LlamaIndex settings
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"

# Supported file types
SUPPORTED_DOCUMENT_TYPES = [
    # Text files
    ".txt", ".md", ".csv", ".json",
    
    # Office documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    
    # Web content
    ".html", ".htm", ".xml",
]

def get_llama_index_config() -> Dict[str, Any]:
    """
    Get LlamaIndex configuration.
    
    Returns:
        Dictionary with LlamaIndex configuration
    """
    return {
        "persist_dir": PERSIST_DIR,
        "use_gpu": USE_GPU,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": EMBEDDING_DEVICE
    }

def get_document_processor_config() -> Dict[str, Any]:
    """
    Get document processor configuration.
    
    Returns:
        Dictionary with document processor configuration
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "supported_types": SUPPORTED_DOCUMENT_TYPES
    }

def get_embedding_config() -> Dict[str, Any]:
    """
    Get embedding generator configuration.
    
    Returns:
        Dictionary with embedding configuration
    """
    return {
        "model_name": EMBEDDING_MODEL,
        "device": EMBEDDING_DEVICE,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "dimension": VECTOR_DIMENSION
    }

def get_vector_db_config() -> Dict[str, Any]:
    """
    Get vector database configuration.
    
    Returns:
        Dictionary with vector database configuration
    """
    # Import Pinecone config for consistency
    from knowledge_base.openai_pinecone_config import get_pinecone_config
    pinecone_config = get_pinecone_config()
    
    # Combine configurations
    return {
        "api_key": pinecone_config["api_key"],
        "environment": pinecone_config["environment"],
        "index_name": pinecone_config["index_name"],
        "namespace": pinecone_config["namespace"],
        "vector_size": VECTOR_DIMENSION
    }

def get_retriever_config() -> Dict[str, Any]:
    """
    Get retriever configuration.
    
    Returns:
        Dictionary with retriever configuration
    """
    return {
        "top_k": DEFAULT_RETRIEVE_COUNT,
        "min_score": MINIMUM_RELEVANCE_SCORE,
        "reranking_enabled": RERANKING_ENABLED
    }