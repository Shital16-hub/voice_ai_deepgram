"""
Embedding model setup for LlamaIndex with OpenAI.
"""
import logging
import os
from typing import Dict, Optional, Any
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding

from knowledge_base.config import get_embedding_config
from knowledge_base.openai_pinecone_config import get_openai_config

logger = logging.getLogger(__name__)

def get_embedding_model(config: Optional[Dict[str, Any]] = None) -> BaseEmbedding:
    """
    Get the OpenAI embedding model for document indexing and querying.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LlamaIndex embedding model
    """
    # Get embedding configuration
    if config is None:
        config = get_embedding_config()
    
    # Get OpenAI configuration
    openai_config = get_openai_config()
    
    # Use OpenAI config for embedding model
    embed_model_name = openai_config["embedding_model"]
    api_key = openai_config["api_key"]
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    try:
        logger.info(f"Initializing OpenAI embedding model: {embed_model_name}")
        
        # Create the OpenAI embedding model
        embed_model = OpenAIEmbedding(
            model=embed_model_name,
            api_key=api_key,
            dimensions=config.get("vector_size", 1536),
            api_base=openai_config.get("api_base"),
            api_type=openai_config.get("api_type"),
            api_version=openai_config.get("api_version")
        )
        
        logger.info(f"Successfully initialized OpenAI embedding model: {embed_model_name}")
        return embed_model
        
    except ImportError as e:
        logger.error(f"Error importing required libraries: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise