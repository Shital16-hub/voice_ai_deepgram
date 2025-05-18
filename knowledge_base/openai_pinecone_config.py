"""
Configuration settings for OpenAI and Pinecone integrations.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "open_ai")  # Can be set to "azure" for Azure OpenAI
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-05-15")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # Optional API base URL

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-ai-agent")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "voice-assistant")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))  # 1536 for text-embedding-ada-002
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

def get_openai_config():
    """
    Get OpenAI API configuration.
    
    Returns:
        Dictionary with OpenAI configuration
    """
    return {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "api_type": OPENAI_API_TYPE,
        "api_version": OPENAI_API_VERSION,
        "api_base": OPENAI_API_BASE
    }

def get_pinecone_config():
    """
    Get Pinecone configuration.
    
    Returns:
        Dictionary with Pinecone configuration
    """
    return {
        "api_key": PINECONE_API_KEY,
        "environment": PINECONE_ENVIRONMENT,
        "index_name": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "dimension": PINECONE_DIMENSION,
        "metric": PINECONE_METRIC
    }