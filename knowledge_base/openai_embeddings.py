"""
OpenAI embeddings integration for the knowledge base.
Uses latest OpenAI embedding models for optimal performance.
CLEAN VERSION - Simplified and fixed
"""
import logging
import asyncio
import os
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIEmbeddings:
    """OpenAI embeddings client optimized for telephony."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI embeddings client."""
        # Load environment variables first
        load_dotenv(override=True)
        
        # Get API key from multiple sources
        api_key = None
        
        # 1. From config parameter
        if config and config.get("api_key"):
            api_key = config["api_key"]
        
        # 2. From environment variables
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        # 3. Try reading from .env file directly
        if not api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except FileNotFoundError:
                pass
        
        # Final check
        if not api_key:
            logger.error("OpenAI API key not found. Please check your .env file.")
            raise ValueError("OpenAI API key is required")
        
        # Initialize configuration
        self.config = config or {}
        self.config["api_key"] = api_key
        
        # Set model parameters
        self.model = self.config.get("embedding_model", "text-embedding-3-small")
        self.dimensions = self.config.get("embedding_dimensions", 1536)
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text using OpenAI's latest embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            return [0.0] * self.dimensions
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                for data in response.data:
                    batch_embeddings.append(data.embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed embeddings
                for _ in batch:
                    batch_embeddings.append([0.0] * self.dimensions)
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions

async def get_embedding_model(config: Optional[Dict[str, Any]] = None) -> OpenAIEmbeddings:
    """
    Get OpenAI embedding model instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        OpenAI embeddings instance
    """
    return OpenAIEmbeddings(config)