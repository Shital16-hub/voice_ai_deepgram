"""
Pinecone vector database management.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pinecone import Pinecone, PodSpec, ServerlessSpec
import uuid

from knowledge_base.config import get_pinecone_config, get_openai_config
from knowledge_base.exceptions import PineconeError
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class PineconeManager:
    """Manage Pinecone vector database operations."""
    
    def __init__(self):
        """Initialize Pinecone Manager."""
        self.config = get_pinecone_config()
        self.openai_config = get_openai_config()
        self.client = Pinecone(api_key=self.config["api_key"])
        self.openai_client = AsyncOpenAI(api_key=self.openai_config["api_key"])
        self.index = None
        self.index_name = self.config["index_name"]
        
    async def init(self):
        """Initialize Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = self.client.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                # Create index if it doesn't exist
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.config["dimension"],
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.config.get("environment", "us-east-1")
                    )
                )
                
                # Wait for index to be ready
                await asyncio.sleep(5)
            
            # Connect to index
            self.index = self.client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise PineconeError(f"Failed to initialize Pinecone: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            # Process in batches to respect API limits
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.openai_client.embeddings.create(
                    input=batch,
                    model=self.openai_config["embedding_model"]
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise PineconeError(f"Failed to generate embeddings: {str(e)}")
    
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert documents to Pinecone."""
        if not self.index:
            await self.init()
        
        try:
            # Extract texts for embedding
            texts = [doc["text"] for doc in documents]
            embeddings = await self.generate_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vector_id = doc.get("id", str(uuid.uuid4()))
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": doc["text"],
                        "source": doc.get("source", "unknown"),
                        "chunk_index": doc.get("chunk_index", 0),
                        "document_id": doc.get("document_id", ""),
                        **doc.get("metadata", {})
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            upserted_count = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                upserted_count += len(batch)
                
                # Add delay between batches
                if i + batch_size < len(vectors):
                    await asyncio.sleep(0.5)
            
            logger.info(f"Upserted {upserted_count} documents to Pinecone")
            return upserted_count
            
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise PineconeError(f"Failed to upsert documents: {str(e)}")
    
    async def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for similar documents."""
        if not self.index:
            await self.init()
        
        try:
            # Generate embedding for query
            embeddings = await self.generate_embeddings([query_text])
            query_embedding = embeddings[0]
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=False
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                
                if include_metadata and match.metadata:
                    result["text"] = match.metadata.get("text", "")
                    result["source"] = match.metadata.get("source", "unknown")
                
                formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            raise PineconeError(f"Failed to query Pinecone: {str(e)}")
    
    async def delete_documents(self, document_ids: List[str]):
        """Delete documents from Pinecone."""
        if not self.index:
            await self.init()
        
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise PineconeError(f"Failed to delete documents: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self.index:
            await self.init()
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "index_fullness": stats.index_fullness,
                "dimension": stats.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}