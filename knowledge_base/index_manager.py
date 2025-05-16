"""
Index management for Pinecone vector store.
Replaces LlamaIndex with pure Pinecone + OpenAI integration.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

from knowledge_base.config import get_pinecone_config
from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.openai_embeddings import OpenAIEmbeddings
from knowledge_base.schema import Document

logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manage Pinecone vector index for document storage and retrieval.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[OpenAIEmbeddings] = None
    ):
        """
        Initialize IndexManager.
        
        Args:
            config: Optional Pinecone configuration
            embedding_model: Optional pre-initialized embeddings model
        """
        self.config = config or get_pinecone_config()
        self.is_initialized = False
        
        # Initialize embeddings
        self.embeddings = embedding_model
        if not self.embeddings:
            self.embeddings = OpenAIEmbeddings()
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            embeddings=self.embeddings,
            config=self.config
        )
        
        logger.info(f"Initialized IndexManager for index: {self.config['index_name']}")
    
    async def init(self):
        """Initialize the index manager and vector store."""
        if self.is_initialized:
            return
        
        try:
            # Initialize vector store
            await self.vector_store.init()
            
            # Get initial stats
            doc_count = await self.vector_store.count_documents()
            logger.info(f"Index initialized with {doc_count} documents")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing IndexManager: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        if not self.is_initialized:
            await self.init()
        
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        try:
            # Add documents to vector store
            doc_ids = await self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully added {len(doc_ids)} documents to index")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            return []
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in the index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            results = await self.vector_store.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filter_metadata=filter_metadata
            )
            
            logger.debug(f"Search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents from the index.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            deleted_count = await self.vector_store.delete_documents(doc_ids)
            logger.info(f"Deleted {deleted_count} documents from index")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    async def count_documents(self) -> int:
        """
        Count documents in the index.
        
        Returns:
            Number of documents
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            count = await self.vector_store.count_documents()
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        Reset the index by deleting all documents.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            success = await self.vector_store.reset_index()
            if success:
                logger.info("Successfully reset index")
            return success
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_stats()
            
            # Add additional stats
            stats = {
                "vector_store": vector_stats,
                "index_name": self.config["index_name"],
                "namespace": self.config["namespace"],
                "embedding_model": self.embeddings.model,
                "embedding_dimensions": self.embeddings.dimensions,
                "is_initialized": self.is_initialized
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def get_embeddings_model(self) -> OpenAIEmbeddings:
        """Get the embeddings model instance."""
        return self.embeddings
    
    def get_vector_store(self) -> PineconeVectorStore:
        """Get the vector store instance."""
        return self.vector_store