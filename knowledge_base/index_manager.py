"""
Index management for Pinecone vector store.
CRITICAL FIXES: Comprehensive timeout handling and retry logic.
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
    FIXED Manage Pinecone vector index with comprehensive error handling and timeouts.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[OpenAIEmbeddings] = None
    ):
        """
        Initialize IndexManager with CRITICAL FIXES.
        
        Args:
            config: Optional Pinecone configuration
            embedding_model: Optional pre-initialized embeddings model
        """
        self.config = config or get_pinecone_config()
        self.is_initialized = False
        
        # CRITICAL FIX: Extract timeout from config
        self.timeout = self.config.get("timeout", 20.0)
        
        # Initialize embeddings
        self.embeddings = embedding_model
        if not self.embeddings:
            self.embeddings = OpenAIEmbeddings(config={"timeout": 10.0})
        
        # Initialize vector store with timeout config
        self.vector_store = PineconeVectorStore(
            embeddings=self.embeddings,
            config=self.config
        )
        
        logger.info(f"Initialized IndexManager for index: {self.config['index_name']}")
        logger.info(f"Configured timeout: {self.timeout}s")
    
    async def init(self):
        """CRITICAL FIX: Initialize with retry logic and proper timeout handling."""
        if self.is_initialized:
            return
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing Pinecone (attempt {attempt + 1}/{max_retries})")
                
                # CRITICAL FIX: Initialize vector store with timeout
                await asyncio.wait_for(
                    self.vector_store.init(),
                    timeout=self.timeout
                )
                
                # CRITICAL FIX: Verify initialization by getting stats
                try:
                    stats = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.vector_store.get_stats
                        ),
                        timeout=10.0
                    )
                    doc_count = stats.get("total_vectors", 0)
                    logger.info(f"Index initialized successfully with {doc_count} documents")
                except Exception as e:
                    logger.warning(f"Could not get initial stats: {e}")
                
                self.is_initialized = True
                return
                
            except asyncio.TimeoutError:
                logger.error(f"Initialization timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Pinecone after {max_retries} attempts due to timeout")
            except Exception as e:
                logger.error(f"Initialization error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize Pinecone after {max_retries} attempts: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        CRITICAL FIX: Add documents with timeout protection and batch processing.
        
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
            # CRITICAL FIX: Process in smaller batches to avoid timeouts
            batch_size = 50  # Smaller batches for stability
            all_doc_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing document batch {i//batch_size + 1} ({len(batch)} docs)")
                
                # Add batch with timeout
                batch_doc_ids = await asyncio.wait_for(
                    self.vector_store.add_documents(batch),
                    timeout=self.timeout * 2  # Double timeout for document addition
                )
                
                all_doc_ids.extend(batch_doc_ids)
                
                # Small delay between batches to prevent rate limiting
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully added {len(all_doc_ids)} documents to index")
            return all_doc_ids
            
        except asyncio.TimeoutError:
            logger.error("Document addition timed out")
            return []
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
        CRITICAL FIX: Search with comprehensive timeout and error handling.
        
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
            # CRITICAL FIX: Search with timeout
            results = await asyncio.wait_for(
                self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    filter_metadata=filter_metadata
                ),
                timeout=self.timeout
            )
            
            logger.debug(f"Search returned {len(results)} results for query: {query}")
            return results
            
        except asyncio.TimeoutError:
            logger.error(f"Search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        CRITICAL FIX: Delete documents with timeout protection.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # CRITICAL FIX: Delete with timeout
            deleted_count = await asyncio.wait_for(
                self.vector_store.delete_documents(doc_ids),
                timeout=self.timeout
            )
            logger.info(f"Deleted {deleted_count} documents from index")
            return deleted_count
            
        except asyncio.TimeoutError:
            logger.error("Document deletion timed out")
            return 0
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    async def count_documents(self) -> int:
        """
        CRITICAL FIX: Count documents with timeout protection.
        
        Returns:
            Number of documents
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # CRITICAL FIX: Count with timeout using executor
            count = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.vector_store.count_documents
                ),
                timeout=5.0  # Shorter timeout for count operation
            )
            return count
        except asyncio.TimeoutError:
            logger.error("Document count timed out")
            return 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        CRITICAL FIX: Reset index with timeout protection.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # CRITICAL FIX: Reset with timeout
            success = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.vector_store.reset_index
                ),
                timeout=self.timeout
            )
            if success:
                logger.info("Successfully reset index")
            return success
        except asyncio.TimeoutError:
            logger.error("Index reset timed out")
            return False
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Get index statistics with timeout protection.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # CRITICAL FIX: Get vector store stats with timeout
            vector_stats = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.vector_store.get_stats
                ),
                timeout=5.0
            )
            
            # Add additional stats
            stats = {
                "vector_store": vector_stats,
                "index_name": self.config["index_name"],
                "namespace": self.config["namespace"],
                "embedding_model": self.embeddings.model,
                "embedding_dimensions": self.embeddings.dimensions,
                "is_initialized": self.is_initialized,
                "timeout_configured": self.timeout
            }
            
            return stats
            
        except asyncio.TimeoutError:
            logger.error("Get index stats timed out")
            return {
                "error": "timeout",
                "index_name": self.config["index_name"],
                "is_initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "error": str(e),
                "index_name": self.config["index_name"],
                "is_initialized": self.is_initialized
            }
    
    def get_embeddings_model(self) -> OpenAIEmbeddings:
        """Get the embeddings model instance."""
        return self.embeddings
    
    def get_vector_store(self) -> PineconeVectorStore:
        """Get the vector store instance."""
        return self.vector_store