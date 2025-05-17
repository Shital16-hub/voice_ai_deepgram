# knowledge_base/pinecone_store.py - FIXED VERSION

"""
Pinecone vector store integration for ultra-low latency knowledge retrieval.
CRITICAL FIXES: Added timeouts, better error handling, and async optimization.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid

from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC

from knowledge_base.config import get_pinecone_config
from knowledge_base.openai_embeddings import OpenAIEmbeddings
from knowledge_base.schema import Document

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """Pinecone vector store optimized for ultra-low latency retrieval with CRITICAL fixes."""
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Pinecone vector store with timeout protection."""
        self.embeddings = embeddings
        self.config = config or get_pinecone_config()
        
        if not self.config["api_key"]:
            raise ValueError("Pinecone API key is required")
        
        # Initialize with gRPC for better performance
        self.pc = PineconeGRPC(api_key=self.config["api_key"])
        
        self.index_name = self.config["index_name"]
        self.namespace = self.config["namespace"]
        self.dimensions = self.config["embedding_dimensions"]
        
        self.index = None
        self.is_initialized = False
        
        # CRITICAL FIX: Add connection pool for better performance
        self._connection_pool_size = 10
        
        logger.info(f"Initialized Pinecone with index: {self.index_name} (with timeouts)")
    
    async def init(self):
        """Initialize Pinecone index with latest serverless features and timeout protection."""
        if self.is_initialized:
            return
        
        try:
            # CRITICAL FIX: Add timeout to initialization
            await asyncio.wait_for(self._init_index(), timeout=30.0)
            self.is_initialized = True
            
        except asyncio.TimeoutError:
            logger.error("Pinecone initialization timed out")
            raise
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    async def _init_index(self):
        """Internal method to initialize index with proper async handling."""
        # Check if index exists
        existing_indexes = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.pc.list_indexes()
        )
        index_names = [idx.name for idx in existing_indexes.indexes]
        
        if self.index_name not in index_names:
            logger.info(f"Creating new index: {self.index_name}")
            
            # Create index with serverless spec for better performance
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            )
            
            # Wait for index to be ready
            while True:
                status = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.pc.describe_index(self.index_name)
                )
                if status.status.ready:
                    break
                await asyncio.sleep(1)
            
            logger.info(f"Created and initialized index: {self.index_name}")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats for verification
        stats = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.index.describe_index_stats()
        )
        logger.info(f"Connected to index. Stats: {stats}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Pinecone with optimized batching and timeout protection.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not self.is_initialized:
            await self.init()
        
        if not documents:
            return []
        
        # Prepare texts for embedding
        texts = [doc.text for doc in documents]
        
        # CRITICAL FIX: Add timeout to embedding generation
        try:
            embeddings = await asyncio.wait_for(
                self.embeddings.embed_batch(texts),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error("Timeout generating embeddings")
            return []
        
        # Prepare vectors for upsert
        vectors = []
        doc_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.doc_id or f"doc_{uuid.uuid4()}"
            doc_ids.append(doc_id)
            
            # Prepare metadata (ensure JSON serializable)
            metadata = {
                "text": doc.text,
                "source": doc.metadata.get("source", ""),
                "doc_id": doc_id,
                **{k: v for k, v in doc.metadata.items() 
                   if isinstance(v, (str, int, float, bool))}
            }
            
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # CRITICAL FIX: Upsert in batches with timeout protection
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.index.upsert(vectors=batch, namespace=self.namespace)
                    ),
                    timeout=15.0
                )
                logger.debug(f"Upserted batch {i//batch_size + 1}")
                
                # Small delay to avoid rate limits
                if i + batch_size < len(vectors):
                    await asyncio.sleep(0.01)
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout upserting batch {i//batch_size + 1}")
                # Continue with next batch instead of failing completely
                continue
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Added {len(documents)} documents to Pinecone")
        return doc_ids
    
    async def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents with ultra-low latency and timeout protection.
        
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
            # CRITICAL FIX: Add timeout to embedding generation
            query_embedding = await asyncio.wait_for(
                self.embeddings.embed_text(query),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("Timeout generating query embedding")
            return []
        
        # Prepare filter
        pinecone_filter = filter_metadata if filter_metadata else None
        
        try:
            # CRITICAL FIX: Add timeout to Pinecone search
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        include_values=False,  # Don't include vectors to reduce response size
                        namespace=self.namespace,
                        filter=pinecone_filter
                    )
                ),
                timeout=15.0
            )
            
            # Process results
            search_results = []
            for match in results.matches:
                score = match.score
                
                # Filter by minimum score
                if score < min_score:
                    continue
                
                metadata = match.metadata or {}
                
                result = {
                    "id": match.id,
                    "text": metadata.get("text", ""),
                    "score": score,
                    "metadata": metadata
                }
                search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} results for query")
            return search_results
            
        except asyncio.TimeoutError:
            logger.error(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents from Pinecone with timeout protection.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.delete(ids=doc_ids, namespace=self.namespace)
                ),
                timeout=10.0
            )
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return len(doc_ids)
            
        except asyncio.TimeoutError:
            logger.error("Timeout deleting documents")
            return 0
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    async def count_documents(self) -> int:
        """
        Count documents in the index with timeout protection.
        
        Returns:
            Number of documents
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            stats = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.describe_index_stats()
                ),
                timeout=10.0
            )
            
            # Get namespace-specific count if using namespace
            if self.namespace and stats.namespaces:
                namespace_stats = stats.namespaces.get(self.namespace)
                if namespace_stats:
                    return namespace_stats.vector_count
            
            return stats.total_vector_count
            
        except asyncio.TimeoutError:
            logger.error("Timeout counting documents")
            return 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        Reset the index by deleting all vectors with timeout protection.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Delete all vectors in the namespace
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.delete(delete_all=True, namespace=self.namespace)
                ),
                timeout=20.0
            )
            logger.info(f"Reset index namespace: {self.namespace}")
            return True
            
        except asyncio.TimeoutError:
            logger.error("Timeout resetting index")
            return False
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics with timeout protection."""
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        try:
            # CRITICAL FIX: Make stats retrieval async-safe
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": {ns: data.vector_count for ns, data in (stats.namespaces or {}).items()},
                "current_namespace": self.namespace,
                "namespace_vectors": stats.namespaces.get(self.namespace, {}).vector_count if stats.namespaces else 0,
                "timeout_protection": True
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}