"""
Pinecone vector store integration for ultra-low latency knowledge retrieval.
Uses latest Pinecone features for optimal performance.
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
    """Pinecone vector store optimized for ultra-low latency retrieval."""
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Pinecone vector store."""
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
        
        logger.info(f"Initialized Pinecone with index: {self.index_name}")
    
    async def init(self):
        """Initialize Pinecone index with latest serverless features."""
        if self.is_initialized:
            return
        
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new index: {self.index_name}")
                
                # Create index with serverless spec for better performance
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status.ready:
                    await asyncio.sleep(1)
                
                logger.info(f"Created and initialized index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats for verification
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to index. Stats: {stats}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Pinecone with optimized batching.
        
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
        
        # Generate embeddings in batches
        embeddings = await self.embeddings.embed_batch(texts)
        
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
        
        # Upsert in batches for better performance
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.debug(f"Upserted batch {i//batch_size + 1}")
                
                # Small delay to avoid rate limits
                if i + batch_size < len(vectors):
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                raise
        
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
        Search for similar documents with ultra-low latency.
        
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
        
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)
        
        # Prepare filter
        pinecone_filter = None
        if filter_metadata:
            pinecone_filter = filter_metadata
        
        try:
            # Perform search with latest Pinecone features
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False,  # Don't include vectors to reduce response size
                namespace=self.namespace,
                filter=pinecone_filter
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
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents from Pinecone.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            self.index.delete(ids=doc_ids, namespace=self.namespace)
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return len(doc_ids)
            
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
            stats = self.index.describe_index_stats()
            total_count = stats.total_vector_count
            
            # Get namespace-specific count if using namespace
            if self.namespace and stats.namespaces:
                namespace_stats = stats.namespaces.get(self.namespace)
                if namespace_stats:
                    return namespace_stats.vector_count
            
            return total_count
            
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        Reset the index by deleting all vectors.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Reset index namespace: {self.namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": {ns: data.vector_count for ns, data in (stats.namespaces or {}).items()},
                "current_namespace": self.namespace,
                "namespace_vectors": stats.namespaces.get(self.namespace, {}).vector_count if stats.namespaces else 0
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}