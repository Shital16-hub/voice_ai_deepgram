"""
Index management for LlamaIndex with Pinecone vector store.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union

from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument

from knowledge_base.openai_pinecone_config import get_pinecone_config
from knowledge_base.llama_index.schema import Document
from knowledge_base.llama_index.embedding_setup import get_embedding_model

logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manage Pinecone vector indexes for document storage and retrieval.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        storage_dir: Optional[str] = None,
        embed_model: Optional[Any] = None
    ):
        """
        Initialize IndexManager with Pinecone.
        
        Args:
            config: Optional configuration dictionary
            storage_dir: Directory for local storage (not used with Pinecone but kept for compatibility)
            embed_model: Optional embedding model to use
        """
        self.config = config or get_pinecone_config()
        self.api_key = self.config.get("api_key")
        self.environment = self.config.get("environment")
        self.index_name = self.config.get("index_name")
        self.namespace = self.config.get("namespace")
        self.dimension = self.config.get("dimension", 1536)  # Default to OpenAI embed dimension
        
        # Keep storage directory for compatibility
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "storage")
        
        # Store initialization state
        self.is_initialized = False
        self.vector_store = None
        self.index = None
        self._embed_model = embed_model
        
        logger.info(f"Initialized IndexManager with Pinecone: {self.index_name}")
    
    async def init(self):
        """Initialize the index and vector store with Pinecone."""
        if self.is_initialized:
            return
        
        try:
            # Import Pinecone
            import pinecone
            from llama_index.core import Settings
            
            # Set up the embedding model
            embed_model = self._embed_model
            if embed_model is None:
                embed_model = get_embedding_model()
            
            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = None  # Explicitly disable LLM usage until needed
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                # Create index with the proper dimension
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.config.get("metric", "cosine")
                )
            
            # Connect to the index
            pinecone_index = pinecone.Index(self.index_name)
            
            # Create vector store
            self.vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace=self.namespace
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create vector store index
            self.index = VectorStoreIndex.from_documents(
                documents=[],  # Empty initially
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            # Get stats
            stats = pinecone_index.describe_index_stats()
            namespace_stats = stats.get("namespaces", {}).get(self.namespace, {})
            doc_count = namespace_stats.get("vector_count", 0)
            
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {doc_count} documents in namespace '{self.namespace}'")
            
            self.is_initialized = True
            
        except ImportError as e:
            logger.error(f"Error importing required packages: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing IndexManager with Pinecone: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the Pinecone index.
        
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
            # Convert to LlamaIndex documents
            llama_docs = [doc.to_llama_index_document() for doc in documents]
            
            # Add to index
            doc_ids = []
            for doc in llama_docs:
                self.index.insert(doc)
                doc_ids.append(doc.id_)
            
            logger.info(f"Added {len(doc_ids)} documents to Pinecone index")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents from the Pinecone index.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Delete from index
            for doc_id in doc_ids:
                self.index.delete(doc_id)
            
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone index")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    async def count_documents(self) -> int:
        """
        Count documents in the Pinecone index.
        
        Returns:
            Number of documents
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Get Pinecone index
            import pinecone
            
            pinecone_index = pinecone.Index(self.index_name)
            stats = pinecone_index.describe_index_stats()
            
            # Get count for our namespace
            namespace_stats = stats.get("namespaces", {}).get(self.namespace, {})
            doc_count = namespace_stats.get("vector_count", 0)
            
            return doc_count
            
        except Exception as e:
            logger.error(f"Error counting documents in Pinecone: {e}")
            return 0
    
    async def reset_index(self) -> bool:
        """
        Reset the Pinecone index by deleting all vectors in the namespace.
        
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.init()
        
        try:
            # Get Pinecone index
            import pinecone
            
            pinecone_index = pinecone.Index(self.index_name)
            
            # Delete all vectors in the namespace
            pinecone_index.delete(
                delete_all=True,
                namespace=self.namespace
            )
            
            logger.info(f"Reset Pinecone index namespace: {self.namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting Pinecone index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False