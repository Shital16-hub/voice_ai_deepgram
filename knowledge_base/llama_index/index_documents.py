#!/usr/bin/env python3
"""
Document indexing script for Voice AI Agent.
This script indexes documents into the Pinecone vector store.
"""
import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Import components
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.embedding_setup import get_embedding_model
from knowledge_base.openai_pinecone_config import get_openai_config, get_pinecone_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def index_documents(
    directory: str, 
    storage_dir: str = './storage',
    reset_index: bool = False
) -> int:
    """
    Index documents from a directory using OpenAI embeddings and Pinecone.
    
    Args:
        directory: Directory containing documents
        storage_dir: Directory for persistent storage
        reset_index: Whether to reset the index before indexing
        
    Returns:
        Number of documents indexed
    """
    logger.info(f"Indexing documents from {directory}")
    
    # Validate OpenAI and Pinecone API keys
    openai_config = get_openai_config()
    pinecone_config = get_pinecone_config()
    
    if not openai_config["api_key"]:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    if not pinecone_config["api_key"]:
        raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable.")
    
    # Initialize document store
    doc_store = DocumentStore()
    
    # Initialize index manager with Pinecone
    index_manager = IndexManager(storage_dir=storage_dir)
    await index_manager.init()
    
    # Reset index if requested
    if reset_index:
        logger.info("Resetting Pinecone index...")
        success = await index_manager.reset_index()
        if success:
            logger.info("Index reset successfully")
        else:
            logger.error("Failed to reset index")
            return 0
    
    # Load and process documents
    documents = doc_store.load_documents_from_directory(directory)
    
    if not documents:
        logger.warning(f"No documents found in {directory}")
        return 0
    
    # Add documents to index
    doc_ids = await index_manager.add_documents(documents)
    
    # Get stats
    doc_count = await index_manager.count_documents()
    logger.info(f"Indexed {len(doc_ids)} new documents, total count: {doc_count}")
    
    return len(doc_ids)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Index documents for Voice AI Agent")
    parser.add_argument('--directory', '-d', type=str, default='./knowledge_base/knowledge_docs',
                        help='Directory containing documents to index')
    parser.add_argument('--storage', '-s', type=str, default='./storage',
                        help='Storage directory')
    parser.add_argument('--reset', '-r', action='store_true',
                        help='Reset index before indexing')
    
    args = parser.parse_args()
    
    try:
        # Check if directory exists
        if not os.path.exists(args.directory):
            logger.error(f"Directory not found: {args.directory}")
            return 1
        
        # Index documents
        indexed_count = await index_documents(args.directory, args.storage, args.reset)
        
        if indexed_count > 0:
            logger.info(f"Successfully indexed {indexed_count} documents")
            return 0
        else:
            logger.warning("No documents were indexed")
            return 1
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))