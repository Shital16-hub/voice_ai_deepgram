"""
Example script for using OpenAI + Pinecone integration.
Replaces the LlamaIndex example with OpenAI and Pinecone.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_base.document_store import DocumentStore
from knowledge_base.index_manager import IndexManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.openai_embeddings import OpenAIEmbeddings
from knowledge_base.pinecone_store import PineconeVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def index_documents(directory: str) -> int:
    """
    Index documents from a directory using OpenAI + Pinecone.
    
    Args:
        directory: Directory containing documents
        
    Returns:
        Number of documents indexed
    """
    logger.info(f"Indexing documents from {directory} using OpenAI + Pinecone")
    
    # Initialize document store
    doc_store = DocumentStore()
    
    # Load and process documents
    documents = doc_store.load_documents_from_directory(directory)
    
    # Initialize embeddings and index manager
    embeddings = OpenAIEmbeddings()
    index_manager = IndexManager(embedding_model=embeddings)
    await index_manager.init()
    
    # Add documents to index
    doc_ids = await index_manager.add_documents(documents)
    
    # Get stats
    doc_count = await index_manager.count_documents()
    logger.info(f"Indexed {len(doc_ids)} new documents, total count: {doc_count}")
    
    return len(doc_ids)

async def query_knowledge_base():
    """Interactive query example with OpenAI + Pinecone."""
    # Initialize components
    embeddings = OpenAIEmbeddings()
    index_manager = IndexManager(embedding_model=embeddings)
    await index_manager.init()
    
    query_engine = QueryEngine(index_manager=index_manager)
    await query_engine.init()
    
    # Get stats
    stats = await query_engine.get_stats()
    doc_count = stats["index_stats"]["vector_store"]["total_vectors"]
    
    print("\n" + "="*50)
    print(f"OpenAI + Pinecone Knowledge Base Query Example")
    print(f"Documents in knowledge base: {doc_count}")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    if doc_count == 0:
        print("Warning: No documents found in knowledge base.")
        should_index = input("Would you like to index some documents? (y/n): ")
        if should_index.lower() == 'y':
            directory = input("Enter the directory path containing documents: ")
            await index_documents(directory)
    
    while True:
        # Get query
        query = input("\nEnter your query: ")
        
        # Check for exit
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Process query
        start_time = asyncio.get_event_loop().time()
        results = await query_engine.retrieve_with_sources(query)
        query_time = asyncio.get_event_loop().time() - start_time
        
        # Display results
        print(f"\nRetrieved {len(results['results'])} results in {query_time:.3f}s:")
        for i, doc in enumerate(results['results']):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {doc.get('score', 0):.4f}")
            source = doc["metadata"].get("source", "Unknown")
            print(f"Source: {source}")
            
            # Print truncated text
            text = doc["text"]
            if len(text) > 500:
                text = text[:497] + "..."
            print(f"Content: {text}")
        
        # Try a direct query with OpenAI
        print("\nGenerating a response using OpenAI...")
        response = await query_engine.query(query)
        
        print(f"\nResponse: {response['response']}")
        print(f"Based on {len(response['sources'])} sources")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI + Pinecone Knowledge Base Example")
    parser.add_argument('--index', action='store_true', help='Index documents')
    parser.add_argument('--query', action='store_true', help='Query knowledge base')
    parser.add_argument('--directory', type=str, help='Directory containing documents to index')
    
    args = parser.parse_args()
    
    # Verify API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable not set")
        return 1
    
    if args.index:
        if not args.directory:
            print("Error: --directory is required when using --index")
            return 1
        
        await index_documents(args.directory)
    
    if args.query or (not args.index):
        await query_knowledge_base()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)