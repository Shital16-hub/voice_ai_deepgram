#!/usr/bin/env python3
"""
FIXED Script to index documents from knowledge_docs to Pinecone.
This version handles metadata properly and has better error handling.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Load .env file before importing other modules
from dotenv import load_dotenv
load_dotenv()

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from knowledge_base.document_store import DocumentStore
from knowledge_base.index_manager import IndexManager
from knowledge_base.openai_embeddings import OpenAIEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def index_documents():
    """Index documents from the knowledge_docs directory to Pinecone."""
    
    print("🔄 Loading environment variables...")
    
    print("🔍 Checking API keys...")
    # Verify environment variables
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return False
    
    print("✅ API keys found")
    print("🚀 Starting document indexing...")
    
    # Set the documents directory
    docs_directory = "knowledge_base/knowledge_docs"
    print(f"📁 Using documents directory: {docs_directory}")
    
    # Check if directory exists
    if not os.path.exists(docs_directory):
        print(f"❌ Directory not found: {docs_directory}")
        return False
    
    try:
        print("📄 Loading documents...")
        # Initialize document store
        doc_store = DocumentStore()
        
        # Load documents from the directory
        documents = doc_store.load_documents_from_directory(docs_directory)
        
        if not documents:
            print("❌ No documents found in the directory")
            return False
        
        print(f"✅ Found {len(documents)} document chunks")
        
        print("🔧 Initializing OpenAI embeddings...")
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        
        print("🔧 Initializing Pinecone...")
        # Initialize Pinecone index manager
        index_manager = IndexManager(embedding_model=embeddings)
        await index_manager.init()
        
        # Check current document count
        current_count = await index_manager.count_documents()
        print(f"📊 Current documents in Pinecone: {current_count}")
        
        print("⬆️ Adding documents to Pinecone...")
        # Add documents to Pinecone
        doc_ids = await index_manager.add_documents(documents)
        
        # Get final count
        final_count = await index_manager.count_documents()
        
        print(f"✅ Successfully indexed {len(doc_ids)} documents!")
        print(f"📊 Total documents in Pinecone: {final_count}")
        
        print("\n📋 Indexed documents:")
        # Display indexed documents - fixed metadata access
        for i, doc in enumerate(documents[:5]):  # Show first 5
            # Access metadata properly - it should be a dict now
            if isinstance(doc.metadata, dict):
                source = doc.metadata.get('source', 'Unknown')
            else:
                # If it's still a Pydantic model, convert to dict
                source = getattr(doc.metadata, 'source', 'Unknown')
            
            print(f"  {i+1}. {source} ({len(doc.text)} characters)")
        
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more documents")
        
        # Test query
        print("\n🧪 Testing with a sample query...")
        test_query = "What are your pricing plans?"
        results = await index_manager.search_documents(test_query, top_k=2)
        
        print(f"Query: '{test_query}'")
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            score = result.get('score', 0)
            text_preview = result.get('text', '')[:100] + '...'
            print(f"  {i+1}. Score: {score:.3f} - {text_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        logger.error(f"Error during indexing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main function."""
    print("🚀 Document Indexing Tool")
    print("=" * 40)
    
    success = await index_documents()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Indexing completed successfully!")
        print("Your knowledge base is now ready for queries.")
        print("\nNext steps:")
        print("1. Start your Voice AI server: python twilio_app_simplified.py")
        print("2. Test with Twilio phone calls")
    else:
        print("❌ Indexing failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())