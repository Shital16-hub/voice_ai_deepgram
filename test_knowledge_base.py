#!/usr/bin/env python3
"""
Test script to verify your OpenAI API key and test knowledge base queries
"""
import os
import sys
import asyncio
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_openai_api():
    """Test OpenAI API connection and key."""
    print("🧪 Testing OpenAI API...")
    
    try:
        from openai import AsyncOpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
        
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
        
        # Test with a simple embedding request
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        
        print("✅ OpenAI API key is valid!")
        print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

async def test_pinecone_api():
    """Test Pinecone API connection."""
    print("\n🧪 Testing Pinecone API...")
    
    try:
        from pinecone.grpc import PineconeGRPC
        
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            print("❌ PINECONE_API_KEY not found")
            return False
        
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
        
        # Initialize Pinecone client
        pc = PineconeGRPC(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes()
        print(f"✅ Pinecone API key is valid!")
        print(f"   Available indexes: {[idx.name for idx in indexes.indexes]}")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone API error: {e}")
        return False

async def test_knowledge_base():
    """Test knowledge base query."""
    print("\n🧪 Testing Knowledge Base...")
    
    try:
        from knowledge_base.index_manager import IndexManager
        from knowledge_base.query_engine import QueryEngine
        from knowledge_base.openai_embeddings import OpenAIEmbeddings
        
        # Initialize components
        embeddings = OpenAIEmbeddings()
        index_manager = IndexManager(embedding_model=embeddings)
        await index_manager.init()
        
        # Check document count
        doc_count = await index_manager.count_documents()
        print(f"📊 Documents in knowledge base: {doc_count}")
        
        if doc_count == 0:
            print("❌ No documents in knowledge base")
            print("   Run the indexing script first: python index_knowledge_docs.py")
            return False
        
        # Initialize query engine
        query_engine = QueryEngine(index_manager=index_manager)
        await query_engine.init()
        
        # Test a query
        test_query = "What are your pricing plans?"
        print(f"\n🔍 Testing query: '{test_query}'")
        
        result = await query_engine.query(test_query)
        
        print(f"✅ Query successful!")
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Sources found: {len(result.get('sources', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 VOICE AI KNOWLEDGE BASE TESTS")
    print("=" * 50)
    
    # Test OpenAI API
    openai_ok = await test_openai_api()
    
    # Test Pinecone API
    pinecone_ok = await test_pinecone_api()
    
    # Test Knowledge Base (only if both APIs work)
    if openai_ok and pinecone_ok:
        kb_ok = await test_knowledge_base()
    else:
        print("\n⏭️ Skipping knowledge base test (API keys invalid)")
        kb_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"OpenAI API:      {'✅ PASS' if openai_ok else '❌ FAIL'}")
    print(f"Pinecone API:    {'✅ PASS' if pinecone_ok else '❌ FAIL'}")
    print(f"Knowledge Base:  {'✅ PASS' if kb_ok else '❌ FAIL'}")
    
    if all([openai_ok, pinecone_ok, kb_ok]):
        print("\n🎉 ALL TESTS PASSED! Your system is ready!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())