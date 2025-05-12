#!/usr/bin/env python3
"""
Updated test script to verify knowledge base functionality with better run handling.
"""
import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_knowledge_base():
    """Test the knowledge base components."""
    print("Testing Knowledge Base Components...")
    print("=" * 50)
    
    try:
        # Test 1: Initialize Pinecone Manager
        print("\n1. Testing Pinecone Manager...")
        from knowledge_base.pinecone_manager import PineconeManager
        
        pinecone_manager = PineconeManager()
        await pinecone_manager.init()
        print("✓ Pinecone Manager initialized successfully")
        
        # Test 2: Get Pinecone stats
        stats = await pinecone_manager.get_stats()
        print(f"✓ Pinecone stats: {stats}")
        
        # Test 3: Test Pinecone search
        print("\n2. Testing direct Pinecone search...")
        results = await pinecone_manager.query(
            query_text="pricing plans",
            top_k=3,
            include_metadata=True
        )
        print(f"✓ Found {len(results)} results for 'pricing plans'")
        
        # Display some results
        if results:
            for i, result in enumerate(results[:2]):
                score = result.get("score", 0)
                metadata = result.get("metadata", {})
                text = metadata.get("text", "No text")[:100]
                print(f"  Result {i+1}: Score={score:.3f}, Text={text}...")
        
        # Test 4: Initialize OpenAI Assistant Manager
        print("\n3. Testing OpenAI Assistant Manager...")
        from knowledge_base.openai_assistant_manager import OpenAIAssistantManager
        
        openai_manager = OpenAIAssistantManager()
        openai_manager.set_pinecone_manager(pinecone_manager)
        assistant_id = await openai_manager.get_or_create_assistant()
        print(f"✓ OpenAI Assistant created/found: {assistant_id}")
        
        # Test 5: Test Conversation Manager
        print("\n4. Testing Conversation Manager...")
        from knowledge_base.conversation_manager import ConversationManager
        
        conversation_manager = ConversationManager()
        await conversation_manager.init()
        print("✓ Conversation Manager initialized successfully")
        
        # Test 6: Test conversations with better handling
        print("\n5. Testing conversation flow...")
        test_queries = [
            "What are your pricing plans?",
            "Tell me about your features",
            "What is the cost of the basic plan?"
        ]
        
        user_id = "test_user"
        
        # Reset conversation first
        await conversation_manager.reset_conversation(user_id)
        print("✓ Reset conversation for new test")
        
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}: {query}")
            
            # Add delay between queries to ensure proper handling
            if i > 0:
                print("Waiting 3 seconds between queries...")
                await asyncio.sleep(3)
            
            result = await conversation_manager.handle_user_input(user_id, query)
            
            if result.get("error"):
                print(f"✗ Error: {result['error']}")
                
                # If there's an error, reset the conversation and try again
                print("Resetting conversation and trying again...")
                await conversation_manager.reset_conversation(user_id)
                await asyncio.sleep(2)
                
                result = await conversation_manager.handle_user_input(user_id, query)
                
                if result.get("error"):
                    print(f"✗ Still error after reset: {result['error']}")
                    continue
            
            response = result.get("response", "")
            print(f"✓ Response: {response[:150]}...")
            
            # Check if response contains knowledge base information
            if len(response) > 50 and response != "I'm sorry, I couldn't generate a response.":
                print("✓ Response appears to contain knowledge base information")
                
                # Check for specific pricing information
                if any(word in response.lower() for word in ["$", "plan", "pricing", "cost"]):
                    print("✓ Response contains pricing information")
            else:
                print("✗ Response appears to be a fallback/generic response")
        
        # Test 7: Test direct search functionality
        print("\n6. Testing direct search fallback...")
        search_result = await conversation_manager._direct_knowledge_search("pricing plans")
        if search_result:
            print(f"✓ Direct search works: {search_result[:100]}...")
        else:
            print("✗ Direct search returned no results")
        
        print("\n" + "=" * 50)
        print("Knowledge Base Test Complete!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

async def check_environment():
    """Check if all required environment variables are set."""
    print("Checking Environment Variables...")
    print("-" * 30)
    
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"✗ {var}: Not set")
        else:
            print(f"✓ {var}: Set")
    
    if missing_vars:
        print(f"\nError: Missing required environment variables: {missing_vars}")
        return False
    
    print("\n✓ All required environment variables are set")
    return True

async def check_knowledge_base_content():
    """Check what's actually in the knowledge base."""
    print("\n7. Checking Knowledge Base Content...")
    print("-" * 30)
    
    try:
        from knowledge_base.pinecone_manager import PineconeManager
        
        pinecone_manager = PineconeManager()
        await pinecone_manager.init()
        
        # Get stats
        stats = await pinecone_manager.get_stats()
        total_vectors = stats.get('total_vectors', 0)
        print(f"Total vectors in knowledge base: {total_vectors}")
        
        if total_vectors > 0:
            # Try different search queries
            search_queries = ["pricing", "features", "plan", "cost"]
            
            for query in search_queries:
                results = await pinecone_manager.query(
                    query_text=query,
                    top_k=1,
                    include_metadata=True
                )
                
                if results:
                    metadata = results[0].get("metadata", {})
                    text = metadata.get("text", "No text")[:100]
                    score = results[0].get("score", 0)
                    print(f"✓ Query '{query}': Found result (score={score:.3f}) - {text}...")
                else:
                    print(f"✗ Query '{query}': No results found")
        else:
            print("✗ Knowledge base appears to be empty!")
            print("\nTo add content to your knowledge base, run:")
            print("python -m knowledge_base.examples.add_sample_data")
            
    except Exception as e:
        print(f"✗ Error checking knowledge base content: {e}")

async def main():
    """Main test function."""
    print("Knowledge Base Integration Test")
    print("=" * 50)
    
    # Check environment first
    if not await check_environment():
        print("\nPlease set the required environment variables and try again.")
        return
    
    # Check knowledge base content
    await check_knowledge_base_content()
    
    # Run the knowledge base test
    await test_knowledge_base()

if __name__ == "__main__":
    asyncio.run(main())