"""
Query engine combining OpenAI Assistants and Pinecone.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator

from knowledge_base.pinecone_manager import PineconeManager
from knowledge_base.openai_assistant_manager import OpenAIAssistantManager
from knowledge_base.config import get_retrieval_config
from knowledge_base.exceptions import KnowledgeBaseError

logger = logging.getLogger(__name__)

class QueryEngine:
    """Query engine using OpenAI Assistants and Pinecone."""
    
    def __init__(self):
        """Initialize query engine."""
        self.pinecone_manager = PineconeManager()
        self.openai_manager = OpenAIAssistantManager()
        self.config = get_retrieval_config()
    
    async def init(self):
        """Initialize all components."""
        await self.pinecone_manager.init()
        await self.openai_manager.get_or_create_assistant()
        logger.info("Query engine initialized")
    
    async def query(self, query_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the knowledge base."""
        try:
            # Step 1: Search Pinecone for relevant documents
            search_results = await self.pinecone_manager.query(
                query_text=query_text,
                top_k=self.config["top_k"]
            )
            
            # Step 2: Format context from search results
            context = self._format_context(search_results)
            
            # Step 3: Create a temporary thread for this query
            thread_id = await self.openai_manager.create_thread()
            
            # Step 4: Add query with context to thread
            message_with_context = f"""Question: {query_text}

Relevant information from knowledge base:
{context}

Please answer the question based on the provided information. If the information is not sufficient, please say so."""

            await self.openai_manager.add_message_to_thread(thread_id, message_with_context)
            
            # Step 5: Get response from assistant
            response_text = ""
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    response_text += event["content"]
                elif event["type"] == "completed":
                    break
                elif event["type"] == "error":
                    return {
                        "query": query_text,
                        "response": "Error processing your query.",
                        "error": event["error"]
                    }
            
            # Clean up thread
            await self.openai_manager.delete_thread(thread_id)
            
            return {
                "query": query_text,
                "response": response_text,
                "sources": search_results,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "query": query_text,
                "response": "I'm sorry, I encountered an error processing your query.",
                "error": str(e)
            }
    
    async def query_with_streaming(self, query_text: str, user_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Query with streaming response."""
        try:
            # Step 1: Search Pinecone for relevant documents
            search_results = await self.pinecone_manager.query(
                query_text=query_text,
                top_k=self.config["top_k"]
            )
            
            # Step 2: Format context from search results
            context = self._format_context(search_results)
            
            # Step 3: Create a temporary thread for this query
            thread_id = await self.openai_manager.create_thread()
            
            # Step 4: Add query with context to thread
            message_with_context = f"""Question: {query_text}

Relevant information from knowledge base:
{context}

Please answer the question based on the provided information. If the information is not sufficient, please say so."""

            await self.openai_manager.add_message_to_thread(thread_id, message_with_context)
            
            # Step 5: Stream response from assistant
            full_response = ""
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    chunk = event["content"]
                    full_response += chunk
                    yield {
                        "chunk": chunk,
                        "done": False,
                        "sources": search_results
                    }
                elif event["type"] == "completed":
                    yield {
                        "chunk": "",
                        "done": True,
                        "full_response": full_response,
                        "sources": search_results
                    }
                    break
                elif event["type"] == "error":
                    yield {
                        "chunk": f"Error: {event['error']}",
                        "done": True,
                        "error": event["error"]
                    }
                    break
            
            # Clean up thread
            await self.openai_manager.delete_thread(thread_id)
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {
                "chunk": "I'm sorry, I encountered an error processing your query.",
                "done": True,
                "error": str(e)
            }
    
    async def retrieve_with_sources(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve documents with source information."""
        try:
            results = await self.pinecone_manager.query(
                query_text=query_text,
                top_k=top_k or self.config["top_k"]
            )
            
            formatted_results = []
            sources = set()
            
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "text": result.get("text", ""),
                    "score": result["score"],
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
                
                # Extract unique sources
                source = result.get("source", "unknown")
                sources.add(source)
            
            return {
                "query": query_text,
                "results": formatted_results,
                "sources": list(sources)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "query": query_text,
                "results": [],
                "sources": [],
                "error": str(e)
            }
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved context for use in prompts."""
        return self._format_context(results)
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into context string."""
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            source = result.get("source", "Unknown")
            score = result.get("score", 0)
            
            context_parts.append(f"""
Document {i} (Source: {source}, Relevance: {score:.3f}):
{text}
""")
        
        return "\n".join(context_parts)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        try:
            pinecone_stats = await self.pinecone_manager.get_stats()
            return {
                "pinecone_stats": pinecone_stats,
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}