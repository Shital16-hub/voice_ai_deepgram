"""
Knowledge Base integration module for Voice AI Agent.
Updated to use OpenAI + Pinecone instead of Ollama + Chroma.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List

from knowledge_base.conversation_manager import ConversationManager, ConversationState
from knowledge_base.query_engine import QueryEngine
from knowledge_base.utils.cache_utils import StreamingResponseCache

logger = logging.getLogger(__name__)

class KnowledgeBaseIntegration:
    """
    Knowledge Base integration for Voice AI Agent using OpenAI + Pinecone.
    
    Provides an abstraction layer for knowledge base functionality,
    handling query processing and response generation with OpenAI.
    """
    
    def __init__(
        self,
        query_engine: QueryEngine,
        conversation_manager: ConversationManager,
        use_cache: bool = True
    ):
        """
        Initialize the Knowledge Base integration.
        
        Args:
            query_engine: Initialized QueryEngine instance (OpenAI + Pinecone)
            conversation_manager: Initialized ConversationManager instance
            use_cache: Whether to use response caching
        """
        self.query_engine = query_engine
        self.conversation_manager = conversation_manager
        self.initialized = True
        self.use_cache = use_cache
        
        # Initialize response cache
        if self.use_cache:
            self.response_cache = StreamingResponseCache()
    
    async def query(self, text: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Query the knowledge base using OpenAI and generate a response.
        
        Args:
            text: Query text
            include_context: Whether to include context in the response
            
        Returns:
            Dictionary with query response information
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            return {"error": "Knowledge Base integration not initialized"}
        
        # Reset conversation manager state if needed
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Track timing
        start_time = time.time()
        
        # Check cache first for fast responses
        if self.use_cache:
            cached_result = self.response_cache.get(text)
            if cached_result:
                logger.info(f"Cache hit for query: '{text}'")
                cached_result["total_time"] = time.time() - start_time
                cached_result["cache_hit"] = True
                return cached_result
        
        try:
            # Use timeout for the entire process
            try:
                # Query using OpenAI + Pinecone
                response_task = self.query_engine.query(text)
                result = await asyncio.wait_for(response_task, timeout=5.0)
                
                response = result.get("response", "")
                retrieval_results = result.get("sources", [])
                
            except asyncio.TimeoutError:
                logger.warning(f"Knowledge base query timed out for: '{text}'")
                response = "I'm processing your question, but it's taking longer than expected. Could you please rephrase your question?"
                retrieval_results = []
            
            # Prepare the result
            query_result = {
                "query": text,
                "response": response,
                "total_time": time.time() - start_time,
                "engine": "openai_pinecone"
            }
            
            # Include context if requested
            if include_context:
                query_result["sources"] = []
                
                # Extract source information
                for i, doc in enumerate(retrieval_results):
                    metadata = doc.get("metadata", {})
                    source = metadata.get("source", f"Source {i+1}")
                    query_result["sources"].append({
                        "id": i,
                        "source": source,
                        "metadata": metadata,
                        "score": doc.get("score", 0.0)
                    })
                
                # Format context for display
                if retrieval_results:
                    query_result["context"] = self.query_engine.format_retrieved_context(retrieval_results)
                else:
                    query_result["context"] = ""
            
            # Store in cache for future queries
            if self.use_cache:
                self.response_cache.set(text, query_result)
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error in Knowledge Base query: {e}")
            return {
                "error": str(e),
                "query": text,
                "total_time": time.time() - start_time,
                "engine": "openai_pinecone"
            }
    
    async def query_streaming(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Query the knowledge base with streaming response generation using OpenAI.
        
        Args:
            text: Query text
            
        Yields:
            Response chunks
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            yield {"error": "Knowledge Base integration not initialized", "done": True}
            return
        
        # Reset conversation manager state if needed
        if self.conversation_manager.current_state != ConversationState.WAITING_FOR_QUERY:
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
        
        # Check cache for fast responses
        if self.use_cache:
            cached_result = self.response_cache.get(text)
            if cached_result and "response" in cached_result:
                # For cached responses, yield the entire response as a single chunk
                yield {
                    "chunk": cached_result["response"],
                    "done": False,
                    "cache_hit": True
                }
                yield {
                    "chunk": "",
                    "full_response": cached_result["response"],
                    "done": True,
                    "cache_hit": True,
                    "sources": cached_result.get("sources", [])
                }
                return
        
        try:
            # Use the query engine's streaming capability with OpenAI
            async for chunk in self.query_engine.query_with_streaming(text):
                yield chunk
                
                # Cache the final result
                if chunk.get("done", False) and chunk.get("full_response") and self.use_cache:
                    cache_result = {
                        "query": text,
                        "response": chunk.get("full_response"),
                        "sources": chunk.get("sources", []),
                        "engine": "openai_pinecone"
                    }
                    self.response_cache.set(text, cache_result)
                    
        except Exception as e:
            logger.error(f"Error in streaming Knowledge Base query: {e}")
            yield {
                "error": str(e),
                "done": True
            }
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        if self.conversation_manager:
            self.conversation_manager.reset()
            self.conversation_manager.current_state = ConversationState.WAITING_FOR_QUERY
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        if not self.conversation_manager:
            return []
        
        return self.conversation_manager.get_history()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        if not self.query_engine:
            return {"error": "Query engine not initialized"}
        
        # Get stats from query engine (OpenAI + Pinecone)
        kb_stats = await self.query_engine.get_stats()
        
        # Add integration-specific stats
        stats = {
            "kb_stats": kb_stats,
            "engine_type": "openai_pinecone",
        }
        
        # Add cache stats if enabled
        if self.use_cache:
            stats["cache_stats"] = self.response_cache.get_stats()
        
        # Add conversation stats if available
        if self.conversation_manager and self.conversation_manager.history:
            conversation_stats = {
                "total_turns": len(self.conversation_manager.history),
                "current_state": self.conversation_manager.current_state
            }
            stats["conversation"] = conversation_stats
        
        return stats