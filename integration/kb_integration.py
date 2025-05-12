"""
Knowledge Base integration for Voice AI Agent - Updated for OpenAI + Pinecone.
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List

from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.utils.cache_utils import CacheManager

logger = logging.getLogger(__name__)

class KnowledgeBaseIntegration:
    """
    Knowledge Base integration using OpenAI Assistants and Pinecone.
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the Knowledge Base integration.
        
        Args:
            user_id: User identifier for conversation tracking
            use_cache: Whether to use response caching
        """
        self.user_id = user_id
        self.use_cache = use_cache
        
        self.conversation_manager = ConversationManager()
        self.query_engine = QueryEngine()
        self.initialized = False
        
        if self.use_cache:
            self.response_cache = CacheManager()
    
    async def init(self):
        """Initialize the knowledge base integration."""
        try:
            await self.conversation_manager.init()
            await self.query_engine.init()
            self.initialized = True
            logger.info("Knowledge Base integration initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Knowledge Base integration: {e}")
            raise
    
    async def query(self, text: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Query the knowledge base and generate a response.
        
        Args:
            text: Query text
            include_context: Whether to include context in the response
            
        Returns:
            Dictionary with query response information
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            return {"error": "Knowledge Base integration not initialized"}
        
        # Track timing
        start_time = time.time()
        
        # Check cache first if enabled
        if self.use_cache:
            cached_result = await self.response_cache.get(text, {"user_id": self.user_id})
            if cached_result:
                logger.info(f"Cache hit for query: '{text[:50]}...'")
                return {
                    **cached_result,
                    "total_time": time.time() - start_time,
                    "cache_hit": True
                }
        
        try:
            # Handle conversation through OpenAI Assistant
            result = await self.conversation_manager.handle_user_input(
                user_id=self.user_id or "default_user",
                message=text
            )
            
            # Get retrieval results if context is needed
            context_data = {}
            if include_context:
                retrieval_results = await self.query_engine.retrieve_with_sources(text)
                context_data = {
                    "context": self.query_engine.format_retrieved_context(retrieval_results["results"]),
                    "sources": retrieval_results["sources"]
                }
            
            # Prepare final result
            final_result = {
                "query": text,
                "response": result.get("response", ""),
                "total_time": time.time() - start_time,
                **context_data
            }
            
            # Cache result if enabled and no error
            if self.use_cache and not result.get("error"):
                await self.response_cache.set(text, final_result, {"user_id": self.user_id})
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in Knowledge Base query: {e}")
            return {
                "error": str(e),
                "query": text,
                "total_time": time.time() - start_time
            }
    
    async def query_streaming(self, text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Query the knowledge base with streaming response generation.
        
        Args:
            text: Query text
            
        Yields:
            Response chunks
        """
        if not self.initialized:
            logger.error("Knowledge Base integration not properly initialized")
            yield {"error": "Knowledge Base integration not initialized", "done": True}
            return
        
        # Check cache for fast responses
        if self.use_cache:
            cached_result = await self.response_cache.get(text, {"user_id": self.user_id})
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
            # Stream response from conversation manager
            full_response = ""
            async for chunk in self.conversation_manager.handle_user_input_streaming(
                user_id=self.user_id or "default_user",
                message=text
            ):
                # Pass through chunks directly
                yield chunk
                
                # Accumulate full response for caching
                if not chunk.get("done", False):
                    full_response += chunk.get("chunk", "")
                elif chunk.get("done", False) and chunk.get("full_response"):
                    full_response = chunk["full_response"]
                    
            # Cache the full response
            if self.use_cache and full_response:
                await self.response_cache.set(text, {
                    "response": full_response,
                    "query": text
                }, {"user_id": self.user_id})
                
        except Exception as e:
            logger.error(f"Error in streaming Knowledge Base query: {e}")
            yield {
                "error": str(e),
                "done": True
            }
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        asyncio.create_task(self.conversation_manager.reset_conversation(
            user_id=self.user_id or "default_user"
        ))
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        # This would need to be implemented with async context
        # For now, return empty list
        logger.warning("get_conversation_history needs async implementation")
        return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        if not self.initialized:
            return {"error": "Not initialized"}
        
        try:
            # Get stats from query engine
            kb_stats = await self.query_engine.get_stats()
            
            # Add integration-specific stats
            stats = {
                "kb_stats": kb_stats,
                "user_id": self.user_id,
                "cache_enabled": self.use_cache
            }
            
            # Add cache stats if enabled
            if self.use_cache:
                stats["cache_stats"] = {
                    "cache_enabled": True,
                    "cache_type": "Redis"
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}