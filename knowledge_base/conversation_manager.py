"""
Conversation management using OpenAI Assistants API with Pinecone integration.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import json

from knowledge_base.openai_assistant_manager import OpenAIAssistantManager
from knowledge_base.pinecone_manager import PineconeManager
from knowledge_base.exceptions import ConversationError
from knowledge_base.utils.cache_utils import CacheManager
from knowledge_base.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manage conversations using OpenAI Assistants API with Pinecone integration."""
    
    def __init__(self):
        """Initialize conversation manager."""
        self.openai_manager = OpenAIAssistantManager()
        self.pinecone_manager = PineconeManager()
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()
        self.user_threads: Dict[str, str] = {}
        self.initialized = False
    
    async def init(self):
        """Initialize all components."""
        try:
            # Initialize Pinecone first
            await self.pinecone_manager.init()
            
            # Set Pinecone manager in OpenAI manager for search functionality
            self.openai_manager.set_pinecone_manager(self.pinecone_manager)
            
            # Create or get assistant
            await self.openai_manager.get_or_create_assistant()
            
            self.initialized = True
            logger.info("Conversation manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing conversation manager: {e}")
            raise ConversationError(f"Failed to initialize: {str(e)}")
    
    async def get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user."""
        if user_id not in self.user_threads:
            thread_id = await self.openai_manager.create_thread()
            self.user_threads[user_id] = thread_id
            logger.debug(f"Created new thread for user {user_id}: {thread_id}")
        
        return self.user_threads[user_id]
    
    async def handle_user_input(self, user_id: str, message: str) -> Dict[str, Any]:
        """Handle user input and return response."""
        if not self.initialized:
            logger.warning("Conversation manager not initialized")
            return {"response": "System is initializing. Please try again.", "error": "not_initialized"}
        
        try:
            # Check rate limits
            if not await self.rate_limiter.check_rate_limit(user_id, 1000):
                return {
                    "response": "You've reached your rate limit. Please try again later.",
                    "error": "rate_limit_exceeded"
                }
            
            # Check cache first
            cached_response = await self.cache.get(message, {"user_id": user_id})
            if cached_response:
                return {
                    "response": cached_response,
                    "cached": True
                }
            
            # Get or create thread for user
            thread_id = await self.get_or_create_thread(user_id)
            
            # Add user message to thread
            await self.openai_manager.add_message_to_thread(thread_id, message)
            
            # Process the message and get response
            response = await self._process_message(thread_id, user_id)
            
            # Cache the response
            if response and not response.get("error"):
                await self.cache.set(message, response["response"], {"user_id": user_id})
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your request.",
                "error": str(e)
            }
    
    async def handle_user_input_streaming(self, user_id: str, message: str) -> AsyncIterator[Dict[str, Any]]:
        """Handle user input with streaming response."""
        if not self.initialized:
            logger.warning("Conversation manager not initialized")
            yield {
                "chunk": "System is initializing. Please try again.",
                "done": True,
                "error": "not_initialized"
            }
            return
        
        try:
            # Check rate limits
            if not await self.rate_limiter.check_rate_limit(user_id, 1000):
                yield {
                    "chunk": "You've reached your rate limit. Please try again later.",
                    "done": True,
                    "error": "rate_limit_exceeded"
                }
                return
            
            # Get or create thread for user
            thread_id = await self.get_or_create_thread(user_id)
            
            # Add user message to thread
            await self.openai_manager.add_message_to_thread(thread_id, message)
            
            # Process with streaming
            full_response = ""
            pending_function_calls = []
            run_id = None
            
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    chunk = event["content"]
                    full_response += chunk
                    yield {
                        "chunk": chunk,
                        "done": False
                    }
                
                elif event["type"] == "function_calls":
                    pending_function_calls = event["tool_calls"]
                    run_id = event.get("run_id")
                    
                    # Process function calls
                    tool_outputs = await self._process_function_calls(pending_function_calls)
                    
                    # Submit tool outputs
                    if run_id:
                        await self.openai_manager.submit_tool_outputs(thread_id, run_id, tool_outputs)
                
                elif event["type"] == "completed":
                    yield {
                        "chunk": "",
                        "done": True,
                        "full_response": full_response
                    }
                    break
                
                elif event["type"] == "error":
                    yield {
                        "chunk": f"Error: {event['error']}",
                        "done": True,
                        "error": event["error"]
                    }
                    break
            
            # Cache the full response
            if full_response:
                await self.cache.set(message, full_response, {"user_id": user_id})
                
        except Exception as e:
            logger.error(f"Error handling streaming input: {e}")
            yield {
                "chunk": "I'm sorry, I encountered an error processing your request.",
                "done": True,
                "error": str(e)
            }
    
    async def _process_message(self, thread_id: str, user_id: str) -> Dict[str, Any]:
        """Process message with non-streaming response."""
        try:
            response_text = ""
            run_completed = False
            
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    response_text += event["content"]
                
                elif event["type"] == "function_calls":
                    # Process function calls
                    tool_outputs = await self._process_function_calls(event["tool_calls"])
                    
                    # Submit tool outputs (this will continue the run)
                    await self.openai_manager.submit_tool_outputs(
                        thread_id, 
                        event.get("run_id"), 
                        tool_outputs
                    )
                
                elif event["type"] == "completed":
                    run_completed = True
                    break
                
                elif event["type"] == "error":
                    return {
                        "response": "I encountered an error processing your request.",
                        "error": event["error"]
                    }
            
            if run_completed and response_text:
                return {
                    "response": response_text,
                    "user_id": user_id
                }
            else:
                return {
                    "response": "I'm sorry, I couldn't generate a response.",
                    "error": "no_response_generated"
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I'm sorry, I encountered an error.",
                "error": str(e)
            }
    
    async def _process_function_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process function calls from the assistant."""
        tool_outputs = []
        
        for tool_call in tool_calls:
            if tool_call.function.name == "search_knowledge_base":
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments.get("query", "")
                    filters = arguments.get("filters")
                    top_k = arguments.get("top_k", 5)
                    
                    # Search Pinecone
                    results = await self.pinecone_manager.query(
                        query_text=query,
                        top_k=top_k,
                        filter_dict=filters
                    )
                    
                    # Format results for the assistant
                    formatted_results = self._format_search_results(results)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": formatted_results
                    })
                    
                    logger.info(f"Knowledge base search for '{query}' returned {len(results)} results")
                    
                except Exception as e:
                    logger.error(f"Error in search_knowledge_base: {e}")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Error searching knowledge base: {str(e)}"
                    })
        
        return tool_outputs
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for the assistant."""
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted = "Found the following relevant information:\n\n"
        
        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            text = metadata.get("text", result.get("text", "No content available"))
            source = metadata.get("source", "Unknown")
            
            formatted += f"**Source {i}** (Relevance: {score:.3f})\n"
            formatted += f"Document: {source}\n"
            formatted += f"Content: {text[:500]}...\n\n"  # Limit content length
        
        return formatted
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        if user_id not in self.user_threads:
            return []
        
        thread_id = self.user_threads[user_id]
        return await self.openai_manager.get_thread_messages(thread_id, limit)
    
    async def reset_conversation(self, user_id: str):
        """Reset conversation for a user."""
        if user_id in self.user_threads:
            # Delete old thread
            await self.openai_manager.delete_thread(self.user_threads[user_id])
            
            # Create new thread
            self.user_threads[user_id] = await self.openai_manager.create_thread()
            
            logger.info(f"Reset conversation for user {user_id}")