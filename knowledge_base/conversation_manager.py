"""
Complete fixed conversation management with proper run handling.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import json
import time

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
        self.active_runs: Dict[str, str] = {}  # Track active runs per thread
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
    
    async def _wait_for_run_completion(self, thread_id: str, max_wait: float = 10.0):
        """Wait for any active run on the thread to complete or expire."""
        if thread_id not in self.active_runs:
            return
        
        run_id = self.active_runs[thread_id]
        logger.info(f"Waiting for active run {run_id} to complete")
        
        start_time = time.time()
        while (time.time() - start_time) < max_wait:
            try:
                # Check run status
                run = await self.openai_manager.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                if run.status in ['completed', 'failed', 'cancelled', 'expired']:
                    logger.info(f"Run {run_id} is now {run.status}")
                    del self.active_runs[thread_id]
                    return
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error checking run status: {e}")
                # If we can't check the run, assume it's done
                del self.active_runs[thread_id]
                return
        
        # If we timeout, cancel the run
        logger.warning(f"Run {run_id} did not complete in {max_wait}s, attempting to cancel")
        try:
            await self.openai_manager.client.beta.threads.runs.cancel(
                thread_id=thread_id,
                run_id=run_id
            )
            del self.active_runs[thread_id]
        except Exception as e:
            logger.error(f"Error cancelling run: {e}")
    
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
            
            # Wait for any active run to complete
            await self._wait_for_run_completion(thread_id)
            
            # Add user message to thread
            await self.openai_manager.add_message_to_thread(thread_id, message)
            
            # Process the message and get response
            response = await self._process_message_with_timeout(thread_id, user_id, timeout=7.0)
            
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
    
    async def _process_message_with_timeout(self, thread_id: str, user_id: str, timeout: float = 7.0) -> Dict[str, Any]:
        """Process message with timeout and proper error handling."""
        try:
            # Create a task to process the message
            task = asyncio.create_task(self._process_message(thread_id, user_id))
            
            # Wait for the task with timeout
            result = await asyncio.wait_for(task, timeout=timeout)
            
            # If the result is empty or contains an error, try to handle it
            if not result.get("response") or result.get("error"):
                # If we got a timeout or error, try a direct search as fallback
                logger.warning("Assistant response was empty or errored, trying direct search")
                # Get the last message from the thread
                messages = await self.openai_manager.get_thread_messages(thread_id, limit=2)
                for msg in messages:
                    if msg.get("role") == "user":
                        last_message = msg.get("content", "")
                        search_result = await self._direct_knowledge_search(last_message)
                        if search_result:
                            return {"response": search_result}
                        break
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Message processing timeout after {timeout}s")
            # Try to cancel any active run
            if thread_id in self.active_runs:
                run_id = self.active_runs[thread_id]
                try:
                    await self.openai_manager.client.beta.threads.runs.cancel(
                        thread_id=thread_id,
                        run_id=run_id
                    )
                    del self.active_runs[thread_id]
                except Exception as e:
                    logger.error(f"Error cancelling run after timeout: {e}")
            
            # Try direct search as fallback
            messages = await self.openai_manager.get_thread_messages(thread_id, limit=2)
            for msg in messages:
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
                    search_result = await self._direct_knowledge_search(last_message)
                    if search_result:
                        return {"response": search_result}
                    break
            
            return {
                "response": "I'm processing your request. Let me help you with that.",
                "error": "timeout"
            }
    
    async def _direct_knowledge_search(self, query: str) -> Optional[str]:
        """Perform direct knowledge base search."""
        try:
            # Search Pinecone directly
            results = await self.pinecone_manager.query(
                query_text=query,
                top_k=3,
                include_metadata=True
            )
            
            if results:
                # Format results into a response
                context = ""
                for result in results:
                    if result.get("metadata") and result["metadata"].get("text"):
                        text = result["metadata"]["text"]
                        context += f"{text}\n\n"
                
                # Create a simple response based on the context
                if any(word in query.lower() for word in ["price", "pricing", "cost", "plan"]):
                    response = f"Here's information about our pricing:\n\n{context}"
                elif any(word in query.lower() for word in ["feature", "features", "capability"]):
                    response = f"Here are our features:\n\n{context}"
                else:
                    response = f"Here's the relevant information:\n\n{context}"
                
                logger.info(f"Direct search found {len(results)} results")
                return response
            
        except Exception as e:
            logger.error(f"Error in direct knowledge search: {e}")
        
        return None
    
    async def _process_message(self, thread_id: str, user_id: str) -> Dict[str, Any]:
        """Process message with non-streaming response."""
        try:
            response_text = ""
            run_completed = False
            function_calls_processed = 0
            run_id = None
            
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    response_text += event["content"]
                
                elif event["type"] == "function_calls":
                    # Process function calls
                    function_calls_processed += 1
                    run_id = event.get("run_id")
                    
                    # Track the active run
                    if run_id:
                        self.active_runs[thread_id] = run_id
                    
                    tool_outputs = await self._process_function_calls(event["tool_calls"])
                    
                    # Submit tool outputs (this will continue the run)
                    await self.openai_manager.submit_tool_outputs(
                        thread_id, 
                        run_id, 
                        tool_outputs
                    )
                
                elif event["type"] == "completed":
                    run_completed = True
                    # Clear the active run
                    if thread_id in self.active_runs:
                        del self.active_runs[thread_id]
                    break
                
                elif event["type"] == "error":
                    logger.error(f"Assistant error: {event['error']}")
                    # Clear the active run
                    if thread_id in self.active_runs:
                        del self.active_runs[thread_id]
                    return {
                        "response": "",
                        "error": event["error"]
                    }
            
            # If we processed function calls but got no response, try again
            if function_calls_processed > 0 and not response_text:
                logger.warning("Function calls processed but no response text received")
                # Wait a bit and check for new messages
                await asyncio.sleep(1)
                messages = await self.openai_manager.get_thread_messages(thread_id, limit=1)
                if messages and messages[0].get("role") == "assistant":
                    response_text = messages[0].get("content", "")
            
            if response_text:
                logger.info(f"Assistant response received: {response_text[:100]}...")
                return {
                    "response": response_text,
                    "user_id": user_id
                }
            else:
                logger.warning("No valid response from assistant")
                return {
                    "response": "",
                    "error": "no_response_generated"
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Clear the active run
            if thread_id in self.active_runs:
                del self.active_runs[thread_id]
            return {
                "response": "",
                "error": str(e)
            }
    
    async def _process_function_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process function calls from the assistant."""
        tool_outputs = []
        
        for tool_call in tool_calls:
            logger.info(f"Processing tool call: {tool_call.function.name}")
            
            if tool_call.function.name == "search_knowledge_base":
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments.get("query", "")
                    filters = arguments.get("filters")
                    top_k = arguments.get("top_k", 5)
                    
                    logger.info(f"Searching knowledge base for: '{query}'")
                    
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
            formatted += f"Content: {text}\n\n"
        
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
            thread_id = self.user_threads[user_id]
            
            # Cancel any active run
            if thread_id in self.active_runs:
                run_id = self.active_runs[thread_id]
                try:
                    await self.openai_manager.client.beta.threads.runs.cancel(
                        thread_id=thread_id,
                        run_id=run_id
                    )
                    del self.active_runs[thread_id]
                except Exception as e:
                    logger.error(f"Error cancelling run during reset: {e}")
            
            # Delete old thread
            await self.openai_manager.delete_thread(thread_id)
            
            # Create new thread
            self.user_threads[user_id] = await self.openai_manager.create_thread()
            
            logger.info(f"Reset conversation for user {user_id}")