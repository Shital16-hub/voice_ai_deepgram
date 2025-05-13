"""
Fixed conversation management with optimized error handling and fallbacks.
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
    """Manage conversations using OpenAI Assistants API with optimized error handling."""
    
    def __init__(self):
        """Initialize conversation manager."""
        self.openai_manager = OpenAIAssistantManager()
        self.pinecone_manager = PineconeManager()
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()
        self.user_threads: Dict[str, str] = {}
        self.active_runs: Dict[str, str] = {}
        self.run_timeouts: Dict[str, float] = {}  # Track run start times
        self.initialized = False
        
        # Performance settings
        self.max_run_timeout = 5.0  # Reduced from 7.0
        self.fallback_enabled = True
    
    async def init(self):
        """Initialize all components."""
        try:
            # Initialize Pinecone first
            await self.pinecone_manager.init()
            
            # Set Pinecone manager in OpenAI manager
            self.openai_manager.set_pinecone_manager(self.pinecone_manager)
            
            # Create or get assistant
            await self.openai_manager.get_or_create_assistant()
            
            self.initialized = True
            logger.info("Conversation manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing conversation manager: {e}")
            if self.fallback_enabled:
                logger.info("Enabling fallback mode")
                self.initialized = True  # Enable fallback mode
            else:
                raise ConversationError(f"Failed to initialize: {str(e)}")
    
    async def get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user."""
        if user_id not in self.user_threads:
            try:
                thread_id = await self.openai_manager.create_thread()
                self.user_threads[user_id] = thread_id
                logger.debug(f"Created new thread for user {user_id}: {thread_id}")
            except Exception as e:
                logger.error(f"Error creating thread for user {user_id}: {e}")
                # Use a fallback thread ID
                thread_id = f"fallback_{user_id}_{int(time.time())}"
                self.user_threads[user_id] = thread_id
        
        return self.user_threads[user_id]
    
    async def _cleanup_stale_runs(self, thread_id: str):
        """Clean up stale runs that might be blocking the thread."""
        if thread_id not in self.active_runs:
            return
        
        run_id = self.active_runs[thread_id]
        start_time = self.run_timeouts.get(run_id, time.time())
        
        if time.time() - start_time > self.max_run_timeout:
            logger.warning(f"Cleaning up stale run {run_id} (timeout)")
            try:
                await self.openai_manager.client.beta.threads.runs.cancel(
                    thread_id=thread_id,
                    run_id=run_id
                )
            except Exception as e:
                logger.error(f"Error cancelling stale run: {e}")
            finally:
                del self.active_runs[thread_id]
                if run_id in self.run_timeouts:
                    del self.run_timeouts[run_id]
    
    async def handle_user_input(self, user_id: str, message: str) -> Dict[str, Any]:
        """Handle user input with optimized error handling and fallbacks."""
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
            
            # Clean up any stale runs
            await self._cleanup_stale_runs(thread_id)
            
            # Process message with timeout and fallback
            response = await self._process_message_with_fallback(thread_id, user_id, message)
            
            # Cache successful responses
            if response and response.get("response") and not response.get("error"):
                await self.cache.set(message, response["response"], {"user_id": user_id})
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling user input: {e}", exc_info=True)
            # Return fallback response
            return await self._get_fallback_response(message)
    
    async def _process_message_with_fallback(self, thread_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """Process message with multiple fallback strategies."""
        # Strategy 1: Try OpenAI Assistant
        try:
            response = await self._try_openai_assistant(thread_id, user_id, message)
            if response and response.get("response"):
                return response
        except Exception as e:
            logger.warning(f"OpenAI Assistant failed: {e}")
        
        # Strategy 2: Direct Pinecone search with OpenAI
        if self.fallback_enabled:
            try:
                response = await self._try_direct_search(message)
                if response and response.get("response"):
                    return response
            except Exception as e:
                logger.warning(f"Direct search failed: {e}")
        
        # Strategy 3: Final fallback
        return await self._get_fallback_response(message)
    
    async def _try_openai_assistant(self, thread_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """Try to get response from OpenAI Assistant."""
        try:
            # Check if thread starts with fallback (indicating error state)
            if thread_id.startswith("fallback_"):
                raise Exception("Thread is in fallback mode")
            
            # Add message to thread
            await self.openai_manager.add_message_to_thread(thread_id, message)
            
            # Process with timeout
            start_time = time.time()
            response_task = asyncio.create_task(self._process_with_assistant(thread_id, user_id))
            
            response = await asyncio.wait_for(response_task, timeout=self.max_run_timeout)
            
            # Check response quality
            if response and response.get("response") and len(response["response"].strip()) > 0:
                return response
            else:
                logger.warning("Empty or invalid response from assistant")
                return None
                
        except asyncio.TimeoutError:
            logger.warning(f"OpenAI Assistant timeout after {self.max_run_timeout}s")
            # Cancel any active run
            if thread_id in self.active_runs:
                await self._cleanup_stale_runs(thread_id)
            return None
        except Exception as e:
            logger.error(f"Error with OpenAI Assistant: {e}")
            return None
    
    async def _process_with_assistant(self, thread_id: str, user_id: str) -> Dict[str, Any]:
        """Process message with OpenAI Assistant."""
        response_text = ""
        run_completed = False
        function_calls_processed = 0
        run_id = None
        
        try:
            async for event in self.openai_manager.run_assistant(thread_id):
                if event["type"] == "text_delta":
                    response_text += event["content"]
                
                elif event["type"] == "function_calls":
                    function_calls_processed += 1
                    run_id = event.get("run_id")
                    
                    # Track the active run
                    if run_id:
                        self.active_runs[thread_id] = run_id
                        self.run_timeouts[run_id] = time.time()
                    
                    # Process function calls with timeout
                    try:
                        tool_outputs = await asyncio.wait_for(
                            self._process_function_calls(event["tool_calls"]),
                            timeout=2.0
                        )
                        
                        await self.openai_manager.submit_tool_outputs(thread_id, run_id, tool_outputs)
                    except asyncio.TimeoutError:
                        logger.warning("Function call processing timeout")
                        raise
                
                elif event["type"] == "completed":
                    run_completed = True
                    if thread_id in self.active_runs:
                        del self.active_runs[thread_id]
                    if run_id and run_id in self.run_timeouts:
                        del self.run_timeouts[run_id]
                    break
                
                elif event["type"] == "error":
                    logger.error(f"Assistant error: {event['error']}")
                    # Clean up
                    if thread_id in self.active_runs:
                        del self.active_runs[thread_id]
                    if run_id and run_id in self.run_timeouts:
                        del self.run_timeouts[run_id]
                    return None
            
            # If we got function calls but no response, try to get the latest message
            if function_calls_processed > 0 and not response_text:
                logger.info("Function calls processed, checking for response in thread")
                await asyncio.sleep(0.5)  # Brief wait for processing
                messages = await self.openai_manager.get_thread_messages(thread_id, limit=1)
                if messages and messages[0].get("role") == "assistant":
                    response_text = messages[0].get("content", "")
            
            if response_text:
                return {
                    "response": response_text,
                    "user_id": user_id,
                    "source": "openai_assistant"
                }
            else:
                logger.warning("No response from assistant")
                return None
                
        except Exception as e:
            logger.error(f"Error processing with assistant: {e}")
            # Clean up any active runs
            if thread_id in self.active_runs:
                del self.active_runs[thread_id]
            if run_id and run_id in self.run_timeouts:
                del self.run_timeouts[run_id]
            raise
    
    async def _try_direct_search(self, message: str) -> Dict[str, Any]:
        """Try direct Pinecone search as fallback."""
        try:
            # Search Pinecone directly
            results = await self.pinecone_manager.query(
                query_text=message,
                top_k=3,
                include_metadata=True
            )
            
            if results:
                # Format results into a response
                context = ""
                sources = []
                
                for result in results:
                    if result.get("metadata") and result["metadata"].get("text"):
                        text = result["metadata"]["text"]
                        source = result["metadata"].get("source", "Unknown")
                        context += f"{text}\n\n"
                        sources.append(source)
                
                if context:
                    # Create a context-aware response
                    response = self._format_direct_search_response(message, context)
                    return {
                        "response": response,
                        "sources": sources,
                        "source": "direct_search"
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error in direct search: {e}")
            return None
    
    def _format_direct_search_response(self, query: str, context: str) -> str:
        """Format a response from direct search results."""
        query_lower = query.lower()
        
        # Determine response type based on query
        if any(word in query_lower for word in ["price", "pricing", "cost", "plan"]):
            return f"Based on our pricing information: {context[:300]}..."
        elif any(word in query_lower for word in ["feature", "features", "capability"]):
            return f"Here are our key features: {context[:300]}..."
        elif any(word in query_lower for word in ["how", "what", "tell", "explain"]):
            return f"Let me explain: {context[:300]}..."
        else:
            return f"Here's what I found: {context[:300]}..."
    
    async def _get_fallback_response(self, message: str) -> Dict[str, Any]:
        """Get generic fallback response."""
        message_lower = message.lower()
        
        # Context-aware fallback responses
        if any(word in message_lower for word in ["price", "pricing", "cost", "plan"]):
            response = "I can help you with pricing information. Our basic plan starts at $499/month. Would you like more details about our pricing options?"
        elif any(word in message_lower for word in ["feature", "features", "capability"]):
            response = "We offer voice recognition, natural language processing, and automated customer service solutions. Would you like to know more about any specific feature?"
        elif any(word in message_lower for word in ["help", "support"]):
            response = "I'm here to help! You can ask me about our pricing, features, or how our voice AI system works."
        else:
            response = "I understand you have a question. Could you please rephrase it, or ask about our pricing, features, or services?"
        
        return {
            "response": response,
            "source": "fallback"
        }
    
    async def _process_function_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process function calls with timeout protection."""
        tool_outputs = []
        
        for tool_call in tool_calls:
            logger.info(f"Processing tool call: {tool_call.function.name}")
            
            if tool_call.function.name == "search_knowledge_base":
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments.get("query", "")
                    filters = arguments.get("filters")
                    top_k = arguments.get("top_k", 3)  # Reduced default
                    
                    logger.info(f"Searching knowledge base for: '{query}'")
                    
                    # Search with timeout
                    search_task = asyncio.create_task(
                        self.pinecone_manager.query(
                            query_text=query,
                            top_k=top_k,
                            filter_dict=filters
                        )
                    )
                    results = await asyncio.wait_for(search_task, timeout=1.5)
                    
                    # Format results
                    formatted_results = self._format_search_results(results)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": formatted_results
                    })
                    
                    logger.info(f"Knowledge base search returned {len(results)} results")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Search timeout for query: '{query}'")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": "Search timeout. Please try a more specific query."
                    })
                except Exception as e:
                    logger.error(f"Error in search_knowledge_base: {e}")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Search error. Please try rephrasing your question."
                    })
        
        return tool_outputs
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for the assistant."""
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Limit to top 2 results to avoid token limits
        results = results[:2]
        
        formatted = "Found the following information:\n\n"
        
        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            text = metadata.get("text", result.get("text", "No content available"))
            source = metadata.get("source", "Unknown")
            
            # Limit text length
            if len(text) > 200:
                text = text[:197] + "..."
            
            formatted += f"**Source {i}** (Relevance: {score:.3f})\n"
            formatted += f"Document: {source}\n"
            formatted += f"Content: {text}\n\n"
        
        return formatted
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        if user_id not in self.user_threads:
            return []
        
        thread_id = self.user_threads[user_id]
        
        # Skip if it's a fallback thread
        if thread_id.startswith("fallback_"):
            return []
        
        try:
            return await self.openai_manager.get_thread_messages(thread_id, limit)
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
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
                    if run_id in self.run_timeouts:
                        del self.run_timeouts[run_id]
                except Exception as e:
                    logger.error(f"Error cancelling run during reset: {e}")
            
            # Don't delete fallback threads
            if not thread_id.startswith("fallback_"):
                try:
                    await self.openai_manager.delete_thread(thread_id)
                except Exception as e:
                    logger.warning(f"Could not delete thread: {e}")
            
            # Create new thread
            try:
                new_thread_id = await self.openai_manager.create_thread()
                self.user_threads[user_id] = new_thread_id
            except Exception as e:
                logger.error(f"Error creating new thread: {e}")
                # Create fallback thread
                self.user_threads[user_id] = f"fallback_{user_id}_{int(time.time())}"
            
            logger.info(f"Reset conversation for user {user_id}")