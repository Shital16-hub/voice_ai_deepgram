"""
Fixed OpenAI Assistant management for the knowledge base with integrated Pinecone search.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from openai import AsyncOpenAI
import json

from knowledge_base.config import get_openai_config
from knowledge_base.exceptions import OpenAIError
from knowledge_base.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)

class OpenAIAssistantManager:
    """Manage OpenAI Assistants for the voice AI agent with knowledge base integration."""
    
    def __init__(self, pinecone_manager=None):
        """Initialize OpenAI Assistant Manager."""
        self.config = get_openai_config()
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        self.assistant_id = None
        self.token_counter = TokenCounter(self.config["model"])
        self.system_instructions = self._get_system_instructions()
        self.pinecone_manager = pinecone_manager
    
    def _get_system_instructions(self) -> str:
        """Get comprehensive system instructions for the assistant."""
        return """You are a helpful voice assistant for customer service that has access to a knowledge base. Your role is to help customers with their questions about products, services, pricing, and features.

IMPORTANT GUIDELINES:
1. Always search the knowledge base when customers ask about:
   - Pricing information (plans, costs, subscription details)
   - Product features and capabilities
   - Service details and offerings
   - Company information
   - Technical specifications

2. Use the search_knowledge_base function for ALL substantive questions
3. Keep responses conversational and concise for voice interactions (1-2 sentences)
4. Always provide specific information when available from the knowledge base
5. If the knowledge base doesn't have the information, say so clearly

VOICE INTERACTION BEST PRACTICES:
- Speak naturally as if talking on the phone
- Use brief pauses (commas) for better speech synthesis
- Avoid long lists or complex formatting in speech
- Summarize key points clearly

SEARCH STRATEGY:
- Extract key terms from customer questions to search effectively
- Try different search terms if initial search doesn't yield results
- Search the knowledge base for any question about products, services, pricing, or features

Remember: Your primary job is to find and communicate information from the knowledge base to help customers."""
    
    async def create_assistant(self) -> str:
        """Create a new OpenAI assistant with function calling capabilities."""
        try:
            assistant = await self.client.beta.assistants.create(
                name="Voice AI Knowledge Assistant",
                instructions=self.system_instructions,
                model=self.config["model"],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "description": "Search the knowledge base for relevant information about products, services, pricing, features, or company information. Use this function for any customer question that requires specific information.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query using relevant keywords extracted from the customer's question. Examples: 'pricing plans', 'voice features', 'subscription cost', 'company information'"
                                    },
                                    "filters": {
                                        "type": "object",
                                        "description": "Optional metadata filters for the search (usually not needed)"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Number of results to return (default: 5, max: 10)"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent responses
                top_p=0.95,
                response_format={"type": "text"}
            )
            
            self.assistant_id = assistant.id
            logger.info(f"Created OpenAI Assistant: {self.assistant_id}")
            return self.assistant_id
            
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            raise OpenAIError(f"Failed to create assistant: {str(e)}")
    
    async def get_or_create_assistant(self) -> str:
        """Get existing assistant or create new one."""
        if not self.assistant_id:
            # Try to find existing assistant first
            try:
                assistants = await self.client.beta.assistants.list(limit=20)
                for assistant in assistants.data:
                    if assistant.name == "Voice AI Knowledge Assistant":
                        self.assistant_id = assistant.id
                        logger.info(f"Found existing assistant: {self.assistant_id}")
                        return self.assistant_id
            except Exception as e:
                logger.warning(f"Error listing assistants: {e}")
            
            # Create new if not found
            return await self.create_assistant()
        
        return self.assistant_id
    
    async def create_thread(self) -> str:
        """Create a new conversation thread."""
        try:
            thread = await self.client.beta.threads.create()
            logger.debug(f"Created thread: {thread.id}")
            return thread.id
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise OpenAIError(f"Failed to create thread: {str(e)}")
    
    async def add_message_to_thread(self, thread_id: str, message: str, role: str = "user"):
        """Add a message to a thread."""
        try:
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=message
            )
            logger.debug(f"Added message to thread {thread_id}: {message[:50]}...")
        except Exception as e:
            logger.error(f"Error adding message to thread: {e}")
            raise OpenAIError(f"Failed to add message: {str(e)}")
    
    async def run_assistant(self, thread_id: str, assistant_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the assistant on a thread with streaming support."""
        if not assistant_id:
            assistant_id = await self.get_or_create_assistant()
        
        logger.info(f"Running assistant {assistant_id} on thread {thread_id}")
        
        try:
            # Create run with streaming
            async with self.client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                max_prompt_tokens=4000,
                max_completion_tokens=1000,
                temperature=0.3
            ) as stream:
                async for event in stream:
                    # Handle different event types
                    if event.event == "thread.message.delta":
                        # Extract text delta
                        if event.data.delta.content:
                            for content in event.data.delta.content:
                                if content.type == "text":
                                    yield {
                                        "type": "text_delta",
                                        "content": content.text.value,
                                        "done": False
                                    }
                    
                    elif event.event == "thread.run.requires_action":
                        # Handle function calls
                        tool_calls = event.data.required_action.submit_tool_outputs.tool_calls
                        logger.info(f"Assistant requires action: {len(tool_calls)} tool calls")
                        yield {
                            "type": "function_calls",
                            "tool_calls": tool_calls,
                            "run_id": event.data.id,
                            "done": False
                        }
                    
                    elif event.event == "thread.run.completed":
                        logger.info("Assistant run completed successfully")
                        yield {
                            "type": "completed",
                            "done": True
                        }
                    
                    elif event.event == "thread.run.failed":
                        error_msg = event.data.last_error.message if event.data.last_error else "Unknown error"
                        logger.error(f"Assistant run failed: {error_msg}")
                        yield {
                            "type": "error",
                            "error": error_msg,
                            "done": True
                        }
                    
                    elif event.event == "thread.run.expired":
                        logger.error("Assistant run expired")
                        yield {
                            "type": "error",
                            "error": "Run expired",
                            "done": True
                        }
                    
                    elif event.event == "thread.run.cancelled":
                        logger.warning("Assistant run cancelled")
                        yield {
                            "type": "error",
                            "error": "Run cancelled",
                            "done": True
                        }
                    
        except Exception as e:
            logger.error(f"Error running assistant: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "done": True
            }
    
    async def submit_tool_outputs(self, thread_id: str, run_id: str, tool_outputs: List[Dict[str, Any]]):
        """Submit tool outputs for function calls."""
        try:
            logger.info(f"Submitting {len(tool_outputs)} tool outputs for run {run_id}")
            
            # Submit tool outputs with streaming
            async with self.client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs
            ) as stream:
                async for event in stream:
                    if event.event == "thread.message.delta":
                        # Process any additional text output
                        if event.data.delta.content:
                            for content in event.data.delta.content:
                                if content.type == "text":
                                    logger.debug(f"Additional content after tool submission: {content.text.value[:50]}...")
                    elif event.event == "thread.run.failed":
                        error_msg = event.data.last_error.message if event.data.last_error else "Unknown error"
                        logger.error(f"Run failed after tool submission: {error_msg}")
                        raise OpenAIError(f"Run failed after tool submission: {error_msg}")
                    elif event.event == "thread.run.completed":
                        logger.info("Tool outputs processed successfully")
                        break
            
        except Exception as e:
            logger.error(f"Error submitting tool outputs: {e}")
            raise OpenAIError(f"Failed to submit tool outputs: {str(e)}")
    
    def set_pinecone_manager(self, pinecone_manager):
        """Set the Pinecone manager for search functionality."""
        self.pinecone_manager = pinecone_manager
        logger.info("Pinecone manager set for search functionality")
    
    async def get_thread_messages(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get messages from a thread."""
        try:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit,
                order="desc"
            )
            
            result = []
            for msg in messages.data:
                content = ""
                if msg.content and len(msg.content) > 0:
                    if hasattr(msg.content[0], 'text'):
                        content = msg.content[0].text.value
                    elif hasattr(msg.content[0], 'value'):
                        content = msg.content[0].value
                
                result.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": content,
                    "created_at": msg.created_at
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting thread messages: {e}")
            raise OpenAIError(f"Failed to get thread messages: {str(e)}")
    
    async def delete_thread(self, thread_id: str):
        """Delete a thread."""
        try:
            await self.client.beta.threads.delete(thread_id)
            logger.debug(f"Deleted thread: {thread_id}")
        except Exception as e:
            logger.error(f"Error deleting thread: {e}")
            # Don't raise error as thread cleanup is not critical