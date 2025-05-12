"""
OpenAI Assistant management for the knowledge base.
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
    """Manage OpenAI Assistants for the voice AI agent."""
    
    def __init__(self):
        """Initialize OpenAI Assistant Manager."""
        self.config = get_openai_config()
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        self.assistant_id = None
        self.token_counter = TokenCounter(self.config["model"])
        self.system_instructions = self._get_system_instructions()
    
    def _get_system_instructions(self) -> str:
        """Get system instructions for the assistant."""
        return """You are a helpful assistant for a voice AI agent that handles customer inquiries.

IMPORTANT GUIDELINES:
1. Keep responses conversational and concise for voice interactions
2. When you need specific information, use the search_knowledge_base function
3. Always maintain context within the conversation
4. For complex queries, break down information into digestible parts
5. If you don't know something, admit it and offer to search for information
6. Prioritize accuracy and cite sources when using retrieved information

VOICE INTERACTION CONSIDERATIONS:
- Speak naturally, as if in a phone conversation
- Use brief pauses (commas) for better speech synthesis
- Avoid long lists or complex formatting
- Summarize key points clearly"""
    
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
                            "description": "Search the knowledge base for relevant information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query"
                                    },
                                    "filters": {
                                        "type": "object",
                                        "description": "Optional metadata filters for the search"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Number of results to return (default: 5)"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ]
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
            assistants = await self.client.beta.assistants.list()
            for assistant in assistants.data:
                if assistant.name == "Voice AI Knowledge Assistant":
                    self.assistant_id = assistant.id
                    logger.info(f"Found existing assistant: {self.assistant_id}")
                    return self.assistant_id
            
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
            logger.debug(f"Added message to thread {thread_id}")
        except Exception as e:
            logger.error(f"Error adding message to thread: {e}")
            raise OpenAIError(f"Failed to add message: {str(e)}")
    
    async def run_assistant(self, thread_id: str, assistant_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the assistant on a thread with streaming support."""
        if not assistant_id:
            assistant_id = await self.get_or_create_assistant()
        
        try:
            # Create run with streaming
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                stream=True
            )
            
            # Process streaming events
            async for event in run:
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
                    yield {
                        "type": "function_calls",
                        "tool_calls": tool_calls,
                        "done": False
                    }
                
                elif event.event == "thread.run.completed":
                    yield {
                        "type": "completed",
                        "done": True
                    }
                
                elif event.event == "thread.run.failed":
                    yield {
                        "type": "error",
                        "error": event.data.last_error.message,
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
            await self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs
            )
            logger.debug(f"Submitted tool outputs for run {run_id}")
        except Exception as e:
            logger.error(f"Error submitting tool outputs: {e}")
            raise OpenAIError(f"Failed to submit tool outputs: {str(e)}")
    
    async def get_thread_messages(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get messages from a thread."""
        try:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit
            )
            
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content[0].text.value if msg.content else "",
                    "created_at": msg.created_at
                }
                for msg in messages.data
            ]
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