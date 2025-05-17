# knowledge_base/conversation_manager.py - OPTIMIZED VERSION

"""
Conversation manager optimized for OpenAI and telephony conversations.
IMPROVED: Faster processing, better response quality, and fixed session management.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from enum import Enum
import time
import json

from knowledge_base.query_engine import QueryEngine
from knowledge_base.openai_llm import create_telephony_optimized_messages

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enum for conversation states optimized for telephony."""
    GREETING = "greeting"
    WAITING_FOR_QUERY = "waiting_for_query"
    RETRIEVING = "retrieving"
    GENERATING_RESPONSE = "generating_response"
    CLARIFYING = "clarifying"
    HUMAN_HANDOFF = "human_handoff"
    CONTINUOUS = "continuous"
    ENDED = "ended"

class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    def __init__(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        state: ConversationState = ConversationState.WAITING_FOR_QUERY,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None
    ):
        """Initialize ConversationTurn."""
        self.query = query
        self.response = response
        self.retrieved_context = retrieved_context or []
        self.state = state
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.turn_id = turn_id or f"turn_{int(time.time() * 1000)}"
        self.processing_time = 0.0
        self.start_time = time.time()  # IMPROVED: Track start time for latency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "retrieved_context": self.retrieved_context,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "latency": time.time() - self.start_time  # IMPROVED: Include latency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        turn = cls(
            query=data.get("query"),
            response=data.get("response"),
            retrieved_context=data.get("retrieved_context", []),
            state=ConversationState(data.get("state", ConversationState.WAITING_FOR_QUERY)),
            metadata=data.get("metadata", {}),
            turn_id=data.get("turn_id")
        )
        turn.timestamp = data.get("timestamp", turn.timestamp)
        turn.processing_time = data.get("processing_time", 0.0)
        return turn

class ConversationManager:
    """
    IMPROVED: Optimized conversation manager for <2s voice latency.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        session_id: Optional[str] = None,
        skip_greeting: bool = True,
        max_history_turns: int = 2,  # IMPROVED: Reduced from 4 to 2
        context_window_tokens: int = 512  # IMPROVED: Reduced from 1024 to 512
    ):
        """Initialize ConversationManager with improved telephony optimizations."""
        self.query_engine = query_engine
        self.session_id = session_id or f"session_{int(time.time())}"
        self.skip_greeting = skip_greeting
        self.max_history_turns = max_history_turns
        self.context_window_tokens = context_window_tokens
        
        # Initialize conversation state
        self.current_state = ConversationState.CONTINUOUS if skip_greeting else ConversationState.GREETING
        self.history: List[ConversationTurn] = []
        
        # Telephony-specific tracking
        self.conversation_started = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        self.topic_context = []
        
        # IMPROVED: Session state tracking for multiple calls
        self.session_active = True
        self.last_response_time = None
        self.response_count = 0
        self.session_restart_count = 0  # IMPROVED: Track session restarts
        
        logger.info(f"Initialized ConversationManager for OpenAI telephony - Session: {self.session_id}")
    
    async def init(self):
        """Initialize dependencies with improved error handling."""
        # Initialize query engine if provided
        if self.query_engine and not self.query_engine.is_initialized:
            try:
                await asyncio.wait_for(
                    self.query_engine.init(),
                    timeout=10.0  # IMPROVED: Added timeout
                )
            except asyncio.TimeoutError:
                logger.error("Query engine initialization timed out")
                raise
            except Exception as e:
                logger.error(f"Error initializing query engine: {e}")
                raise
        
        logger.info("ConversationManager initialized with OpenAI")
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """IMPROVED: Process user input with optimized processing."""
        start_time = time.time()
        self.last_activity = time.time()
        self.turn_count += 1
        
        # IMPROVED: Validate and reset session state when needed
        if not self.session_active:
            logger.info("Reactivating inactive session")
            self.session_active = True
            self.session_restart_count += 1
        
        # Create new turn
        turn = ConversationTurn(
            query=user_input,
            state=self.current_state,
            turn_id=f"{self.session_id}_turn_{self.turn_count}",
            metadata={
                "turn_number": self.turn_count,
                "session_duration": time.time() - self.conversation_started,
                "time_since_last": time.time() - self.last_activity if self.history else 0,
                "session_restart_count": self.session_restart_count  # IMPROVED: Track restarts
            }
        )
        
        # Always handle as continuous conversation for telephony
        if self.skip_greeting or self.current_state == ConversationState.CONTINUOUS:
            turn.state = ConversationState.CONTINUOUS
            self.current_state = ConversationState.CONTINUOUS
        
        try:
            # Process based on current state - simplified for lower latency
            if self.current_state == ConversationState.GREETING:
                response = await self._handle_greeting(turn)
            else:
                # IMPROVED: All states go through optimized query path
                response = await self._handle_query_optimized(turn)
            
            # Update turn with response
            turn.response = response["response"]
            turn.state = response["state"]
            turn.processing_time = time.time() - start_time
            
            # Add to history
            self._add_to_history(turn)
            
            # Update current state
            self.current_state = response["state"]
            
            # Track response for session management
            self.last_response_time = time.time()
            self.response_count += 1
            
            # IMPROVED: Include timing data
            response["latency"] = turn.processing_time
            response["session_id"] = self.session_id
            response["turn_id"] = turn.turn_id
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            error_response = {
                "response": "I'm sorry, I encountered an error. Could you please try again?",
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": None,
                "error": str(e),
                "latency": time.time() - start_time  # IMPROVED: Include latency
            }
            
            turn.response = error_response["response"]
            turn.state = error_response["state"]
            turn.processing_time = time.time() - start_time
            turn.metadata["error"] = str(e)
            
            self._add_to_history(turn)
            return error_response
    
    def _add_to_history(self, turn: ConversationTurn):
        """Add turn to history with strict size management."""
        self.history.append(turn)
        
        # IMPROVED: Stricter history management
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]
        
        logger.debug(f"Added turn to history: {len(self.history)} total turns")
    
    async def _handle_greeting(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle greeting state with quicker transition."""
        # If there's a query, process it immediately
        if turn.query and len(turn.query.split()) > 1:
            logger.info("Detected query in greeting state, transitioning to continuous conversation")
            return await self._handle_query_optimized(turn)
        
        # IMPROVED: More engaging greeting
        response_text = "Hello! I'm here to help. What can I do for you?"
        
        return {
            "response": response_text,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query_optimized(self, turn: ConversationTurn) -> Dict[str, Any]:
        """IMPROVED: Optimized query handler for <2s latency."""
        query = turn.query
        
        # Check for human handoff request
        if self._check_for_human_handoff(query):
            return {
                "response": "I'll connect you with a human agent right away. Please hold on.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        
        # Validate query engine
        if not self.query_engine:
            logger.error("No query engine available")
            return {
                "response": "I'm having trouble accessing my knowledge. Please try again.",
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": None,
                "error": "No query engine"
            }
        
        # IMPROVED: Parallel context retrieval and response generation
        # Start context retrieval immediately
        turn.state = ConversationState.RETRIEVING
        context_task = asyncio.create_task(self._get_context_optimized(query))
        
        # Set state to generating response
        turn.state = ConversationState.GENERATING_RESPONSE
        
        try:
            # Wait for context with timeout
            retrieval_timeout = 2.5  # IMPROVED: Reduced timeout
            try:
                context_result = await asyncio.wait_for(context_task, timeout=retrieval_timeout)
                context = context_result.get("context")
                turn.retrieved_context = context_result.get("results", [])
            except asyncio.TimeoutError:
                logger.warning(f"Context retrieval timed out for: {query}")
                context = None
                context_task.cancel()  # Cancel the task to free resources
            
            # Generate response with even stricter timeout
            generation_timeout = 3.0  # IMPROVED: Reduced timeout
            if self.query_engine and self.query_engine.llm:
                # Get minimal conversation history
                conversation_history = self._format_minimal_history()
                
                try:
                    # Generate response with strict timeout
                    response_text = await asyncio.wait_for(
                        self.query_engine.llm.generate_response(
                            query=query,
                            context=context,
                            chat_history=conversation_history
                        ),
                        timeout=generation_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Response generation timed out for: {query}")
                    response_text = "I need a moment to think about that. Could you repeat your question?"
            else:
                # Fallback response
                response_text = "I understand your question, but I'm having trouble accessing my knowledge right now."
            
            # Optimize response for telephony
            response_text = self._optimize_for_telephony(response_text)
            
            logger.info(f"Generated response: {response_text[:100]}...")
            
            return {
                "response": response_text,
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": context,
                "retrieval_results": turn.retrieved_context if hasattr(turn, 'retrieved_context') else []
            }
            
        except Exception as e:
            logger.error(f"Error in query handling: {e}")
            return {
                "response": "I'm having trouble right now. Could you try again?",
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": None
            }
    
    async def _get_context_optimized(self, query: str) -> Dict[str, Any]:
        """IMPROVED: Optimized context retrieval with lower latency."""
        try:
            results = await self.query_engine.retrieve_with_sources(
                query=query,
                top_k=1,  # IMPROVED: Get just 1 result for speed
                min_score=0.6  # IMPROVED: Slightly lower threshold for better matches
            )
            
            # Format context
            context = self.query_engine.format_retrieved_context(results.get("results", []))
            
            return {
                "context": context,
                "results": results.get("results", [])
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {
                "context": None,
                "results": []
            }
    
    def _optimize_for_telephony(self, response: str) -> str:
        """IMPROVED: Better response optimization for telephony."""
        # Remove markdown formatting
        import re
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Italic
        response = re.sub(r'`(.*?)`', r'\1', response)        # Code
        
        # Convert lists to natural speech
        response = re.sub(r'^\s*[-*]\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)
        
        # IMPROVED: Better length management - allow 2-3 sentences for engagement
        sentences = response.split('. ')
        if len(sentences) > 3:
            # For telephony, keep response concise but conversational
            response = '. '.join(sentences[:3]) + '.'
        
        # IMPROVED: Better word limit - allow up to 30 words for engagement
        words = response.split()
        if len(words) > 30:
            response = ' '.join(words[:30]) + '.'
        
        # Replace complex terms with simpler alternatives
        replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'set up',
            'configure': 'set up',
            'documentation': 'guide',
            'specifications': 'details'
        }
        
        for formal, casual in replacements.items():
            response = response.replace(formal, casual)
        
        # IMPROVED: Ensure proper ending punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response.strip()
    
    def _format_minimal_history(self) -> List[Dict[str, str]]:
        """IMPROVED: Ultra-minimal conversation history for speed."""
        formatted_history = []
        
        # Only use the most recent turn for lower latency
        if self.history:
            last_turn = self.history[-1]
            
            if last_turn.query:
                formatted_history.append({
                    "role": "user",
                    "content": last_turn.query
                })
            
            if last_turn.response:
                formatted_history.append({
                    "role": "assistant", 
                    "content": last_turn.response
                })
        
        return formatted_history
    
    def _check_for_human_handoff(self, query: str) -> bool:
        """Detect human handoff requests."""
        query_lower = query.lower()
        
        handoff_keywords = [
            "speak to a human", "talk to a person", "human agent", "real person",
            "speak to someone", "talk to someone", "connect me with",
            "customer service", "representative", "agent", "supervisor"
        ]
        
        for keyword in handoff_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """IMPROVED: Optimized streaming with better chunk management."""
        self.last_activity = time.time()
        self.turn_count += 1
        
        # Create turn for tracking
        turn = ConversationTurn(
            query=user_input,
            state=ConversationState.CONTINUOUS,
            turn_id=f"{self.session_id}_turn_{self.turn_count}"
        )
        
        # Check for human handoff
        if self._check_for_human_handoff(user_input):
            result = {
                "chunk": "I'll connect you with a human agent right away. Please hold on.",
                "done": True,
                "requires_human": True,
                "state": ConversationState.HUMAN_HANDOFF,
                "conversation_summary": self._generate_history_summary()
            }
            
            turn.response = result["chunk"]
            turn.state = ConversationState.HUMAN_HANDOFF
            self.history.append(turn)
            self.current_state = ConversationState.HUMAN_HANDOFF
            
            yield result
            return
        
        # Validate query engine
        if not hasattr(self, 'query_engine') or not self.query_engine:
            yield {
                "chunk": "I'm having trouble with my knowledge system. Please try again.",
                "done": True,
                "requires_human": False,
                "state": ConversationState.CONTINUOUS,
                "error": "No query engine available"
            }
            return
        
        # IMPROVED: More efficient streaming with lower latency
        full_response = ""
        
        try:
            # Get minimal conversation history
            conversation_history = self._format_minimal_history()
            
            # IMPROVED: Use a faster streaming approach
            async def stream_with_reduced_timeout():
                # Start retrieval task for context
                context_task = asyncio.create_task(self._get_context_optimized(user_input))
                
                # Set a timeout for context retrieval
                try:
                    context_result = await asyncio.wait_for(context_task, timeout=1.5)  # IMPROVED: Reduced from 2.5s
                    context = context_result.get("context")
                except (asyncio.TimeoutError, Exception):
                    logger.warning("Context retrieval timed out, proceeding without context")
                    context = None
                    # Attempt to cancel the task
                    if not context_task.done():
                        context_task.cancel()
                
                # Stream with the retrieved context (or None if timed out)
                async for chunk in self.query_engine.query_with_streaming(
                    user_input, 
                    chat_history=conversation_history
                ):
                    yield chunk
            
            # Stream response using query engine with lower timeout
            stream_iter = stream_with_reduced_timeout()
            stream_timeout = 15.0  # IMPROVED: Reduced from 25.0s
            
            try:
                async for chunk in asyncio.wait_for(stream_iter, timeout=stream_timeout):
                    chunk_text = chunk.get("chunk", "")
                    
                    if chunk_text:
                        full_response += chunk_text
                        
                        # IMPROVED: Better word limit for engagement (up to 30 words)
                        word_count = len(full_response.split())
                        if word_count >= 30:
                            # Truncate to natural sentence boundary
                            sentences = full_response.split('.')
                            if sentences:
                                full_response = '.'.join(sentences[:3]) + '.'
                                if not full_response.endswith('.'):
                                    full_response += '.'
                            
                            yield {
                                "chunk": "",
                                "full_response": full_response,
                                "done": True,
                                "requires_human": False,
                                "state": ConversationState.CONTINUOUS,
                                "truncated": True
                            }
                            break
                        
                        # Yield each chunk for real-time TTS
                        yield {
                            "chunk": chunk_text,
                            "done": False,
                            "requires_human": False,
                            "state": ConversationState.GENERATING_RESPONSE
                        }
                    
                    # Handle final result
                    if chunk.get("done", False):
                        full_response = chunk.get("full_response", full_response)
                        
                        # Optimize final response for telephony
                        optimized_response = self._optimize_for_telephony(full_response)
                        
                        yield {
                            "chunk": "",
                            "full_response": optimized_response,
                            "done": True,
                            "requires_human": False,
                            "state": ConversationState.CONTINUOUS,
                            "sources": chunk.get("sources", [])
                        }
                        
                        # Update turn and history
                        turn.response = optimized_response
                        turn.state = ConversationState.CONTINUOUS
                        self._add_to_history(turn)
                        self.current_state = ConversationState.CONTINUOUS
                        
                        # Update session tracking
                        self.last_response_time = time.time()
                        self.response_count += 1
                        return
                        
            except asyncio.TimeoutError:
                logger.error("Streaming response timed out")
                yield {
                    "chunk": "I'm taking longer than usual. Please try again.",
                    "done": True,
                    "requires_human": False,
                    "state": ConversationState.CONTINUOUS,
                    "error": "Timeout"
                }
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            
            error_message = "I'm sorry, I'm having trouble right now. Please try again."
            yield {
                "chunk": error_message,
                "done": True,
                "requires_human": False,
                "state": ConversationState.CONTINUOUS,
                "error": str(e)
            }
            
            turn.response = error_message
            turn.state = ConversationState.CONTINUOUS
            turn.metadata["error"] = str(e)
            self._add_to_history(turn)
    
    def _generate_history_summary(self) -> str:
        """Generate a summary of conversation history."""
        if not self.history:
            return "No conversation history."
        
        summary_parts = [f"Conversation with {len(self.history)} exchanges:"]
        
        for turn in self.history[-2:]:  # Last 2 turns
            if turn.query:
                summary_parts.append(f"User: {turn.query}")
            if turn.response:
                response = turn.response
                if len(response) > 100:
                    response = response[:97] + "..."
                summary_parts.append(f"AI: {response}")
        
        return "\n".join(summary_parts)
    
    def reset(self):
        """IMPROVED: Reset conversation state with better cleanup."""
        self.current_state = ConversationState.CONTINUOUS if self.skip_greeting else ConversationState.GREETING
        self.history = []
        self.topic_context = []
        
        # Reset timing but keep session ID
        self.conversation_started = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        
        # Track session restart
        self.session_active = True
        self.last_response_time = None
        self.response_count = 0
        self.session_restart_count += 1  # IMPROVED: Track restart
        
        logger.info(f"Reset conversation for session: {self.session_id} (restart #{self.session_restart_count})")