"""
Conversation manager optimized for OpenAI and telephony conversations.
Enhanced with OpenAI-specific optimizations and telephony-focused features.
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
            "processing_time": self.processing_time
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
    Manage conversation state and flow optimized for OpenAI and telephony.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        session_id: Optional[str] = None,
        skip_greeting: bool = True,
        max_history_turns: int = 6,  # Shorter for telephony
        context_window_tokens: int = 2048  # Optimized for telephony
    ):
        """Initialize ConversationManager for telephony."""
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
        
        logger.info(f"Initialized ConversationManager for OpenAI telephony - Session: {self.session_id}")
    
    async def init(self):
        """Initialize dependencies."""
        # Initialize query engine if provided
        if self.query_engine and not self.query_engine.is_initialized:
            await self.query_engine.init()
        
        logger.info("ConversationManager initialized with OpenAI")
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input optimized for telephony conversations."""
        start_time = time.time()
        self.last_activity = time.time()
        self.turn_count += 1
        
        # Create new turn
        turn = ConversationTurn(
            query=user_input,
            state=self.current_state,
            turn_id=f"{self.session_id}_turn_{self.turn_count}",
            metadata={
                "turn_number": self.turn_count,
                "session_duration": time.time() - self.conversation_started,
                "time_since_last": time.time() - self.last_activity if self.history else 0
            }
        )
        
        # Always handle as continuous conversation for telephony
        if self.skip_greeting or self.current_state == ConversationState.CONTINUOUS:
            turn.state = ConversationState.CONTINUOUS
            self.current_state = ConversationState.CONTINUOUS
        
        try:
            # Process based on current state
            if self.current_state == ConversationState.GREETING:
                response = await self._handle_greeting(turn)
            elif self.current_state in [ConversationState.WAITING_FOR_QUERY, ConversationState.CONTINUOUS]:
                response = await self._handle_query(turn)
            elif self.current_state == ConversationState.CLARIFYING:
                response = await self._handle_clarification(turn)
            elif self.current_state == ConversationState.HUMAN_HANDOFF:
                response = await self._handle_human_handoff(turn)
            else:
                response = await self._handle_query(turn)
            
            # Update turn with response
            turn.response = response["response"]
            turn.state = response["state"]
            turn.processing_time = time.time() - start_time
            
            # Add to history
            self._add_to_history(turn)
            
            # Update current state
            self.current_state = response["state"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            error_response = {
                "response": "I'm sorry, I encountered an error. Could you please try again?",
                "state": ConversationState.CONTINUOUS,
                "requires_human": False,
                "context": None,
                "error": str(e)
            }
            
            turn.response = error_response["response"]
            turn.state = error_response["state"]
            turn.processing_time = time.time() - start_time
            turn.metadata["error"] = str(e)
            
            self._add_to_history(turn)
            return error_response
    
    def _add_to_history(self, turn: ConversationTurn):
        """Add turn to history with size management."""
        self.history.append(turn)
        
        # Manage history size for telephony
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]
        
        logger.debug(f"Added turn to history: {len(self.history)} total turns")
    
    async def _handle_greeting(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle greeting state with quick transition."""
        # For telephony, quickly move to continuous conversation
        if turn.query and len(turn.query.split()) > 2:
            logger.info("Detected query in greeting state, transitioning to continuous conversation")
            return await self._handle_query(turn)
        
        # Brief greeting for telephony
        response_text = "Hello! How can I help you today?"
        
        return {
            "response": response_text,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle user query with OpenAI and telephony optimization."""
        query = turn.query
        
        # Check for human handoff request
        if self._check_for_human_handoff(query):
            return {
                "response": "I'll connect you with a human agent right away. Please hold on.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        
        # Retrieve relevant documents if query engine available
        context = None
        if self.query_engine:
            turn.state = ConversationState.RETRIEVING
            
            try:
                # Get relevant documents
                retrieval_results = await self.query_engine.retrieve_with_sources(query)
                turn.retrieved_context = retrieval_results["results"]
                
                # Format context for OpenAI
                context = self.query_engine.format_retrieved_context(turn.retrieved_context)
                
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        
        # Generate response with OpenAI
        turn.state = ConversationState.GENERATING_RESPONSE
        
        try:
            # Get conversation history for context
            conversation_history = self._format_conversation_history()
            
            # Use OpenAI directly for telephony-optimized response
            if self.query_engine and self.query_engine.llm:
                response_text = await self.query_engine.llm.generate_response(
                    query=query,
                    context=context,
                    chat_history=conversation_history
                )
            else:
                # Fallback response
                response_text = "I understand your question, but I'm having trouble accessing my knowledge base right now. Could you please try again?"
            
            # Optimize response for telephony
            response_text = self._optimize_for_telephony(response_text)
            
            logger.info(f"Generated response: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            response_text = "I'm sorry, I'm having trouble processing that right now. Could you rephrase your question?"
        
        return {
            "response": response_text,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": context,
            "retrieval_results": turn.retrieved_context if hasattr(turn, 'retrieved_context') else []
        }
    
    def _optimize_for_telephony(self, response: str) -> str:
        """Optimize response for telephony (shorter, clearer, more natural)."""
        # Remove markdown formatting
        import re
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Italic
        response = re.sub(r'`(.*?)`', r'\1', response)        # Code
        
        # Convert lists to natural speech
        response = re.sub(r'^\s*[-*]\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)
        
        # Simplify for telephony - keep it under 30 words when possible
        sentences = response.split('. ')
        if len(sentences) > 2:
            # For telephony, keep responses shorter
            response = '. '.join(sentences[:2]) + '.'
        
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
        
        return response.strip()
    
    async def _handle_clarification(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle clarification with conversation context."""
        # Get recent context
        recent_queries = [t.query for t in self.history[-3:] if t.query]
        
        if recent_queries:
            # Combine recent context with current clarification
            context_query = f"Previous context: {' '.join(recent_queries[-2:])} Current: {turn.query}"
            
            # Create new turn with combined context
            combined_turn = ConversationTurn(
                query=context_query,
                state=ConversationState.CONTINUOUS,
                metadata={**turn.metadata, "clarification": True}
            )
            
            return await self._handle_query(combined_turn)
        
        return await self._handle_query(turn)
    
    async def _handle_human_handoff(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle human handoff with conversation summary."""
        return {
            "response": "I've notified a human agent. They'll be with you shortly with access to our conversation history.",
            "state": ConversationState.HUMAN_HANDOFF,
            "requires_human": True,
            "context": self._generate_handoff_summary(),
            "conversation_summary": self._generate_history_summary()
        }
    
    def _check_for_human_handoff(self, query: str) -> bool:
        """Enhanced human handoff detection."""
        query_lower = query.lower()
        
        handoff_keywords = [
            "speak to a human", "talk to a person", "human agent", "real person",
            "speak to someone", "talk to someone", "connect me with",
            "customer service", "representative", "agent", "supervisor",
            "escalate", "complaint", "billing issue", "account problem",
            "not satisfied", "frustrated", "this isn't working"
        ]
        
        for keyword in handoff_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """Format conversation history for OpenAI."""
        formatted_history = []
        
        # Include recent turns for context
        recent_turns = self.history[-4:] if len(self.history) >= 4 else self.history
        
        for turn in recent_turns:
            if turn.query:
                formatted_history.append({
                    "role": "user",
                    "content": turn.query
                })
            
            if turn.response:
                formatted_history.append({
                    "role": "assistant", 
                    "content": turn.response
                })
        
        return formatted_history
    
    def _generate_handoff_summary(self) -> str:
        """Generate summary for human handoff."""
        if not self.history:
            return "No conversation history available."
        
        summary_parts = [
            f"Conversation Summary (Session: {self.session_id})",
            f"Duration: {time.time() - self.conversation_started:.1f} seconds",
            f"Total turns: {len(self.history)}"
        ]
        
        # Add recent context
        recent_context = []
        for turn in self.history[-3:]:
            if turn.query:
                recent_context.append(f"User: {turn.query}")
            if turn.response:
                recent_context.append(f"AI: {turn.response[:100]}...")
        
        if recent_context:
            summary_parts.append("Recent conversation:")
            summary_parts.extend(recent_context)
        
        return "\n".join(summary_parts)
    
    def _generate_history_summary(self) -> str:
        """Generate a summary of conversation history."""
        if not self.history:
            return "No conversation history."
        
        summary_parts = [f"Conversation with {len(self.history)} exchanges:"]
        
        for turn in self.history[-3:]:  # Last 3 turns
            if turn.query:
                summary_parts.append(f"User: {turn.query}")
            if turn.response:
                response = turn.response
                if len(response) > 100:
                    response = response[:97] + "..."
                summary_parts.append(f"AI: {response}")
        
        return "\n".join(summary_parts)
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """Generate streaming response optimized for telephony."""
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
        
        # Use query engine with streaming for real-time response
        full_response = ""
        
        try:
            # Get conversation history
            conversation_history = self._format_conversation_history()
            
            # Stream response using query engine
            if self.query_engine:
                async for chunk in self.query_engine.query_with_streaming(
                    user_input, 
                    chat_history=conversation_history
                ):
                    chunk_text = chunk.get("chunk", "")
                    
                    if chunk_text:
                        full_response += chunk_text
                        
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
                        return
            else:
                # Fallback if no query engine
                fallback_response = "I understand your question, but I'm having trouble accessing my knowledge base right now."
                
                yield {
                    "chunk": fallback_response,
                    "done": True,
                    "requires_human": False,
                    "state": ConversationState.CONTINUOUS
                }
                
                turn.response = fallback_response
                turn.state = ConversationState.CONTINUOUS
                self._add_to_history(turn)
        
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}", exc_info=True)
            
            error_message = "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
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
    
    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Get detailed conversation metrics."""
        if not self.history:
            return {"error": "No conversation history"}
        
        total_turns = len(self.history)
        user_turns = len([t for t in self.history if t.query])
        ai_turns = len([t for t in self.history if t.response])
        avg_processing_time = sum(t.processing_time for t in self.history) / total_turns if total_turns > 0 else 0
        
        return {
            "session_id": self.session_id,
            "conversation_duration": time.time() - self.conversation_started,
            "total_turns": total_turns,
            "user_turns": user_turns,
            "ai_turns": ai_turns,
            "avg_processing_time": avg_processing_time,
            "current_state": self.current_state,
            "last_activity": self.last_activity,
            "session_health": "active" if time.time() - self.last_activity < 60 else "idle"
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return [turn.to_dict() for turn in self.history]
    
    def get_latest_context(self) -> Optional[List[Dict[str, Any]]]:
        """Get most recently retrieved context."""
        for turn in reversed(self.history):
            if turn.retrieved_context:
                return turn.retrieved_context
        return None
    
    def reset(self):
        """Reset conversation state."""
        self.current_state = ConversationState.CONTINUOUS if self.skip_greeting else ConversationState.GREETING
        self.history = []
        self.topic_context = []
        
        # Reset timing but keep session ID
        self.conversation_started = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        
        logger.info(f"Reset conversation for session: {self.session_id}")
    
    def get_state_for_transfer(self) -> Dict[str, Any]:
        """Get comprehensive conversation state for human handoff."""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history_summary": self._generate_handoff_summary(),
            "conversation_metrics": self.get_conversation_metrics(),
            "last_query": self.history[-1].query if self.history else None,
            "last_response": self.history[-1].response if self.history else None,
            "recent_context": self.get_latest_context(),
            "full_history": [turn.to_dict() for turn in self.history[-5:]]  # Last 5 turns
        }