"""
Conversation manager optimized for continuous telephony conversations.
Enhanced with better state management and context preservation.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator, Union
from enum import Enum
import time
import json

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from knowledge_base.llama_index.query_engine import QueryEngine
from knowledge_base.llama_index.llm_setup import get_ollama_llm, format_system_prompt, create_chat_messages

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enum for conversation states optimized for telephony."""
    GREETING = "greeting"
    WAITING_FOR_QUERY = "waiting_for_query"
    RETRIEVING = "retrieving"
    GENERATING_RESPONSE = "generating_response"
    CLARIFYING = "clarifying"
    HUMAN_HANDOFF = "human_handoff"
    CONTINUOUS = "continuous"  # New state for ongoing conversation
    ENDED = "ended"

class ConversationTurn:
    """Represents a single turn in the conversation with enhanced context."""
    
    def __init__(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        state: ConversationState = ConversationState.WAITING_FOR_QUERY,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None
    ):
        """Initialize ConversationTurn with enhanced tracking."""
        self.query = query
        self.response = response
        self.retrieved_context = retrieved_context or []
        self.state = state
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.turn_id = turn_id or f"turn_{int(time.time() * 1000)}"
        self.processing_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with additional metadata."""
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
    Manage conversation state and flow optimized for continuous telephony interactions.
    Enhanced with better context preservation and state management.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        session_id: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_temperature: float = 0.7,
        use_langgraph: bool = False,
        skip_greeting: bool = True,  # Default to skip for telephony
        max_history_turns: int = 10,  # Increased for better context
        context_window_tokens: int = 3000  # Manage context window size
    ):
        """Initialize ConversationManager with telephony optimizations."""
        self.query_engine = query_engine
        self.session_id = session_id or f"session_{int(time.time())}"
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.use_langgraph = use_langgraph
        self.skip_greeting = skip_greeting
        self.max_history_turns = max_history_turns
        self.context_window_tokens = context_window_tokens
        
        # LLM setup
        self.llm = None
        
        # Initialize conversation state
        self.current_state = ConversationState.CONTINUOUS if skip_greeting else ConversationState.GREETING
        self.history: List[ConversationTurn] = []
        self.context_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Enhanced state tracking
        self.conversation_started = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        self.topic_context = []  # Track conversation topics for better context
        
        # State for LangGraph (will be expanded later)
        self.graph_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history": [],
            "context": None,
            "metadata": {},
            "conversation_topics": []
        }
        
        logger.info(f"Initialized ConversationManager for continuous telephony - Session: {self.session_id}")
    
    async def init(self):
        """Initialize dependencies with telephony optimizations."""
        # Initialize query engine if provided
        if self.query_engine:
            await self.query_engine.init()
        
        # Initialize LLM with telephony-optimized settings
        if not Settings.llm:
            self.llm = get_ollama_llm(
                model_name=self.llm_model_name,
                temperature=self.llm_temperature,
                max_tokens=512,  # Shorter responses for telephony
                context_window=self.context_window_tokens
            )
        else:
            self.llm = Settings.llm
        
        # Initialize LangGraph components if enabled
        if self.use_langgraph:
            logger.info("LangGraph integration will be implemented in a future update")
    
    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input with enhanced context preservation for continuous conversation."""
        start_time = time.time()
        self.last_activity = time.time()
        self.turn_count += 1
        
        # Create new turn with enhanced metadata
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
        
        # Process based on current state
        try:
            if self.current_state == ConversationState.GREETING:
                response = await self._handle_greeting(turn)
            elif self.current_state in [ConversationState.WAITING_FOR_QUERY, ConversationState.CONTINUOUS]:
                response = await self._handle_query(turn)
            elif self.current_state == ConversationState.CLARIFYING:
                response = await self._handle_clarification(turn)
            elif self.current_state == ConversationState.HUMAN_HANDOFF:
                response = await self._handle_human_handoff(turn)
            else:
                # Default to query handling for continuous conversation
                response = await self._handle_query(turn)
            
            # Update turn with response and processing time
            turn.response = response["response"]
            turn.state = response["state"]
            turn.processing_time = time.time() - start_time
            
            # Add to history with context management
            self._add_to_history(turn)
            
            # Update current state
            self.current_state = response["state"]
            
            # Update graph state for future LangGraph compatibility
            self._update_graph_state(turn)
            
            # Extract and track conversation topics
            self._extract_conversation_topics(user_input, response["response"])
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            error_response = {
                "response": "I'm sorry, I encountered an error processing your request. Could you please try again?",
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
        
        # Manage history size to prevent memory issues
        if len(self.history) > self.max_history_turns:
            # Keep the most recent turns and important context
            self.history = self.history[-self.max_history_turns:]
        
        logger.debug(f"Added turn to history: {len(self.history)} total turns")
    
    def _extract_conversation_topics(self, query: str, response: str):
        """Extract and track conversation topics for better context."""
        # Simple keyword extraction - could be enhanced with NLP
        combined_text = f"{query} {response}".lower()
        
        # Common topic indicators
        topic_keywords = {
            "pricing": ["price", "cost", "plan", "pricing", "subscription", "fee"],
            "features": ["feature", "capability", "function", "feature", "what can"],
            "support": ["help", "support", "assistance", "problem", "issue"],
            "product": ["product", "service", "offering", "solution"],
            "technical": ["technical", "integration", "api", "setup", "configuration"]
        }
        
        current_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                current_topics.append(topic)
        
        # Update topic context (keep last 5 topics)
        self.topic_context.extend(current_topics)
        self.topic_context = list(set(self.topic_context[-5:]))
        
        logger.debug(f"Current conversation topics: {self.topic_context}")
    
    async def _handle_greeting(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle greeting state with quick transition to continuous mode."""
        # For telephony, quickly move to continuous conversation
        if turn.query and len(turn.query.split()) > 2:
            logger.info("Detected query in greeting state, transitioning to continuous conversation")
            return await self._handle_query(turn)
        
        # Generate brief greeting for telephony
        response_text = "Hello! How can I help you today?"
        
        return {
            "response": response_text,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": None
        }
    
    async def _handle_query(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle user query with enhanced context from conversation history."""
        query = turn.query
        
        # Check for human handoff request
        if self._check_for_human_handoff(query):
            return {
                "response": "I'll connect you with a human agent right away. Please hold on.",
                "state": ConversationState.HUMAN_HANDOFF,
                "requires_human": True,
                "context": None
            }
        
        # Retrieve relevant documents with conversation context
        context = None
        if self.query_engine:
            turn.state = ConversationState.RETRIEVING
            
            try:
                # Enhance query with conversation context
                enhanced_query = self._enhance_query_with_context(query)
                logger.debug(f"Enhanced query: {enhanced_query}")
                
                # Get relevant documents
                retrieval_results = await self.query_engine.retrieve_with_sources(enhanced_query)
                turn.retrieved_context = retrieval_results["results"]
                
                # Format context for LLM
                context = self.query_engine.format_retrieved_context(turn.retrieved_context)
                
                # Check if we have enough context
                if not turn.retrieved_context:
                    # Try with original query if enhanced query didn't work
                    if enhanced_query != query:
                        retrieval_results = await self.query_engine.retrieve_with_sources(query)
                        turn.retrieved_context = retrieval_results["results"]
                        context = self.query_engine.format_retrieved_context(turn.retrieved_context)
                
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        
        # Generate response with conversation context
        turn.state = ConversationState.GENERATING_RESPONSE
        
        try:
            # Get conversation history for context
            conversation_history = self._format_conversation_history_with_context()
            
            # Create enhanced system prompt with conversation context
            system_prompt = self._create_contextual_system_prompt(context)
            
            # Create messages with conversation history
            messages = create_chat_messages(
                system_prompt=system_prompt,
                user_message=query,
                chat_history=conversation_history
            )
            
            # Generate response with telephony optimization
            try:
                response = await self.llm.achat(messages)
                response_text = response.message.content
            except AttributeError:
                # Fallback to synchronous chat if async is not available
                logger.info("Falling back to synchronous chat method")
                response = self.llm.chat(messages)
                response_text = response.message.content
            
            # Optimize response for telephony (shorter, clearer)
            response_text = self._optimize_for_telephony(response_text)
            
            logger.info(f"Generated response: {response_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            response_text = "I'm sorry, I'm having trouble processing that right now. Could you rephrase your question?"
        
        # Return response with continuous state for ongoing conversation
        return {
            "response": response_text,
            "state": ConversationState.CONTINUOUS,
            "requires_human": False,
            "context": context,
            "retrieval_results": turn.retrieved_context if hasattr(turn, 'retrieved_context') else []
        }
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with conversation context for better retrieval."""
        if not self.history:
            return query
        
        # Get recent conversation context
        recent_turns = self.history[-3:] if len(self.history) >= 3 else self.history
        
        # Extract relevant context terms
        context_terms = []
        for turn in recent_turns:
            if turn.query:
                # Extract key terms from previous queries
                words = turn.query.lower().split()
                important_words = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'why']]
                context_terms.extend(important_words[:2])  # Take up to 2 key terms per turn
        
        # Add current conversation topics
        context_terms.extend(self.topic_context)
        
        # Remove duplicates and limit context
        context_terms = list(set(context_terms))[:5]
        
        if context_terms:
            # Create enhanced query with context
            context_str = " ".join(context_terms)
            enhanced_query = f"{query} (context: {context_str})"
            return enhanced_query
        
        return query
    
    def _create_contextual_system_prompt(self, retrieved_context: Optional[str] = None) -> str:
        """Create system prompt with conversation and topic context."""
        base_prompt = """You are a helpful AI assistant for customer support. You provide clear, concise answers optimized for phone conversations. 

Key guidelines:
- Keep responses conversational and natural for speech
- Avoid lists and bullet points - use flowing speech instead
- Be direct and helpful
- If you don't know something, say so clearly
- Maintain continuity with the ongoing conversation"""
        
        # Add conversation topics context
        if self.topic_context:
            topic_context = f"\nThis conversation has covered: {', '.join(self.topic_context)}"
            base_prompt += topic_context
        
        # Add retrieved context if available
        if retrieved_context:
            return format_system_prompt(base_prompt, retrieved_context)
        
        return base_prompt
    
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
        
        # Simplify sentence structure for speech
        sentences = response.split('. ')
        if len(sentences) > 3:
            # Keep first 3 sentences for conciseness
            response = '. '.join(sentences[:3]) + '.'
        
        # Replace technical terms with more conversational language
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
        # Get original query from conversation context
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
            
            # Handle as normal query with enhanced context
            return await self._handle_query(combined_turn)
        
        # If no context, handle as normal query
        return await self._handle_query(turn)
    
    async def _handle_human_handoff(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Handle human handoff with conversation summary."""
        return {
            "response": "I've notified a human agent about your request. They'll be with you shortly with access to our full conversation history.",
            "state": ConversationState.HUMAN_HANDOFF,
            "requires_human": True,
            "context": self._generate_handoff_summary(),
            "conversation_summary": self._generate_history_summary()
        }
    
    def _check_for_human_handoff(self, query: str) -> bool:
        """Enhanced human handoff detection with conversation context."""
        query_lower = query.lower()
        
        # Expanded handoff keywords for telephony
        handoff_keywords = [
            "speak to a human", "talk to a person", "human agent", "real person",
            "speak to someone", "talk to someone", "connect me with",
            "customer service", "representative", "agent", "supervisor",
            "escalate", "complaint", "billing issue", "account problem",
            "not satisfied", "frustrated", "this isn't working"
        ]
        
        # Check for explicit handoff requests
        for keyword in handoff_keywords:
            if keyword in query_lower:
                return True
        
        # Check for repeated issues (context-based handoff)
        if len(self.history) >= 3:
            recent_queries = [turn.query.lower() for turn in self.history[-3:] if turn.query]
            
            # Check for repeated similar queries (user frustration)
            if len(set(recent_queries)) < len(recent_queries) * 0.7:  # High similarity
                logger.info("Detected repeated similar queries, suggesting handoff")
                return True
        
        return False
    
    def _format_conversation_history_with_context(self) -> List[Dict[str, str]]:
        """Format conversation history with enhanced context for better responses."""
        formatted_history = []
        
        # Include more history for better context (last 8 turns)
        recent_turns = self.history[-8:] if len(self.history) >= 8 else self.history
        
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
        """Generate comprehensive summary for human handoff."""
        if not self.history:
            return "No conversation history available."
        
        summary_parts = [
            f"Conversation Summary (Session: {self.session_id})",
            f"Duration: {time.time() - self.conversation_started:.1f} seconds",
            f"Total turns: {len(self.history)}",
            f"Topics discussed: {', '.join(self.topic_context) if self.topic_context else 'General inquiry'}"
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
        
        # Count turns
        num_turns = len(self.history)
        
        # Get key exchanges
        summary_parts = [f"Conversation with {num_turns} exchanges:"]
        
        for i, turn in enumerate(self.history[-5:]):  # Last 5 turns
            if turn.query:
                summary_parts.append(f"User: {turn.query}")
            if turn.response:
                # Truncate long responses
                response = turn.response
                if len(response) > 100:
                    response = response[:97] + "..."
                summary_parts.append(f"AI: {response}")
        
        return "\n".join(summary_parts)
    
    def _update_graph_state(self, turn: ConversationTurn):
        """Update graph state with enhanced context tracking."""
        # Update current state
        self.graph_state["current_state"] = turn.state
        
        # Add to history with metadata
        history_entry = {
            "role": "user",
            "content": turn.query,
            "turn_id": turn.turn_id,
            "timestamp": turn.timestamp
        }
        self.graph_state["history"].append(history_entry)
        
        if turn.response:
            response_entry = {
                "role": "assistant",
                "content": turn.response,
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp,
                "processing_time": turn.processing_time
            }
            self.graph_state["history"].append(response_entry)
        
        # Update context and topics
        if turn.retrieved_context:
            self.graph_state["context"] = turn.retrieved_context
        
        self.graph_state["conversation_topics"] = self.topic_context
        self.graph_state["metadata"]["last_activity"] = self.last_activity
        self.graph_state["metadata"]["turn_count"] = self.turn_count
    
    async def generate_streaming_response(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """Generate streaming response optimized for continuous conversation."""
        # Update activity tracking
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
            
            # Update turn and history
            turn.response = result["chunk"]
            turn.state = ConversationState.HUMAN_HANDOFF
            self.history.append(turn)
            self.current_state = ConversationState.HUMAN_HANDOFF
            self._update_graph_state(turn)
            
            yield result
            return
        
        # Retrieve relevant documents with conversation context
        context = None
        if self.query_engine:
            try:
                turn.state = ConversationState.RETRIEVING
                
                # Enhance query with conversation context
                enhanced_query = self._enhance_query_with_context(user_input)
                retrieval_results = await self.query_engine.retrieve_with_sources(enhanced_query)
                turn.retrieved_context = retrieval_results["results"]
                context = self.query_engine.format_retrieved_context(turn.retrieved_context)
                
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = None
        
        # Stream response with conversation context
        turn.state = ConversationState.GENERATING_RESPONSE
        full_response = ""
        
        try:
            # Get conversation history
            conversation_history = self._format_conversation_history_with_context()
            
            # Create enhanced system prompt
            system_prompt = self._create_contextual_system_prompt(context)
            
            # Create messages with conversation history
            messages = create_chat_messages(
                system_prompt=system_prompt,
                user_message=user_input,
                chat_history=conversation_history
            )
            
            try:
                # Stream response
                stream_response = await self.llm.astream_chat(messages)
                
                word_buffer = ""
                
                async for chunk in stream_response:
                    chunk_text = chunk.delta if hasattr(chunk, 'delta') else chunk.content
                    full_response += chunk_text
                    word_buffer += chunk_text
                    
                    # Send words individually for better TTS processing
                    if ' ' in word_buffer:
                        words = word_buffer.split(' ')
                        for word in words[:-1]:
                            if word.strip():
                                yield {
                                    "chunk": word + " ",
                                    "done": False,
                                    "requires_human": False,
                                    "state": ConversationState.GENERATING_RESPONSE
                                }
                        word_buffer = words[-1]  # Keep the last partial word
                
                # Send any remaining text
                if word_buffer.strip():
                    yield {
                        "chunk": word_buffer,
                        "done": False,
                        "requires_human": False,
                        "state": ConversationState.GENERATING_RESPONSE
                    }
                
            except AttributeError:
                # Fallback to non-streaming
                logger.info("Async streaming not available, falling back to regular chat")
                response = self.llm.chat(messages)
                response_text = response.message.content
                full_response = response_text
                
                # Optimize and send as chunks
                optimized_response = self._optimize_for_telephony(response_text)
                words = optimized_response.split()
                
                for i, word in enumerate(words):
                    yield {
                        "chunk": word + (" " if i < len(words) - 1 else ""),
                        "done": False,
                        "requires_human": False,
                        "state": ConversationState.GENERATING_RESPONSE
                    }
            
            # Optimize final response for telephony
            optimized_response = self._optimize_for_telephony(full_response)
            
            # Final result
            yield {
                "chunk": "",
                "full_response": optimized_response,
                "done": True,
                "requires_human": False,
                "state": ConversationState.CONTINUOUS,  # Keep conversation going
                "sources": retrieval_results.get("sources", []) if 'retrieval_results' in locals() else []
            }
            
            # Update turn and history
            turn.response = optimized_response
            turn.state = ConversationState.CONTINUOUS
            self._add_to_history(turn)
            self.current_state = ConversationState.CONTINUOUS
            self._update_graph_state(turn)
            self._extract_conversation_topics(user_input, optimized_response)
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}", exc_info=True)
            
            error_message = "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
            yield {
                "chunk": error_message,
                "done": True,
                "requires_human": False,
                "state": ConversationState.CONTINUOUS
            }
            
            # Update turn and history
            turn.response = error_message
            turn.state = ConversationState.CONTINUOUS
            turn.metadata["error"] = str(e)
            self._add_to_history(turn)
            self._update_graph_state(turn)
    
    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Get detailed conversation metrics for monitoring."""
        if not self.history:
            return {"error": "No conversation history"}
        
        # Calculate metrics
        total_turns = len(self.history)
        user_turns = len([t for t in self.history if t.query])
        ai_turns = len([t for t in self.history if t.response])
        avg_processing_time = sum(t.processing_time for t in self.history) / total_turns
        
        # Response quality metrics
        short_responses = len([t for t in self.history if t.response and len(t.response.split()) < 10])
        long_responses = len([t for t in self.history if t.response and len(t.response.split()) > 50])
        
        return {
            "session_id": self.session_id,
            "conversation_duration": time.time() - self.conversation_started,
            "total_turns": total_turns,
            "user_turns": user_turns,
            "ai_turns": ai_turns,
            "avg_processing_time": avg_processing_time,
            "conversation_topics": self.topic_context,
            "current_state": self.current_state,
            "response_distribution": {
                "short_responses": short_responses,
                "long_responses": long_responses,
                "optimal_responses": ai_turns - short_responses - long_responses
            },
            "last_activity": self.last_activity,
            "session_health": "active" if time.time() - self.last_activity < 60 else "idle"
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return [turn.to_dict() for turn in self.history]
    
    def get_latest_context(self) -> Optional[List[Dict[str, Any]]]:
        """Get most recently retrieved context."""
        # Find most recent turn with context
        for turn in reversed(self.history):
            if turn.retrieved_context:
                return turn.retrieved_context
        
        return None
    
    def reset(self):
        """Reset conversation state while preserving session information."""
        # Reset to continuous state for telephony
        self.current_state = ConversationState.CONTINUOUS if self.skip_greeting else ConversationState.GREETING
        self.history = []
        self.context_cache = {}
        self.topic_context = []
        
        # Reset timing but keep session ID
        self.conversation_started = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        
        # Reset graph state
        self.graph_state = {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history": [],
            "context": None,
            "metadata": {},
            "conversation_topics": []
        }
        
        logger.info(f"Reset conversation for session: {self.session_id} (keeping session ID)")
    
    def get_state_for_transfer(self) -> Dict[str, Any]:
        """Get comprehensive conversation state for human handoff."""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "history_summary": self._generate_handoff_summary(),
            "conversation_metrics": self.get_conversation_metrics(),
            "last_query": self.history[-1].query if self.history else None,
            "last_response": self.history[-1].response if self.history else None,
            "conversation_topics": self.topic_context,
            "recent_context": self.get_latest_context(),
            "full_history": [turn.to_dict() for turn in self.history[-10:]]  # Last 10 turns
        }
    
    # LangGraph preparation methods (for future use)
    def get_graph_state(self) -> Dict[str, Any]:
        """Get the current graph state."""
        return self.graph_state
    
    def serialize_state(self) -> str:
        """Serialize the current state for LangGraph."""
        return json.dumps(self.graph_state)
    
    @classmethod
    def deserialize_state(cls, serialized_state: str) -> Dict[str, Any]:
        """Deserialize a state for LangGraph."""
        return json.loads(serialized_state)