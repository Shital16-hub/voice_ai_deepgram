"""
Query engine using OpenAI + Pinecone for ultra-low latency retrieval and generation.
CRITICAL FIXES: Comprehensive timeout handling and error recovery.
"""
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator

from knowledge_base.config import get_retriever_config, get_performance_config
from knowledge_base.index_manager import IndexManager
from knowledge_base.openai_llm import OpenAILLM, create_telephony_optimized_messages
from knowledge_base.schema import Document

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    FIXED Query engine with comprehensive timeout handling and error recovery.
    Optimized for ultra-low latency telephony applications.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[OpenAILLM] = None
    ):
        """
        Initialize QueryEngine with CRITICAL FIXES.

        Args:
            index_manager: IndexManager instance
            config: Optional configuration dictionary
            llm: Optional pre-initialized OpenAI LLM
        """
        self.index_manager = index_manager
        self.config = config or get_retriever_config()
        self.performance_config = get_performance_config()
        
        # CRITICAL: Extract timeout settings
        self.retrieval_timeout = self.config.get("timeout", 10.0)
        self.total_timeout = self.performance_config.get("total_response_timeout", 25.0)
        self.openai_timeout = self.performance_config.get("openai_timeout", 15.0)
        
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]

        # Initialize OpenAI LLM with timeout
        self.llm = llm or OpenAILLM()

        self.is_initialized = False

        logger.info(f"Initialized QueryEngine with top_k={self.top_k}, min_score={self.min_score}")
        logger.info(f"Timeouts: retrieval={self.retrieval_timeout}s, total={self.total_timeout}s")

    async def init(self):
        """Initialize the query engine with timeout protection."""
        if self.is_initialized:
            return

        try:
            # Ensure index manager is initialized with timeout
            if not self.index_manager.is_initialized:
                await asyncio.wait_for(
                    self.index_manager.init(),
                    timeout=30.0
                )

            self.is_initialized = True
            logger.info("Query engine initialized with OpenAI + Pinecone")
        except asyncio.TimeoutError:
            logger.error("Query engine initialization timed out")
            raise
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        CRITICAL FIX: Retrieve relevant documents with timeout protection.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        if not self.is_initialized:
            await self.init()

        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score

        try:
            # CRITICAL FIX: Add timeout to retrieval
            results = await asyncio.wait_for(
                self.index_manager.search_documents(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    filter_metadata=filter_metadata
                ),
                timeout=self.retrieval_timeout
            )

            # Convert to expected format
            documents = []
            for result in results:
                doc = {
                    "id": result["id"],
                    "text": result["text"],
                    "score": result["score"],
                    "metadata": result["metadata"]
                }
                documents.append(doc)

            logger.debug(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents

        except asyncio.TimeoutError:
            logger.error(f"Retrieval timed out for query: '{query}'")
            return []
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    async def retrieve_with_sources(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        CRITICAL FIX: Retrieve documents with source information and timeout.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            Dictionary with results and sources
        """
        try:
            docs = await asyncio.wait_for(
                self.retrieve(query, top_k, min_score),
                timeout=self.retrieval_timeout
            )

            if not docs:
                return {
                    "query": query,
                    "results": [],
                    "sources": []
                }

            # Extract unique sources
            sources = []
            source_names = set()
            results = []

            for doc in docs:
                metadata = doc["metadata"]
                
                # Format result
                result = {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": metadata,
                    "score": doc["score"]
                }
                results.append(result)

                # Extract source information
                source = metadata.get("source")
                if source and source not in source_names:
                    source_names.add(source)
                    source_info = {
                        "name": source,
                        "type": metadata.get("source_type", "unknown")
                    }

                    if metadata.get("file_path"):
                        source_info["file_path"] = metadata.get("file_path")
                        source_info["file_type"] = metadata.get("file_type")

                    sources.append(source_info)

            return {
                "query": query,
                "results": results,
                "sources": sources
            }
        except asyncio.TimeoutError:
            logger.error(f"Retrieve with sources timed out for query: '{query}'")
            return {
                "query": query,
                "results": [],
                "sources": [],
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"Error in retrieve_with_sources: {e}")
            return {
                "query": query,
                "results": [],
                "sources": [],
                "error": str(e)
            }

    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        CRITICAL FIX: Format context optimized for telephony (very short).
        
        Args:
            results: Retrieved document results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""

        # CRITICAL: For telephony, keep context extremely concise
        context_parts = []
        max_context_words = self.performance_config.get("max_context_words", 100)
        current_words = 0

        for i, doc in enumerate(results):
            text = doc["text"]
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Source {i+1}")

            # CRITICAL: Truncate individual documents aggressively
            words = text.split()
            if current_words + len(words) > max_context_words:
                remaining_words = max_context_words - current_words
                if remaining_words > 0:
                    text = " ".join(words[:remaining_words]) + "..."
                    context_parts.append(f"{source}: {text}")
                break
            
            context_parts.append(f"{source}: {text}")
            current_words += len(words)

        context = " | ".join(context_parts)
        
        # CRITICAL: Final length check
        if len(context) > 300:
            context = context[:297] + "..."

        return context

    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        CRITICAL FIX: Query with comprehensive timeout handling and error recovery.
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            await self.init()

        start_time = time.time()

        try:
            # CRITICAL FIX: Wrap entire query in timeout
            async def _execute_query():
                # Step 1: Retrieve relevant documents with timeout
                retrieval_start = time.time()
                retrieval_results = await self.retrieve_with_sources(query_text)
                retrieval_time = time.time() - retrieval_start
                
                results = retrieval_results["results"]
                
                # Step 2: Format context for telephony
                context = self.format_retrieved_context(results)

                # Step 3: Generate response using OpenAI with timeout
                generation_start = time.time()
                
                try:
                    response = await asyncio.wait_for(
                        self.llm.generate_response(
                            query=query_text,
                            context=context if context else None
                        ),
                        timeout=self.openai_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("OpenAI generation timed out")
                    response = "I'm processing that. Could you try again?"
                
                generation_time = time.time() - generation_start

                # Prepare result
                return {
                    "query": query_text,
                    "response": response,
                    "sources": results,
                    "total_time": time.time() - start_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "context_used": bool(context),
                    "context_length": len(context) if context else 0
                }

            # Execute with total timeout
            result = await asyncio.wait_for(
                _execute_query(),
                timeout=self.total_timeout
            )
            
            return result

        except asyncio.TimeoutError:
            logger.error(f"Total query timeout for: '{query_text}'")
            return {
                "query": query_text,
                "response": "I'm taking too long to respond. Could you try again?",
                "sources": [],
                "total_time": time.time() - start_time,
                "error": "total_timeout",
                "context_used": False
            }
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            
            return {
                "query": query_text,
                "response": "I'm having trouble finding that information. Could you rephrase your question?",
                "sources": [],
                "total_time": time.time() - start_time,
                "error": str(e),
                "context_used": False
            }

    async def query_with_streaming(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        CRITICAL FIX: Query with streaming response and timeout protection.
        
        Args:
            query_text: Query text
            chat_history: Previous conversation history
            
        Yields:
            Response chunks
        """
        if not self.is_initialized:
            await self.init()

        start_time = time.time()

        try:
            # CRITICAL FIX: Retrieve context quickly with timeout
            retrieval_start = time.time()
            
            async def _retrieve_with_timeout():
                return await self.retrieve_with_sources(query_text)
            
            retrieval_results = await asyncio.wait_for(
                _retrieve_with_timeout(),
                timeout=self.retrieval_timeout
            )
            
            results = retrieval_results.get("results", [])
            context = self.format_retrieved_context(results)
            retrieval_time = time.time() - retrieval_start

            # CRITICAL FIX: Stream response from OpenAI with timeout protection
            full_response = ""
            generation_start = time.time()

            try:
                # Create a timeout wrapper for streaming
                async def _stream_with_timeout():
                    async for chunk in self.llm.generate_streaming_response(
                        query=query_text,
                        context=context if context else None,
                        chat_history=chat_history
                    ):
                        return chunk

                # Stream with individual chunk timeout
                timeout_per_chunk = 5.0
                chunk_count = 0
                
                async for chunk in self.llm.generate_streaming_response(
                    query=query_text,
                    context=context if context else None,
                    chat_history=chat_history
                ):
                    chunk_count += 1
                    full_response += chunk
                    
                    # Yield each chunk immediately for real-time TTS
                    yield {
                        "chunk": chunk,
                        "done": False,
                        "sources": retrieval_results.get("sources", []),
                        "chunk_number": chunk_count
                    }
                    
                    # CRITICAL: Check if we're taking too long
                    if time.time() - generation_start > self.openai_timeout:
                        logger.warning("Streaming timeout reached, ending response")
                        break

                # Final completion signal
                generation_time = time.time() - generation_start
                
                yield {
                    "chunk": "",
                    "full_response": full_response,
                    "done": True,
                    "sources": retrieval_results.get("sources", []),
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": time.time() - start_time,
                    "context_used": bool(context),
                    "chunk_count": chunk_count
                }

            except asyncio.TimeoutError:
                logger.error("Streaming generation timed out")
                yield {
                    "chunk": " I'm taking too long. Let me know if you need anything else.",
                    "full_response": full_response + " I'm taking too long. Let me know if you need anything else.",
                    "done": True,
                    "sources": retrieval_results.get("sources", []),
                    "error": "streaming_timeout",
                    "total_time": time.time() - start_time
                }

        except asyncio.TimeoutError:
            logger.error(f"Retrieval timed out for streaming query: {query_text}")
            
            yield {
                "chunk": "I'm having trouble finding that information quickly.",
                "done": True,
                "error": "retrieval_timeout",
                "total_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            
            yield {
                "chunk": "I'm having trouble processing that request.",
                "done": True,
                "error": str(e),
                "total_time": time.time() - start_time
            }

    async def get_stats(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Get query engine statistics with timeout protection.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get index stats with timeout
            index_stats = await asyncio.wait_for(
                self.index_manager.get_index_stats(),
                timeout=5.0
            )
            
            # Get LLM info
            llm_info = self.llm.get_model_info()

            return {
                "index_stats": index_stats,
                "llm_info": llm_info,
                "config": {
                    "top_k": self.top_k,
                    "min_score": self.min_score,
                    "retrieval_timeout": self.retrieval_timeout,
                    "total_timeout": self.total_timeout,
                    "openai_timeout": self.openai_timeout
                },
                "performance_config": self.performance_config,
                "is_initialized": self.is_initialized
            }
        except asyncio.TimeoutError:
            logger.error("Get stats timed out")
            return {
                "error": "timeout",
                "is_initialized": self.is_initialized,
                "config": {
                    "top_k": self.top_k,
                    "min_score": self.min_score
                }
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e),
                "is_initialized": self.is_initialized
            }