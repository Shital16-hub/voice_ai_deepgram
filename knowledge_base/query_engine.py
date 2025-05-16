"""
Query engine using OpenAI + Pinecone for ultra-low latency retrieval and generation.
Optimized specifically for telephony conversations.
"""
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator

from knowledge_base.config import get_retriever_config
from knowledge_base.index_manager import IndexManager
from knowledge_base.openai_llm import OpenAILLM, create_telephony_optimized_messages
from knowledge_base.schema import Document

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Query engine using OpenAI + Pinecone for knowledge retrieval and response generation.
    Optimized for ultra-low latency telephony applications.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[OpenAILLM] = None
    ):
        """
        Initialize QueryEngine.

        Args:
            index_manager: IndexManager instance
            config: Optional configuration dictionary
            llm: Optional pre-initialized OpenAI LLM
        """
        self.index_manager = index_manager
        self.config = config or get_retriever_config()
        self.top_k = self.config["top_k"]
        self.min_score = self.config["min_score"]

        # Initialize OpenAI LLM
        self.llm = llm or OpenAILLM()

        self.is_initialized = False

        logger.info(f"Initialized QueryEngine with top_k={self.top_k}, min_score={self.min_score}")

    async def init(self):
        """Initialize the query engine."""
        if self.is_initialized:
            return

        # Ensure index manager is initialized
        if not self.index_manager.is_initialized:
            await self.index_manager.init()

        self.is_initialized = True
        logger.info("Query engine initialized with OpenAI + Pinecone")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
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
            # Search using Pinecone
            results = await self.index_manager.search_documents(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filter_metadata=filter_metadata
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
        Retrieve documents with source information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            Dictionary with results and sources
        """
        docs = await self.retrieve(query, top_k, min_score)

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

    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string optimized for telephony.
        
        Args:
            results: Retrieved document results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""

        # For telephony, keep context concise and relevant
        context_parts = []

        for i, doc in enumerate(results):
            text = doc["text"]
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Source {i+1}")

            # Truncate text for telephony (keep most relevant parts)
            if len(text) > 300:
                text = text[:297] + "..."

            context_parts.append(f"From {source}: {text}")

        # Join and limit total context length for telephony
        context = "\n\n".join(context_parts)
        if len(context) > 800:
            context = context[:797] + "..."

        return context

    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query the knowledge base using OpenAI + Pinecone.
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            await self.init()

        try:
            # Track timing
            start_time = time.time()

            # Retrieve relevant documents
            retrieval_results = await self.retrieve_with_sources(query_text)
            results = retrieval_results["results"]
            
            # Format context for telephony
            context = self.format_retrieved_context(results)

            # Generate response using OpenAI
            response = await self.llm.generate_response(
                query=query_text,
                context=context if context else None
            )

            # Prepare result
            result = {
                "query": query_text,
                "response": response,
                "sources": results,
                "total_time": time.time() - start_time,
                "context_used": bool(context)
            }

            return result

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            
            return {
                "query": query_text,
                "response": "I'm sorry, I'm having trouble accessing that information right now. Could you please rephrase your question?",
                "sources": [],
                "total_time": time.time() - start_time if 'start_time' in locals() else 0,
                "error": str(e)
            }

    async def query_with_streaming(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Query with streaming response optimized for telephony.
        
        Args:
            query_text: Query text
            chat_history: Previous conversation history
            
        Yields:
            Response chunks
        """
        if not self.is_initialized:
            await self.init()

        try:
            # Retrieve relevant context quickly
            retrieval_start = time.time()
            retrieval_results = await self.retrieve_with_sources(query_text)
            results = retrieval_results.get("results", [])
            context = self.format_retrieved_context(results)
            retrieval_time = time.time() - retrieval_start

            full_response = ""

            # Stream response from OpenAI
            async for chunk in self.llm.generate_streaming_response(
                query=query_text,
                context=context if context else None,
                chat_history=chat_history
            ):
                full_response += chunk
                
                # Yield each chunk immediately for real-time TTS
                yield {
                    "chunk": chunk,
                    "done": False,
                    "sources": retrieval_results.get("sources", [])
                }

            # Final completion signal
            yield {
                "chunk": "",
                "full_response": full_response,
                "done": True,
                "sources": retrieval_results.get("sources", []),
                "retrieval_time": retrieval_time,
                "context_used": bool(context)
            }

        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            
            yield {
                "chunk": "I'm sorry, I'm having trouble processing that right now.",
                "done": True,
                "error": str(e)
            }

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get query engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Get index stats
        index_stats = await self.index_manager.get_index_stats()
        
        # Get LLM info
        llm_info = self.llm.get_model_info()

        return {
            "index_stats": index_stats,
            "llm_info": llm_info,
            "config": {
                "top_k": self.top_k,
                "min_score": self.min_score
            },
            "is_initialized": self.is_initialized
        }