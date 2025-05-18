"""
Query Engine API for telephony integration.
Provides a simplified interface to the knowledge base.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QueryEngineAPI:
    """Simple API for accessing the query engine from telephony components."""
    
    def __init__(self, query_engine):
        """Initialize with a query engine."""
        self.query_engine = query_engine
        
    async def query(self, text: str) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            text: Query text
            
        Returns:
            Response dictionary
        """
        if not self.query_engine:
            logger.error("No query engine available")
            return {
                "response": "I'm sorry, my knowledge base is not available.",
                "error": "No query engine"
            }
        
        try:
            # Query with timeout
            result = await self.query_engine.query(text)
            return result
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return {
                "response": "I encountered an error processing your question.",
                "error": str(e)
            }
            
    def set_query_engine(self, query_engine):
        """Set the query engine."""
        self.query_engine = query_engine