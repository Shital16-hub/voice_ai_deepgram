"""
OpenAI Assistants API + Pinecone integration for Voice AI Agent.
"""
from knowledge_base.openai_assistant_manager import OpenAIAssistantManager
from knowledge_base.pinecone_manager import PineconeManager
from knowledge_base.document_processor import DocumentProcessor
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine

__version__ = "2.0.0"

__all__ = [
    "OpenAIAssistantManager",
    "PineconeManager", 
    "DocumentProcessor",
    "ConversationManager",
    "QueryEngine",
]