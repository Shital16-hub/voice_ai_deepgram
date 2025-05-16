"""
Knowledge base component for Voice AI Agent.
Updated to use OpenAI + Pinecone instead of Ollama + Chroma.
"""
from knowledge_base.document_store import DocumentStore
from knowledge_base.openai_embeddings import OpenAIEmbeddings, get_embedding_model
from knowledge_base.pinecone_store import PineconeVectorStore
from knowledge_base.index_manager import IndexManager
from knowledge_base.query_engine import QueryEngine
from knowledge_base.openai_llm import OpenAILLM
from knowledge_base.schema import Document, DocumentMetadata
from knowledge_base.conversation_manager import ConversationManager, ConversationState, ConversationTurn

__version__ = "0.3.0"

__all__ = [
    "Document",
    "DocumentMetadata",
    "DocumentStore",
    "OpenAIEmbeddings",
    "get_embedding_model",
    "PineconeVectorStore",
    "IndexManager",
    "QueryEngine",
    "OpenAILLM",
    "ConversationManager",
    "ConversationState",
    "ConversationTurn",
]