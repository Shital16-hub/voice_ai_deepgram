"""
Custom exceptions for the knowledge base module.
"""

class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass

class OpenAIError(KnowledgeBaseError):
    """OpenAI API related errors."""
    pass

class PineconeError(KnowledgeBaseError):
    """Pinecone related errors."""
    pass

class DocumentProcessingError(KnowledgeBaseError):
    """Document processing errors."""
    pass

class ConversationError(KnowledgeBaseError):
    """Conversation management errors."""
    pass

class RateLimitError(KnowledgeBaseError):
    """Rate limiting errors."""
    pass

class TokenLimitError(KnowledgeBaseError):
    """Token limit exceeded errors."""
    pass