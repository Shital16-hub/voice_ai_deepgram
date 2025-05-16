"""
Schema definitions for documents.
FIXED VERSION - Ensures metadata is always a dict, not a Pydantic model
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import hashlib

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str = Field(description="Source of the document (e.g., filename)")
    source_type: str = Field(description="Type of source (e.g., file, url, text)")
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[float] = None
    modified_at: Optional[float] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create from dictionary."""
        return cls(**data)

class Document:
    """
    Document class that ALWAYS uses dict for metadata to avoid Pydantic issues.
    FIXED VERSION - Ensures compatibility with Pinecone and OpenAI.
    """
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize Document with dict metadata only.
        
        Args:
            text: The document text content
            metadata: Metadata as a dictionary (NOT a Pydantic model)
            doc_id: Optional document ID
        """
        self.text = text
        
        # Ensure metadata is always a dict
        if metadata is None:
            self.metadata = {}
        elif isinstance(metadata, dict):
            self.metadata = metadata
        elif hasattr(metadata, 'to_dict'):
            # If it's a Pydantic model, convert to dict
            self.metadata = metadata.to_dict()
        elif hasattr(metadata, 'model_dump'):
            # For newer Pydantic versions
            self.metadata = metadata.model_dump(exclude_none=True)
        else:
            # Try to convert to dict
            self.metadata = dict(metadata) if metadata else {}
        
        self.doc_id = doc_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content."""
        content_hash = hashlib.md5(self.text.encode('utf-8')).hexdigest()
        return f"doc_{content_hash}"
    
    def __str__(self) -> str:
        return f"Document(id={self.doc_id}, text={self.text[:50]}..., metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("id")
        )
    
    def to_llama_index_document(self) -> 'llama_index.core.schema.Document':
        """Convert to LlamaIndex Document (if needed for compatibility)."""
        try:
            from llama_index.core.schema import Document as LlamaDocument
            
            # Create a LlamaIndex document with dict metadata
            return LlamaDocument(
                text=self.text,
                metadata=self.metadata,  # Pass dict directly
                id_=self.doc_id
            )
        except ImportError:
            raise ImportError("LlamaIndex not available")
    
    @classmethod
    def from_llama_index_document(cls, llama_doc: 'llama_index.core.schema.Document') -> 'Document':
        """Create from LlamaIndex Document."""
        return cls(
            text=llama_doc.text,
            metadata=llama_doc.metadata,  # Should be a dict
            doc_id=llama_doc.id_
        )