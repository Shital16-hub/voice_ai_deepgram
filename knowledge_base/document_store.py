"""
Document processing and handling for knowledge base.
FIXED VERSION - Properly handles DocumentMetadata
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union
from pathlib import Path
import hashlib

from knowledge_base.config import get_document_processor_config
from knowledge_base.utils.file_utils import list_documents, check_supported_file
from knowledge_base.schema import Document

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Store and process documents for knowledge base ingestion.
    FIXED VERSION - Creates proper Document objects with dict metadata
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DocumentStore.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_document_processor_config()
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]
        self.max_document_size = self.config["max_document_size_mb"] * 1024 * 1024
        self.supported_types = self.config["supported_types"]
        
        logger.info(f"Initialized DocumentStore with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_types
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to split at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                search_text = text[search_start:search_end]
                
                # Find last sentence ending
                sentence_endings = ['.', '!', '?', '\n']
                last_sentence_end = -1
                
                for ending in sentence_endings:
                    pos = search_text.rfind(ending)
                    if pos > last_sentence_end:
                        last_sentence_end = pos
                
                # If we found a sentence ending, adjust the end position
                if last_sentence_end > 0:
                    end = search_start + last_sentence_end + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if end >= len(text):
                break
            
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load and process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with proper dict metadata
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file type is supported
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_document_size:
            raise ValueError(f"File too large: {file_size} bytes "
                           f"(max: {self.max_document_size} bytes)")
        
        # Extract file metadata as dict (not Pydantic model)
        file_metadata = self._extract_file_metadata(file_path)
        
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Load text content
            if ext in ['.txt', '.md']:
                # Simple text files
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            else:
                # For other file types, try to read as text for now
                # In a real implementation, you'd use appropriate parsers
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            
            # Split into chunks
            chunks = self._chunk_text(text)
            
            # Create Document objects with dict metadata
            documents = []
            for i, chunk in enumerate(chunks):
                # Create metadata for this chunk as a regular dict
                chunk_metadata = file_metadata.copy()  # This is already a dict
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(chunks)
                
                # Create Document with dict metadata
                doc = Document(
                    text=chunk,
                    metadata=chunk_metadata,  # Pass dict directly, not DocumentMetadata
                    doc_id=f"{os.path.basename(file_path)}_{i}"
                )
                documents.append(doc)
            
            logger.info(f"Processed document {file_path}: created {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def load_text(self, text: str, source_name: str = "text_input") -> List[Document]:
        """
        Load and process text directly.
        
        Args:
            text: Text content
            source_name: Name to use as source identifier
            
        Returns:
            List of Document objects with dict metadata
        """
        # Create basic metadata as dict
        metadata = {
            "source": source_name,
            "source_type": "direct_text",
            "file_path": None,
            "file_type": None,
            "file_name": None,
            "created_at": None,
            "modified_at": None
        }
        
        # Split into chunks
        chunks = self._chunk_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            # Create metadata for this chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            
            # Create Document with dict metadata
            doc = Document(
                text=chunk,
                metadata=chunk_metadata,  # Pass dict directly
                doc_id=f"{source_name}_{i}"
            )
            documents.append(doc)
        
        logger.info(f"Processed text input '{source_name}': created {len(documents)} chunks")
        return documents
    
    def load_documents_from_directory(
        self, 
        directory: str, 
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory path
            extensions: Optional list of file extensions to include
            max_files: Optional maximum number of files to process
            
        Returns:
            List of Document objects
        """
        # Get list of files
        file_paths = list_documents(directory, extensions=extensions)
        
        if max_files is not None and len(file_paths) > max_files:
            logger.info(f"Limiting to {max_files} files")
            file_paths = file_paths[:max_files]
        
        # Process each file
        all_documents = []
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Processed {len(file_paths)} files into {len(all_documents)} document chunks")
        return all_documents
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file as a regular dictionary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata (not DocumentMetadata model)
        """
        file_stat = os.stat(file_path)
        return {
            "source": os.path.basename(file_path),
            "source_type": "file",
            "file_path": os.path.abspath(file_path),
            "file_type": os.path.splitext(file_path)[1].lower(),
            "file_name": os.path.basename(file_path),
            "file_size": file_stat.st_size,
            "created_at": file_stat.st_ctime,
            "modified_at": file_stat.st_mtime
        }