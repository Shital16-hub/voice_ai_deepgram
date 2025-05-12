"""
Document processing for the knowledge base.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import tiktoken

from knowledge_base.config import get_processing_config
from knowledge_base.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for the knowledge base."""
    
    def __init__(self):
        """Initialize document processor."""
        self.config = get_processing_config()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_text(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """Process text into chunks suitable for vector storage."""
        try:
            # Split text into chunks
            chunks = self._split_text(text)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_id = self._generate_document_id(chunk, source, i)
                
                documents.append({
                    "id": doc_id,
                    "text": chunk,
                    "source": source,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "metadata": {
                        "token_count": self._count_tokens(chunk),
                        "word_count": len(chunk.split()),
                        "char_count": len(chunk)
                    }
                })
            
            logger.info(f"Processed text into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise DocumentProcessingError(f"Failed to process text: {str(e)}")
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file into document chunks."""
        try:
            # Check file exists and is supported
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.config["supported_types"]:
                raise DocumentProcessingError(f"Unsupported file type: {file_ext}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = self.config["max_document_size_mb"] * 1024 * 1024
            if file_size > max_size:
                raise DocumentProcessingError(f"File too large: {file_size} bytes")
            
            # Read file content based on type
            if file_ext == ".txt":
                text = self._read_text_file(file_path)
            elif file_ext == ".pdf":
                text = self._read_pdf_file(file_path)
            elif file_ext == ".docx":
                text = self._read_docx_file(file_path)
            else:
                text = self._read_text_file(file_path)  # Fallback
            
            # Process the extracted text
            source = Path(file_path).name
            return self.process_text(text, source)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to process file: {str(e)}")
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]
        
        # Simple token-based chunking
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        # Ensure we have at least one chunk
        if not chunks and text:
            chunks = [text]
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _generate_document_id(self, text: str, source: str, chunk_index: int) -> str:
        """Generate unique document ID."""
        content = f"{source}_{chunk_index}_{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _read_text_file(self, file_path: str) -> str:
        """Read plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _read_pdf_file(self, file_path: str) -> str:
        """Read PDF file."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.warning("PyPDF2 not installed, falling back to text reading")
            return self._read_text_file(file_path)
    
    def _read_docx_file(self, file_path: str) -> str:
        """Read DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.warning("python-docx not installed, falling back to text reading")
            return self._read_text_file(file_path)