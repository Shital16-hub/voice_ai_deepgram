"""
Mulaw buffer processor for handling small audio chunks.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MulawBufferProcessor:
    """
    Buffer processor for small mulaw audio chunks.
    Accumulates small chunks to avoid processing many small chunks.
    """
    
    def __init__(self, min_chunk_size=640):  # 80ms at 8kHz
        """
        Initialize buffer processor.
        
        Args:
            min_chunk_size: Minimum chunk size to process (default 640 bytes = 80ms at 8kHz)
        """
        self.buffer = bytearray()
        self.min_chunk_size = min_chunk_size
        logger.info(f"Initialized MulawBufferProcessor with min_chunk_size={min_chunk_size}")
    
    def process(self, data: bytes) -> Optional[bytes]:
        """
        Process incoming mulaw data by buffering until minimum size is reached.
        
        Args:
            data: New audio data
            
        Returns:
            Processed data of minimum size or None if still buffering
        """
        # Skip empty chunks
        if not data:
            return None
            
        # Log small chunks at debug level
        if 0 < len(data) < 320:  # Less than 40ms at 8kHz
            logger.debug(f"Small mulaw data: {len(data)} bytes (accumulating)")
            
        # Add to buffer
        self.buffer.extend(data)
        
        # Only process if we have enough data
        if len(self.buffer) >= self.min_chunk_size:
            # Get the buffered data
            result = bytes(self.buffer)
            
            # Clear buffer
            self.buffer = bytearray()
            
            logger.debug(f"Processed mulaw buffer: {len(result)} bytes")
            return result
        
        # Not enough data yet
        return None