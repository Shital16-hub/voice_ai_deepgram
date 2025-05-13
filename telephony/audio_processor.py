# telephony/audio_processor.py

"""
Optimized audio processing utilities for telephony integration.
"""
import audioop
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Optimized audio converter between Twilio and Voice AI formats with minimal quality loss.
    """
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM with optimized processing.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Skip very small chunks
            if len(mulaw_data) < 160:  # Less than 20ms at 8kHz
                return np.array([], dtype=np.float32)
            
            # Convert mulaw to 16-bit PCM using built-in audioop
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Convert to numpy array (float32)
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Minimal processing for better accuracy
            # Only apply gentle normalization if needed
            max_val = np.max(np.abs(audio_array))
            if max_val > 0.95:
                # Gentle normalization to prevent clipping
                audio_array = audio_array * (0.95 / max_val)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return np.array([], dtype=np.float32)
    
    def pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """
        Convert PCM audio from Voice AI to mulaw for Twilio.
        
        Args:
            pcm_data: Audio data in PCM format
            
        Returns:
            Audio data in mulaw format
        """
        try:
            # Ensure even byte count
            if len(pcm_data) % 2 != 0:
                pcm_data = pcm_data + b'\x00'
            
            # Convert to mulaw at 8kHz
            mulaw_data = audioop.lin2ulaw(pcm_data, 2)
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}")
            return b''
    
    def convert_to_mulaw_direct(self, audio_data: bytes) -> bytes:
        """
        Convert audio directly to mulaw format for Twilio.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Audio data converted to mulaw format
        """
        # If it's already mulaw (8-bit), return as-is
        if self.is_mulaw(audio_data):
            return audio_data
        
        # For other formats, convert through PCM
        return self.pcm_to_mulaw(audio_data)
    
    def is_mulaw(self, audio_data: bytes) -> bool:
        """
        Check if audio data is in mulaw format.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            True if audio is in mulaw format
        """
        if len(audio_data) < 100:
            return False
        
        # mulaw uses full 8-bit range
        sample = audio_data[:100]
        unique_values = len(set(sample))
        
        # mulaw typically has good distribution across full range
        return unique_values > 50  # Simple heuristic


class MulawBufferProcessor:
    """
    Optimized buffer processor for mulaw audio chunks with improved buffering strategy.
    """
    
    def __init__(self, min_chunk_size=6400):  # Changed from 1600 to 6400 (800ms)
        """
        Initialize buffer processor.
        
        Args:
            min_chunk_size: Minimum chunk size to process (default 6400 bytes = 800ms)
        """
        self.buffer = bytearray()
        self.min_chunk_size = min_chunk_size
        logger.info(f"MulawBufferProcessor initialized with min_chunk_size={min_chunk_size}")
    
    def process(self, data: bytes) -> Optional[bytes]:
        """
        Process incoming mulaw data by buffering until minimum size is reached.
        
        Args:
            data: New audio data
            
        Returns:
            Processed data of minimum size or None if still buffering
        """
        if not data:
            return None
        
        # Add to buffer
        self.buffer.extend(data)
        
        # Only process if we have enough data
        if len(self.buffer) >= self.min_chunk_size:
            # Get the buffered data
            result = bytes(self.buffer)
            
            # Clear buffer
            self.buffer = bytearray()
            
            return result
        
        return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush any remaining data in the buffer.
        
        Returns:
            Remaining data or None if buffer is empty
        """
        if self.buffer:
            result = bytes(self.buffer)
            self.buffer = bytearray()
            return result
        return None
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear_buffer(self) -> int:
        """
        Clear the buffer and return the size that was cleared.
        
        Returns:
            Size of data that was cleared
        """
        size = len(self.buffer)
        self.buffer = bytearray()
        return size