"""
Optimized audio processing utilities for telephony integration.

Simplified version that reduces latency and improves accuracy.
"""
import audioop
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Optimized audio converter between Twilio and Voice AI formats.
    
    Simplified to reduce latency and improve accuracy.
    """
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM with minimal processing.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Skip very small chunks
            if len(mulaw_data) < 160:  # Less than 20ms at 8kHz
                return np.array([], dtype=np.float32)
            
            # For Google STT telephony optimization, we'll keep at 8kHz
            # Convert mulaw to 16-bit PCM
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Convert to numpy array (float32)
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Minimal processing for better accuracy
            # Only apply basic normalization
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                # Gentle normalization to preserve dynamics
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
    Optimized buffer processor for mulaw audio chunks.
    """
    
    def __init__(self, min_chunk_size=1600):  # 200ms at 8kHz - better for STT
        """
        Initialize buffer processor.
        
        Args:
            min_chunk_size: Minimum chunk size to process (default 1600 bytes = 200ms)
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