# telephony/audio_processor.py

"""
Optimized audio processing utilities for telephony integration.
Minimal processing to preserve speech recognition quality.
"""
import audioop
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Optimized audio converter that preserves quality for speech recognition.
    """
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert mulaw to PCM with quality preservation.
        """
        try:
            # Skip very small chunks
            if len(mulaw_data) < 160:  # Less than 20ms at 8kHz
                return np.array([], dtype=np.float32)
            
            # Convert mulaw to 16-bit PCM directly (no processing)
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Convert to float32 array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # NO additional processing - preserve original quality
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return np.array([], dtype=np.float32)
    
    def pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """
        Convert PCM to mulaw for Twilio.
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
        Convert audio directly to mulaw format.
        """
        # If it's already mulaw, return as-is
        if self.is_mulaw(audio_data):
            return audio_data
        
        # Otherwise convert
        return self.pcm_to_mulaw(audio_data)
    
    def is_mulaw(self, audio_data: bytes) -> bool:
        """
        Check if audio data is in mulaw format.
        """
        if len(audio_data) < 100:
            return False
        
        # Simple heuristic: mulaw uses full 8-bit range
        sample = audio_data[:100]
        unique_values = len(set(sample))
        
        # mulaw typically has good distribution
        return unique_values > 50
    
    def analyze_audio_quality(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Basic audio quality analysis for mulaw data.
        """
        if len(audio_data) < 100:
            return {"valid": False}
        
        try:
            # Basic statistics on mulaw data
            sample = list(audio_data[:1000])  # Sample first 1000 bytes
            
            mean_val = np.mean(sample)
            std_val = np.std(sample)
            min_val = min(sample)
            max_val = max(sample)
            
            # Check if it looks like valid audio
            # Mulaw audio should have reasonable variation
            valid = (
                std_val > 5  # Some variation
                and min_val < 100  # Not all high values
                and max_val > 150  # Not all low values
                and len(audio_data) >= 160  # At least 20ms
            )
            
            return {
                "valid": valid,
                "mean": mean_val,
                "std": std_val,
                "size": len(audio_data),
                "likely_speech": std_val > 10 and 50 < mean_val < 200
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"valid": False}


class MulawBufferProcessor:
    """
    Optimized buffer processor with better accumulation strategy.
    """
    
    def __init__(self, min_chunk_size=6400):  # 800ms at 8kHz
        """
        Initialize buffer processor.
        """
        self.buffer = bytearray()
        self.min_chunk_size = min_chunk_size
        self.last_flush_time = 0
        self.max_buffer_age = 2.0  # Maximum buffer age in seconds
        
        logger.info(f"MulawBufferProcessor initialized with min_chunk_size={min_chunk_size}")
    
    def process(self, data: bytes) -> Optional[bytes]:
        """
        Process incoming mulaw data with time-based flushing.
        """
        if not data:
            return None
        
        import time
        current_time = time.time()
        
        # Add to buffer
        self.buffer.extend(data)
        
        # Check if we should process based on size OR time
        should_process = (
            len(self.buffer) >= self.min_chunk_size
            or (len(self.buffer) > 1600 and current_time - self.last_flush_time > self.max_buffer_age)
        )
        
        if should_process:
            # Get the buffered data
            result = bytes(self.buffer)
            
            # Clear buffer
            self.buffer = bytearray()
            self.last_flush_time = current_time
            
            return result
        
        return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush any remaining data in the buffer.
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
        """Clear the buffer and return the size that was cleared."""
        size = len(self.buffer)
        self.buffer = bytearray()
        return size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        import time
        return {
            "buffer_size": len(self.buffer),
            "min_chunk_size": self.min_chunk_size,
            "buffer_age": time.time() - self.last_flush_time if self.last_flush_time else 0
        }