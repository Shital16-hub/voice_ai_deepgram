# telephony/websocket/audio_manager.py

"""
Fixed audio manager that properly handles MULAW audio for Google Cloud STT v2.
"""
import base64
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class AudioManager:
    """
    Audio manager that properly handles MULAW audio without unnecessary conversions.
    """
    
    def __init__(self):
        """Initialize audio manager for MULAW processing."""
        logger.info("Initializing AudioManager for MULAW processing")
        
        self.input_buffer = bytearray()
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Optimized settings for 8kHz MULAW
        self.min_chunk_size = 1600      # 200ms at 8kHz
        self.max_buffer_size = 8000     # 1 second max
        self.max_chunk_for_stt = 20480  # 20KB limit for Google Cloud STT
        
        # Statistics
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        
        # Processing control
        self.last_processing_time = 0
        self.min_processing_interval = 0.1  # 100ms minimum between chunks
        
        logger.info(f"AudioManager initialized - min_chunk={self.min_chunk_size}, max={self.max_buffer_size}")
    
    def _analyze_mulaw_quality(self, mulaw_data: bytes) -> Dict[str, Any]:
        """
        Analyze MULAW data quality without conversion.
        """
        if len(mulaw_data) < 160:  # Less than 20ms
            return {"valid": False, "reason": "too_short"}
        
        # Simple MULAW analysis
        sample = list(mulaw_data[:min(1000, len(mulaw_data))])
        
        # Check for silence (all zeros or very low values)
        non_zero_count = sum(1 for x in sample if x != 0)
        if non_zero_count < len(sample) * 0.1:  # Less than 10% non-zero
            return {"valid": False, "reason": "mostly_silence"}
        
        # Check for reasonable distribution
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        
        # MULAW should have reasonable variation
        is_valid = std_val > 5 and 20 < mean_val < 235
        
        return {
            "valid": is_valid,
            "mean": mean_val,
            "std": std_val,
            "size": len(mulaw_data),
            "non_zero_ratio": non_zero_count / len(sample)
        }
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process Twilio media data - keep MULAW as raw bytes."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.debug(f"Media event #{self.media_events_received}: No payload")
            return None
        
        try:
            # Decode base64 to get raw MULAW bytes
            mulaw_data = base64.b64decode(payload)
            self.total_audio_bytes += len(mulaw_data)
            
            # Skip if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Check minimum time interval
            current_time = time.time()
            if current_time - self.last_processing_time < self.min_processing_interval:
                # Add to buffer for later processing
                self.input_buffer.extend(mulaw_data)
                return None
            
            # Add to buffer
            self.input_buffer.extend(mulaw_data)
            
            # Check if we have enough data to process
            if len(self.input_buffer) >= self.min_chunk_size:
                # Respect Google Cloud STT 25KB limit
                chunk_size = min(len(self.input_buffer), self.max_chunk_for_stt)
                
                # Get chunk as raw MULAW bytes
                audio_chunk = bytes(self.input_buffer[:chunk_size])
                
                # Remove processed data from buffer
                self.input_buffer = self.input_buffer[chunk_size:]
                
                # Analyze quality
                quality = self._analyze_mulaw_quality(audio_chunk)
                
                if quality["valid"]:
                    self.last_processing_time = current_time
                    self.audio_chunks_sent += 1
                    
                    logger.info(f"Sending MULAW chunk #{self.audio_chunks_sent}: {len(audio_chunk)} bytes "
                               f"(std={quality['std']:.1f}, mean={quality['mean']:.1f})")
                    
                    return audio_chunk
                else:
                    logger.debug(f"Skipping invalid MULAW chunk: {quality['reason']}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state and manage buffer."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
            if speaking:
                # Clear buffer when starting to speak
                buffer_size = len(self.input_buffer)
                self.input_buffer.clear()
                logger.info(f"Cleared {buffer_size} bytes from buffer (started speaking)")
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        buffer_size = len(self.input_buffer)
        if buffer_size > 0:
            logger.debug(f"Clearing audio buffer: {buffer_size} bytes")
            self.input_buffer.clear()
    
    def update_response_time(self) -> None:
        """Update the last response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics."""
        current_time = time.time()
        success_rate = (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "current_buffer_size": len(self.input_buffer),
            "is_speaking": self.is_speaking,
            "success_rate": round(success_rate, 2),
            "time_since_response": current_time - self.last_response_time
        }