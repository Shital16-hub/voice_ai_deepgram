"""
Optimized audio manager with minimal processing for better speech recognition.
Remove all audio processing and let Google Cloud STT handle it.
"""
import base64
import asyncio
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AudioManager:
    """
    Simplified audio manager that passes audio directly to STT.
    No buffering, no processing - just stream directly for lowest latency.
    """
    
    def __init__(self):
        """Initialize audio manager with minimal processing."""
        logger.info("Initializing AudioManager (minimal processing)")
        
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Minimal buffering for very small chunks only
        self.min_chunk_size = 160  # 20ms at 8kHz
        self.buffer = bytearray()
        
        # Statistics
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        
        logger.info("AudioManager initialized for direct streaming")
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media - pass through with minimal buffering."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return None
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            audio_size = len(audio_data)
            self.total_audio_bytes += audio_size
            
            # Skip if system is speaking
            if self.is_speaking:
                return None
            
            # If chunk is too small, buffer it
            if audio_size < self.min_chunk_size:
                self.buffer.extend(audio_data)
                # If buffer is still too small, wait for more
                if len(self.buffer) < self.min_chunk_size:
                    return None
                # Return buffered data
                result = bytes(self.buffer)
                self.buffer.clear()
                self.audio_chunks_sent += 1
                return result
            
            # Chunk is large enough, add any buffered data and return
            if self.buffer:
                result = bytes(self.buffer) + audio_data
                self.buffer.clear()
            else:
                result = audio_data
            
            self.audio_chunks_sent += 1
            return result
            
        except Exception as e:
            logger.error(f"Error processing media: {e}")
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state and clear buffer if starting to speak."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
            if speaking:
                # Clear buffer when starting to speak
                buffer_size = len(self.buffer)
                self.buffer.clear()
                if buffer_size > 0:
                    logger.debug(f"Cleared {buffer_size} bytes from buffer")
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self.buffer.clear()
    
    def update_response_time(self) -> None:
        """Update response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio statistics."""
        current_time = time.time()
        success_rate = (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        avg_chunk_size = self.total_audio_bytes / max(self.media_events_received, 1)
        
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "avg_chunk_size_bytes": round(avg_chunk_size, 2),
            "current_buffer_size": len(self.buffer),
            "is_speaking": self.is_speaking,
            "time_since_response": current_time - self.last_response_time,
            "success_rate": round(success_rate, 2),
            "compression_ratio": round(self.audio_chunks_sent / max(self.media_events_received, 1), 3)
        }