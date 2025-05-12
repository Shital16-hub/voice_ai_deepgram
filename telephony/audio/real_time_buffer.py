"""
Real-time audio buffer for prioritizing recent audio.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

class RealTimeAudioBuffer:
    """Specialized buffer for real-time audio processing that prioritizes recent audio."""
    
    def __init__(self, max_size=32000):
        self.buffer = bytearray()
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def add(self, data):
        """Add data to the buffer, keeping only the most recent data if size exceeded."""
        async with self.lock:
            self.buffer.extend(data)
            # Keep only the most recent data
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]
    
    async def get(self, size=None):
        """Get data from the buffer."""
        async with self.lock:
            if size is None:
                return bytes(self.buffer)
            else:
                return bytes(self.buffer[-size:])
    
    async def clear(self):
        """Clear the buffer."""
        async with self.lock:
            self.buffer = bytearray()