"""
WebSocket connection management utilities.
"""
import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connection state and operations."""
    
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.stream_sid = None
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        self._last_ping_time = 0
    
    def set_connected(self, connected: bool) -> None:
        """Set connection state."""
        self.connected = connected
        if connected:
            self.connection_active.set()
        else:
            self.connection_active.clear()
    
    def set_stream_sid(self, stream_sid: str) -> None:
        """Set stream SID."""
        self.stream_sid = stream_sid
    
    def is_ready_for_audio(self) -> bool:
        """Check if connection is ready for audio streaming."""
        return self.connected and self.stream_sid is not None
    
    async def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout, False otherwise
        """
        try:
            await asyncio.wait_for(self.connection_active.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for connection for call {self.call_sid}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "connected": self.connected,
            "ready_for_audio": self.is_ready_for_audio()
        }