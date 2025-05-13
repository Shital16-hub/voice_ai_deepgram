"""
WebSocket connection management for Twilio streams.
"""
import asyncio
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connection state and keep-alive functionality."""
    
    def __init__(self, call_sid: str):
        """Initialize connection manager."""
        self.call_sid = call_sid
        self.stream_sid = None
        self.connected = False
        self.connection_active = asyncio.Event()
        self.keep_alive_task = None
        
    async def handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle WebSocket connected event."""
        logger.info(f"WebSocket connected for call {self.call_sid}")
        logger.info(f"Connected data: {data}")
        
        self.connected = True
        self.connection_active.set()
        
        # Start keep-alive task
        if not self.keep_alive_task:
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
    
    async def handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle stream start event."""
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        logger.info(f"Start data: {data}")
        
        # Signal that stream is ready
        self.connection_active.set()
    
    async def handle_stop(self, data: Dict[str, Any]) -> None:
        """Handle stream stop event."""
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        self.connected = False
        self.connection_active.clear()
        
        # Cancel keep-alive task
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
    
    async def _keep_alive_loop(self, ws) -> None:
        """Send periodic keep-alive messages."""
        try:
            while self.connected:
                await asyncio.sleep(10)
                
                if not self.stream_sid or not self.connected:
                    continue
                    
                try:
                    message = {
                        "event": "ping",
                        "streamSid": self.stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.connected = False
                        self.connection_active.clear()
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")