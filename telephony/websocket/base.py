"""
Base classes and utilities for WebSocket handling.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"

@dataclass
class CallInfo:
    """Information about the current call."""
    call_sid: str
    stream_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    start_time: Optional[float] = None
    state: ConnectionState = ConnectionState.DISCONNECTED

class WebSocketEventHandler:
    """Base class for WebSocket event handling."""
    
    def __init__(self, call_sid: str):
        """Initialize the event handler."""
        self.call_sid = call_sid
        self.call_info = CallInfo(call_sid=call_sid)
        self.event_handlers = {}
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Timing tracking
        self.last_activity_time = time.time()
        self.last_response_time = time.time()
        
        # State tracking
        self.is_processing = False
        self.processing_lock = asyncio.Lock()
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for different WebSocket events."""
        self.event_handlers = {
            'connected': self._handle_connected,
            'start': self._handle_start,
            'media': self._handle_media,
            'stop': self._handle_stop,
            'mark': self._handle_mark
        }
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], ws) -> None:
        """Handle a WebSocket event."""
        self.last_activity_time = time.time()
        
        handler = self.event_handlers.get(event_type)
        if handler:
            await handler(data, ws)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def _handle_connected(self, data: Dict[str, Any], ws) -> None:
        """Handle connected event - to be implemented by subclasses."""
        pass
    
    async def _handle_start(self, data: Dict[str, Any], ws) -> None:
        """Handle start event - to be implemented by subclasses."""
        pass
    
    async def _handle_media(self, data: Dict[str, Any], ws) -> None:
        """Handle media event - to be implemented by subclasses."""
        pass
    
    async def _handle_stop(self, data: Dict[str, Any], ws) -> None:
        """Handle stop event - to be implemented by subclasses."""
        pass
    
    async def _handle_mark(self, data: Dict[str, Any], ws) -> None:
        """Handle mark event - to be implemented by subclasses."""
        pass

class RealTimeAudioBuffer:
    """Specialized buffer for real-time audio processing."""
    
    def __init__(self, max_size: int = 32000):
        """Initialize the buffer."""
        self.buffer = bytearray()
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def add(self, data: bytes):
        """Add data to the buffer."""
        async with self.lock:
            self.buffer.extend(data)
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]
    
    async def get(self, size: Optional[int] = None) -> bytes:
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

class KeepAliveManager:
    """Manages keep-alive functionality for WebSocket connections."""
    
    def __init__(self, interval: float = 10.0):
        """Initialize the keep-alive manager."""
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self, ws, stream_sid: str):
        """Start the keep-alive task."""
        if self.task and not self.task.done():
            self.task.cancel()
        
        self.running = True
        self.task = asyncio.create_task(self._keep_alive_loop(ws, stream_sid))
    
    async def stop(self):
        """Stop the keep-alive task."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    async def _keep_alive_loop(self, ws, stream_sid: str):
        """Send periodic keep-alive messages."""
        try:
            while self.running:
                await asyncio.sleep(self.interval)
                
                if not self.running:
                    break
                
                try:
                    import json
                    message = {
                        "event": "ping",
                        "streamSid": stream_sid
                    }
                    ws.send(json.dumps(message))
                    logger.debug("Sent keep-alive ping")
                except Exception as e:
                    logger.error(f"Error sending keep-alive: {e}")
                    if "Connection closed" in str(e):
                        self.running = False
                        break
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")