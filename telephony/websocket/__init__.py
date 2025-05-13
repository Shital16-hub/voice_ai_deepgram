"""
WebSocket handler components for Twilio media streams - Updated for v2 components.
"""

from .connection_manager import ConnectionManager
from .audio_manager_v2 import AudioManager  # Changed from audio_manager to audio_manager_v2
from .speech_processor_v3 import SpeechProcessor  # Changed to v3
from .response_generator import ResponseGenerator
from .message_router_v2 import MessageRouter  # Changed to v2

__all__ = [
    'ConnectionManager',
    'AudioManager',
    'SpeechProcessor',
    'ResponseGenerator',
    'MessageRouter'
]