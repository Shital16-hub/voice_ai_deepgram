# text_to_speech/__init__.py
"""
Text-to-Speech module for Voice AI Agent.

This module provides functionality for converting text to speech
using the ElevenLabs TTS API, optimized for real-time voice applications.
"""

from .elevenlabs_tts import ElevenLabsTTS
from .streaming import TTSStreamer, RealTimeResponseHandler
from .audio_utils import AudioProcessor
from .config import config, TTSConfig
from .exceptions import TTSError, TTSAPIError, TTSStreamingError, TTSConfigError, TTSAudioError

__all__ = [
    'ElevenLabsTTS',
    'TTSStreamer',
    'RealTimeResponseHandler',
    'AudioProcessor',
    'config',
    'TTSConfig',
    'TTSError',
    'TTSAPIError',
    'TTSStreamingError', 
    'TTSConfigError',
    'TTSAudioError'
]