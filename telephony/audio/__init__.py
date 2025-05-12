"""
Audio processing components for telephony integration.
"""
from .audio_stream_processor import AudioStreamProcessor
from .mulaw_buffer_processor import MulawBufferProcessor
from .real_time_buffer import RealTimeAudioBuffer

__all__ = [
    'AudioStreamProcessor',
    'MulawBufferProcessor',
    'RealTimeAudioBuffer'
]