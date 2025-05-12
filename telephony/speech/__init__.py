"""
Speech recognition components for telephony integration.
"""
from .speech_recognition_manager import SpeechRecognitionManager
from .transcription_cleaner import TranscriptionCleaner

__all__ = [
    'SpeechRecognitionManager',
    'TranscriptionCleaner'
]