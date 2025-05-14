# speech_to_text/config.py
"""
Configuration settings for Google Cloud Speech-to-Text v2 with latest features.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class STTConfig(BaseSettings):
    """Configuration for Speech-to-Text v2 module."""
    
    # Google Cloud credentials
    credentials_file: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # Project settings
    project_id: str = Field(
        default=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        description="Google Cloud Project ID"
    )
    
    # STT v2 settings
    model_name: str = Field(
        default="telephony_short",
        description="Google Cloud Speech-to-Text v2 model name"
    )
    
    language: str = Field(
        default="en-US",
        description="Language code for speech recognition"
    )
    
    sample_rate: int = Field(
        default=8000,
        description="Audio sample rate in Hz (8kHz for telephony)"
    )
    
    encoding: str = Field(
        default="MULAW",
        description="Audio encoding (MULAW for telephony compatibility)"
    )
    
    channels: int = Field(
        default=1,
        description="Number of audio channels"
    )
    
    # Streaming settings
    interim_results: bool = Field(
        default=False,
        description="Whether to return interim results (disabled for better accuracy)"
    )
    
    enable_automatic_punctuation: bool = Field(
        default=True,
        description="Enable automatic punctuation in transcriptions"
    )
    
    enable_word_time_offsets: bool = Field(
        default=True,
        description="Enable word-level timestamps"
    )
    
    enable_word_confidence: bool = Field(
        default=True,
        description="Enable word-level confidence scores"
    )
    
    # Telephony optimizations
    use_enhanced_model: bool = Field(
        default=True,
        description="Use enhanced model for telephony"
    )
    
    enable_speaker_diarization: bool = Field(
        default=False,
        description="Enable speaker diarization (disabled for single speaker)"
    )
    
    # Voice activity detection
    enable_voice_activity_events: bool = Field(
        default=True,
        description="Enable voice activity events for better endpoint detection"
    )
    
    speech_start_timeout: int = Field(
        default=5,
        description="Speech start timeout in seconds"
    )
    
    speech_end_timeout: int = Field(
        default=1,
        description="Speech end timeout in seconds"
    )
    
    # Performance settings
    max_alternatives: int = Field(
        default=1,
        description="Maximum number of alternative transcriptions"
    )
    
    profanity_filter: bool = Field(
        default=False,
        description="Filter profanity from transcription results"
    )
    
    # Regional settings
    location: str = Field(
        default="global",
        description="Google Cloud region (global or specific region)"
    )
    
    recognizer_id: str = Field(
        default="_",
        description="Recognizer ID for v2 API"
    )
    
    # Keywords to boost recognition for telephony
    speech_contexts: list = Field(
        default=[
            "pricing", "plan", "cost", "subscription", "service", "features", 
            "support", "premium", "basic", "enterprise", "monthly", "annual",
            "upgrade", "downgrade", "cancel", "billing", "payment", "trial"
        ],
        description="Keywords to boost in telephony context"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create a global config instance
config = STTConfig()

# Available models in Google Cloud Speech-to-Text v2
AVAILABLE_MODELS = {
    "telephony_short": {
        "description": "Optimized for telephony with utterances under 60 seconds",
        "best_for": "Phone calls, IVR, call centers",
        "sample_rate": [8000, 16000],
        "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"]
    },
    "telephony": {
        "description": "General telephony model for longer conversations",
        "best_for": "Extended phone conversations",
        "sample_rate": [8000, 16000],
        "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"]
    },
    "latest_short": {
        "description": "Latest model for short utterances",
        "best_for": "Voice commands, short phrases",
        "sample_rate": [16000, 44100, 48000],
        "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"]
    },
    "latest_long": {
        "description": "Latest model for long-form audio",
        "best_for": "Meetings, lectures, long recordings",
        "sample_rate": [16000, 44100, 48000],
        "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"]
    },
    "medical_conversation": {
        "description": "Specialized for medical conversations",
        "best_for": "Medical consultations, healthcare",
        "sample_rate": [16000],
        "languages": ["en-US"]
    },
    "medical_dictation": {
        "description": "Specialized for medical dictation",
        "best_for": "Medical notes, dictation",
        "sample_rate": [16000],
        "languages": ["en-US"]
    }
}

# Recommended configurations by use case
RECOMMENDED_CONFIGS = {
    "telephony": {
        "model": "telephony_short",
        "sample_rate": 8000,
        "encoding": "MULAW",
        "interim_results": False,
        "enable_automatic_punctuation": True,
        "enable_word_time_offsets": True,
        "use_enhanced_model": True
    },
    "ivr": {
        "model": "telephony_short",
        "sample_rate": 8000,
        "encoding": "MULAW",
        "interim_results": False,
        "enable_automatic_punctuation": False,
        "enable_word_time_offsets": False,
        "use_enhanced_model": True
    },
    "call_center": {
        "model": "telephony",
        "sample_rate": 8000,
        "encoding": "MULAW",
        "interim_results": True,
        "enable_automatic_punctuation": True,
        "enable_word_time_offsets": True,
        "enable_speaker_diarization": True,
        "use_enhanced_model": True
    },
    "voice_assistant": {
        "model": "latest_short",
        "sample_rate": 16000,
        "encoding": "LINEAR16",
        "interim_results": True,
        "enable_automatic_punctuation": True,
        "enable_word_time_offsets": True,
        "use_enhanced_model": True
    }
}

def get_recommended_config(use_case: str) -> dict:
    """
    Get recommended configuration for a specific use case.
    
    Args:
        use_case: One of 'telephony', 'ivr', 'call_center', 'voice_assistant'
        
    Returns:
        Dictionary with recommended configuration
    """
    return RECOMMENDED_CONFIGS.get(use_case, RECOMMENDED_CONFIGS["telephony"])

def validate_model_config(model: str, language: str, sample_rate: int) -> bool:
    """
    Validate that a model supports the given language and sample rate.
    
    Args:
        model: Model name
        language: Language code
        sample_rate: Sample rate in Hz
        
    Returns:
        True if the configuration is valid
    """
    if model not in AVAILABLE_MODELS:
        return False
    
    model_info = AVAILABLE_MODELS[model]
    
    # Check language support
    if language not in model_info["languages"]:
        return False
    
    # Check sample rate support
    if sample_rate not in model_info["sample_rate"]:
        return False
    
    return True