"""
Configuration settings for the Text-to-Speech module.
Updated with Google Cloud TTS settings and latest voice types.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List, Union

# Load environment variables from .env file
load_dotenv()

class TTSConfig(BaseSettings):
    """Configuration for Text-to-Speech module with Google Cloud TTS."""
    
    # Google Cloud TTS settings
    voice_name: str = Field(
        default=os.getenv("TTS_VOICE_NAME", "en-US-Neural2-C"),
        description="Google Cloud TTS voice name"
    )
    
    voice_gender: str = Field(
        default=os.getenv("TTS_VOICE_GENDER", "NEUTRAL"),
        description="Google Cloud TTS voice gender (MALE, FEMALE, NEUTRAL)"
    )
    
    language_code: str = Field(
        default=os.getenv("TTS_LANGUAGE_CODE", "en-US"),
        description="Google Cloud TTS language code"
    )
    
    # Voice type selection (latest models)
    voice_type: str = Field(
        default=os.getenv("TTS_VOICE_TYPE", "NEURAL2"),
        description="Voice type: CHIRP_3_HD, NEURAL2, STUDIO, WAVENET, STANDARD"
    )
    
    # Audio settings optimized for Twilio
    sample_rate: int = Field(
        default=8000,  # 8kHz for Twilio telephony
        description="Audio sample rate in Hz"
    )
    
    container_format: str = Field(
        default="mulaw",  # mulaw for Twilio compatibility
        description="Audio container format (mp3, wav, mulaw)"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of TTS results"
    )
    
    cache_dir: str = Field(
        default="./cache/tts_cache",
        description="Directory for caching TTS results"
    )
    
    # Streaming settings
    chunk_size: int = Field(
        default=1024,
        description="Size of audio chunks to process at once in bytes"
    )
    
    max_text_chunk_size: int = Field(
        default=100,
        description="Maximum text chunk size to send to TTS at once"
    )
    
    stream_timeout: float = Field(
        default=10.0,
        description="Timeout for streaming operations in seconds"
    )
    
    # Telephony optimization settings
    optimize_streaming_latency: int = Field(
        default=4,  # Maximum optimization
        description="Streaming latency optimization level (1-4)"
    )
    
    # Google Cloud TTS specific settings
    enable_telephony_effects: bool = Field(
        default=True,
        description="Enable telephony-class audio effects profile"
    )
    
    # Deprecated ElevenLabs settings (kept for backward compatibility)
    elevenlabs_api_key: str = Field(
        default=os.getenv("ELEVENLABS_API_KEY", ""),
        description="ElevenLabs API Key (deprecated - use Google Cloud TTS)"
    )
    
    model_id: str = Field(
        default=os.getenv("TTS_MODEL_ID", "eleven_turbo_v2"),
        description="ElevenLabs model ID (deprecated)"
    )
    
    voice_id: str = Field(
        default=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        description="ElevenLabs Voice ID (deprecated)"
    )
    
    class Config:
        env_prefix = "TTS_"
        case_sensitive = False

# Create a global config instance
config = TTSConfig()

# Voice type mappings for different use cases
VOICE_MAPPINGS = {
    # Chirp 3 HD voices (latest and highest quality)
    "CHIRP_3_HD": {
        "en-US": "en-US-Chirp3-HD-C",
        "en-GB": "en-GB-Chirp3-HD-B",
        "es-ES": "es-ES-Chirp3-HD-A",
        "fr-FR": "fr-FR-Chirp3-HD-A",
        "de-DE": "de-DE-Chirp3-HD-A",
        "ja-JP": "ja-JP-Chirp3-HD-A",
        "ko-KR": "ko-KR-Chirp3-HD-A",
        "pt-BR": "pt-BR-Chirp3-HD-A",
        "zh-CN": "zh-CN-Chirp3-HD-A",
    },
    
    # Neural2 voices (custom voice technology)
    "NEURAL2": {
        "en-US": "en-US-Neural2-C",
        "en-US-MALE": "en-US-Neural2-D",
        "en-US-FEMALE": "en-US-Neural2-F",
        "en-GB": "en-GB-Neural2-A",
        "en-AU": "en-AU-Neural2-A",
        "es-ES": "es-ES-Neural2-A",
        "fr-FR": "fr-FR-Neural2-A",
        "de-DE": "de-DE-Neural2-A",
        "it-IT": "it-IT-Neural2-A",
        "ja-JP": "ja-JP-Neural2-B",
        "ko-KR": "ko-KR-Neural2-A",
        "pt-BR": "pt-BR-Neural2-A",
        "zh-CN": "zh-CN-Neural2-A",
    },
    
    # Studio voices (professional quality)
    "STUDIO": {
        "en-US": "en-US-Studio-Q",
        "en-GB": "en-GB-Studio-B",
        "en-AU": "en-AU-Studio-A",
    },
    
    # WaveNet voices (high quality neural)
    "WAVENET": {
        "en-US": "en-US-Wavenet-C",
        "en-US-MALE": "en-US-Wavenet-D",
        "en-US-FEMALE": "en-US-Wavenet-F",
        "en-GB": "en-GB-Wavenet-A",
        "en-AU": "en-AU-Wavenet-A",
        "es-ES": "es-ES-Wavenet-A",
        "fr-FR": "fr-FR-Wavenet-A",
        "de-DE": "de-DE-Wavenet-A",
        "it-IT": "it-IT-Wavenet-A",
        "ja-JP": "ja-JP-Wavenet-A",
        "ko-KR": "ko-KR-Wavenet-A",
        "pt-BR": "pt-BR-Wavenet-A",
        "zh-CN": "zh-CN-Wavenet-A",
    },
    
    # Standard voices (basic quality)
    "STANDARD": {
        "en-US": "en-US-Standard-C",
        "en-US-MALE": "en-US-Standard-D",
        "en-US-FEMALE": "en-US-Standard-F",
        "en-GB": "en-GB-Standard-A",
        "en-AU": "en-AU-Standard-A",
        "es-ES": "es-ES-Standard-A",
        "fr-FR": "fr-FR-Standard-A",
        "de-DE": "de-DE-Standard-A",
        "it-IT": "it-IT-Standard-A",
        "ja-JP": "ja-JP-Standard-A",
        "ko-KR": "ko-KR-Standard-A",
        "pt-BR": "pt-BR-Standard-A",
        "zh-CN": "zh-CN-Standard-A",
    }
}

# Recommended voice types by use case
RECOMMENDED_VOICES = {
    "telephony": "NEURAL2",  # Best for phone calls
    "conversational_ai": "CHIRP_3_HD",  # Highest quality for AI assistants
    "ivr": "STUDIO",  # Professional IVR systems
    "cost_effective": "STANDARD",  # Budget option
    "high_quality": "WAVENET",  # Good quality option
}

def get_recommended_voice(language_code: str, voice_type: str, gender: Optional[str] = None) -> str:
    """
    Get recommended voice name based on language, type, and gender.
    
    Args:
        language_code: Language code (e.g., 'en-US')
        voice_type: Voice type (CHIRP_3_HD, NEURAL2, etc.)
        gender: Optional gender (MALE, FEMALE)
        
    Returns:
        Recommended voice name
    """
    voice_mapping = VOICE_MAPPINGS.get(voice_type, VOICE_MAPPINGS["NEURAL2"])
    
    # Try to find gender-specific voice
    if gender and gender.upper() in ["MALE", "FEMALE"]:
        gender_key = f"{language_code}-{gender.upper()}"
        if gender_key in voice_mapping:
            return voice_mapping[gender_key]
    
    # Fall back to default voice for language
    return voice_mapping.get(language_code, voice_mapping.get("en-US", "en-US-Neural2-C"))