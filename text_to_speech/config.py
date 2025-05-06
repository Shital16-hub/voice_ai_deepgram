# text_to_speech/config.py

"""
Configuration settings for the Text-to-Speech module.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class TTSConfig(BaseSettings):
    """Configuration for Text-to-Speech module."""
    
    # ElevenLabs API settings
    elevenlabs_api_key: str = Field(
        default=os.getenv("ELEVENLABS_API_KEY", ""),
        description="ElevenLabs API Key for TTS services"
    )
    
    # TTS settings
    model_id: str = Field(
        default=os.getenv("TTS_MODEL_ID", "eleven_turbo_v2"),  # Upgraded to latest model
        description="ElevenLabs model ID to use"
    )
    
    voice_id: str = Field(
        default=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),  # Default to Rachel voice
        description="Voice ID for the TTS system"
    )
    
    sample_rate: int = Field(
        default=8000,  # Set to 8000Hz for Twilio compatibility
        description="Audio sample rate in Hz"
    )
    
    container_format: str = Field(
        default="mulaw",  # mulaw for Twilio compatibility
        description="Audio container format (mp3, wav, mulaw)"
    )
    
    # Streaming settings
    chunk_size: int = Field(
        default=1024,
        description="Size of audio chunks to process at once in bytes"
    )
    
    max_text_chunk_size: int = Field(
        default=100,
        description="Maximum text chunk size to send to ElevenLabs at once"
    )
    
    stream_timeout: float = Field(
        default=10.0,
        description="Timeout for streaming operations in seconds"
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
    
    # Optimization settings for telephony
    optimize_streaming_latency: int = Field(
        default=4,  # Maximum optimization
        description="Streaming latency optimization level (1-4)"
    )
    
    class Config:
        env_prefix = "TTS_"
        case_sensitive = False

# Create a global config instance
config = TTSConfig()