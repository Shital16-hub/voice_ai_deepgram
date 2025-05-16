"""
Configuration settings for the speech-to-text module.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class STTConfig(BaseSettings):
    """Configuration for Speech-to-Text module."""
    
    # Google Cloud credentials
    credentials_file: str = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        description="Path to Google Cloud credentials JSON file"
    )
    
    # STT settings
    model_name: str = Field(
        default="latest_long",
        description="Google Cloud Speech-to-Text model name"
    )
    
    language: str = Field(
        default="en-US",
        description="Language code for speech recognition"
    )
    
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    
    # Streaming settings
    interim_results: bool = Field(
        default=True,
        description="Whether to return interim results"
    )
    
    enable_automatic_punctuation: bool = Field(
        default=True,
        description="Enable automatic punctuation in transcriptions"
    )
    
    enable_word_time_offsets: bool = Field(
        default=True,
        description="Enable word-level timestamps"
    )
    
    # Telephony optimizations
    use_enhanced_model: bool = Field(
        default=True,
        description="Use enhanced model for telephony"
    )
    
    enable_speaker_diarization: bool = Field(
        default=False,
        description="Enable speaker diarization"
    )
    
    diarization_speaker_count: int = Field(
        default=1,
        description="Expected number of speakers in the audio"
    )
    
    # Keywords to boost recognition for telephony
    speech_contexts: list = Field(
        default=["price", "plan", "cost", "subscription", "service", "features", "support"],
        description="Keywords to boost in telephony context"
    )
    
    # Performance settings
    use_enhanced_telephony: bool = Field(
        default=True,
        description="Use telephony model for optimized speech recognition"
    )
    
    profanity_filter: bool = Field(
        default=False,
        description="Filter profanity from transcription results"
    )
    
    class Config:
        env_prefix = "STT_"
        case_sensitive = False

# Create a global config instance
config = STTConfig()