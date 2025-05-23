"""
Google Cloud Text-to-Speech client optimized for Twilio telephony.
Fixed implementation with proper voice configuration and error handling.
"""
import logging
import hashlib
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from google.cloud import texttospeech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleCloudTTS:
    """Google Cloud Text-to-Speech client optimized for Twilio telephony."""
    
    def __init__(
        self,
        credentials_file: Optional[str] = None,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: str = "en-US",
        container_format: str = "mulaw",
        sample_rate: int = 8000,
        enable_caching: bool = True,
        voice_type: str = "NEURAL2"
    ):
        """
        Initialize Google Cloud TTS client optimized for Twilio.
        
        Args:
            credentials_file: Path to credentials JSON file
            voice_name: Voice name (e.g., "en-US-Neural2-C")
            voice_gender: Voice gender (deprecated for Neural2 voices)
            language_code: Language code (e.g., "en-US")
            container_format: Audio format ("mulaw" for Twilio)
            sample_rate: Sample rate (8000 for Twilio)
            enable_caching: Whether to cache synthesized audio
            voice_type: Voice type ("NEURAL2", "STANDARD", "WAVENET")
        """
        self.credentials_file = credentials_file
        self.language_code = language_code
        self.container_format = container_format.upper()
        self.sample_rate = sample_rate
        self.enable_caching = enable_caching
        self.voice_type = voice_type
        
        # Handle voice configuration properly for different voice types
        if voice_name:
            self.voice_name = voice_name
            # Don't set gender for Neural2 voices to avoid conflicts
            if "Neural2" in voice_name:
                self.voice_gender = None
                logger.info(f"Using Neural2 voice: {voice_name}, gender parameter ignored")
            else:
                self.voice_gender = self._validate_gender(voice_gender) if voice_gender else None
        else:
            # Default voice configuration
            if voice_type == "NEURAL2":
                self.voice_name = "en-US-Neural2-C"  # Default neutral Neural2 voice
                self.voice_gender = None
            else:
                self.voice_name = None
                self.voice_gender = self._validate_gender(voice_gender) if voice_gender else "FEMALE"
        
        # Initialize client with proper error handling
        self._initialize_client()
        
        # Cache setup
        if self.enable_caching:
            self.cache_dir = Path("./cache/tts_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio format configuration
        self.audio_config = self._create_audio_config()
        self.voice_config = self._create_voice_config()
        
        logger.info(f"Initialized Google Cloud TTS - Voice: {self.voice_name or 'default'}, "
                   f"Format: {self.container_format}, Rate: {self.sample_rate}Hz")
    
    def _validate_gender(self, gender: str) -> Optional[str]:
        """Validate and convert gender string."""
        if not gender:
            return None
        
        gender_upper = gender.upper()
        valid_genders = ["MALE", "FEMALE", "NEUTRAL"]
        
        if gender_upper not in valid_genders:
            logger.warning(f"Invalid gender '{gender}', ignoring")
            return None
        
        return gender_upper
    
    def _initialize_client(self):
        """Initialize the Google Cloud TTS client with proper credentials."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                self.client = texttospeech.TextToSpeechClient(credentials=credentials)
                logger.info(f"Initialized TTS client with credentials from {self.credentials_file}")
            else:
                # Use default credentials (ADC)
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Initialized TTS client with default credentials")
                
        except Exception as e:
            logger.error(f"Error initializing TTS client: {e}")
            raise
    
    def _create_audio_config(self) -> texttospeech.AudioConfig:
        """Create audio configuration optimized for Twilio."""
        # Audio encoding based on format
        if self.container_format == "MULAW":
            audio_encoding = texttospeech.AudioEncoding.MULAW
        elif self.container_format == "LINEAR16":
            audio_encoding = texttospeech.AudioEncoding.LINEAR16
        elif self.container_format == "MP3":
            audio_encoding = texttospeech.AudioEncoding.MP3
        else:
            # Default to MULAW for telephony
            audio_encoding = texttospeech.AudioEncoding.MULAW
            logger.warning(f"Unknown format {self.container_format}, using MULAW")
        
        config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            sample_rate_hertz=self.sample_rate,
        )
        
        # Add telephony effects profile for better phone call quality
        if self.container_format == "MULAW":
            config.effects_profile_id = ["telephony-class-application"]
        
        return config
    
    def _create_voice_config(self) -> texttospeech.VoiceSelectionParams:
        """Create voice configuration with proper handling for different voice types."""
        voice_config = texttospeech.VoiceSelectionParams(
            language_code=self.language_code
        )
        
        # Set voice name if specified (preferred method)
        if self.voice_name:
            voice_config.name = self.voice_name
            # For Neural2 voices, don't set gender as it's included in the name
            if not ("Neural2" in self.voice_name or "Studio" in self.voice_name):
                if self.voice_gender:
                    if self.voice_gender == "MALE":
                        voice_config.ssml_gender = texttospeech.SsmlVoiceGender.MALE
                    elif self.voice_gender == "FEMALE":
                        voice_config.ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
                    elif self.voice_gender == "NEUTRAL":
                        voice_config.ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL
        else:
            # Set gender only if no specific voice name
            if self.voice_gender:
                if self.voice_gender == "MALE":
                    voice_config.ssml_gender = texttospeech.SsmlVoiceGender.MALE
                elif self.voice_gender == "FEMALE":
                    voice_config.ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
                elif self.voice_gender == "NEUTRAL":
                    voice_config.ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL
        
        return voice_config
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include voice and audio config in cache key
        cache_data = {
            "text": text,
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "format": self.container_format,
            "sample_rate": self.sample_rate
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.audio"
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech optimized for Twilio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes (MULAW format for Twilio)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return b""
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(text)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cache_path.read_bytes()
        
        try:
            # Prepare the synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Make the synthesis request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice_config,
                audio_config=self.audio_config
            )
            
            audio_content = response.audio_content
            
            # Cache the result
            if self.enable_caching and audio_content:
                cache_path.write_bytes(audio_content)
                logger.debug(f"Cached audio for text: {text[:50]}...")
            
            logger.debug(f"Synthesized {len(audio_content)} bytes for text: {text[:50]}...")
            return audio_content
            
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            # Log more details about the error
            if "voice" in str(e).lower():
                logger.error(f"Voice configuration: {self.voice_config}")
                logger.error(f"Available voices might need to be checked")
            raise
    
    def get_available_voices(self, language_code: Optional[str] = None) -> list:
        """Get list of available voices for debugging."""
        try:
            request = texttospeech.ListVoicesRequest(
                language_code=language_code or self.language_code
            )
            voices = self.client.list_voices(request=request)
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "gender": voice.ssml_gender.name,
                    "natural_sample_rate": voice.natural_sample_rate_hertz
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS client statistics."""
        stats = {
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "audio_format": self.container_format,
            "sample_rate": self.sample_rate,
            "caching_enabled": self.enable_caching,
            "voice_type": self.voice_type
        }
        
        if self.enable_caching and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.audio"))
            stats["cache_entries"] = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            stats["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats