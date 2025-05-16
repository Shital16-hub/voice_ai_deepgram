"""
Google Cloud Speech-to-Text client for batch processing.
"""
import os
import logging
import asyncio
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, BinaryIO

from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio

from ..config import config
from .models import TranscriptionResult, TranscriptionConfig
from .exceptions import STTError, STTAPIError

logger = logging.getLogger(__name__)

class GoogleCloudSTT:
    """
    Client for the Google Cloud Speech-to-Text API.
    
    This class handles batch STT operations using Google Cloud's API,
    with configurations optimized for telephony voice applications.
    """
    
    def __init__(
        self, 
        credentials_file: Optional[str] = None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            model_name: STT model to use (defaults to config)
            language: Language code for recognition (defaults to config)
            sample_rate: Audio sample rate (defaults to config)
            enable_caching: Whether to cache results (defaults to True)
        """
        self.credentials_file = credentials_file or config.credentials_file
        
        # Set environment variable for credentials if provided
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        
        self.model_name = model_name or config.model_name
        self.language = language or config.language
        self.sample_rate = sample_rate or config.sample_rate
        self.enable_caching = enable_caching
        
        # Create Google Cloud Speech client
        try:
            self.client = speech.SpeechClient()
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Speech client: {e}")
            raise STTConfigError(f"Failed to initialize Google Cloud Speech client: {e}")
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path('./cache/stt_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_config(
        self, 
        config_obj: Optional[TranscriptionConfig] = None, 
        **kwargs
    ) -> RecognitionConfig:
        """
        Get the recognition configuration for an STT request.
        
        Args:
            config_obj: Optional configuration object
            **kwargs: Additional parameters to override defaults
            
        Returns:
            RecognitionConfig object for Google Cloud Speech-to-Text
        """
        # Default encoding for PCM audio
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Create base config
        recognition_config = {
            "encoding": encoding,
            "sample_rate_hertz": self.sample_rate,
            "language_code": self.language,
            "model": self.model_name,
            "enable_automatic_punctuation": config.enable_automatic_punctuation,
            "enable_word_time_offsets": config.enable_word_time_offsets,
            "profanity_filter": config.profanity_filter,
            "use_enhanced": config.use_enhanced_model,
        }
        
        # Add speech contexts (keywords) for better recognition
        if config.speech_contexts:
            recognition_config["speech_contexts"] = [
                {"phrases": config.speech_contexts, "boost": 15.0}
            ]
        
        # Add telephony-specific optimizations
        if config.use_enhanced_telephony:
            recognition_config["use_enhanced"] = True
            recognition_config["model"] = "phone_call"
        
        # Override with any config object settings
        if config_obj:
            config_dict = config_obj.dict(exclude_none=True, exclude_unset=True)
            for key, value in config_dict.items():
                recognition_config[key] = value
        
        # Override with any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                recognition_config[key] = value
        
        # Create RecognitionConfig object
        return RecognitionConfig(**recognition_config)
    
    def _get_cache_path(self, audio_bytes: bytes, config: Dict[str, Any]) -> Path:
        """
        Generate a cache file path based on audio content and configuration.
        
        Args:
            audio_bytes: Audio data
            config: Recognition configuration
            
        Returns:
            Path to cache file
        """
        # Create a unique hash based on audio and config
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        cache_key = f"{audio_hash}_{config_hash}"
        
        return self.cache_dir / f"{cache_key}.json"
    
    async def transcribe(
        self, 
        audio_bytes: bytes,
        config_obj: Optional[TranscriptionConfig] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_bytes: Audio data
            config_obj: Optional configuration object
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        if not audio_bytes:
            logger.warning("Empty audio provided to transcribe")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                words=[],
                alternatives=[],
                audio_duration=0.0
            )
        
        # Get recognition config
        recognition_config = self._get_config(config_obj, **kwargs)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(
                audio_bytes, 
                {k: v for k, v in recognition_config.items() if not callable(v)}
            )
            if cache_path.exists():
                logger.debug(f"Found cached STT result for audio hash: {cache_path.stem}")
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    return TranscriptionResult(**cached_data)
                except Exception as e:
                    logger.warning(f"Error loading cached result: {e}, will transcribe again")
        
        # Create recognition audio
        audio = RecognitionAudio(content=audio_bytes)
        
        try:
            # Use an event loop to run the transcription asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.recognize(
                    config=recognition_config,
                    audio=audio
                )
            )
            
            # Process results
            result = self._process_response(response)
            
            # Cache result if enabled
            if self.enable_caching and result.text:
                with open(cache_path, 'w') as f:
                    json.dump(result.dict(), f)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during STT transcription: {str(e)}")
            raise STTError(f"Error during STT transcription: {str(e)}")
    
    def _process_response(self, response) -> TranscriptionResult:
        """
        Process Google Cloud Speech-to-Text response into structured result.
        
        Args:
            response: Response from Google Cloud Speech-to-Text API
            
        Returns:
            Structured transcription result
        """
        # Default values
        text = ""
        confidence = 0.0
        words = []
        alternatives = []
        audio_duration = 0.0
        
        # Extract results
        if response.results:
            # Get first result (most likely)
            first_result = response.results[0]
            
            if first_result.alternatives:
                # Get best alternative
                best_alternative = first_result.alternatives[0]
                
                # Extract transcript and confidence
                text = best_alternative.transcript
                confidence = best_alternative.confidence
                
                # Extract words with timestamps if available
                if hasattr(best_alternative, 'words') and best_alternative.words:
                    for word_info in best_alternative.words:
                        word = {
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds(),
                            "confidence": confidence  # Word-level confidence not provided
                        }
                        words.append(word)
                
                # Extract other alternatives
                alternatives = [alt.transcript for alt in first_result.alternatives[1:]]
            
            # Try to estimate audio duration from words or results
            if words:
                audio_duration = words[-1]["end_time"]
            elif hasattr(response, 'total_billed_time'):
                audio_duration = response.total_billed_time.total_seconds()
        
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            words=words,
            alternatives=alternatives,
            audio_duration=audio_duration
        )
    
    async def transcribe_file(
        self, 
        file_path: str,
        config_obj: Optional[TranscriptionConfig] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            config_obj: Optional configuration object
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            
            return await self.transcribe(audio_bytes, config_obj, **kwargs)
        except FileNotFoundError:
            raise STTError(f"File not found: {file_path}")
        except Exception as e:
            raise STTError(f"Error transcribing file: {str(e)}")