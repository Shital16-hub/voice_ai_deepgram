"""
Google Cloud Text-to-Speech client for the Voice AI Agent.
Enhanced with latest voice types including Chirp 3 HD and Neural2 voices.
"""
import os
import logging
import asyncio
import hashlib
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Any, Union, List

# Import Google Cloud TTS
from google.cloud import texttospeech

from .config import config
from .exceptions import TTSError

logger = logging.getLogger(__name__)

class GoogleCloudTTS:
    """
    Client for the Google Cloud Text-to-Speech API with support for streaming.
    
    This class handles both batch and streaming TTS operations using Google Cloud's API,
    optimized for low-latency voice AI applications with support for latest voice types.
    """
    
    # Voice type constants
    CHIRP_3_HD = "CHIRP_3_HD"
    NEURAL2 = "NEURAL2"
    STUDIO = "STUDIO"
    WAVENET = "WAVENET"
    STANDARD = "STANDARD"
    
    def __init__(
        self, 
        credentials_file: Optional[str] = None,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: Optional[str] = "en-US",
        sample_rate: Optional[int] = None,
        container_format: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        voice_type: Optional[str] = None
    ):
        """
        Initialize the Google Cloud TTS client.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            voice_name: Voice name (e.g., 'en-US-Neural2-C')
            voice_gender: Voice gender (MALE, FEMALE, NEUTRAL)
            language_code: Language code (defaults to en-US)
            sample_rate: Audio sample rate (defaults to config)
            container_format: Audio format (defaults to config)
            enable_caching: Whether to cache results (defaults to config)
            voice_type: Type of voice (CHIRP_3_HD, NEURAL2, STUDIO, WAVENET, STANDARD)
        """
        self.credentials_file = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not self.credentials_file:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Using default credentials.")
            
        # Set environment variable for credentials if provided
        if self.credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            
        self.voice_name = voice_name
        self.voice_gender = voice_gender or "NEUTRAL"
        self.language_code = language_code
        self.sample_rate = sample_rate or config.sample_rate
        self.container_format = container_format or config.container_format
        self.enable_caching = enable_caching if enable_caching is not None else config.enable_caching
        self.voice_type = voice_type or self.NEURAL2  # Default to Neural2
        
        # Auto-detect voice type from voice name if not specified
        if self.voice_name and not voice_type:
            self.voice_type = self._detect_voice_type(self.voice_name)
        
        # Create cache directory if enabled
        if self.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"TTS caching enabled. Cache directory: {self.cache_dir}")
            
        # Initialize Google Cloud TTS client
        try:
            self.client = texttospeech.TextToSpeechClient()
            logger.info(f"Google Cloud TTS client initialized successfully with voice type: {self.voice_type}")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS client: {e}")
            raise TTSError(f"Failed to initialize Google Cloud TTS client: {e}")
    
    def _detect_voice_type(self, voice_name: str) -> str:
        """Detect voice type from voice name."""
        voice_lower = voice_name.lower()
        if "chirp" in voice_lower:
            return self.CHIRP_3_HD
        elif "neural2" in voice_lower:
            return self.NEURAL2
        elif "studio" in voice_lower:
            return self.STUDIO
        elif "wavenet" in voice_lower:
            return self.WAVENET
        else:
            return self.STANDARD
    
    def _get_audio_config(self) -> texttospeech.AudioConfig:
        """Get the audio configuration for Google Cloud TTS."""
        # Map container format to appropriate audio encoding
        if self.container_format == "mp3":
            encoding = texttospeech.AudioEncoding.MP3
        elif self.container_format == "wav":
            encoding = texttospeech.AudioEncoding.LINEAR16
        elif self.container_format in ["mulaw", "ulaw"]:
            encoding = texttospeech.AudioEncoding.MULAW
        else:
            # Default to MULAW for Twilio compatibility
            encoding = texttospeech.AudioEncoding.MULAW
        
        # Create audio config with telephony optimization
        config_params = {
            "audio_encoding": encoding,
            "sample_rate_hertz": self.sample_rate,
        }
        
        # Add telephony-class effects profile for phone calls
        if self.container_format in ["mulaw", "ulaw"] and self.sample_rate == 8000:
            config_params["effects_profile_id"] = ["telephony-class-application"]
        
        return texttospeech.AudioConfig(**config_params)
    
    def _get_voice(self) -> texttospeech.VoiceSelectionParams:
        """Get the voice selection parameters for Google Cloud TTS."""
        # Map gender string to enum
        gender_map = {
            "MALE": texttospeech.SsmlVoiceGender.MALE,
            "FEMALE": texttospeech.SsmlVoiceGender.FEMALE,
            "NEUTRAL": texttospeech.SsmlVoiceGender.NEUTRAL
        }
        
        gender = gender_map.get(self.voice_gender, texttospeech.SsmlVoiceGender.NEUTRAL)
        
        # Create voice selection
        voice_params = {
            "language_code": self.language_code,
            "ssml_gender": gender
        }
        
        # Add specific voice name if provided
        if self.voice_name:
            voice_params["name"] = self.voice_name
        
        return texttospeech.VoiceSelectionParams(**voice_params)
    
    def _get_cache_path(self, text: str) -> Path:
        """
        Generate a cache file path based on text and parameters.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to the cache file
        """
        # Create a unique hash based on text and params
        params = {
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "sample_rate": self.sample_rate,
            "container_format": self.container_format,
            "voice_type": self.voice_type
        }
        cache_key = hashlib.md5(f"{text}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        # Determine file extension based on format
        ext = "wav"
        if self.container_format == "mp3":
            ext = "mp3"
        elif self.container_format in ["mulaw", "ulaw"]:
            ext = "ulaw"
            
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text into smaller chunks that won't exceed API limits.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Google Cloud TTS has a limit of 5000 characters
        max_chunk_size = 4800  # Leave some margin
        
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        sentences = []
        
        # First try to split by sentences (periods, exclamation marks, question marks)
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            if sentence.strip():
                sentences.append(sentence.strip() + '.')
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > max_chunk_size:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If a single sentence is longer than the limit, split it by words
                if len(sentence) > max_chunk_size:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > max_chunk_size:
                            chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += " " + word
                    
                    if word_chunk.strip():
                        current_chunk = word_chunk.strip()
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Split text of length {len(text)} into {len(chunks)} chunks")
        return chunks
    
    async def synthesize(
        self, 
        text: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech in a single request.
        
        Args:
            text: Text to synthesize
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        if not text:
            logger.warning("Empty text provided to synthesize")
            return b''
            
        # Check if text exceeds Google Cloud TTS limit
        if len(text) > 5000:
            logger.info(f"Text length ({len(text)}) exceeds Google Cloud TTS 5000 character limit. Splitting into chunks.")
            chunks = self._split_long_text(text)
            audio_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
                    chunk_audio = await self._synthesize_chunk(chunk, **kwargs)
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i+1}: {e}")
                    # Continue with next chunk instead of failing completely
            
            # Combine audio chunks - for simplicity, just concatenate
            # In a production implementation, you'd properly combine the audio files
            return b''.join(audio_chunks)
        else:
            # Standard case - text is within limits
            return await self._synthesize_chunk(text, **kwargs)
    
    async def _synthesize_chunk(
        self, 
        text: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize a single chunk of text.
        
        Args:
            text: Text chunk to synthesize
            **kwargs: Additional parameters
            
        Returns:
            Audio data as bytes
        """
        # Check cache first if enabled
        if self.enable_caching:
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                logger.debug(f"Found cached TTS result for: {text[:30]}...")
                return cache_path.read_bytes()
        
        # Create input text
        input_text = texttospeech.SynthesisInput(text=text)
        
        # Get voice and audio config
        voice = self._get_voice()
        audio_config = self._get_audio_config()
        
        # Run the synthesis in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config
                )
            )
            
            # Get audio data
            audio_data = response.audio_content
            
            # Cache result if enabled
            if self.enable_caching and audio_data:
                cache_path = self._get_cache_path(text)
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {str(e)}")
            raise TTSError(f"Error during TTS synthesis: {str(e)}")
    
    async def synthesize_streaming(
        self, 
        text_stream: AsyncGenerator[str, None],
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech synthesis for real-time applications using Google Cloud TTS streaming.
        
        Args:
            text_stream: Async generator producing text chunks
            **kwargs: Additional parameters to pass to the API
            
        Yields:
            Audio data chunks as they are generated
        """
        try:
            # Get voice and audio config
            voice = self._get_voice()
            
            # Create streaming config
            streaming_config = texttospeech.StreamingSynthesizeConfig(
                voice=voice
            )
            
            # Prepare the configuration request
            config_request = texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config
            )
            
            def request_generator():
                """Generate streaming requests."""
                yield config_request
                
                # Convert async generator to sync iterator for the request
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    for text_chunk in loop.run_until_complete(self._collect_text_stream(text_stream)):
                        if text_chunk:
                            yield texttospeech.StreamingSynthesizeRequest(
                                input=texttospeech.StreamingSynthesisInput(text=text_chunk)
                            )
                finally:
                    loop.close()
            
            # Start streaming synthesis
            streaming_responses = self.client.streaming_synthesize(request_generator())
            
            # Yield audio chunks as they arrive
            for response in streaming_responses:
                if response.audio_content:
                    yield response.audio_content
                    
        except Exception as e:
            logger.error(f"Error in streaming TTS: {str(e)}")
            raise TTSError(f"Streaming TTS error: {str(e)}")
    
    async def _collect_text_stream(self, text_stream: AsyncGenerator[str, None]) -> List[str]:
        """Collect text chunks from async generator."""
        text_chunks = []
        async for chunk in text_stream:
            if chunk:
                text_chunks.append(chunk)
        return text_chunks
    
    async def synthesize_with_ssml(
        self, 
        ssml: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize speech using SSML markup for advanced control.
        
        Args:
            ssml: SSML-formatted text
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Audio data as bytes
        """
        # Ensure SSML is properly formatted
        if not ssml.startswith('<speak>'):
            ssml = f"<speak>{ssml}</speak>"
            
        # Check length
        if len(ssml) > 5000:
            logger.warning("SSML text exceeds 5000 character limit. Will be truncated by Google Cloud TTS.")
            
        # Create input text
        input_text = texttospeech.SynthesisInput(ssml=ssml)
        
        # Get voice and audio config
        voice = self._get_voice()
        audio_config = self._get_audio_config()
        
        # Run the synthesis in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config
                )
            )
            
            # Get audio data
            audio_data = response.audio_content
            
            # Cache result if enabled
            if self.enable_caching:
                cache_key = hashlib.md5(f"{ssml}:{json.dumps(kwargs, sort_keys=True)}".encode()).hexdigest()
                cache_path = self.cache_dir / f"{cache_key}.{self.container_format}"
                cache_path.write_bytes(audio_data)
                
            return audio_data
            
        except Exception as e:
            logger.error(f"Error during SSML TTS synthesis: {str(e)}")
            raise TTSError(f"Error during SSML TTS synthesis: {str(e)}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices for the current language.
        
        Returns:
            List of available voice information
        """
        try:
            voices = self.client.list_voices(language_code=self.language_code)
            return [
                {
                    "name": voice.name,
                    "language_codes": voice.language_codes,
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
                }
                for voice in voices.voices
            ]
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    # Alias for compatibility with existing code that calls text_to_speech
    text_to_speech = synthesize