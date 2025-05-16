"""
Google Cloud Text-to-Speech client enhanced with WebRTC integration and echo prevention.
Provides audio fingerprinting for echo cancellation system integration.
"""
import logging
import hashlib
import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from google.cloud import texttospeech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleCloudTTS:
    """Enhanced Google Cloud Text-to-Speech client with WebRTC integration and echo prevention."""
    
    def __init__(
        self,
        credentials_file: Optional[str] = None,
        voice_name: Optional[str] = None,
        voice_gender: Optional[str] = None,
        language_code: str = "en-US",
        container_format: str = "mulaw",
        sample_rate: int = 8000,
        enable_caching: bool = True,
        voice_type: str = "NEURAL2",
        echo_suppression_callback: Optional[callable] = None
    ):
        """
        Initialize enhanced Google Cloud TTS client.
        
        Args:
            credentials_file: Path to credentials JSON file
            voice_name: Voice name (e.g., "en-US-Neural2-C")
            voice_gender: Voice gender (deprecated for Neural2 voices)
            language_code: Language code (e.g., "en-US")
            container_format: Audio format ("mulaw" for Twilio)
            sample_rate: Sample rate (8000 for Twilio)
            enable_caching: Whether to cache synthesized audio
            voice_type: Voice type ("NEURAL2", "STANDARD", "WAVENET")
            echo_suppression_callback: Callback to register TTS output for echo suppression
        """
        self.credentials_file = credentials_file
        self.language_code = language_code
        self.container_format = container_format.upper()
        self.sample_rate = sample_rate
        self.enable_caching = enable_caching
        self.voice_type = voice_type
        self.echo_suppression_callback = echo_suppression_callback
        
        # Enhanced voice configuration
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
        
        # Initialize client with enhanced error handling
        self._initialize_client()
        
        # Cache setup
        if self.enable_caching:
            self.cache_dir = Path("./cache/tts_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio format configuration with WebRTC optimization
        self.audio_config = self._create_audio_config()
        self.voice_config = self._create_voice_config()
        
        # Enhanced metrics tracking
        self.synthesis_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_synthesis_time = 0.0
        self.last_synthesis_time = None
        
        logger.info(f"Initialized enhanced Google Cloud TTS - Voice: {self.voice_name or 'default'}, "
                   f"Format: {self.container_format}, Rate: {self.sample_rate}Hz, "
                   f"Echo suppression: {'enabled' if echo_suppression_callback else 'disabled'}")
    
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
        """Initialize the Google Cloud TTS client with enhanced credentials handling."""
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
        """Create audio configuration optimized for telephony and WebRTC."""
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
        
        # Enhanced effects for telephony and WebRTC optimization
        if self.container_format == "MULAW":
            # Add multiple effects for better telephony quality
            config.effects_profile_id = [
                "telephony-class-application",
                "wearable-class-device"  # Additional enhancement for clarity
            ]
        
        return config
    
    def _create_voice_config(self) -> texttospeech.VoiceSelectionParams:
        """Create voice configuration with enhanced handling for different voice types."""
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
        """Generate cache key for text with enhanced parameters."""
        # Include voice and audio config in cache key
        cache_data = {
            "text": text,
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "format": self.container_format,
            "sample_rate": self.sample_rate,
            "voice_type": self.voice_type
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.audio"
    
    def _calculate_audio_fingerprint(self, audio_data: bytes) -> Dict[str, Any]:
        """Calculate audio fingerprint for echo suppression."""
        try:
            # Calculate energy fingerprint
            if self.container_format == "MULAW":
                # For MULAW, calculate simple energy
                energy = sum(abs(b) for b in audio_data[:160])  # First 20ms
            else:
                # For PCM, calculate RMS energy
                if len(audio_data) >= 320:  # 20ms at 8kHz 16-bit
                    audio_array = np.frombuffer(audio_data[:320], dtype=np.int16)
                    energy = float(np.sqrt(np.mean(audio_array**2)))
                else:
                    energy = 0.0
            
            # Calculate spectral centroid (simplified)
            spectral_centroid = self._calculate_spectral_centroid(audio_data)
            
            return {
                'energy': energy,
                'spectral_centroid': spectral_centroid,
                'duration': len(audio_data) / (self.sample_rate if self.container_format != "MULAW" else self.sample_rate),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.debug(f"Error calculating audio fingerprint: {e}")
            return {'energy': 0.0, 'spectral_centroid': 0.0, 'duration': 0.0, 'timestamp': time.time()}
    
    def _calculate_spectral_centroid(self, audio_data: bytes) -> float:
        """Calculate spectral centroid for audio fingerprinting."""
        try:
            if self.container_format == "MULAW":
                # For MULAW, convert to linear PCM first
                import audioop
                pcm_data = audioop.ulaw2lin(audio_data[:160], 2)
                audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            else:
                audio_array = np.frombuffer(audio_data[:320], dtype=np.int16).astype(np.float32)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Calculate FFT
            fft = np.fft.fft(audio_array)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Calculate spectral centroid
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                return float(centroid)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating spectral centroid: {e}")
            return 0.0
    
    async def synthesize(self, text: str) -> bytes:
        """
        Enhanced synthesize with WebRTC integration and echo suppression support.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes (MULAW format for Twilio)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return b""
        
        synthesis_start = time.time()
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(text)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                logger.debug(f"Cache hit for text: {text[:50]}...")
                audio_data = cache_path.read_bytes()
                self.cache_hits += 1
                
                # Register with echo suppression system
                if self.echo_suppression_callback:
                    fingerprint = self._calculate_audio_fingerprint(audio_data)
                    self.echo_suppression_callback(text, audio_data, fingerprint)
                
                return audio_data
            else:
                self.cache_misses += 1
        
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
            
            # Update metrics
            self.synthesis_count += 1
            synthesis_time = time.time() - synthesis_start
            self.total_synthesis_time += synthesis_time
            self.last_synthesis_time = time.time()
            
            # Cache the result
            if self.enable_caching and audio_content:
                cache_path.write_bytes(audio_content)
                logger.debug(f"Cached audio for text: {text[:50]}...")
            
            # Register with echo suppression system
            if self.echo_suppression_callback:
                fingerprint = self._calculate_audio_fingerprint(audio_content)
                self.echo_suppression_callback(text, audio_content, fingerprint)
            
            logger.debug(f"Synthesized {len(audio_content)} bytes for text: {text[:50]}... "
                        f"(time: {synthesis_time:.3f}s)")
            return audio_content
            
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            # Log more details about the error
            if "voice" in str(e).lower():
                logger.error(f"Voice configuration: {self.voice_config}")
                logger.error(f"Available voices might need to be checked")
            raise
    
    async def synthesize_with_optimization(self, text: str, optimization_level: int = 1) -> bytes:
        """
        Synthesize with WebRTC optimization for real-time communication.
        
        Args:
            text: Text to synthesize
            optimization_level: 1-4, higher means more optimized for real-time
            
        Returns:
            Optimized audio data
        """
        # Standard synthesis
        audio_data = await self.synthesize(text)
        
        if optimization_level <= 1:
            return audio_data
        
        # Apply WebRTC-style optimizations
        try:
            if optimization_level >= 2:
                # Apply noise reduction (simplified)
                audio_data = self._apply_noise_reduction(audio_data)
            
            if optimization_level >= 3:
                # Apply dynamic range compression
                audio_data = self._apply_compression(audio_data)
            
            if optimization_level >= 4:
                # Apply additional telephony enhancements
                audio_data = self._apply_telephony_enhancement(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Error applying optimizations: {e}, returning standard audio")
            return audio_data
    
    def _apply_noise_reduction(self, audio_data: bytes) -> bytes:
        """Apply basic noise reduction to audio."""
        try:
            if self.container_format == "MULAW":
                # For MULAW, apply simple thresholding
                return bytes(b if abs(b) > 2 else 0 for b in audio_data)
            else:
                # For PCM, apply more sophisticated processing
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # Apply simple gate filter
                threshold = np.std(audio_array) * 0.1
                audio_array[np.abs(audio_array) < threshold] = 0
                return audio_array.tobytes()
        except Exception as e:
            logger.debug(f"Error in noise reduction: {e}")
            return audio_data
    
    def _apply_compression(self, audio_data: bytes) -> bytes:
        """Apply dynamic range compression to audio."""
        try:
            if self.container_format != "MULAW":
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                # Simple compressor
                threshold = 0.7
                ratio = 4.0
                
                # Convert to [0, 1] range
                normalized = audio_array / 32768.0
                
                # Apply compression
                compressed = np.where(
                    np.abs(normalized) > threshold,
                    np.sign(normalized) * (threshold + (np.abs(normalized) - threshold) / ratio),
                    normalized
                )
                
                # Convert back to int16
                compressed_int = (compressed * 32768.0).astype(np.int16)
                return compressed_int.tobytes()
            
            return audio_data
        except Exception as e:
            logger.debug(f"Error in compression: {e}")
            return audio_data
    
    def _apply_telephony_enhancement(self, audio_data: bytes) -> bytes:
        """Apply telephony-specific enhancements."""
        try:
            if self.container_format != "MULAW":
                # Apply high-pass filter to remove low frequencies
                from scipy import signal
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
                # High-pass filter at 300Hz
                sos = signal.butter(2, 300, 'hp', fs=self.sample_rate, output='sos')
                filtered = signal.sosfilt(sos, audio_array)
                
                # Convert back to int16
                enhanced = np.clip(filtered, -32768, 32767).astype(np.int16)
                return enhanced.tobytes()
            
            return audio_data
        except Exception as e:
            logger.debug(f"Error in telephony enhancement: {e}")
            return audio_data
    
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
        """Get enhanced TTS client statistics."""
        stats = {
            "voice_name": self.voice_name,
            "voice_gender": self.voice_gender,
            "language_code": self.language_code,
            "audio_format": self.container_format,
            "sample_rate": self.sample_rate,
            "caching_enabled": self.enable_caching,
            "voice_type": self.voice_type,
            "echo_suppression_enabled": self.echo_suppression_callback is not None,
            # Enhanced metrics
            "synthesis_count": self.synthesis_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100, 2),
            "avg_synthesis_time": round(self.total_synthesis_time / max(self.synthesis_count, 1), 3),
            "last_synthesis_time": self.last_synthesis_time
        }
        
        if self.enable_caching and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.audio"))
            stats["cache_entries"] = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            stats["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats