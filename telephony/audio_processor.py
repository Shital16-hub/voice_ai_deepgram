"""
Enhanced audio processing utilities for telephony integration with ElevenLabs TTS.

Handles audio format conversion between Twilio and Voice AI Agent.
"""
import audioop
import numpy as np
import logging
import os
import tempfile
import subprocess
import io
import wave
from typing import Tuple, Dict, Any
from scipy import signal

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats with improved noise handling.
    Optimized for ElevenLabs TTS integration.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    def __init__(self):
        """Initialize audio processor with enhanced voice quality settings."""
        # Noise reduction parameters
        self.noise_gate_threshold = 0.015  # Noise gate threshold
        self.highpass_cutoff = 100         # High-pass filter cutoff (Hz)
        self.lowpass_cutoff = 3400         # Low-pass filter cutoff (Hz)
        self.pre_emphasis = 0.97           # Pre-emphasis factor
        
        # Voice optimization parameters
        self.compression_threshold = 0.3    # Compression threshold
        self.compression_ratio = 0.7        # Compression ratio
        self.target_volume = 0.8            # Target volume (0-1)
        
        # Audio quality checks
        self.min_acceptable_quality = 0.2   # Minimum energy level for acceptable quality
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with enhanced noise filtering.
        Optimized for speech recognition processing.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Check if we have enough data to convert
            if len(mulaw_data) < 1000:
                logger.warning(f"Very small mulaw data: {len(mulaw_data)} bytes")
            
            # Convert mulaw to 16-bit PCM
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Resample from 8kHz to 16kHz
            pcm_data_16k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                SAMPLE_RATE_TWILIO, 
                SAMPLE_RATE_AI, 
                None
            )
            
            # Convert to numpy array (float32)
            audio_array = np.frombuffer(pcm_data_16k, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Apply enhanced audio filtering optimized for speech recognition
            
            # 1. Apply pre-emphasis filter to boost higher frequencies (improves speech detection)
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
            
            # 2. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            nyquist = SAMPLE_RATE_AI / 2
            high_pass = 80 / nyquist
            b_high, a_high = signal.butter(6, high_pass, 'highpass')
            high_passed = signal.filtfilt(b_high, a_high, emphasized_audio)
            
            # 3. Apply band-pass filter for telephony freq range (300-3400 Hz)
            low_band = 300 / nyquist
            high_band = 3400 / nyquist
            b_band, a_band = signal.butter(4, [low_band, high_band], 'band')
            band_passed = signal.filtfilt(b_band, a_band, high_passed)
            
            # 4. Apply a simple noise gate
            noise_threshold = 0.015  # Adjusted threshold
            noise_gate = np.where(np.abs(band_passed) < noise_threshold, 0, band_passed)
            
            # 5. Normalize for consistent volume
            max_val = np.max(np.abs(noise_gate))
            if max_val > 0:
                normalized = noise_gate * (0.95 / max_val)
            else:
                normalized = noise_gate
            
            # Check audio levels
            audio_level = np.mean(np.abs(normalized)) * 100
            logger.debug(f"Converted {len(mulaw_data)} bytes to {len(normalized)} samples. Audio level: {audio_level:.1f}%")
            
            # Apply a gain if audio is very quiet
            if audio_level > 0:
                # Safely apply gain to quiet audio
                normalized = normalized * min(5.0, 5.0/audio_level)
                logger.debug(f"Applied gain to quiet audio. New level: {np.mean(np.abs(normalized)) * 100:.1f}%")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            # Return an empty array rather than raising an exception
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def pcm_to_mulaw(pcm_data: bytes) -> bytes:
        """
        Convert PCM audio to mulaw for Twilio.
        
        Args:
            pcm_data: Audio data in PCM format
            
        Returns:
            Audio data in mulaw format
        """
        try:
            # Check if the data length is a multiple of 2 (for 16-bit samples)
            if len(pcm_data) % 2 != 0:
                # Pad with a zero byte to make it even
                pcm_data = pcm_data + b'\x00'
                logger.debug("Padded audio data to make even length")
            
            # Resample from 16kHz to 8kHz
            pcm_data_8k, _ = audioop.ratecv(
                pcm_data, 2, 1, 
                SAMPLE_RATE_AI, 
                SAMPLE_RATE_TWILIO, 
                None
            )
            
            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data_8k, 2)
            
            logger.debug(f"Converted {len(pcm_data)} bytes of PCM to {len(mulaw_data)} bytes of mulaw")
            
            return mulaw_data
            
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}")
            # Return empty data rather than raising an exception
            return b''
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range with improved loudness.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        if len(audio_data) == 0:
            return audio_data
            
        # Calculate RMS (power) for audio leveling
        rms = np.sqrt(np.mean(np.square(audio_data)))
        target_rms = 0.2  # Target RMS power (good loudness for speech)
        
        # If audio is already loud enough, just do peak normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            if rms >= target_rms:
                # Just prevent clipping
                return audio_data * (0.95 / max_val)
            else:
                # Apply gain to reach target RMS
                gain = min(target_rms / rms, 5.0)  # Limit gain to 5x to prevent noise amplification
                return np.clip(audio_data * gain, -0.95, 0.95)
        
        return audio_data
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality by reducing noise and improving speech clarity.
        Optimized for speech recognition processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        try:
            # Skip processing if audio is empty
            if len(audio_data) == 0:
                return audio_data
                
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            # Telephone lines often have low frequency hum
            b, a = signal.butter(4, self.highpass_cutoff/(SAMPLE_RATE_AI/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply pre-emphasis filter to boost higher frequencies (for better speech detection)
            emphasized = np.append(filtered_audio[0], filtered_audio[1:] - self.pre_emphasis * filtered_audio[:-1])
            
            # 3. Apply a mild de-emphasis filter to reduce hissing sounds in phone calls
            b, a = signal.butter(1, self.lowpass_cutoff/(SAMPLE_RATE_AI/2), 'low')
            de_emphasis = signal.filtfilt(b, a, emphasized)
            
            # 4. Apply a simple noise gate to remove background noise
            noise_gate = np.where(np.abs(de_emphasis) < self.noise_gate_threshold, 0, de_emphasis)
            
            # 5. Apply dynamic range compression for more consistent volume
            # Calculate the signal magnitude
            magnitude = np.abs(noise_gate)
            
            # Determine which samples exceed the threshold
            above_threshold = magnitude > self.compression_threshold
            
            # Prepare the gain array (initialize with ones)
            gain = np.ones_like(magnitude)
            
            # Calculate the gain to apply for samples above threshold
            if np.any(above_threshold):
                compressed_magnitude = self.compression_threshold + \
                                      (magnitude[above_threshold] - self.compression_threshold) * \
                                      self.compression_ratio
                gain[above_threshold] = compressed_magnitude / magnitude[above_threshold]
            
            # Apply the gain
            compressed = noise_gate * gain
            
            # 6. Normalize to target volume
            max_val = np.max(np.abs(compressed))
            if max_val > 0:
                normalized = compressed * (self.target_volume / max_val)
            else:
                normalized = compressed
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            # Return original audio if enhancement fails
            return audio_data
    
    def prepare_audio_for_telephony(
        self,
        audio_data: bytes,
        format: str = 'mp3',
        target_sample_rate: int = 8000,
        target_channels: int = 1
    ) -> bytes:
        """
        Prepare audio data for telephony systems with ElevenLabs support.
        Optimized with telephony-specific filters for clearer speech.
        
        Args:
            audio_data: Audio data as bytes
            format: Source format ('mp3', 'wav', etc.)
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels
            
        Returns:
            Processed audio data as bytes
        """
        try:
            # Verify ffmpeg is installed
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.error("ffmpeg not found! Please install it with: apt-get install -y ffmpeg")
                raise RuntimeError("ffmpeg not installed")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as src_file:
                src_file.write(audio_data)
                src_path = src_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name
            
            # Build ffmpeg command with SPECIFIC telephony optimizations:
            cmd = [
                'ffmpeg',
                '-i', src_path,
                '-acodec', 'pcm_mulaw',  # μ-law encoding for telephony
                '-ar', str(target_sample_rate),  # 8kHz for telephony
                '-ac', str(target_channels),     # Mono for telephony
                
                # Critical audio filter chain for telephony - do not modify!
                '-af', 'highpass=f=200,lowpass=f=3400,compand=0.02|0.05:-60/-60|-40/-10|-20/-8|0/-6:6:0:-90:0.2,volume=3.0',
                # This filter chain:
                # 1. Removes frequencies below 200Hz (eliminates rumble/hum)
                # 2. Removes frequencies above 3400Hz (telephone bandpass)
                # 3. Applies compression for consistent volume
                # 4. Increases overall volume for better audibility
                
                '-y',  # Overwrite output file if it exists
                wav_path
            ]
            
            # Run ffmpeg with detailed error capture
            logger.debug(f"Running ffmpeg with telephony-optimized command: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True)
            
            if process.stderr:
                stderr_output = process.stderr.decode('utf-8', errors='ignore')
                if "Error" in stderr_output or "Invalid" in stderr_output:
                    logger.warning(f"FFmpeg warnings or errors: {stderr_output}")
            
            # Read the output file
            with open(wav_path, 'rb') as f:
                processed_audio = f.read()
            
            # Log successful conversion details
            logger.info(f"Converted {len(audio_data)} bytes of {format} to {len(processed_audio)} bytes of telephony audio")
            
            # Clean up temporary files
            os.unlink(src_path)
            os.unlink(wav_path)
            
            return processed_audio
            
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
            logger.error(f"FFmpeg error: {stderr}")
            raise RuntimeError(f"Error preparing audio for telephony: {stderr}")
        except Exception as e:
            logger.error(f"Error in prepare_audio_for_telephony: {str(e)}")
            raise