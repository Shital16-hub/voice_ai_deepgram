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
            if audio_level < 1.0:  # Very quiet audio
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
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    @staticmethod
    def enhance_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality by reducing noise and improving speech clarity.
        Optimized for speech recognition processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        try:
            # 1. Apply high-pass filter to remove low-frequency noise (below 80Hz)
            # Telephone lines often have low frequency hum
            b, a = signal.butter(4, 80/(SAMPLE_RATE_AI/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Apply pre-emphasis filter to boost higher frequencies (for better speech detection)
            pre_emphasis = 0.97
            emphasized = np.append(filtered_audio[0], filtered_audio[1:] - pre_emphasis * filtered_audio[:-1])
            
            # 3. Apply a mild de-emphasis filter to reduce hissing sounds in phone calls
            b, a = signal.butter(1, 3000/(SAMPLE_RATE_AI/2), 'low')
            de_emphasis = signal.filtfilt(b, a, emphasized)
            
            # 4. Apply a simple noise gate to remove background noise
            noise_threshold = 0.005  # Adjust based on expected noise level
            noise_gate = np.where(np.abs(de_emphasis) < noise_threshold, 0, de_emphasis)
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
            
            # 6. Apply a mild compression to even out volumes
            # Compression ratio 2:1 for values above threshold
            threshold = 0.2
            ratio = 0.5  # 2:1 compression
            
            def compressor(x, threshold, ratio):
                # If below threshold, leave it alone
                # If above threshold, compress it
                mask = np.abs(x) > threshold
                sign = np.sign(x)
                mag = np.abs(x)
                compressed = np.where(
                    mask,
                    threshold + (mag - threshold) * ratio,
                    mag
                )
                return sign * compressed
            
            compressed = compressor(normalized, threshold, ratio)
            
            # Re-normalize after compression
            if np.max(np.abs(compressed)) > 0:
                result = compressed / np.max(np.abs(compressed)) * 0.95
            else:
                result = compressed
                
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            # Return original audio if enhancement fails
            return audio_data
    
    @staticmethod
    def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Enhanced silence detection with frequency analysis.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Silence threshold
            
        Returns:
            True if audio is considered silence
        """
        try:
            # 1. Check energy level
            energy = np.mean(np.abs(audio_data))
            energy_silence = energy < threshold
            
            # Only do more expensive analysis if the energy check isn't conclusive
            if energy < threshold * 2:  # If energy is low but not definitely silent
                # 2. Check zero-crossing rate (white noise has high ZCR)
                zcr = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
                
                # 3. Check spectral flatness (noise typically has flatter spectrum)
                # Approximate with FFT magnitude variance
                fft_data = np.abs(np.fft.rfft(audio_data))
                spectral_flatness = np.std(fft_data) / (np.mean(fft_data) + 1e-10)
                
                # Combined decision - true silence has low energy, low-moderate ZCR, and low spectral flatness
                return energy_silence and zcr < 0.1 and spectral_flatness < 2.0
            
            # If energy is very low or very high, just use that criterion
            return energy_silence
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            # Fall back to simple energy threshold
            return np.mean(np.abs(audio_data)) < threshold
    
    @staticmethod
    def get_audio_info(audio_data: bytes) -> Dict[str, Any]:
        """
        Get information about audio data.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with audio information
        """
        info = {
            "size_bytes": len(audio_data)
        }
        
        # Try to determine if mulaw or pcm
        if len(audio_data) > 0:
            # Check first few bytes for common patterns
            if audio_data[:4] == b'RIFF':
                info["format"] = "wav"
            else:
                # Rough guess based on values
                sample_values = np.frombuffer(audio_data[:100], dtype=np.uint8)
                if np.any(sample_values > 127):
                    info["format"] = "mulaw"
                else:
                    info["format"] = "pcm"
        
        return info
    
    @staticmethod
    def float32_to_pcm16(audio_data: np.ndarray) -> bytes:
        """
        Convert float32 audio to 16-bit PCM bytes.
        
        Args:
            audio_data: Audio data as numpy array (float32)
            
        Returns:
            Audio data as 16-bit PCM bytes
        """
        # Ensure audio is in [-1, 1] range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_int16.tobytes()
    
    @staticmethod
    def pcm16_to_float32(audio_data: bytes) -> np.ndarray:
        """
        Convert 16-bit PCM bytes to float32 audio.
        
        Args:
            audio_data: Audio data as 16-bit PCM bytes
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Convert to numpy array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32
            return audio_int16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Error converting PCM16 to float32: {e}")
            return np.array([], dtype=np.float32)
    
    @staticmethod
    def prepare_audio_for_telephony(
        audio_data: bytes,
        format: str = 'mp3',
        target_sample_rate: int = 8000,
        target_channels: int = 1
    ) -> bytes:
        """
        Prepare audio data for telephony systems with ElevenLabs support.
        
        Converts audio to the format and parameters expected by
        telephony systems like Twilio.
        
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
            
            # Build ffmpeg command for direct conversion to Twilio format
            cmd = [
                'ffmpeg',
                '-i', src_path,
                '-acodec', 'pcm_mulaw',  # μ-law encoding for telephony
                '-ar', str(target_sample_rate),  # 8kHz for telephony
                '-ac', str(target_channels),     # Mono for telephony
                '-y',  # Overwrite output file if it exists
                wav_path
            ]
            
            # Run ffmpeg
            logger.debug(f"Running ffmpeg with command: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the output file
            with open(wav_path, 'rb') as f:
                processed_audio = f.read()
            
            # Clean up temporary files
            os.unlink(src_path)
            os.unlink(wav_path)
            
            logger.info(f"Converted {len(audio_data)} bytes of {format} to {len(processed_audio)} bytes of telephony audio")
            return processed_audio
            
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            logger.error(f"FFmpeg error: {stderr}")
            raise RuntimeError(f"Error preparing audio for telephony: {stderr}")
        except Exception as e:
            logger.error(f"Error in prepare_audio_for_telephony: {str(e)}")
            raise