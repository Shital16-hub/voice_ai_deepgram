"""
Enhanced audio processing utilities for telephony integration.

Handles audio format conversion between Twilio and Voice AI Agent with improved
barge-in detection capabilities.
"""
import audioop
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from scipy import signal

from telephony.config import SAMPLE_RATE_TWILIO, SAMPLE_RATE_AI

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between Twilio and Voice AI formats with improved noise handling
    and enhanced barge-in detection capabilities.
    
    Twilio uses 8kHz mulaw encoding, while our Voice AI uses 16kHz PCM.
    """
    
    def mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert Twilio's mulaw audio to PCM for Voice AI with enhanced noise filtering.
        Modified to handle small chunks more efficiently.
        
        Args:
            mulaw_data: Audio data in mulaw format
            
        Returns:
            Audio data as numpy array (float32)
        """
        try:
            # Check if we have enough data to convert - log at debug level instead of warning
            if len(mulaw_data) < 1000:
                logger.debug(f"Small mulaw data: {len(mulaw_data)} bytes - accumulating")
                # Return empty array for very small chunks instead of processing
                if len(mulaw_data) < 320:  # Less than 20ms at 16kHz
                    return np.array([], dtype=np.float32)
            
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
            
            # Apply enhanced audio filtering optimized for barge-in detection
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(6, 100/(SAMPLE_RATE_AI/2), 'highpass')
            audio_array = signal.filtfilt(b, a, audio_array)
            
            # Apply band-pass filter for telephony freq range (300-3400 Hz)
            b, a = signal.butter(4, [300/(SAMPLE_RATE_AI/2), 3400/(SAMPLE_RATE_AI/2)], 'band')
            audio_array = signal.filtfilt(b, a, audio_array)
            
            # Apply a simple noise gate with lower threshold for better barge-in
            noise_threshold = 0.01  # Lower threshold (was 0.015) to detect quieter speech for barge-in
            audio_array = np.where(np.abs(audio_array) < noise_threshold, 0, audio_array)
            
            # Apply pre-emphasis filter to boost higher frequencies (improves speech detection)
            audio_array = np.append(audio_array[0], audio_array[1:] - 0.97 * audio_array[:-1])
            
            # Normalize for consistent volume but preserve relative energy for better barge-in detection
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array * (0.95 / max_val)
            
            # Check audio levels at debug level instead of always logging
            audio_level = np.mean(np.abs(audio_array)) * 100
            if audio_level > 5.0:  # Only log significant audio
                logger.debug(f"Converted {len(mulaw_data)} bytes to {len(audio_array)} samples. Audio level: {audio_level:.1f}%")
            
            # Apply a gain if audio is very quiet
            if audio_level < 1.0:  # Very quiet audio
                audio_array = audio_array * min(5.0, 5.0/audio_level)
                logger.debug(f"Applied gain to quiet audio. New level: {np.mean(np.abs(audio_array)) * 100:.1f}%")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            # Return an empty array rather than raising an exception
            return np.array([], dtype=np.float32)
    
    def pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """
        Convert PCM audio from Voice AI to mulaw for Twilio with optimizations
        for improved barge-in response times.
        
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
        Normalize audio to [-1, 1] range while preserving dynamics
        for better barge-in detection.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality by reducing noise and improving speech clarity
        with optimizations for better barge-in detection.
        
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
            
            # 2. Apply a milder de-emphasis filter to reduce hissing but preserve speech onset
            b, a = signal.butter(1, 3400/(SAMPLE_RATE_AI/2), 'low')  # Higher cutoff (was 3000Hz)
            de_emphasis = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Apply a simple noise gate with lower threshold for better barge-in detection
            noise_threshold = 0.004  # Lower threshold (was 0.005) for better barge-in detection
            noise_gate = np.where(np.abs(de_emphasis) < noise_threshold, 0, de_emphasis)
            
            # 4. Apply pre-emphasis filter to boost higher frequencies (for better speech detection)
            pre_emphasis = np.append(noise_gate[0], noise_gate[1:] - 0.97 * noise_gate[:-1])
            
            # 5. Normalize audio to have consistent volume
            if np.max(np.abs(pre_emphasis)) > 0:
                normalized = pre_emphasis / np.max(np.abs(pre_emphasis)) * 0.95
            else:
                normalized = pre_emphasis
            
            # 6. Apply a milder compression to preserve dynamics for better barge-in detection
            # Reduced compression ratio from 0.5 to 0.7 (closer to 1:1) and lower threshold
            threshold = 0.15  # Lower threshold (was 0.2)
            ratio = 0.7  # Milder compression (was 0.5)
            
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
            
    def detect_speech_for_barge_in(self, audio_data: np.ndarray, threshold: float = 0.005) -> bool:
        """
        Enhanced speech detection specifically optimized for barge-in detection
        with multi-factor analysis to reduce false positives.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Speech detection threshold (lower than normal)
            
        Returns:
            True if speech is detected for barge-in
        """
        try:
            # Apply a band-pass filter to focus on speech frequencies (300-3400 Hz)
            # This helps eliminate background noise and some echo
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 1. Energy level check with adaptive threshold
            energy = np.mean(np.abs(filtered_audio))
            energy_detected = energy > threshold
            
            # 2. Check for speech patterns - meaningful energy variations
            # Speech typically has variations in energy
            frame_size = min(len(filtered_audio), 320)  # 20ms at 16kHz
            if frame_size > 0:
                frames = [filtered_audio[i:i+frame_size] for i in range(0, len(filtered_audio), frame_size)]
                frame_energies = [np.mean(np.abs(frame)) for frame in frames if len(frame) == frame_size]
                
                if len(frame_energies) >= 3:
                    # Calculate energy variance (speech has more variance than steady noise)
                    energy_std = np.std(frame_energies)
                    energy_mean = np.mean(frame_energies)
                    variation_ratio = energy_std / energy_mean if energy_mean > 0 else 0
                    variation_detected = variation_ratio > 0.2  # Speech typically has high variation
                else:
                    variation_detected = False
            else:
                variation_detected = False
            
            # 3. Peak detection - speech often has stronger peaks
            if len(filtered_audio) > 0:
                peaks = np.max(np.abs(filtered_audio))
                peak_detected = peaks > (threshold * 3)  # Higher threshold for peaks
            else:
                peak_detected = False
                
            # Decision logic: Multiple factors must indicate speech to reduce false positives
            # This helps avoid triggering on echoes or background noise
            is_speech = energy_detected and (variation_detected or peak_detected)
            
            # Log detection results for debugging
            logger.debug(f"Barge-in speech detection: energy={energy:.6f}, threshold={threshold:.6f}, "
                         f"variation={variation_ratio if 'variation_ratio' in locals() else 0:.6f}, "
                         f"peaks={peaks if 'peaks' in locals() else 0:.6f}, "
                         f"detected={is_speech}")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Error in barge-in speech detection: {e}")
            # Return False if there's an error
            return False
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
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
    
    def float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
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
    
    def pcm16_to_float32(self, audio_data: bytes) -> np.ndarray:
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
            
    def convert_to_mulaw(self, audio_data: bytes, sample_rate: int = 8000) -> bytes:
        """
        Convert audio from any format to µ-law format specifically for Twilio.
        Used for ElevenLabs TTS output processing.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate for Twilio
            
        Returns:
            Audio data converted to µ-law format
        """
        try:
            import wave
            import io
            
            # If it's already µ-law, don't convert
            if self.is_mulaw(audio_data):
                return audio_data
                
            # If it's WAV, extract PCM data
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                with io.BytesIO(audio_data) as f:
                    with wave.open(f, 'rb') as wav:
                        pcm_data = wav.readframes(wav.getnframes())
                        src_sample_rate = wav.getframerate()
                        src_channels = wav.getnchannels()
                        src_width = wav.getsampwidth()
                        
                        # Resample if needed
                        if src_sample_rate != sample_rate:
                            pcm_data, _ = audioop.ratecv(
                                pcm_data, src_width, src_channels, 
                                src_sample_rate, sample_rate, None
                            )
                        
                        # Convert to mono if needed
                        if src_channels > 1:
                            pcm_data = audioop.tomono(pcm_data, src_width, 1, 0)
                            
                        # Convert to 16-bit if needed
                        if src_width != 2:  # 2 bytes = 16-bit
                            pcm_data = audioop.lin2lin(pcm_data, src_width, 2)
                            
                        # Convert to µ-law
                        mulaw_data = audioop.lin2ulaw(pcm_data, 2)  # 2 bytes = 16-bit
                        
                        return mulaw_data
            else:
                # For MP3 or other formats from ElevenLabs, convert to WAV first
                # and then to µ-law using an external utility (simulated here)
                
                # This would normally use ffmpeg or another converter
                # For our purposes, treat as 16-bit PCM and convert directly
                return audioop.lin2ulaw(audio_data, 2)
                
        except Exception as e:
            logger.error(f"Error converting to mulaw: {e}")
            return b''
            
    def is_mulaw(self, audio_data: bytes) -> bool:
        """
        Check if audio data is in µ-law format.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            True if audio is in µ-law format
        """
        # Simple heuristic check
        if len(audio_data) < 100:
            return False
            
        # Sample the first 100 bytes and check value distribution
        sample = audio_data[:100]
        
        # µ-law typically uses the full 8-bit range
        value_count = {}
        for b in sample:
            range_key = b // 32  # Group into 8 ranges (0-31, 32-63, etc.)
            value_count[range_key] = value_count.get(range_key, 0) + 1
            
        # If we have values across most ranges, it's likely µ-law
        return len(value_count) >= 6  # At least 6 out of 8 ranges have values


class MulawBufferProcessor:
    """
    Buffer processor for small mulaw audio chunks.
    
    This class accumulates small mulaw chunks to avoid processing many small
    chunks and reduce "Very small mulaw data" warnings.
    """
    
    def __init__(self, min_chunk_size=640):  # 80ms at 8kHz
        """
        Initialize buffer processor.
        
        Args:
            min_chunk_size: Minimum chunk size to process (default 640 bytes = 80ms at 8kHz)
        """
        self.buffer = bytearray()
        self.min_chunk_size = min_chunk_size
        logger.info(f"Initialized MulawBufferProcessor with min_chunk_size={min_chunk_size}")
    
    def process(self, data: bytes) -> Optional[bytes]:
        """
        Process incoming mulaw data by buffering until minimum size is reached.
        
        Args:
            data: New audio data
            
        Returns:
            Processed data of minimum size or None if still buffering
        """
        # Skip empty chunks
        if not data:
            return None
            
        # Log small chunks at debug level instead of warning
        if 0 < len(data) < 320:  # Less than 40ms at 8kHz
            logger.debug(f"Small mulaw data: {len(data)} bytes (accumulating)")
            
        # Add to buffer
        self.buffer.extend(data)
        
        # Only process if we have enough data
        if len(self.buffer) >= self.min_chunk_size:
            # Get the buffered data
            result = bytes(self.buffer)
            
            # Clear buffer
            self.buffer = bytearray()
            
            logger.debug(f"Processed mulaw buffer: {len(result)} bytes")
            return result
        
        # Not enough data yet
        return None


def convert_mulaw_to_float(mulaw_data: bytes) -> np.ndarray:
    """
    Convert mulaw audio to float32 array.
    
    Args:
        mulaw_data: Mulaw audio data
        
    Returns:
        Float32 array in [-1.0, 1.0] range
    """
    # Convert to unsigned 8-bit integers
    mulaw_samples = np.frombuffer(mulaw_data, dtype=np.uint8)
    
    # Convert mulaw to linear PCM (simplified conversion)
    # This uses the μ-law algorithm to convert back to linear
    mulaw_quantized = mulaw_samples.astype(np.float32) / 255.0  # Normalize to [0, 1]
    mulaw_quantized = 2 * mulaw_quantized - 1  # Scale to [-1, 1]
    
    # Apply inverse μ-law transform
    sign = np.sign(mulaw_quantized)
    magnitude = (1.0 / 255.0) * (1.0 + 255.0 * np.abs(mulaw_quantized))
    pcm_samples = sign * magnitude
    
    return pcm_samples


def convert_float_to_mulaw(float_data: np.ndarray) -> bytes:
    """
    Convert float32 array to mulaw audio.
    
    Args:
        float_data: Float32 array in [-1.0, 1.0] range
        
    Returns:
        Mulaw audio data
    """
    # Clip to [-1.0, 1.0] range
    float_data = np.clip(float_data, -1.0, 1.0)
    
    # Apply μ-law transform
    sign = np.sign(float_data)
    magnitude = np.log(1.0 + 255.0 * np.abs(float_data)) / np.log(1.0 + 255.0)
    mulaw_quantized = sign * magnitude
    
    # Convert to [0, 1] range and then to 8-bit
    mulaw_quantized = (mulaw_quantized + 1.0) / 2.0  # Convert to [0, 1]
    mulaw_samples = (mulaw_quantized * 255.0).astype(np.uint8)  # Convert to 8-bit
    
    # Convert to bytes
    return mulaw_samples.tobytes()


def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Detect if audio chunk is silence.
    
    Args:
        audio_data: Audio data as numpy array
        threshold: Energy threshold for silence detection
        
    Returns:
        True if audio is considered silence
    """
    energy = np.mean(np.abs(audio_data))
    return energy < threshold