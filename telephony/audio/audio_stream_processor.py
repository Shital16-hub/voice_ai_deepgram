"""
Audio stream processing for telephony integration.
"""
import logging
import numpy as np
from typing import Optional, List
from scipy import signal

from telephony.audio.mulaw_buffer_processor import MulawBufferProcessor
from telephony.audio.real_time_buffer import RealTimeAudioBuffer
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, MAX_BUFFER_SIZE

logger = logging.getLogger(__name__)

class AudioStreamProcessor:
    """Handles audio stream processing for telephony."""
    
    def __init__(self):
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)
        self.rt_audio_buffer = RealTimeAudioBuffer(max_size=48000)
        self.input_buffer = bytearray()
        self.ambient_noise_level = 0.008
        self.noise_samples = []
        self.max_noise_samples = 20
    
    def process_incoming_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Process incoming mulaw audio data."""
        # Process with mulaw buffer processor
        processed_data = self.mulaw_processor.process(audio_data)
        
        if processed_data is None:
            return None
        
        # Add to input buffer
        self.input_buffer.extend(processed_data)
        
        # Limit buffer size
        if len(self.input_buffer) > MAX_BUFFER_SIZE:
            excess = len(self.input_buffer) - MAX_BUFFER_SIZE
            self.input_buffer = self.input_buffer[excess:]
            logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
        
        # Convert to PCM for processing
        pcm_audio = self._mulaw_to_pcm(processed_data)
        self._update_ambient_noise_level(pcm_audio)
        
        return processed_data
    
    def should_process_buffer(self) -> bool:
        """Check if buffer is ready for processing."""
        return len(self.input_buffer) >= AUDIO_BUFFER_SIZE
    
    def get_audio_for_processing(self) -> Optional[np.ndarray]:
        """Get audio data for speech processing."""
        if not self.input_buffer:
            return None
        
        # Convert mulaw to PCM
        mulaw_bytes = bytes(self.input_buffer)
        logger.debug(f"Converting {len(mulaw_bytes)} mulaw bytes to PCM")
        
        pcm_audio = self._mulaw_to_pcm(mulaw_bytes)
        
        if len(pcm_audio) == 0:
            logger.warning("No PCM audio generated from mulaw data")
            return None
        
        # Apply preprocessing
        processed_audio = self._preprocess_audio(pcm_audio)
        
        # Check audio levels
        audio_level = np.mean(np.abs(processed_audio))
        logger.info(f"Processed audio - samples: {len(processed_audio)}, level: {audio_level:.4f}")
        
        # Clear buffer after successful processing
        self.input_buffer.clear()
        
        return processed_audio
    
    def prepare_outgoing_audio(self, audio_data: bytes) -> bytes:
        """Prepare audio data for sending to Twilio."""
        return self._convert_to_mulaw_if_needed(audio_data)
    
    def split_audio_for_streaming(self, audio_data: bytes) -> List[bytes]:
        """Split audio into chunks for streaming."""
        chunk_size = 800  # 100ms at 8kHz
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        
        return chunks
    
    def reset(self) -> None:
        """Reset processor state."""
        self.input_buffer.clear()
        self.noise_samples.clear()
        self.ambient_noise_level = 0.008
    
    def _mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """Convert mulaw to PCM audio."""
        try:
            import audioop
            
            if len(mulaw_data) < 320:  # Less than 20ms at 16kHz
                return np.array([], dtype=np.float32)
            
            # Convert mulaw to 16-bit PCM
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Resample from 8kHz to 16kHz
            pcm_data_16k, _ = audioop.ratecv(
                pcm_data, 2, 1, 8000, 16000, None
            )
            
            # Convert to numpy array
            audio_array = np.frombuffer(pcm_data_16k, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Apply gain if audio is very quiet (helps with low audio levels)
            audio_level = np.mean(np.abs(audio_array))
            if 0 < audio_level < 0.01:  # Very quiet audio
                gain = min(5.0, 0.05/audio_level)  # Max 5x gain
                audio_array = audio_array * gain
                logger.debug(f"Applied gain {gain:.1f}x to quiet audio")
            
            return audio_array
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return np.array([], dtype=np.float32)

    def _is_mulaw_format(self, audio_data: bytes) -> bool:
        """Check if audio data is already in mulaw format."""
        # Simple heuristic - mulaw typically has most values in full 8-bit range
        if len(audio_data) < 100:
            return False
        
        # Sample the first 100 bytes
        sample = audio_data[:100]
        
        # Count values in different ranges
        value_count = {}
        for b in sample:
            range_key = b // 32  # Group into 8 ranges
            value_count[range_key] = value_count.get(range_key, 0) + 1
        
        # If we have values across most ranges, it's likely mulaw
        return len(value_count) >= 6
    
    def _convert_to_mulaw_if_needed(self, audio_data: bytes) -> bytes:
        
        """Convert audio to mulaw format if needed."""
        try:
            import audioop
            
            # Check if it's already mulaw (from ElevenLabs)
            if self._is_mulaw_format(audio_data):
                return audio_data
            
            # If it's WAV, convert to mulaw
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                import wave
                import io
                
                with io.BytesIO(audio_data) as f:
                    with wave.open(f, 'rb') as wav:
                        pcm_data = wav.readframes(wav.getnframes())
                        sample_rate = wav.getframerate()
                        channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        
                        # Resample to 8kHz if needed
                        if sample_rate != 8000:
                            pcm_data, _ = audioop.ratecv(
                                pcm_data, sample_width, channels,
                                sample_rate, 8000, None
                            )
                        
                        # Convert to mono if needed
                        if channels > 1:
                            pcm_data = audioop.tomono(pcm_data, sample_width, 1, 0)
                        
                        # Convert to 16-bit if needed
                        if sample_width != 2:
                            pcm_data = audioop.lin2lin(pcm_data, sample_width, 2)
                        
                        # Convert to mulaw
                        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
                        return mulaw_data
            
            # If it's MP3, we need to convert it (requires ffmpeg)
            # For now, assume it's already in the right format
            return audio_data
        
        except Exception as e:
            logger.error(f"Error converting audio to mulaw: {e}")
            return audio_data
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """Update ambient noise level based on audio energy."""
        if len(audio_data) == 0:
            return
        
        energy = np.mean(np.abs(audio_data))
        
        if energy < 0.02:  # Very quiet audio
            self.noise_samples.append(energy)
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            if self.noise_samples:
                self.ambient_noise_level = max(
                    0.005,
                    np.percentile(self.noise_samples, 95) * 2.0
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        
        """Preprocess audio data to reduce noise."""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Simple noise gate
            noise_gate_threshold = max(0.015, self.ambient_noise_level)
            noise_gate = np.where(np.abs(filtered_audio) < noise_gate_threshold, 0, filtered_audio)
            
            # Normalize audio
            if np.max(np.abs(noise_gate)) > 0:
                normalized = noise_gate / np.max(np.abs(noise_gate)) * 0.95
            else:
                normalized = noise_gate
            
            return normalized
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data