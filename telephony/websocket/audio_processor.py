"""
Audio processing for WebSocket connections.
"""
import base64
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from scipy import signal

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor
from telephony.config import CHUNK_SIZE, AUDIO_BUFFER_SIZE, MAX_BUFFER_SIZE
from .base import RealTimeAudioBuffer

logger = logging.getLogger(__name__)

class AudioHandler:
    """Handles audio processing for WebSocket connections."""
    
    def __init__(self):
        """Initialize the audio handler."""
        self.audio_processor = AudioProcessor()
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)
        self.rt_audio_buffer = RealTimeAudioBuffer(max_size=48000)
        
        # Audio processing state
        self.input_buffer = bytearray()
        self.max_buffer_size = MAX_BUFFER_SIZE  # Use config value
        
        # Noise level tracking
        self.ambient_noise_level = 0.008
        self.noise_samples = []
        self.max_noise_samples = 20
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """Update ambient noise level based on audio energy."""
        energy = np.mean(np.abs(audio_data))
        
        # If audio is silence, use it to update noise floor
        if energy < 0.02:
            self.noise_samples.append(energy)
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level
            if self.noise_samples:
                self.ambient_noise_level = max(
                    0.005,  # Minimum threshold
                    np.percentile(self.noise_samples, 95) * 2.0
                )
                logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data to reduce noise."""
        try:
            # Apply high-pass filter
            b, a = signal.butter(4, 80/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Apply noise gate
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
    
    def process_media_payload(self, payload: str) -> Optional[bytes]:
        """Process incoming media payload."""
        if not payload:
            return None
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Process with MulawBufferProcessor
            processed_data = self.mulaw_processor.process(audio_data)
            
            # Skip if still buffering
            if processed_data is None:
                return None
            
            # Add to input buffer
            self.input_buffer.extend(processed_data)
            
            # Limit buffer size
            if len(self.input_buffer) > self.max_buffer_size:
                excess = len(self.input_buffer) - self.max_buffer_size
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            return processed_data
        except Exception as e:
            logger.error(f"Error processing media payload: {e}")
            return None
    
    def should_process_audio(self, audio_interval: float = 0.5) -> bool:
        """Determine if audio buffer should be processed."""
        # Check if buffer is large enough (reduced threshold for faster response)
        if len(self.input_buffer) < 16000:  # 1 second at 16kHz
            return False
        
        # Add any additional logic here (e.g., silence detection)
        return True
    
    def get_audio_for_processing(self) -> tuple[np.ndarray, int]:
        """Get audio data for speech recognition processing."""
        if not self.input_buffer:
            return np.array([]), 0
        
        # Convert mulaw to PCM
        mulaw_bytes = bytes(self.input_buffer)
        pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
        
        # Preprocess audio
        pcm_audio = self._preprocess_audio(pcm_audio)
        
        # Update noise level tracking
        self._update_ambient_noise_level(pcm_audio)
        
        # Convert to bytes for speech recognition
        audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
        
        return pcm_audio, len(audio_bytes)
    
    def clear_buffer(self):
        """Clear the audio input buffer."""
        self.input_buffer.clear()
    
    def reduce_buffer(self, factor: float = 0.5):
        """Reduce buffer size by a factor."""
        new_size = int(len(self.input_buffer) * factor)
        self.input_buffer = self.input_buffer[len(self.input_buffer) - new_size:]
    
    def split_audio_into_chunks(self, audio_data: bytes, chunk_size: int = 800) -> List[bytes]:
        """Split audio into smaller chunks for streaming."""
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        return chunks