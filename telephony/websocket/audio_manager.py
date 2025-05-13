"""
Audio processing and buffering for WebSocket streams.
"""
import base64
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
from scipy import signal

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor

logger = logging.getLogger(__name__)

class AudioManager:
    """Manages audio processing, buffering, and format conversion."""
    
    def __init__(self):
        """Initialize audio manager."""
        self.audio_processor = AudioProcessor()
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)
        self.input_buffer = bytearray()
        self.is_speaking = False
        self.last_response_time = 0
        
        # Audio processing parameters
        self.max_buffer_size = 48000  # 3 seconds at 16kHz
        self.min_processing_size = 32000  # 2 seconds at 16kHz
        self.pause_after_response = 0.3
        
        # Noise tracking
        self.ambient_noise_level = 0.008
        self.noise_samples = []
        self.max_noise_samples = 20
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """
        Process incoming media data.
        
        Args:
            data: Media event data from Twilio
            
        Returns:
            Processed audio data if ready for speech recognition
        """
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning("Received media event with no payload")
            return None
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            
            # Skip processing if system is currently speaking
            if self.is_speaking:
                return None
            
            # Process with buffer processor
            processed_data = self.mulaw_processor.process(audio_data)
            
            if processed_data is None:
                return None
            
            # Add to input buffer
            self.input_buffer.extend(processed_data)
            
            # Limit buffer size
            if len(self.input_buffer) > self.max_buffer_size:
                excess = len(self.input_buffer) - self.max_buffer_size
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
            
            # Convert to PCM and update noise level
            pcm_audio = self.audio_processor.mulaw_to_pcm(processed_data)
            self._update_ambient_noise_level(pcm_audio)
            
            # Check if ready for processing
            if (len(self.input_buffer) >= self.min_processing_size and 
                self._time_since_last_response() >= self.pause_after_response):
                return bytes(self.input_buffer)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}")
            return None
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """Update ambient noise level based on audio energy."""
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
    
    def _time_since_last_response(self) -> float:
        """Get time since last response."""
        import time
        return time.time() - self.last_response_time
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set whether the system is currently speaking."""
        self.is_speaking = speaking
    
    def clear_buffer(self) -> None:
        """Clear the input buffer."""
        self.input_buffer.clear()
    
    def update_response_time(self) -> None:
        """Update the last response time."""
        import time
        self.last_response_time = time.time()