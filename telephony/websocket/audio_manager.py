"""
Enhanced audio manager with comprehensive debugging and audio analysis.
"""
import base64
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import time

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor

logger = logging.getLogger(__name__)

class AudioManager:
    """Audio manager with enhanced debugging and monitoring."""
    
    def __init__(self):
        """Initialize audio manager with debugging capabilities."""
        logger.info("Initializing AudioManager")
        
        self.audio_processor = AudioProcessor()
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=800)
        self.input_buffer = bytearray()
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Optimized parameters
        self.max_buffer_size = 12800  # 1.6 seconds
        self.min_processing_size = 1600  # 200ms
        self.pause_after_response = 0.1
        
        # Debugging counters and stats
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        self.last_audio_time = 0
        self.audio_gap_warnings = 0
        
        logger.info("AudioManager initialized successfully")
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media with comprehensive debugging."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.warning(f"Media event #{self.media_events_received} has no payload")
            return None
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            audio_size = len(audio_data)
            self.total_audio_bytes += audio_size
            
            # Check for audio gaps
            current_time = time.time()
            if self.last_audio_time > 0:
                gap = current_time - self.last_audio_time
                if gap > 0.1:  # More than 100ms gap
                    self.audio_gap_warnings += 1
                    logger.warning(f"Audio gap detected: {gap*1000:.1f}ms (warning #{self.audio_gap_warnings})")
            self.last_audio_time = current_time
            
            logger.debug(f"Media event #{self.media_events_received}: {audio_size} bytes "
                        f"(total: {self.total_audio_bytes} bytes)")
            
            # Skip if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Process with buffer
            processed_data = self.mulaw_processor.process(audio_data)
            
            if processed_data is None:
                logger.debug("Buffer not ready yet")
                return None
            
            # Add to input buffer
            old_buffer_size = len(self.input_buffer)
            self.input_buffer.extend(processed_data)
            
            logger.debug(f"Buffer size: {old_buffer_size} -> {len(self.input_buffer)} bytes")
            
            # Limit buffer size
            if len(self.input_buffer) > self.max_buffer_size:
                excess = len(self.input_buffer) - self.max_buffer_size
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed buffer by {excess} bytes")
            
            # Check if ready for processing
            time_since_response = self._time_since_last_response()
            
            if (len(self.input_buffer) >= self.min_processing_size and 
                time_since_response >= self.pause_after_response):
                
                # Return the raw mulaw data
                result = bytes(self.input_buffer)
                self.clear_buffer()
                self.audio_chunks_sent += 1
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: {len(result)} bytes "
                           f"(gap since response: {time_since_response:.2f}s)")
                
                # Analyze audio content
                self._analyze_audio(result)
                
                return result
            else:
                logger.debug(f"Not ready: buffer={len(self.input_buffer)}/{self.min_processing_size}, "
                            f"pause={time_since_response:.3f}/{self.pause_after_response}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def _analyze_audio(self, audio_data: bytes) -> None:
        """Analyze audio data for debugging purposes."""
        if len(audio_data) < 100:
            return
        
        # Convert to numpy for analysis
        audio_array = np.frombuffer(audio_data, dtype=np.uint8)
        
        # Calculate basic statistics
        mean_val = np.mean(audio_array)
        std_val = np.std(audio_array)
        min_val = np.min(audio_array)
        max_val = np.max(audio_array)
        
        # Check for silence (all values near 127 for mulaw)
        is_silence = np.abs(mean_val - 127) < 2 and std_val < 5
        
        # Check for clipping
        clipped_samples = np.sum((audio_array <= 2) | (audio_array >= 253))
        clipping_percentage = (clipped_samples / len(audio_array)) * 100
        
        logger.debug(f"Audio analysis: mean={mean_val:.1f}, std={std_val:.1f}, "
                    f"range=[{min_val}, {max_val}], silence={is_silence}, "
                    f"clipping={clipping_percentage:.1f}%")
        
        if is_silence:
            logger.debug("Detected silence in audio chunk")
        if clipping_percentage > 10:
            logger.warning(f"High clipping detected: {clipping_percentage:.1f}%")
    
    def _time_since_last_response(self) -> float:
        """Get time since last response."""
        return time.time() - self.last_response_time
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with logging."""
        old_state = self.is_speaking
        self.is_speaking = speaking
        
        if old_state != speaking:
            logger.info(f"Speaking state changed: {old_state} -> {speaking}")
            if speaking:
                # Clear buffer when starting to speak
                buffer_size = len(self.input_buffer)
                self.clear_buffer()
                logger.info(f"Cleared {buffer_size} bytes from buffer (started speaking)")
    
    def clear_buffer(self) -> None:
        """Clear buffer with logging."""
        if len(self.input_buffer) > 0:
            logger.debug(f"Clearing audio buffer: {len(self.input_buffer)} bytes")
        self.input_buffer.clear()
    
    def update_response_time(self) -> None:
        """Update response time with logging."""
        old_time = self.last_response_time
        self.last_response_time = time.time()
        logger.debug(f"Updated response time: gap was {self.last_response_time - old_time:.3f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive audio statistics."""
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "current_buffer_size": len(self.input_buffer),
            "is_speaking": self.is_speaking,
            "time_since_response": self._time_since_last_response(),
            "audio_gap_warnings": self.audio_gap_warnings,
            "success_rate": (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        }