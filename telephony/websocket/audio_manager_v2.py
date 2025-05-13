# telephony/websocket/audio_manager_v2.py

"""
Improved audio manager with better streaming and buffering for continuous speech.
"""
import base64
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import time
import audioop

from telephony.audio_processor import AudioProcessor, MulawBufferProcessor

logger = logging.getLogger(__name__)

class AudioManager:
    """
    Improved audio manager for better continuous speech recognition.
    Handles streaming audio with proper buffering and timing.
    """
    
    def __init__(self):
        """Initialize audio manager with improved streaming settings."""
        logger.info("Initializing AudioManager v2")
        
        self.audio_processor = AudioProcessor()
        # Reduce buffer size for more responsive processing
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=3200)  # 400ms at 8kHz
        
        # State management
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Improved buffering strategy
        self.min_processing_size = 3200   # 400ms - more responsive
        self.max_buffer_size = 16000      # 2 seconds max
        self.silence_threshold = 200.0     # Energy threshold for silence
        
        # Continuous streaming management
        self.input_buffer = bytearray()
        self.last_processing_time = 0
        self.processing_interval = 0.1    # Process every 100ms
        
        # Quality and performance tracking
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        self.silence_frames = 0
        self.speech_frames = 0
        
        # Enhanced silence detection
        self.energy_window = []
        self.energy_window_size = 10
        self.adaptive_threshold = True
        self.current_threshold = self.silence_threshold
        
        logger.info("AudioManager v2 initialized with enhanced streaming")
    
    def _analyze_audio_energy(self, audio_data: bytes) -> Dict[str, float]:
        """
        Analyze audio energy for better speech detection.
        
        Args:
            audio_data: Raw mulaw audio data
            
        Returns:
            Energy analysis metrics
        """
        if len(audio_data) < 160:
            return {"energy": 0.0, "is_speech": False}
        
        try:
            # Convert mulaw to linear for analysis
            linear_audio = audioop.ulaw2lin(audio_data, 2)
            audio_array = np.frombuffer(linear_audio, dtype=np.int16)
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            
            # Add to energy window for adaptive threshold
            self.energy_window.append(energy)
            if len(self.energy_window) > self.energy_window_size:
                self.energy_window.pop(0)
            
            # Update adaptive threshold
            if self.adaptive_threshold and len(self.energy_window) >= 5:
                avg_energy = np.mean(self.energy_window)
                # Set threshold to 3x average energy, with bounds
                self.current_threshold = max(100.0, min(500.0, avg_energy * 3.0))
            
            # Determine if this contains speech
            is_speech = energy > self.current_threshold
            
            return {
                "energy": float(energy),
                "is_speech": is_speech,
                "threshold": self.current_threshold,
                "avg_energy": np.mean(self.energy_window) if self.energy_window else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio energy: {e}")
            return {"energy": 0.0, "is_speech": False}
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media with improved streaming."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.debug(f"Media event #{self.media_events_received}: No payload")
            return None
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(payload)
            audio_size = len(audio_data)
            self.total_audio_bytes += audio_size
            
            # Skip if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Analyze audio energy
            energy_analysis = self._analyze_audio_energy(audio_data)
            
            # Track speech/silence
            if energy_analysis.get("is_speech", False):
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.speech_frames > 0:  # Only reset if we had speech before
                    self.speech_frames = max(0, self.speech_frames - 1)
            
            # Add to input buffer
            self.input_buffer.extend(audio_data)
            
            # Maintain buffer size
            if len(self.input_buffer) > self.max_buffer_size:
                # Remove older audio from the beginning
                excess = len(self.input_buffer) - int(self.max_buffer_size * 0.8)
                self.input_buffer = self.input_buffer[excess:]
            
            # Check if ready for processing
            current_time = time.time()
            time_since_last = current_time - self.last_processing_time
            
            should_process = (
                len(self.input_buffer) >= self.min_processing_size and
                time_since_last >= self.processing_interval
            )
            
            # Also process if we detect speech activity
            if self.speech_frames > 3 and not should_process:
                should_process = True
                logger.debug("Processing due to speech activity")
            
            if should_process:
                # Get buffered audio
                result = bytes(self.input_buffer)
                buffer_size = len(result)
                
                # Clear buffer
                self.input_buffer.clear()
                self.last_processing_time = current_time
                self.audio_chunks_sent += 1
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: {buffer_size} bytes")
                logger.debug(f"Energy: {energy_analysis.get('energy', 0):.1f}, "
                           f"Speech frames: {self.speech_frames}, "
                           f"Silence frames: {self.silence_frames}")
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with improved timing."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
            if speaking:
                # When starting to speak, clear buffer to prevent echo
                buffer_size = len(self.input_buffer)
                self.clear_buffer()
                logger.info(f"Cleared {buffer_size} bytes from buffer (started speaking)")
                
                # Reset speech detection
                self.speech_frames = 0
                self.silence_frames = 0
            else:
                # When stopping speech, update response time
                self.update_response_time()
    
    def clear_buffer(self) -> None:
        """Clear audio buffer."""
        self.input_buffer.clear()
        # Also clear the mulaw processor buffer
        if hasattr(self.mulaw_processor, 'clear_buffer'):
            self.mulaw_processor.clear_buffer()
    
    def update_response_time(self) -> None:
        """Update response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive audio statistics."""
        current_time = time.time()
        
        # Calculate rates
        success_rate = (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        avg_chunk_size = self.total_audio_bytes / max(self.media_events_received, 1)
        processing_rate = self.audio_chunks_sent / max(current_time - self.last_response_time, 1)
        
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "avg_chunk_size_bytes": round(avg_chunk_size, 2),
            "current_buffer_size": len(self.input_buffer),
            "is_speaking": self.is_speaking,
            "time_since_response": current_time - self.last_response_time,
            "success_rate": round(success_rate, 2),
            "processing_rate": round(processing_rate, 2),
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "current_threshold": self.current_threshold,
            "avg_energy": np.mean(self.energy_window) if self.energy_window else 0.0
        }