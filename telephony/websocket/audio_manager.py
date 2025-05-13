# telephony/websocket/audio_manager.py

"""
Optimized audio manager with improved buffering and quality preservation.
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
    Optimized audio manager focusing on quality preservation and better buffering.
    """
    
    def __init__(self):
        """Initialize audio manager with improved settings."""
        logger.info("Initializing AudioManager")
        
        self.audio_processor = AudioProcessor()
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=6400)  # Increased from 800 to 6400
        self.input_buffer = bytearray()
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Improved buffer management
        self.min_processing_size = 6400   # 800ms at 8kHz (was 1600/200ms)
        self.max_buffer_size = 40000      # 5 seconds (was 12800/1.6s)
        self.chunk_accumulation_time = 0.5  # Accumulate for 500ms before processing
        self.last_processing_time = 0
        
        # Quality tracking
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        self.audio_quality_metrics = {
            'silence_ratio': 0.0,
            'avg_energy': 0.0,
            'clip_count': 0
        }
        
        # Voice activity detection
        self.speech_energy_threshold = 500.0  # Energy threshold for speech
        self.last_speech_time = 0
        self.min_speech_duration = 0.3  # Minimum speech duration to consider
        
        logger.info(f"AudioManager initialized with enhanced buffering")
        logger.info(f"Buffer settings: min={self.min_processing_size}, max={self.max_buffer_size}")
    
    def _analyze_audio_quality(self, audio_data: bytes) -> Dict[str, float]:
        """
        Analyze audio quality metrics.
        
        Args:
            audio_data: Raw mulaw audio data
            
        Returns:
            Quality metrics dictionary
        """
        if len(audio_data) < 160:  # Need at least 20ms
            return {}
        
        try:
            # Convert mulaw to linear for analysis
            linear_audio = audioop.ulaw2lin(audio_data, 2)
            
            # Convert to numpy for analysis
            audio_array = np.frombuffer(linear_audio, dtype=np.int16)
            
            # Calculate metrics
            energy = np.mean(np.square(audio_array.astype(np.float32)))
            rms_energy = np.sqrt(energy)
            
            # Check for clipping (values at or near max)
            max_value = np.max(np.abs(audio_array))
            clip_ratio = np.sum(np.abs(audio_array) >= 32000) / len(audio_array)
            
            # Check for silence (very low values)
            silence_threshold = 500  # Linear PCM threshold
            silence_ratio = np.sum(np.abs(audio_array) < silence_threshold) / len(audio_array)
            
            # Dynamic range
            dynamic_range = max_value - np.min(np.abs(audio_array))
            
            metrics = {
                'rms_energy': float(rms_energy),
                'max_value': float(max_value),
                'clip_ratio': float(clip_ratio),
                'silence_ratio': float(silence_ratio),
                'dynamic_range': float(dynamic_range),
                'likely_speech': rms_energy > self.speech_energy_threshold and clip_ratio < 0.1
            }
            
            # Log quality issues
            if clip_ratio > 0.1:
                logger.warning(f"High clipping detected: {clip_ratio:.2%}")
            if silence_ratio > 0.9:
                logger.debug(f"Mostly silence detected: {silence_ratio:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {}
    
    def _should_process_buffer(self) -> bool:
        """
        Determine if buffer should be processed based on multiple criteria.
        
        Returns:
            True if buffer should be processed
        """
        current_time = time.time()
        
        # Check basic size requirement
        if len(self.input_buffer) < self.min_processing_size:
            return False
        
        # Don't process if system is speaking
        if self.is_speaking:
            return False
        
        # Time-based processing (avoid too frequent processing)
        time_since_last_processing = current_time - self.last_processing_time
        if time_since_last_processing < self.chunk_accumulation_time:
            return False
        
        # Analyze recent audio for speech activity
        recent_audio = bytes(self.input_buffer[-6400:])  # Last 800ms
        metrics = self._analyze_audio_quality(recent_audio)
        
        # Process if we detect likely speech or buffer is getting full
        if metrics.get('likely_speech', False):
            logger.debug("Processing due to speech detection")
            return True
        
        if len(self.input_buffer) >= self.max_buffer_size * 0.8:
            logger.debug("Processing due to buffer size")
            return True
        
        # Force processing after reasonable time with content
        if time_since_last_processing >= 2.0 and len(self.input_buffer) >= self.min_processing_size:
            logger.debug("Processing due to timeout with content")
            return True
        
        return False
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media with improved quality handling."""
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
            
            logger.debug(f"Media event #{self.media_events_received}: {audio_size} bytes")
            
            # Validate audio data
            if audio_size < 160:  # Less than 20ms at 8kHz
                logger.debug("Skipping tiny audio chunk")
                return None
            
            # Skip if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Process with mulaw buffer
            processed_data = self.mulaw_processor.process(audio_data)
            
            if processed_data is None:
                logger.debug("Buffer not ready yet")
                return None
            
            # Add to input buffer (preserve raw mulaw)
            old_buffer_size = len(self.input_buffer)
            self.input_buffer.extend(processed_data)
            
            logger.debug(f"Buffer size: {old_buffer_size} -> {len(self.input_buffer)} bytes")
            
            # Analyze audio quality on raw data
            metrics = self._analyze_audio_quality(audio_data)
            if metrics:
                logger.debug(f"Audio quality: RMS={metrics.get('rms_energy', 0):.1f}, "
                           f"silence={metrics.get('silence_ratio', 0):.2%}, "
                           f"speech={metrics.get('likely_speech', False)}")
            
            # Maintain buffer size
            if len(self.input_buffer) > self.max_buffer_size:
                # Remove older audio from the beginning
                excess = len(self.input_buffer) - int(self.max_buffer_size * 0.9)
                self.input_buffer = self.input_buffer[excess:]
                logger.debug(f"Trimmed buffer by {excess} bytes")
            
            # Check if ready for processing
            if self._should_process_buffer():
                # Return the accumulated audio
                result = bytes(self.input_buffer)
                buffer_size = len(result)
                
                # Clear buffer
                self.clear_buffer()
                self.last_processing_time = time.time()
                self.audio_chunks_sent += 1
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: {buffer_size} bytes")
                
                # Update quality metrics
                final_metrics = self._analyze_audio_quality(result)
                if final_metrics:
                    logger.info(f"Processing buffer quality: RMS={final_metrics.get('rms_energy', 0):.1f}, "
                               f"likely_speech={final_metrics.get('likely_speech', False)}")
                
                return result
            else:
                logger.debug(f"Buffer not ready: size={len(self.input_buffer)}/{self.min_processing_size}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with logging."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
            if speaking:
                # Clear buffer when starting to speak to avoid echo processing
                buffer_size = len(self.input_buffer)
                self.clear_buffer()
                logger.info(f"Cleared {buffer_size} bytes from buffer (started speaking)")
    
    def clear_buffer(self) -> None:
        """Clear buffer with logging."""
        if len(self.input_buffer) > 0:
            logger.debug(f"Clearing audio buffer: {len(self.input_buffer)} bytes")
        self.input_buffer.clear()
    
    def update_response_time(self) -> None:
        """Update response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive audio statistics."""
        current_time = time.time()
        
        # Calculate success rate and other metrics
        success_rate = (self.audio_chunks_sent / max(self.media_events_received, 1)) * 100
        avg_chunk_size = self.total_audio_bytes / max(self.media_events_received, 1)
        
        return {
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "avg_chunk_size_bytes": round(avg_chunk_size, 2),
            "current_buffer_size": len(self.input_buffer),
            "is_speaking": self.is_speaking,
            "time_since_response": current_time - self.last_response_time,
            "success_rate": round(success_rate, 2),
            "processing_rate": round(self.audio_chunks_sent / max((current_time - self.last_response_time), 1), 2),
            "quality_metrics": self.audio_quality_metrics
        }