# telephony/websocket/audio_manager.py

"""
Optimized audio manager with minimal processing for better speech recognition.
Following Google Cloud streaming best practices.
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
    Optimized audio manager that preserves audio quality for better recognition.
    Follows Google Cloud 25KB chunk limit recommendation.
    """
    
    def __init__(self):
        """Initialize audio manager with minimal processing."""
        logger.info("Initializing AudioManager (telephony-optimized)")
        
        self.audio_processor = AudioProcessor()
        # Smaller buffer for lower latency
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=3200)  # 400ms at 8kHz
        self.input_buffer = bytearray()
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Optimized buffer settings for lower latency
        self.min_processing_size = 3200      # 400ms at 8kHz (reduced from 6400)
        self.max_buffer_size = 16000         # 2 seconds max (reduced from 40000)
        self.chunk_accumulation_time = 0.4   # Accumulate for 400ms (reduced from 0.8)
        self.last_processing_time = 0
        
        # Quality tracking
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        
        # Speech detection (minimal)
        self.speech_threshold = 100.0
        
        logger.info(f"AudioManager initialized - min_size={self.min_processing_size}, max_size={self.max_buffer_size}")
    
    def _analyze_audio_quality(self, audio_data: bytes) -> Dict[str, float]:
        """
        Basic audio quality analysis without heavy processing.
        """
        if len(audio_data) < 160:  # Need at least 20ms
            return {}
        
        try:
            # Simple energy calculation
            # For mulaw, we can estimate energy without full conversion
            sample_mean = np.mean(list(audio_data))
            sample_std = np.std(list(audio_data))
            
            # Estimate if it's likely speech
            # Mulaw has non-linear encoding, so speech typically has more variation
            likely_speech = sample_std > 10 and 50 < sample_mean < 200
            
            return {
                'mean_value': float(sample_mean),
                'std_value': float(sample_std),
                'likely_speech': likely_speech,
                'size': len(audio_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {}
    
    def _should_process_buffer(self) -> bool:
        """Determine if buffer should be processed - more aggressive for lower latency."""
        current_time = time.time()
        
        # Check basic requirements - reduced minimum size
        if len(self.input_buffer) < 1600:  # 200ms at 8kHz (reduced from previous)
            return False
        
        # Don't process if system is speaking
        if self.is_speaking:
            return False
        
        # More aggressive time-based processing
        time_since_last = current_time - self.last_processing_time
        if time_since_last < 0.2:  # Reduced from 0.8s to 0.2s
            return False
        
        # Analyze recent audio for speech activity
        recent_audio = bytes(self.input_buffer[-1600:])  # Last 200ms
        metrics = self._analyze_audio_quality(recent_audio)
        
        # Process if likely speech or buffer is getting full
        if metrics.get('likely_speech', False):
            logger.debug("Processing due to speech detection")
            return True
        
        if len(self.input_buffer) >= 8000:  # 1 second of audio
            logger.debug("Processing due to buffer size")
            return True
        
        # Force processing after shorter time for better responsiveness
        if time_since_last >= 1.0 and len(self.input_buffer) >= 1600:
            logger.debug("Processing due to timeout")
            return True
        
        return False
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process incoming Twilio media following Google Cloud streaming best practices."""
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
            
            # Skip if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Process with buffer - use smaller chunks for better responsiveness
            processed_data = self.mulaw_processor.process(audio_data)
            
            if processed_data is None:
                return None
            
            # Add to input buffer
            self.input_buffer.extend(processed_data)
            
            # Check if ready for processing - use smaller buffer size
            if self._should_process_buffer():
                # Respect Google Cloud's 25KB limit per request
                MAX_CHUNK_SIZE = 20480  # 20KB to stay safe
                
                # Get chunk size, respecting the limit
                chunk_size = min(len(self.input_buffer), MAX_CHUNK_SIZE)
                result = bytes(self.input_buffer[:chunk_size])
                
                # Remove processed data from buffer
                self.input_buffer = self.input_buffer[chunk_size:]
                self.last_processing_time = time.time()
                self.audio_chunks_sent += 1
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: {len(result)} bytes (mulaw)")
                
                # Final quality check
                final_metrics = self._analyze_audio_quality(result)
                if final_metrics:
                    logger.info(f"Chunk quality: std={final_metrics.get('std_value', 0):.1f}, "
                               f"likely_speech={final_metrics.get('likely_speech', False)}")
                
                return result
            else:
                logger.debug(f"Buffer not ready: size={len(self.input_buffer)}/{self.min_processing_size}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with buffer management."""
        if self.is_speaking != speaking:
            logger.info(f"Speaking state changed: {self.is_speaking} -> {speaking}")
            self.is_speaking = speaking
            
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
        """Update response time."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive audio statistics."""
        current_time = time.time()
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
            "compression_ratio": round(self.audio_chunks_sent / max(self.media_events_received, 1), 3)
        }