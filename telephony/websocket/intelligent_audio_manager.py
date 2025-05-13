# telephony/websocket/intelligent_audio_manager.py

"""
Intelligent audio manager focused on quality preservation and minimal processing.
"""
import base64
import asyncio
import logging
import time
from typing import Optional, Dict, Any

from telephony.audio_processor import TelephonyAudioProcessor, OptimizedMulawBuffer

logger = logging.getLogger(__name__)

class IntelligentAudioManager:
    """
    Intelligent audio manager that minimizes processing overhead
    while maintaining high audio quality for speech recognition.
    """
    
    def __init__(self):
        """Initialize intelligent audio manager."""
        logger.info("Initializing IntelligentAudioManager")
        
        # Use optimized components
        self.audio_processor = TelephonyAudioProcessor()
        self.mulaw_buffer = OptimizedMulawBuffer(
            min_size=6400,   # 800ms - better for recognition
            max_size=16000   # 2s max to prevent delays
        )
        
        # State management
        self.is_speaking = False
        self.last_response_time = time.time()
        
        # Adaptive processing parameters
        self.adaptive_threshold = True
        self.silence_counter = 0
        self.speech_counter = 0
        
        # Performance tracking
        self.media_events_received = 0
        self.audio_chunks_sent = 0
        self.total_audio_bytes = 0
        self.quality_issues = 0
        
        # Timing optimization
        self.last_processing_time = 0
        self.min_processing_interval = 0.05  # 50ms minimum between processing
        
        logger.info("IntelligentAudioManager initialized with optimized settings")
    
    async def process_media(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Process Twilio media with intelligent handling."""
        self.media_events_received += 1
        
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            logger.debug(f"Media event #{self.media_events_received}: No payload")
            return None
        
        try:
            # Decode audio data (still in mulaw format)
            mulaw_data = base64.b64decode(payload)
            audio_size = len(mulaw_data)
            self.total_audio_bytes += audio_size
            
            # Skip processing if system is speaking
            if self.is_speaking:
                logger.debug("Skipping audio - system is speaking")
                return None
            
            # Check minimum time interval to prevent overprocessing
            current_time = time.time()
            if current_time - self.last_processing_time < self.min_processing_interval:
                # Buffer for later processing
                self.mulaw_buffer.add_chunk(mulaw_data)
                return None
            
            # Quick quality check without conversion
            quality_metrics = self.audio_processor.analyze_audio_quality(mulaw_data)
            
            if not quality_metrics.get("valid", False):
                logger.debug("Skipping invalid audio chunk")
                self.quality_issues += 1
                return None
            
            # Add to intelligent buffer
            processed_audio = self.mulaw_buffer.add_chunk(mulaw_data)
            
            if processed_audio:
                self.last_processing_time = current_time
                self.audio_chunks_sent += 1
                
                # Update adaptive thresholds based on content
                self._update_adaptive_processing(quality_metrics)
                
                logger.info(f"Sending audio chunk #{self.audio_chunks_sent}: "
                           f"{len(processed_audio)} bytes "
                           f"(accumulated from {self.media_events_received} events)")
                
                return processed_audio
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return None
    
    def _update_adaptive_processing(self, quality_metrics: Dict[str, Any]) -> None:
        """Update adaptive processing parameters based on audio quality."""
        if quality_metrics.get("likely_speech", False):
            self.speech_counter += 1
            self.silence_counter = 0
            
            # More aggressive processing during speech
            if self.adaptive_threshold and self.speech_counter > 3:
                self.mulaw_buffer.min_size = max(3200, self.mulaw_buffer.min_size - 400)
        else:
            self.silence_counter += 1
            self.speech_counter = 0
            
            # Less aggressive processing during silence
            if self.adaptive_threshold and self.silence_counter > 5:
                self.mulaw_buffer.min_size = min(9600, self.mulaw_buffer.min_size + 400)
    
    def set_speaking_state(self, speaking: bool) -> None:
        """Set speaking state with intelligent buffer management."""
        was_speaking = self.is_speaking
        self.is_speaking = speaking
        
        if was_speaking != speaking:
            logger.info(f"Speaking state changed: {was_speaking} -> {speaking}")
            
            if speaking:
                # When starting to speak, don't clear buffer immediately
                # This allows us to process any final user speech
                asyncio.create_task(self._delayed_buffer_clear())
            else:
                # When stopping speech, ensure we're ready for input
                self.update_response_time()
    
    async def _delayed_buffer_clear(self):
        """Clear buffer after a short delay to catch final speech."""
        await asyncio.sleep(0.1)  # 100ms delay
        if self.is_speaking:  # Still speaking after delay
            buffer_size = len(self.mulaw_buffer.buffer)
            if buffer_size > 0:
                logger.debug(f"Clearing audio buffer during speech: {buffer_size} bytes")
                self.mulaw_buffer.buffer.clear()
    
    def update_response_time(self) -> None:
        """Update response time for tracking."""
        self.last_response_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        current_time = time.time()
        buffer_stats = self.mulaw_buffer.get_stats()
        
        return {
            # Basic counts
            "media_events_received": self.media_events_received,
            "audio_chunks_sent": self.audio_chunks_sent,
            "total_audio_bytes": self.total_audio_bytes,
            "quality_issues": self.quality_issues,
            
            # Performance metrics
            "success_rate": round((self.audio_chunks_sent / max(self.media_events_received, 1)) * 100, 2),
            "compression_ratio": round(self.audio_chunks_sent / max(self.media_events_received, 1), 3),
            "avg_chunk_size": round(self.total_audio_bytes / max(self.media_events_received, 1), 2),
            
            # Timing
            "time_since_response": current_time - self.last_response_time,
            "is_speaking": self.is_speaking,
            
            # Adaptive parameters
            "current_min_buffer_size": self.mulaw_buffer.min_size,
            "speech_counter": self.speech_counter,
            "silence_counter": self.silence_counter,
            
            # Buffer statistics
            "buffer_stats": buffer_stats
        }
    
    async def force_flush(self) -> Optional[bytes]:
        """Force processing of any remaining audio."""
        return self.mulaw_buffer.flush()