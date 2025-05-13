"""
Optimized audio stream processing for telephony integration with direct MULAW support.
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
    """Handles audio stream processing for telephony with MULAW optimization."""
    
    def __init__(self):
        # Increased min_chunk_size for efficiency
        self.mulaw_processor = MulawBufferProcessor(min_chunk_size=640)  
        self.rt_audio_buffer = RealTimeAudioBuffer(max_size=48000)
        self.input_buffer = bytearray()
        self.ambient_noise_level = 0.008
        self.noise_samples = []
        self.max_noise_samples = 20
        
        # Track audio statistics
        self.total_bytes_processed = 0
        self.total_chunks_processed = 0
    
    def process_incoming_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Process incoming mulaw audio data with optimizations."""
        if not audio_data:
            return None
        
        # Process with mulaw buffer processor
        processed_data = self.mulaw_processor.process(audio_data)
        
        if processed_data is None:
            return None
        
        # Add to input buffer
        self.input_buffer.extend(processed_data)
        
        # Limit buffer size to prevent memory buildup
        if len(self.input_buffer) > MAX_BUFFER_SIZE:
            excess = len(self.input_buffer) - MAX_BUFFER_SIZE
            self.input_buffer = self.input_buffer[excess:]
            logger.debug(f"Trimmed input buffer to {len(self.input_buffer)} bytes")
        
        # Update statistics
        self.total_bytes_processed += len(processed_data)
        self.total_chunks_processed += 1
        
        return processed_data
    
    def should_process_buffer(self) -> bool:
        """Check if buffer is ready for processing with reduced threshold."""
        # Reduced threshold for faster processing
        return len(self.input_buffer) >= AUDIO_BUFFER_SIZE // 2
    
    def get_audio_for_processing(self) -> Optional[np.ndarray]:
        """Get audio data for speech processing with MULAW optimization."""
        if not self.input_buffer:
            return None
        
        # Get mulaw bytes for processing
        mulaw_bytes = bytes(self.input_buffer)
        
        # Direct MULAW to LINEAR16 conversion
        pcm_audio = self._mulaw_to_linear16_optimized(mulaw_bytes)
        
        if len(pcm_audio) == 0:
            logger.warning("No PCM audio generated from mulaw data")
            return None
        
        # Apply lightweight preprocessing
        processed_audio = self._preprocess_audio_optimized(pcm_audio)
        
        # Clear buffer after successful processing
        self.input_buffer.clear()
        
        return processed_audio
    
    def prepare_outgoing_audio(self, audio_data: bytes) -> bytes:
        """Prepare audio data for sending to Twilio with format validation."""
        # Check if audio is already in MULAW format
        if self._is_mulaw_format(audio_data):
            return audio_data
        
        return self._convert_to_mulaw(audio_data)
    
    def split_audio_for_streaming(self, audio_data: bytes) -> List[bytes]:
        """Split audio into optimized chunks for streaming."""
        # Optimized chunk size for 8kHz MULAW (50ms chunks)
        chunk_size = 400  
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i+chunk_size])
        
        return chunks
    
    def _mulaw_to_linear16_optimized(self, mulaw_data: bytes) -> np.ndarray:
        """Optimized MULAW to LINEAR16 conversion with direct processing."""
        try:
            import audioop
            
            if len(mulaw_data) < 320:  # Less than 40ms at 8kHz
                return np.array([], dtype=np.float32)
            
            # Direct MULAW to 16-bit PCM conversion
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)
            
            # Resample from 8kHz to 16kHz using higher quality method
            pcm_data_16k, _ = audioop.ratecv(
                pcm_data, 2, 1, 8000, 16000, None
            )
            
            # Convert to numpy array
            audio_array = np.frombuffer(pcm_data_16k, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Apply dynamic range compression for better recognition
            audio_array = self._apply_dynamic_range_compression(audio_array)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return np.array([], dtype=np.float32)
    
    def _apply_dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression for better speech recognition."""
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 0:
                # Adaptive gain based on RMS
                target_rms = 0.1
                gain = min(5.0, target_rms / rms)
                audio = audio * gain
                
                # Soft limiting to prevent clipping
                audio = np.tanh(audio * 0.7) * 0.9
            
            return audio
        except Exception as e:
            logger.error(f"Error in dynamic range compression: {e}")
            return audio
    
    def _preprocess_audio_optimized(self, audio_data: np.ndarray) -> np.ndarray:
        """Optimized audio preprocessing for telephony."""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Lightweight filtering for telephony
            # 1. High-pass filter (reduce low frequency noise)
            b, a = signal.butter(3, 100/(16000/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # 2. Telefony bandpass filter (300-3400 Hz)
            b, a = signal.butter(2, [300/(16000/2), 3400/(16000/2)], 'band')
            bandpass_audio = signal.filtfilt(b, a, filtered_audio)
            
            # 3. Adaptive noise gate
            noise_threshold = max(0.01, self.ambient_noise_level * 2)
            gated_audio = np.where(np.abs(bandpass_audio) < noise_threshold, 
                                   bandpass_audio * 0.1, bandpass_audio)
            
            return gated_audio
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data
    
    def _convert_to_mulaw(self, audio_data: bytes) -> bytes:
        """Convert audio to MULAW format with better detection."""
        try:
            import audioop
            
            # Check if it's already MULAW
            if self._is_mulaw_format(audio_data):
                return audio_data
            
            # If it's WAV format, extract PCM data
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                # Extract PCM data from WAV
                pcm_data = self._extract_pcm_from_wav(audio_data)
                
                # Convert PCM to MULAW
                mulaw_data = audioop.lin2ulaw(pcm_data, 2)
                return mulaw_data
            
            # For other formats (like ElevenLabs MP3), assume it's raw PCM
            # This is a simplification - in production you'd want proper format detection
            try:
                mulaw_data = audioop.lin2ulaw(audio_data, 2)
                return mulaw_data
            except:
                # If conversion fails, return original data
                return audio_data
                
        except Exception as e:
            logger.error(f"Error converting to MULAW: {e}")
            return audio_data
    
    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """Extract PCM data from WAV file."""
        try:
            import wave
            import io
            
            with io.BytesIO(wav_data) as f:
                with wave.open(f, 'rb') as wav:
                    pcm_data = wav.readframes(wav.getnframes())
                    sample_rate = wav.getframerate()
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    
                    # Resample to 8kHz mono 16-bit if needed
                    if sample_rate != 8000:
                        pcm_data, _ = audioop.ratecv(
                            pcm_data, sample_width, channels,
                            sample_rate, 8000, None
                        )
                    
                    if channels > 1:
                        pcm_data = audioop.tomono(pcm_data, sample_width, 1, 0)
                    
                    if sample_width != 2:
                        pcm_data = audioop.lin2lin(pcm_data, sample_width, 2)
                    
                    return pcm_data
        except Exception as e:
            logger.error(f"Error extracting PCM from WAV: {e}")
            return b''
    
    def _is_mulaw_format(self, audio_data: bytes) -> bool:
        """Enhanced MULAW format detection."""
        if len(audio_data) < 100:
            return False
        
        # Check WAV header first
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            return False
        
        # Statistical analysis for MULAW detection
        sample = audio_data[:500]  # Larger sample
        
        # MULAW has characteristic distribution
        value_counts = [0] * 8
        for byte_val in sample:
            range_idx = byte_val // 32
            value_counts[range_idx] += 1
        
        # Calculate distribution variance
        total = sum(value_counts)
        if total == 0:
            return False
        
        # MULAW should have relatively uniform distribution
        max_count = max(value_counts)
        min_count = min(value_counts)
        
        # If distribution is too skewed, it's likely not MULAW
        if max_count / total > 0.7 or (max_count - min_count) / total > 0.6:
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset processor state with statistics."""
        logger.info(f"Resetting audio processor. Processed {self.total_chunks_processed} chunks, {self.total_bytes_processed} bytes")
        self.input_buffer.clear()
        self.noise_samples.clear()
        self.ambient_noise_level = 0.008
        self.total_bytes_processed = 0
        self.total_chunks_processed = 0
    
    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            "total_bytes_processed": self.total_bytes_processed,
            "total_chunks_processed": self.total_chunks_processed,
            "buffer_size": len(self.input_buffer),
            "ambient_noise_level": self.ambient_noise_level,
            "noise_samples_count": len(self.noise_samples)
        }