"""
Audio quality enhancer for telephony integration.
"""
import logging
import numpy as np
from scipy import signal
from typing import Optional

logger = logging.getLogger(__name__)

class AudioEnhancer:
    """
    Enhances audio quality for telephony applications.
    Applies telephony-specific processing to improve voice clarity.
    """
    
    def __init__(
        self,
        enable_compression: bool = True,
        enable_noise_gate: bool = True,
        enable_eq: bool = True
    ):
        """
        Initialize the audio enhancer.
        
        Args:
            enable_compression: Enable dynamic range compression
            enable_noise_gate: Enable noise gate
            enable_eq: Enable telephony equalization
        """
        self.enable_compression = enable_compression
        self.enable_noise_gate = enable_noise_gate
        self.enable_eq = enable_eq
        
        # Parameters
        self.noise_threshold = 0.01    # Noise gate threshold
        self.comp_threshold = 0.3      # Compression threshold
        self.comp_ratio = 0.6          # Compression ratio (lower = more compression)
        
    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Enhance audio quality for telephony.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Enhanced audio data
        """
        # Skip processing if audio is empty
        if len(audio) == 0:
            return audio
            
        # Normalize input to range [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Make a copy to avoid modifying the original
        enhanced = audio.copy()
        
        # Apply telephony EQ (bandpass to focus on voice frequencies)
        if self.enable_eq:
            enhanced = self._apply_telephony_eq(enhanced, sample_rate)
        
        # Apply noise gate
        if self.enable_noise_gate:
            enhanced = self._apply_noise_gate(enhanced)
        
        # Apply compression
        if self.enable_compression:
            enhanced = self._apply_compression(enhanced)
        
        # Final normalization to prevent clipping
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced * (0.95 / max_val)
        
        return enhanced
    
    def _apply_telephony_eq(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply telephony-specific equalization."""
        # Apply pre-emphasis to boost high frequencies
        pre_emphasized = np.append(audio[0], audio[1:] - 0.95 * audio[:-1])
        
        # Apply bandpass filter for telephony range (300-3400 Hz)
        nyquist = sample_rate / 2
        low_cutoff = 300 / nyquist
        high_cutoff = 3400 / nyquist
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered = signal.filtfilt(b, a, pre_emphasized)
        
        # Boost 1-3kHz range (for improved intelligibility)
        for freq, gain in [(1000, 1.2), (2000, 1.5), (3000, 1.3)]:
            freq_norm = freq / nyquist
            width = 0.1
            
            # Create a bell filter
            b, a = signal.butter(2, [freq_norm - width/2, freq_norm + width/2], btype='band')
            bell = signal.filtfilt(b, a, filtered)
            
            # Apply boost
            filtered = filtered + (bell * (gain - 1))
        
        return filtered
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove background noise."""
        # Detect energy levels across frames
        frame_size = 512
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size) if i+frame_size <= len(audio)]
        
        if not frames:
            return audio
            
        # Calculate energy for each frame
        energies = np.array([np.mean(np.square(frame)) for frame in frames])
        
        # Apply smoother noise gate with fade in/out to avoid clicks
        gated = np.zeros_like(audio)
        for i, frame_idx in enumerate(range(0, len(audio), frame_size)):
            if i >= len(frames):
                break
                
            # Get gain for this frame (0 if below threshold, 1 if above)
            gain = 1.0 if energies[i] > self.noise_threshold else 0.0
            
            # Apply smoothing between frames
            for j in range(frame_size):
                if frame_idx + j < len(audio):
                    # Apply fade in/out at frame boundaries (10% of frame size)
                    fade_size = frame_size // 10
                    if j < fade_size:
                        # Fade in
                        fade_gain = gain * (j / fade_size)
                    elif j >= frame_size - fade_size:
                        # Fade out
                        fade_gain = gain * ((frame_size - j) / fade_size)
                    else:
                        fade_gain = gain
                        
                    gated[frame_idx + j] = audio[frame_idx + j] * fade_gain
        
        return gated
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression to make voice more consistent."""
        # Calculate absolute values
        abs_audio = np.abs(audio)
        
        # Create a compression curve
        gain = np.ones_like(audio)
        
        # Compress only signals above threshold
        mask = abs_audio > self.comp_threshold
        gain[mask] = (self.comp_threshold + 
                     (abs_audio[mask] - self.comp_threshold) * self.comp_ratio) / abs_audio[mask]
        
        # Apply gain
        compressed = audio * gain
        
        # Apply makeup gain to bring back volume
        max_val = np.max(np.abs(compressed))
        if max_val > 0:
            makeup_gain = 0.95 / max_val
            compressed = compressed * makeup_gain
        
        return compressed