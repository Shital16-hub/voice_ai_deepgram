"""
Speech activity detection for barge-in functionality.
"""
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

logger = logging.getLogger(__name__)

class SpeechActivityDetector:
    """
    Detects user speech activity during agent's response for barge-in handling.
    """
    
    def __init__(self, energy_threshold=0.05, consecutive_frames=3, frame_duration=0.02):
        """
        Initialize speech activity detector.
        
        Args:
            energy_threshold: Energy threshold for speech detection
            consecutive_frames: Number of consecutive frames needed to confirm speech
            frame_duration: Duration of each frame in seconds
        """
        self.energy_threshold = energy_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_duration = frame_duration
        
        # State variables
        self.speech_frames = 0
        self.last_detection_time = 0
        self.is_speaking = False
        
        # Adaptive threshold variables
        self.background_energy = 0.01
        self.adaptation_rate = 0.05
        self.min_energy_threshold = energy_threshold
        
        logger.info(f"Initialized SpeechActivityDetector with threshold={energy_threshold}")
    
    def update_background_energy(self, energy):
        """
        Update background energy level using exponential smoothing.
        
        Args:
            energy: Current frame energy
        """
        if energy < self.energy_threshold * 0.8:  # Only update during silence
            self.background_energy = (1 - self.adaptation_rate) * self.background_energy + \
                                     self.adaptation_rate * energy
            logger.debug(f"Updated background energy to {self.background_energy:.6f}")
    
    def detect(self, audio_frame: np.ndarray) -> bool:
        """
        Detect speech activity in audio frame.
        
        Args:
            audio_frame: Audio data as numpy array
            
        Returns:
            True if speech detected
        """
        # Calculate energy
        energy = np.mean(np.abs(audio_frame))
        
        # Update background energy
        self.update_background_energy(energy)
        
        # Adaptive threshold - at least 2x background but no less than base threshold
        adaptive_threshold = max(self.min_energy_threshold, self.background_energy * 2.5)
        
        # Check for speech activity
        if energy > adaptive_threshold:
            self.speech_frames += 1
            if self.speech_frames >= self.consecutive_frames:
                if not self.is_speaking:
                    logger.info(f"Speech detected! Energy: {energy:.4f}, Threshold: {adaptive_threshold:.4f}")
                self.is_speaking = True
                self.last_detection_time = time.time()
                return True
        else:
            # Reset counter if not enough consecutive frames
            if self.speech_frames < self.consecutive_frames:
                self.speech_frames = 0
            
            # Check for end of speech
            if self.is_speaking and time.time() - self.last_detection_time > 0.5:
                logger.info("Speech ended")
                self.is_speaking = False
                
        return self.is_speaking
    
    def analyze_spectral_properties(self, audio_frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze spectral properties of audio to better detect speech vs. noise.
        
        Args:
            audio_frame: Audio data as numpy array
            
        Returns:
            Dictionary with spectral features
        """
        # Only analyze if we have enough samples
        if len(audio_frame) < 320:  # Need at least 20ms at 16kHz
            return {"speech_likelihood": 0.0}
        
        try:
            from scipy import signal
            
            # Generate spectrum
            freqs, power = signal.welch(audio_frame, fs=16000, nperseg=512)
            
            # Define speech-relevant frequency bands
            formant1_band = (300, 1000)   # First formant region
            formant2_band = (1000, 2500)  # Second formant region
            high_band = (2500, 4000)      # Higher frequencies
            
            # Extract power in each band
            f1_power = np.mean(power[(freqs >= formant1_band[0]) & (freqs <= formant1_band[1])])
            f2_power = np.mean(power[(freqs >= formant2_band[0]) & (freqs <= formant2_band[1])])
            high_power = np.mean(power[(freqs >= high_band[0]) & (freqs <= high_band[1])])
            
            # Calculate ratios that are characteristic of speech
            f1_f2_ratio = f1_power / f2_power if f2_power > 0 else 0
            speech_high_ratio = (f1_power + f2_power) / high_power if high_power > 0 else 0
            
            # Compute speech likelihood based on spectral features
            # Speech typically has strong formants relative to high frequencies
            # and a specific ratio between first and second formants
            speech_likelihood = 0.0
            
            if 0.8 < f1_f2_ratio < 3.0 and speech_high_ratio > 2.0:
                speech_likelihood = 0.8
            
            return {
                "f1_power": f1_power,
                "f2_power": f2_power,
                "high_power": high_power,
                "f1_f2_ratio": f1_f2_ratio,
                "speech_high_ratio": speech_high_ratio,
                "speech_likelihood": speech_likelihood
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spectral properties: {e}")
            return {"speech_likelihood": 0.0}