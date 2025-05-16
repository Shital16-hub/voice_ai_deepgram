"""
Audio utilities for the text-to-speech module.

This module provides utilities for working with audio data,
including format conversion, streaming, and processing.
"""
import io
import os
import tempfile
import logging
from typing import Optional, Dict, Any, Union, Tuple, BinaryIO
import wave

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Utilities for processing audio data for delivery to telephony systems."""
    
    @staticmethod
    def get_audio_info(audio_data: bytes) -> Dict[str, Any]:
        """
        Extract information from audio data.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with audio information (format, duration, etc.)
        """
        # Currently only supporting WAV format detection
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            with io.BytesIO(audio_data) as f:
                with wave.open(f, 'rb') as wav:
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    frame_rate = wav.getframerate()
                    n_frames = wav.getnframes()
                    duration = n_frames / frame_rate
                    
                    return {
                        'format': 'wav',
                        'channels': channels,
                        'sample_width': sample_width,
                        'frame_rate': frame_rate,
                        'n_frames': n_frames,
                        'duration': duration
                    }
        else:
            # For other formats, return basic info
            return {
                'format': 'unknown',
                'size': len(audio_data)
            }
    
    @staticmethod
    def convert_to_wav(
        audio_data: bytes,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None
    ) -> bytes:
        """
        Convert audio data to WAV format for telephony systems.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate (Hz)
            channels: Target number of channels
            
        Returns:
            WAV audio data as bytes
        """
        # If we receive MP3 from ElevenLabs, we need to convert it
        # This requires ffmpeg to be installed
        try:
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                mp3_file.write(audio_data)
                mp3_path = mp3_file.name
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name
            
            cmd = ['ffmpeg', '-i', mp3_path, '-acodec', 'pcm_s16le']
            
            if sample_rate:
                cmd.extend(['-ar', str(sample_rate)])
            if channels:
                cmd.extend(['-ac', str(channels)])
                
            cmd.append(wav_path)
            
            # Run ffmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the resulting WAV file
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
                
            # Clean up temporary files
            os.unlink(mp3_path)
            os.unlink(wav_path)
            
            return wav_data
            
        except ImportError:
            logger.error("ffmpeg not available for audio conversion")
            raise RuntimeError("ffmpeg is required for audio conversion")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise RuntimeError(f"Audio conversion error: {str(e)}")
    
    @staticmethod
    def split_audio(
        audio_data: bytes,
        chunk_size: int = 4096
    ) -> list:
        """
        Split audio data into manageable chunks for streaming.
        
        Args:
            audio_data: Audio data as bytes
            chunk_size: Size of each chunk in bytes
            
        Returns:
            List of audio data chunks
        """
        return [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
    
    @staticmethod
    def convert_to_mulaw(
        audio_data: bytes,
        sample_rate: int = 8000,
        channels: int = 1
    ) -> bytes:
        """
        Convert audio data to µ-law format for Twilio.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate for Twilio
            channels: Target number of channels for Twilio
            
        Returns:
            Audio data converted to µ-law format
        """
        try:
            import audioop
            
            # If input is already µ-law, return as-is
            if is_mulaw(audio_data):
                return audio_data
                
            # If input is WAV, extract PCM data
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                with io.BytesIO(audio_data) as f:
                    with wave.open(f, 'rb') as wav:
                        pcm_data = wav.readframes(wav.getnframes())
                        src_sample_rate = wav.getframerate()
                        src_channels = wav.getnchannels()
                        src_width = wav.getsampwidth()
                        
                        # Resample if needed
                        if src_sample_rate != sample_rate:
                            pcm_data, _ = audioop.ratecv(
                                pcm_data, src_width, src_channels, 
                                src_sample_rate, sample_rate, None
                            )
                        
                        # Convert to mono if needed
                        if src_channels != channels:
                            pcm_data = audioop.tomono(pcm_data, src_width, 1, 0)
                            
                        # Convert to 16-bit if needed
                        if src_width != 2:  # 2 bytes = 16-bit
                            pcm_data = audioop.lin2lin(pcm_data, src_width, 2)
                            
                        # Convert to µ-law
                        mulaw_data = audioop.lin2ulaw(pcm_data, 2)  # 2 bytes = 16-bit
                        
                        return mulaw_data
            else:
                # For other formats, assume 16-bit PCM and convert directly
                return audioop.lin2ulaw(audio_data, 2)  # 2 bytes = 16-bit
                
        except ImportError:
            logger.error("audioop not available for audio conversion")
            raise RuntimeError("audioop is required for audio conversion")
        except Exception as e:
            logger.error(f"Error converting to mulaw: {e}")
            raise RuntimeError(f"Audio conversion error: {str(e)}")
    
    @staticmethod
    def is_mulaw(audio_data: bytes) -> bool:
        """
        Check if audio data is in µ-law format.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            True if audio is in µ-law format
        """
        # This is a simple heuristic - µ-law data typically has
        # most values in the full 8-bit range (0-255)
        if len(audio_data) < 100:
            return False
            
        # Sample the first 100 bytes
        sample = audio_data[:100]
        
        # Count values in different ranges
        low_range = sum(1 for b in sample if b < 64)
        mid_range = sum(1 for b in sample if 64 <= b < 192)
        high_range = sum(1 for b in sample if b >= 192)
        
        # µ-law typically has a more uniform distribution across the full range
        total = low_range + mid_range + high_range
        if total == 0:
            return False
            
        # If we have a good distribution across all ranges, it's likely µ-law
        return (low_range / total >= 0.2 and 
                mid_range / total >= 0.2 and 
                high_range / total >= 0.2)
    
    @staticmethod
    def prepare_for_twilio(
        audio_data: bytes,
        sample_rate: int = 8000,
        channels: int = 1
    ) -> bytes:
        """
        Prepare audio data for Twilio.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Target sample rate for Twilio
            channels: Target number of channels for Twilio
            
        Returns:
            Audio data converted for Twilio
        """
        # Convert to µ-law format for Twilio
        return AudioProcessor.convert_to_mulaw(
            audio_data, 
            sample_rate=sample_rate, 
            channels=channels
        )