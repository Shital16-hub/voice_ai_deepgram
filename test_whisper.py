#!/usr/bin/env python3
"""
Test script for Whisper STT integration.
"""
import asyncio
import logging
import numpy as np
import os
import sys
from scipy.io import wavfile

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("whisper_test")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR
from speech_to_text.utils.audio_utils import preprocess_telephony_audio

async def test_whisper():
    """Test Whisper STT with a sample audio file."""
    
    # Path to test audio file (provide a test file)
    test_file = "test_audio.wav"
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        logger.info("You can create a test file using:")
        logger.info("  sox -d -r 16000 -c 1 -b 16 test_audio.wav trim 0 5")
        return
    
    # Load audio file
    logger.info(f"Loading test file: {test_file}")
    try:
        sample_rate, audio_data = wavfile.read(test_file)
        
        # Convert to float32 in range [-1.0, 1.0]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        logger.info(f"Loaded audio: {len(audio_data)} samples, {sample_rate}Hz")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple resampling for testing
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return
    
    # Initialize Whisper
    logger.info("Initializing Whisper model")
    model_path = os.getenv("WHISPER_MODEL_PATH", "models/base.en.bin")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("You can download a model using:")
        logger.info("  wget -O models/base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
        return
    
    whisper = StreamingWhisperASR(
        model_path=model_path,
        language="en",
        n_threads=4,
        chunk_size_ms=2000,
        vad_enabled=True,
        temperature=0.0,
        initial_prompt="This is a telephone conversation. Transcribe the exact words spoken.",
        no_context=True,
        single_segment=True
    )
    
    # Process the audio
    logger.info("Processing audio with Whisper")
    
    # Apply telephony preprocessing
    logger.info("Applying telephony preprocessing")
    processed_audio = preprocess_telephony_audio(audio_data)
    
    # Set up transcription result callback
    async def handle_result(result):
        logger.info(f"Transcription result: '{result.text}'")
        logger.info(f"Confidence: {result.confidence:.2f}")
    
    # Start streaming session
    whisper.start_streaming()
    
    # Process the audio
    await whisper.process_audio_chunk(processed_audio, handle_result)
    
    # Get final result
    transcription, duration = await whisper.stop_streaming()
    
    logger.info(f"Final transcription: '{transcription}'")
    logger.info(f"Audio duration: {duration:.2f}s")
    
    return transcription

if __name__ == "__main__":
    asyncio.run(test_whisper())