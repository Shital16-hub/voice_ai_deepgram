# Add this to a tests/debug_utils.py file:

import os
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

async def test_elevenlabs_to_twilio():
    """
    Test ElevenLabs to Twilio audio conversion pipeline.
    
    This function tests the entire audio conversion pipeline:
    1. Generate audio from ElevenLabs
    2. Convert to Twilio-compatible format
    3. Save files at each step for inspection
    """
    try:
        # Import necessary components
        from elevenlabs_tts import ElevenLabsTTS
        from telephony.audio_processor import AudioProcessor
        
        # Create output directory
        output_dir = Path("./debug_audio")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        tts = ElevenLabsTTS()
        processor = AudioProcessor()
        
        # Step 1: Generate MP3 from ElevenLabs
        test_text = "This is a test of the ElevenLabs to Twilio audio conversion pipeline. Testing one, two, three."
        logger.info(f"Generating audio from ElevenLabs for: '{test_text}'")
        
        elevenlabs_mp3 = await tts.synthesize(test_text)
        
        # Save the raw MP3
        mp3_path = output_dir / "1_elevenlabs_raw.mp3"
        with open(mp3_path, "wb") as f:
            f.write(elevenlabs_mp3)
        logger.info(f"Saved raw ElevenLabs MP3 ({len(elevenlabs_mp3)} bytes) to {mp3_path}")
        
        # Step 2: Convert to WAV (intermediate step)
        try:
            wav_audio = processor.convert_to_wav(
                elevenlabs_mp3,
                sample_rate=16000,
                channels=1
            )
            
            # Save the WAV
            wav_path = output_dir / "2_converted.wav"
            with open(wav_path, "wb") as f:
                f.write(wav_audio)
            logger.info(f"Saved converted WAV ({len(wav_audio)} bytes) to {wav_path}")
            
        except Exception as wav_error:
            logger.error(f"Error in WAV conversion: {wav_error}")
        
        # Step 3: Convert to Twilio-compatible format (μ-law)
        try:
            twilio_audio = processor.prepare_audio_for_telephony(
                elevenlabs_mp3,
                format="mp3",
                target_sample_rate=8000,
                target_channels=1
            )
            
            # Save the Twilio audio
            twilio_path = output_dir / "3_twilio_compatible.wav"
            with open(twilio_path, "wb") as f:
                f.write(twilio_audio)
            logger.info(f"Saved Twilio-compatible audio ({len(twilio_audio)} bytes) to {twilio_path}")
            
        except Exception as twilio_error:
            logger.error(f"Error in Twilio format conversion: {twilio_error}")
        
        return {
            "elevenlabs_size": len(elevenlabs_mp3),
            "twilio_size": len(twilio_audio) if 'twilio_audio' in locals() else 0,
            "success": 'twilio_audio' in locals() and len(twilio_audio) > 0
        }
        
    except Exception as e:
        logger.error(f"Error in audio conversion test: {e}")
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the test
    result = asyncio.run(test_elevenlabs_to_twilio())
    print(f"Test result: {result}")