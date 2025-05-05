#!/usr/bin/env python3
"""
Test script for ElevenLabs TTS to Twilio audio conversion.
"""
import os
import sys
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Now import from the text_to_speech package
from text_to_speech import ElevenLabsTTS

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and install it if not."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("ffmpeg is installed and working.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg not found! Installing ffmpeg...")
        try:
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
            logger.info("ffmpeg installed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to install ffmpeg: {e}")
            return False

async def test_elevenlabs_to_twilio():
    """
    Test ElevenLabs to Twilio audio conversion pipeline.
    
    This function tests the entire audio conversion pipeline:
    1. Generate audio from ElevenLabs
    2. Convert to Twilio-compatible format using ffmpeg
    3. Save files at each step for inspection
    """
    try:
        # Check if ffmpeg is installed
        if not check_ffmpeg_installed():
            return {"error": "ffmpeg not installed", "success": False}
        
        # Get API key from environment or input
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            api_key = input("Enter your ElevenLabs API key: ")
            os.environ["ELEVENLABS_API_KEY"] = api_key
        
        # Create output directory
        output_dir = Path("./debug_audio")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize ElevenLabsTTS
        tts = ElevenLabsTTS(api_key=api_key)
        
        # Step 1: Generate MP3 from ElevenLabs
        test_text = "This is a test of the ElevenLabs to Twilio audio conversion pipeline. Testing one, two, three."
        logger.info(f"Generating audio from ElevenLabs for: '{test_text}'")
        
        elevenlabs_mp3 = await tts.synthesize(test_text)
        
        # Save the raw MP3
        mp3_path = output_dir / "1_elevenlabs_raw.mp3"
        with open(mp3_path, "wb") as f:
            f.write(elevenlabs_mp3)
        logger.info(f"Saved raw ElevenLabs MP3 ({len(elevenlabs_mp3)} bytes) to {mp3_path}")
        
        # Step 2: Convert to Twilio-compatible format (μ-law) using ffmpeg
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                mp3_file.write(elevenlabs_mp3)
                mp3_file_path = mp3_file.name
            
            # Create output file path
            twilio_path = output_dir / "2_twilio_compatible.wav"
            
            # Build ffmpeg command for μ-law conversion
            cmd = [
                'ffmpeg',
                '-i', mp3_file_path,
                '-acodec', 'pcm_mulaw',  # μ-law encoding for telephony
                '-ar', '8000',           # 8kHz sample rate (Twilio standard)
                '-ac', '1',              # Mono (1 channel)
                '-y',                    # Overwrite output if exists
                str(twilio_path)
            ]
            
            # Run ffmpeg
            logger.info(f"Running ffmpeg: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True)
            
            # Log output
            logger.info(f"Successfully converted to Twilio format: {twilio_path}")
            
            # Check if output file exists and has content
            if twilio_path.exists():
                twilio_size = twilio_path.stat().st_size
                logger.info(f"Twilio audio file size: {twilio_size} bytes")
                success = twilio_size > 0
            else:
                logger.error(f"Output file not created: {twilio_path}")
                success = False
            
            # Clean up temp file
            os.unlink(mp3_file_path)
            
            return {
                "elevenlabs_size": len(elevenlabs_mp3),
                "twilio_size": twilio_size if 'twilio_size' in locals() else 0,
                "success": success
            }
            
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            logger.error(f"FFmpeg error: {stderr}")
            return {"error": f"FFmpeg error: {e}", "success": False}
            
        except Exception as conv_error:
            logger.error(f"Error in Twilio format conversion: {conv_error}")
            return {"error": str(conv_error), "success": False}
        
    except Exception as e:
        logger.error(f"Error in audio conversion test: {e}")
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_elevenlabs_to_twilio())
    print(f"Test result: {result}")