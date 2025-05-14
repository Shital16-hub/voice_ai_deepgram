# test_stt_v2.py

"""
Test script for Google Cloud Speech-to-Text v2 with Twilio MULAW audio.
"""
import asyncio
import logging
import os
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_stt_v2():
    """Test Google Cloud STT v2 with sample MULAW data."""
    
    # Create STT client
    stt = GoogleCloudStreamingSTT(
        language="en-US",
        sample_rate=8000,
        encoding="MULAW",
        channels=1,
        interim_results=False,
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        enhanced_model=True,
        location="global"
    )
    
    logger.info("Starting STT test...")
    
    # Start streaming
    await stt.start_streaming()
    
    # Send some test audio (silence pattern that should be recognized as valid MULAW)
    # This is just for testing the connection
    test_audio = bytes([128] * 1600)  # 200ms of MULAW neutral value
    
    result = await stt.process_audio_chunk(test_audio)
    logger.info(f"Result: {result}")
    
    # Stop streaming
    await stt.stop_streaming()
    
    # Get stats
    stats = stt.get_stats()
    logger.info(f"STT Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_stt_v2())