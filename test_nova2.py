#!/usr/bin/env python3
import asyncio
import os
import logging
from speech_to_text.deepgram_stt import DeepgramStreamingSTT

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_nova2():
    """Test Deepgram Nova-2 integration"""
    
    # Get API key from environment
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY not set in environment")
        return
    
    logger.info("Initializing Deepgram Nova-2 STT...")
    
    # Initialize the STT client
    stt_client = DeepgramStreamingSTT(
        api_key=api_key,
        model_name="nova-2",
        language="en-US",
        sample_rate=16000,
        encoding="linear16",
        channels=1,
        interim_results=True
    )
    
    # Define a callback to handle results
    async def handle_result(result):
        if result.is_final:
            logger.info(f"Final transcription: {result.text}")
            logger.info(f"Confidence: {result.confidence}")
    
    # Test with a sample WAV file
    test_file = "test_audio.wav"  # Make sure this file exists
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"Testing with audio file: {test_file}")
    
    # Start streaming session
    await stt_client.start_streaming()
    
    # Read and process audio file in chunks
    with open(test_file, 'rb') as f:
        chunk_size = 4096
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            result = await stt_client.process_audio_chunk(chunk, handle_result)
    
    # Stop streaming to get final results
    await stt_client.stop_streaming()
    
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_nova2())