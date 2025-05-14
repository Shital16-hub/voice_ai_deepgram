# debug_stt_v2.py

"""
Debug script for Google Cloud Speech-to-Text v2 to identify the issue.
"""
import asyncio
import logging
import os
import audioop
import numpy as np
import time
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

# Set up very detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also enable debug logging for the STT module
logging.getLogger('speech_to_text.google_cloud_stt').setLevel(logging.DEBUG)

def create_simple_mulaw_audio():
    """Create a simple MULAW audio with a clear tone."""
    # Generate a 1-second 440Hz sine wave
    sample_rate = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    
    # Generate sine wave with moderate amplitude
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    pcm_data = (sine_wave * 32767).astype(np.int16)
    
    # Convert to MULAW
    pcm_bytes = pcm_data.tobytes()
    mulaw_data = audioop.lin2ulaw(pcm_bytes, 2)
    
    return mulaw_data

async def test_basic_connection():
    """Test basic connection and config sending."""
    logger.info("=== Testing Basic Connection ===")
    
    try:
        # Create STT client with debug logging
        stt = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,  # Enable interim results
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            enhanced_model=True,
            location="global"
        )
        
        logger.info("Starting streaming...")
        await stt.start_streaming()
        
        # Wait a moment to ensure streaming is started
        await asyncio.sleep(1.0)
        
        # Create simple audio
        logger.info("Creating test audio...")
        audio_data = create_simple_mulaw_audio()
        logger.info(f"Created {len(audio_data)} bytes of MULAW audio")
        
        # Send in larger chunks
        chunk_size = 3200  # 400ms at 8kHz
        results_received = 0
        
        async def callback(result):
            nonlocal results_received
            results_received += 1
            logger.info(f"*** RESULT {results_received}: '{result.text}' (final: {result.is_final}, confidence: {result.confidence})")
        
        logger.info("Sending audio chunks...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            logger.info(f"Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes")
            
            result = await stt.process_audio_chunk(chunk, callback)
            if result:
                logger.info(f"Got immediate result: {result}")
            
            # Wait between chunks
            await asyncio.sleep(0.5)
        
        # Wait longer for results
        logger.info("Waiting for results...")
        for i in range(10):  # Wait up to 10 seconds
            await asyncio.sleep(1.0)
            logger.info(f"Waiting... {i+1}/10")
            
            # Check for any results
            stats = stt.get_stats()
            if stats['successful_transcriptions'] > 0:
                logger.info(f"Got {stats['successful_transcriptions']} transcription(s)!")
                break
        
        # Stop streaming
        logger.info("Stopping streaming...")
        final_text, duration = await stt.stop_streaming()
        
        # Print detailed results
        logger.info(f"Final transcription: '{final_text}'")
        logger.info(f"Duration: {duration}s")
        logger.info(f"Results received via callback: {results_received}")
        
        stats = stt.get_stats()
        logger.info(f"Final stats: {stats}")
        
        return results_received > 0 or final_text
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

async def test_with_real_speech():
    """Test with synthesized speech-like audio."""
    logger.info("=== Testing With Speech-Like Audio ===")
    
    try:
        # Create STT client
        stt = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            enhanced_model=True,
            location="global"
        )
        
        # Create more speech-like audio
        def create_speech_like_mulaw():
            sample_rate = 8000
            duration = 2.0  # 2 seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create a more complex waveform with multiple harmonics
            signal = 0.3 * np.sin(2 * np.pi * 200 * t)  # Fundamental
            signal += 0.2 * np.sin(2 * np.pi * 400 * t)  # First harmonic
            signal += 0.1 * np.sin(2 * np.pi * 600 * t)  # Second harmonic
            
            # Add formant-like filtering
            signal += 0.15 * np.sin(2 * np.pi * 1000 * t)  # Formant
            signal += 0.1 * np.sin(2 * np.pi * 1500 * t)   # Second formant
            
            # Add some noise for realism
            signal += 0.05 * np.random.normal(0, 1, len(signal))
            
            # Apply envelope to make it more speech-like
            envelope = np.exp(-t * 0.5)  # Decay
            signal *= envelope
            
            # Normalize
            signal = np.clip(signal, -0.8, 0.8)
            
            # Convert to MULAW
            pcm_data = (signal * 32767).astype(np.int16)
            pcm_bytes = pcm_data.tobytes()
            return audioop.lin2ulaw(pcm_bytes, 2)
        
        logger.info("Starting streaming...")
        await stt.start_streaming()
        
        await asyncio.sleep(1.0)
        
        logger.info("Creating speech-like audio...")
        audio_data = create_speech_like_mulaw()
        logger.info(f"Created {len(audio_data)} bytes of speech-like MULAW audio")
        
        # Send in appropriate chunks
        chunk_size = 1600  # 200ms at 8kHz
        results_received = 0
        
        async def callback(result):
            nonlocal results_received
            results_received += 1
            logger.info(f"*** SPEECH RESULT {results_received}: '{result.text}' (final: {result.is_final})")
        
        logger.info("Sending speech-like audio...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            logger.info(f"Sending speech chunk {i//chunk_size + 1}: {len(chunk)} bytes")
            
            await stt.process_audio_chunk(chunk, callback)
            await asyncio.sleep(0.2)  # Real-time pace
        
        # Wait for results
        logger.info("Waiting for speech results...")
        await asyncio.sleep(5.0)
        
        # Stop streaming
        final_text, duration = await stt.stop_streaming()
        
        logger.info(f"Speech final result: '{final_text}'")
        logger.info(f"Speech results received: {results_received}")
        
        return results_received > 0 or final_text
        
    except Exception as e:
        logger.error(f"Speech test failed: {e}", exc_info=True)
        return False

async def test_minimal_example():
    """Test with minimal example following Google's documentation."""
    logger.info("=== Testing Minimal Example ===")
    
    try:
        # Test with Google Cloud Speech v2 direct API
        from google.cloud import speech_v2
        
        # Create client
        client = speech_v2.SpeechClient()
        
        # Create recognizer path
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        recognizer_path = f"projects/{project_id}/locations/global/recognizers/_"
        
        logger.info(f"Using recognizer: {recognizer_path}")
        
        # Create streaming config
        config = speech_v2.StreamingRecognitionConfig(
            config=speech_v2.RecognitionConfig(
                explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                    encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW,
                    sample_rate_hertz=8000,
                    audio_channel_count=1,
                ),
                language_codes=["en-US"],
                model="telephony_short",
            ),
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )
        
        # Generate requests
        def request_generator():
            # Config request
            yield speech_v2.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                streaming_config=config,
            )
            
            # Audio request
            audio_data = create_simple_mulaw_audio()
            logger.info(f"Sending {len(audio_data)} bytes of audio")
            yield speech_v2.StreamingRecognizeRequest(audio=audio_data)
        
        # Make the call
        logger.info("Making direct API call...")
        responses = client.streaming_recognize(request_generator())
        
        # Process responses
        response_count = 0
        for response in responses:
            response_count += 1
            logger.info(f"Received response #{response_count}")
            
            for result in response.results:
                logger.info(f"Result: is_final={result.is_final}, "
                           f"alternatives={len(result.alternatives)}")
                if result.alternatives:
                    alt = result.alternatives[0]
                    logger.info(f"Text: '{alt.transcript}', confidence: {alt.confidence}")
        
        logger.info(f"Total responses: {response_count}")
        return response_count > 0
        
    except Exception as e:
        logger.error(f"Minimal test failed: {e}", exc_info=True)
        return False

async def main():
    """Run all debug tests."""
    logger.info("Starting STT v2 debug tests...")
    
    # Check environment
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Credentials file: {creds_file}")
    
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT not set!")
        return
    
    # Run tests
    tests = [
        ("Minimal Example", test_minimal_example),
        ("Basic Connection", test_basic_connection),
        ("Speech-Like Audio", test_with_real_speech),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name} - {status} ({duration:.2f}s)")
            
            if result:
                logger.info("✓ This test showed some activity!")
                break
        except Exception as e:
            logger.error(f"{test_name} failed: {e}", exc_info=True)
        
        # Wait between tests
        await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(main())