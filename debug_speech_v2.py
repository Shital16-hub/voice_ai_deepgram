# debug_speech_v2.py
"""
Debug script for Google Cloud Speech-to-Text v2 API integration.
"""
import os
import asyncio
import logging
import json
import time
from pathlib import Path

# Import the fixed implementation
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_basic_connection():
    """Test basic connection to Google Cloud Speech v2."""
    logger.info("Testing basic connection to Google Cloud Speech v2...")
    
    # Check credentials
    credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_file:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set")
        return False
    
    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        return False
    
    # Extract project ID
    try:
        with open(credentials_file, 'r') as f:
            creds_data = json.load(f)
            project_id = creds_data.get('project_id')
            logger.info(f"Found project ID: {project_id}")
    except Exception as e:
        logger.error(f"Error reading credentials: {e}")
        return False
    
    # Test client initialization
    try:
        stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=project_id,
            enhanced_model=True,
            location="global"
        )
        logger.info("STT client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing STT client: {e}")
        return False

async def test_streaming_session():
    """Test streaming session lifecycle."""
    logger.info("Testing streaming session lifecycle...")
    
    try:
        # Get project ID
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        with open(credentials_file, 'r') as f:
            creds_data = json.load(f)
            project_id = creds_data.get('project_id')
        
        # Initialize client
        stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=project_id,
            enhanced_model=True,
            location="global"
        )
        
        # Test start streaming
        logger.info("Starting streaming session...")
        await stt_client.start_streaming()
        
        # Wait a bit
        await asyncio.sleep(1)
        
        # Test stop streaming
        logger.info("Stopping streaming session...")
        final_text, duration = await stt_client.stop_streaming()
        
        logger.info(f"Session completed. Final text: '{final_text}', Duration: {duration}")
        return True
        
    except Exception as e:
        logger.error(f"Error in streaming session test: {e}")
        return False

async def test_audio_processing():
    """Test processing actual audio data."""
    logger.info("Testing audio processing...")
    
    try:
        # Get project ID
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        with open(credentials_file, 'r') as f:
            creds_data = json.load(f)
            project_id = creds_data.get('project_id')
        
        # Initialize client
        stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=project_id,
            enhanced_model=True,
            location="global"
        )
        
        # Start streaming
        await stt_client.start_streaming()
        
        # Generate some test audio (silence)
        # In a real scenario, this would be actual audio data
        test_audio = np.zeros(8000, dtype=np.int16)  # 1 second of silence
        
        # Convert to MULAW
        import audioop
        mulaw_audio = audioop.lin2ulaw(test_audio.tobytes(), 2)
        
        # Process audio chunk
        logger.info("Processing test audio chunk...")
        
        results = []
        async def collect_result(result):
            results.append(result)
            logger.info(f"Received result: {result.text} (final: {result.is_final})")
        
        # Process the chunk
        result = await stt_client.process_audio_chunk(mulaw_audio, collect_result)
        
        # Stop streaming
        final_text, duration = await stt_client.stop_streaming()
        
        logger.info(f"Audio processing test completed. Results: {len(results)}")
        return True
        
    except Exception as e:
        logger.error(f"Error in audio processing test: {e}")
        return False

async def diagnose_telephony_settings():
    """Diagnose telephony-specific settings."""
    logger.info("Diagnosing telephony settings...")
    
    # Check required environment variables
    required_env_vars = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT"
    ]
    
    for var in required_env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"✓ {var} is set")
        else:
            logger.warning(f"✗ {var} is not set")
    
    # Check credentials file
    credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_file and os.path.exists(credentials_file):
        try:
            with open(credentials_file, 'r') as f:
                creds_data = json.load(f)
                logger.info(f"✓ Credentials file valid, project: {creds_data.get('project_id')}")
        except Exception as e:
            logger.error(f"✗ Error reading credentials: {e}")
    
    # Test telephony-optimized configuration
    logger.info("Testing telephony configuration...")
    
    # Recommended settings for telephony
    telephony_config = {
        "language": "en-US",
        "sample_rate": 8000,
        "encoding": "MULAW",
        "channels": 1,
        "interim_results": False,
        "enhanced_model": True,
        "location": "global"
    }
    
    logger.info(f"Recommended telephony config: {telephony_config}")
    
    return True

async def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("="*50)
    logger.info("Starting comprehensive Google Cloud Speech v2 test")
    logger.info("="*50)
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Streaming Session", test_streaming_session),
        ("Audio Processing", test_audio_processing),
        ("Telephony Settings", diagnose_telephony_settings)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            start_time = time.time()
            success = await test_func()
            end_time = time.time()
            
            results[test_name] = {
                "success": success,
                "duration": end_time - start_time
            }
            
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {status} ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            results[test_name] = {
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "PASSED" if result["success"] else "FAILED"
        logger.info(f"{test_name}: {status}")
        if "error" in result:
            logger.info(f"  Error: {result['error']}")
    
    # Recommendations
    logger.info("\n--- RECOMMENDATIONS ---")
    
    failed_tests = [name for name, result in results.items() if not result["success"]]
    
    if not failed_tests:
        logger.info("✓ All tests passed! Your Google Cloud Speech v2 setup is working correctly.")
    else:
        logger.info("✗ Some tests failed. Please check the following:")
        
        if "Basic Connection" in failed_tests:
            logger.info("1. Verify GOOGLE_APPLICATION_CREDENTIALS is set correctly")
            logger.info("2. Ensure the credentials file exists and is valid")
            logger.info("3. Check that the project ID in the credentials is correct")
        
        if "Streaming Session" in failed_tests:
            logger.info("1. Verify network connectivity to speech.googleapis.com")
            logger.info("2. Check if the Speech-to-Text API is enabled in your project")
            logger.info("3. Ensure sufficient quota and billing is set up")
        
        if "Audio Processing" in failed_tests:
            logger.info("1. Check audio format and encoding settings")
            logger.info("2. Verify sample rate matches the configuration")
            logger.info("3. Ensure audio data is valid")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())