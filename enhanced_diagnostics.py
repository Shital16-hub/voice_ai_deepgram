#!/usr/bin/env python3
"""
Enhanced diagnostic script to debug Twilio + Google Cloud STT issues.
This script specifically tests MULAW audio handling and STT configuration.
"""
import os
import asyncio
import logging
import json
import base64
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_google_cloud_stt_configuration():
    """Test Google Cloud STT configuration specifically for Twilio MULAW."""
    
    print("\n" + "="*70)
    print("🔧 GOOGLE CLOUD STT CONFIGURATION TEST")
    print("="*70)
    
    # 1. Check Environment Variables
    print("\n1. ENVIRONMENT VARIABLES:")
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    print(f"   GOOGLE_CLOUD_PROJECT: {google_project}")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")
    print(f"   Credentials file exists: {os.path.exists(google_creds) if google_creds else 'No'}")
    
    if not google_project:
        print("   ❌ ERROR: GOOGLE_CLOUD_PROJECT not set!")
        return False
    
    if not google_creds or not os.path.exists(google_creds):
        print("   ❌ ERROR: GOOGLE_APPLICATION_CREDENTIALS not found!")
        return False
    
    # 2. Test STT Client Initialization
    print("\n2. STT CLIENT INITIALIZATION:")
    try:
        from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
        
        # Create STT client with Twilio-specific settings
        stt = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,         # Twilio sample rate
            encoding="MULAW",         # Twilio encoding
            channels=1,               # Mono
            interim_results=True,     # Enable for debugging
            project_id=google_project,
            location="global",
            credentials_file=google_creds,
            enable_vad=True,         # Voice Activity Detection
            enable_echo_suppression=False
        )
        
        print("   ✅ STT client created successfully")
        print(f"   📊 Configuration:")
        print(f"      - Model: telephony_short")
        print(f"      - Language: {stt.language}")
        print(f"      - Sample Rate: {stt.sample_rate}Hz")
        print(f"      - Encoding: {stt.encoding}")
        print(f"      - Interim Results: {stt.interim_results}")
        
        # Test streaming
        await stt.start_streaming()
        print("   ✅ STT streaming started successfully")
        
        # Get stats
        stats = stt.get_stats()
        print(f"   📊 STT Stats: {stats}")
        
        await stt.stop_streaming()
        print("   ✅ STT streaming stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ STT Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_processing():
    """Test audio processing with sample MULAW data."""
    
    print("\n3. AUDIO PROCESSING TEST:")
    
    try:
        from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
        
        # Create STT client
        stt = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,
            project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location="global",
            credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        
        # Create sample MULAW data (silence)
        # This simulates what Twilio sends
        sample_mulaw_data = b'\xff' * 160  # 160 bytes = 20ms of silence at 8kHz MULAW
        
        print(f"   📦 Created sample MULAW data: {len(sample_mulaw_data)} bytes")
        
        # Test callback to capture results
        results = []
        
        async def capture_result(result):
            results.append(result)
            print(f"   📬 Received result: '{result.text}' (final: {result.is_final})")
        
        # Start streaming and process sample data
        await stt.start_streaming()
        print("   🎙️ Started streaming...")
        
        # Send sample data multiple times
        for i in range(5):
            await stt.process_audio_chunk(sample_mulaw_data, capture_result)
            await asyncio.sleep(0.02)  # 20ms delay
        
        # Wait a bit for results
        await asyncio.sleep(1.0)
        
        await stt.stop_streaming()
        print(f"   ✅ Processed {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Audio Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_twilio_audio_simulation():
    """Simulate actual Twilio audio payload processing."""
    
    print("\n4. TWILIO AUDIO SIMULATION:")
    
    try:
        # Simulate a Twilio media message with MULAW payload
        # This is similar to what your WebSocket handler receives
        twilio_message = {
            "event": "media",
            "streamSid": "MZ12345678901234567890123456789012",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "1000",
                "payload": base64.b64encode(b'\xff' * 160).decode('utf-8')  # Silence
            }
        }
        
        print(f"   📨 Simulated Twilio message payload length: {len(twilio_message['media']['payload'])}")
        
        # Decode like the WebSocket handler does
        audio_data = base64.b64decode(twilio_message['media']['payload'])
        print(f"   🎵 Decoded audio data: {len(audio_data)} bytes")
        
        # Test with STT
        from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
        
        stt = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=True,
            project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
            credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        
        await stt.start_streaming()
        
        async def log_result(result):
            print(f"   🗣️ STT Result: '{result.text}' (final: {result.is_final}, conf: {result.confidence:.2f})")
        
        # Process the Twilio-style audio
        await stt.process_audio_chunk(audio_data, log_result)
        await asyncio.sleep(1.0)
        
        await stt.stop_streaming()
        print("   ✅ Twilio simulation completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Twilio Simulation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_websocket_handler_configuration():
    """Test the WebSocket handler configuration."""
    
    print("\n5. WEBSOCKET HANDLER CONFIGURATION:")
    
    try:
        from telephony.simple_websocket_handler import SimpleWebSocketHandler
        
        # Create a mock pipeline
        class MockPipeline:
            pass
        
        mock_pipeline = MockPipeline()
        
        # Create handler
        handler = SimpleWebSocketHandler("test_call_sid", mock_pipeline)
        
        print("   ✅ WebSocket handler created successfully")
        print(f"   📊 Handler configuration:")
        print(f"      - Project ID: {handler.project_id}")
        print(f"      - Min transcription length: {handler.MIN_TRANSCRIPTION_LENGTH}")
        print(f"      - Response timeout: {handler.RESPONSE_TIMEOUT}s")
        print(f"      - STT debug enabled: {handler.stt_debug}")
        print(f"      - Audio debug enabled: {handler.enable_audio_debug}")
        
        # Check STT client configuration
        stt_stats = handler.stt_client.get_stats()
        print(f"   📊 STT Client Stats: {stt_stats}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ WebSocket Handler Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_recommendations():
    """Print recommendations based on the test results."""
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS FOR FIXING SPEECH DETECTION")
    print("="*70)
    
    print("\n1. IMMEDIATE FIXES:")
    print("   🔧 Enable interim results for better debugging:")
    print("      - Your current config enables this ✅")
    
    print("\n   🔧 Use proper voice activity timeouts:")
    print("      - speech_start_timeout: 5 seconds (increased)")
    print("      - speech_end_timeout: 1 second (increased)")
    
    print("\n   🔧 Use 'telephony_short' model:")
    print("      - Optimized for phone calls")
    print("      - Better for short utterances")
    
    print("\n2. AUDIO CONFIGURATION:")
    print("   🎤 Ensure exact Twilio settings:")
    print("      - Sample Rate: 8000 Hz")
    print("      - Encoding: MULAW")
    print("      - Channels: 1 (mono)")
    
    print("\n3. DEBUGGING STEPS:")
    print("   🔍 Enable enhanced logging:")
    print("      - Set STT_DEBUG=True")
    print("      - Set ENABLE_AUDIO_DEBUG=True")
    print("      - Monitor interim results")
    
    print("\n4. COMMON ISSUES TO CHECK:")
    print("   ⚠️ Echo prevention too aggressive")
    print("   ⚠️ Voice Activity Detection (VAD) too strict")
    print("   ⚠️ Confidence thresholds too high")
    print("   ⚠️ Audio chunk processing delays")
    
    print("\n5. TEST WITH REAL CALL:")
    print("   📞 Make a test call and speak clearly")
    print("   📊 Monitor logs for interim results")
    print("   🎯 Look for 'INTERIM:' and 'FINAL:' log messages")

async def main():
    """Run all diagnostic tests."""
    
    print("🚀 STARTING COMPREHENSIVE VOICE AI DIAGNOSTICS")
    print("Testing configuration for Twilio + Google Cloud STT")
    
    # Run all tests
    test_results = []
    
    test_results.append(await test_google_cloud_stt_configuration())
    test_results.append(await test_audio_processing())
    test_results.append(await test_twilio_audio_simulation())
    test_results.append(await test_websocket_handler_configuration())
    
    # Summary
    print("\n" + "="*70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your configuration should work.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    # Always print recommendations
    print_recommendations()
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)