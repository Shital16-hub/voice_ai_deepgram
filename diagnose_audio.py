"""
Diagnostic script to check audio processing and STT configuration.
Run this to debug audio and transcription issues.
"""
import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def diagnose_audio_pipeline():
    """Diagnose common audio and STT issues."""
    
    print("🔍 DIAGNOSING VOICE AI AUDIO PIPELINE")
    print("="*50)
    
    # 1. Check Environment Variables
    print("\n1. ENVIRONMENT VARIABLES:")
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    print(f"   GOOGLE_CLOUD_PROJECT: {google_project}")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")
    print(f"   Credentials file exists: {os.path.exists(google_creds) if google_creds else 'No'}")
    
    # 2. Check Google Cloud STT Configuration
    print("\n2. GOOGLE CLOUD STT CONFIGURATION:")
    stt_config = {
        'STT_PROJECT_ID': os.getenv('STT_PROJECT_ID'),
        'STT_LANGUAGE': os.getenv('STT_LANGUAGE', 'en-US'),
        'STT_MODEL': os.getenv('STT_MODEL', 'telephony_short'),
        'STT_SAMPLE_RATE': os.getenv('STT_SAMPLE_RATE', '8000'),
        'STT_ENCODING': os.getenv('STT_ENCODING', 'MULAW'),
        'STT_INTERIM_RESULTS': os.getenv('STT_INTERIM_RESULTS', 'False'),
    }
    
    for key, value in stt_config.items():
        print(f"   {key}: {value}")
    
    # 3. Test Google Cloud STT Direct Connection
    print("\n3. TESTING GOOGLE CLOUD STT CONNECTION:")
    try:
        from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
        
        # Create STT client
        stt = GoogleCloudStreamingSTT(
            language=stt_config['STT_LANGUAGE'],
            sample_rate=int(stt_config['STT_SAMPLE_RATE']),
            encoding=stt_config['STT_ENCODING'],
            channels=1,
            interim_results=False,
            project_id=google_project,
            location="global",
            credentials_file=google_creds
        )
        
        print("   ✅ STT client created successfully")
        
        # Test streaming
        await stt.start_streaming()
        print("   ✅ STT streaming started successfully")
        
        # Get stats
        stats = stt.get_stats()
        print(f"   📊 STT Stats: {stats}")
        
        await stt.stop_streaming()
        print("   ✅ STT streaming stopped successfully")
        
    except Exception as e:
        print(f"   ❌ STT Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Check Audio Processing Settings
    print("\n4. AUDIO PROCESSING SETTINGS:")
    audio_settings = {
        'MIN_TRANSCRIPTION_LENGTH': os.getenv('MIN_TRANSCRIPTION_LENGTH', '1'),
        'RESPONSE_TIMEOUT': os.getenv('RESPONSE_TIMEOUT', '1.5'),
        'SILENCE_TIMEOUT': os.getenv('SILENCE_TIMEOUT', '3.0'),
        'ECHO_DETECTION_WINDOW': os.getenv('ECHO_DETECTION_WINDOW', '0.8'),
        'STT_ENABLE_VAD': os.getenv('STT_ENABLE_VAD', 'True'),
        'STT_ENABLE_ECHO_SUPPRESSION': os.getenv('STT_ENABLE_ECHO_SUPPRESSION', 'False'),
    }
    
    for key, value in audio_settings.items():
        print(f"   {key}: {value}")
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS:")
    
    # Check if VAD is too aggressive
    if os.getenv('STT_ENABLE_VAD', 'True').lower() == 'true':
        print("   💡 Try disabling VAD if speech is not being detected:")
        print("      Set STT_ENABLE_VAD=False in .env")
    
    # Check speech timeouts
    speech_start = os.getenv('STT_SPEECH_START_TIMEOUT', '1')
    speech_end = os.getenv('STT_SPEECH_END_TIMEOUT', '0.2')
    print(f"   💡 Current speech timeouts: start={speech_start}s, end={speech_end}s")
    print("   💡 For better speech detection, try:")
    print("      STT_SPEECH_START_TIMEOUT=2")
    print("      STT_SPEECH_END_TIMEOUT=0.5")
    
    # Check interim results
    if os.getenv('STT_INTERIM_RESULTS', 'False').lower() == 'false':
        print("   💡 Enable interim results for debugging:")
        print("      Set STT_INTERIM_RESULTS=True in .env")
    
    print("\n✅ DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    asyncio.run(diagnose_audio_pipeline())