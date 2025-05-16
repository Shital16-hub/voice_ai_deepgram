import os
import json
from dotenv import load_dotenv
from google.cloud import speech_v2
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

def test_v2_access():
    print("🚀 Testing Google Speech-to-Text v2 API Access")
    print("=" * 50)
    
    # Get credentials path
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        return False
    
    print(f"Using credentials: {creds_path}")
    
    # Load project ID
    with open(creds_path, 'r') as f:
        project_id = json.load(f)['project_id']
    print(f"Project ID: {project_id}")
    
    # Create v2 client with explicit credentials
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    client = speech_v2.SpeechClient(credentials=credentials)
    print("✅ Created Speech v2 client")
    
    # Test 1: Create recognizer path
    try:
        recognizer_path = client.recognizer_path(project_id, "global", "_")
        print(f"✅ Generated recognizer path: {recognizer_path}")
    except Exception as e:
        print(f"❌ Error generating recognizer path: {e}")
        return False
    
    # Test 2: Create v2 recognition config
    try:
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model="latest_short",
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        print("✅ Created v2 recognition config")
    except Exception as e:
        print(f"❌ Error creating config: {e}")
        return False
    
    # Test 3: Attempt recognition with minimal audio
    try:
        # Create minimal test audio (1000 samples of silence)
        audio_data = b'\x00\x00' * 1000  # 16-bit linear PCM silence
        
        request = speech_v2.RecognizeRequest(
            recognizer=recognizer_path,
            config=config,
            content=audio_data,
        )
        
        print("🔄 Attempting v2 speech recognition...")
        response = client.recognize(request=request)
        print(f"✅ v2 Recognition successful! Results: {len(response.results)}")
        
        # Print any results
        for i, result in enumerate(response.results):
            for j, alternative in enumerate(result.alternatives):
                print(f"   Alternative {j}: {alternative.transcript}")
        
        return True
        
    except Exception as e:
        print(f"❌ Recognition failed: {e}")
        
        # Check if it's a permission error
        if "Permission" in str(e) or "403" in str(e):
            print("\n🔧 SOLUTION: Add proper roles to your service account:")
            print("   1. Go to Google Cloud Console")
            print("   2. Navigate to IAM & Admin > IAM")
            print("   3. Find your service account")
            print("   4. Add role: 'Cloud Speech Client' or 'Speech Administrator'")
        
        return False

if __name__ == "__main__":
    success = test_v2_access()
    if success:
        print("\n🎉 Speech-to-Text v2 API is fully accessible!")
    else:
        print("\n⚠️  v2 API access needs configuration")
