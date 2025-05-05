import os
from google.cloud import speech

def test_credentials():
    try:
        # Print the path to verify it's correct
        print(f"Using credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        
        # Try to create a client
        client = speech.SpeechClient()
        print("Successfully created Speech client")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_credentials()