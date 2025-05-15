"""
Simplified configuration settings for telephony integration.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Audio Configuration - Simplified for direct Google STT v2
SAMPLE_RATE = 8000              # 8kHz for Twilio
CHUNK_SIZE = 800                # 100ms at 8kHz
ENCODING = "MULAW"              # Twilio encoding

# Google Cloud STT v2 Configuration
STT_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not STT_PROJECT_ID:
    # Try to extract from credentials file
    credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_file and os.path.exists(credentials_file):
        try:
            with open(credentials_file, 'r') as f:
                creds_data = json.load(f)
                STT_PROJECT_ID = creds_data.get('project_id')
        except Exception:
            pass

STT_LANGUAGE = os.getenv('STT_LANGUAGE', 'en-US')
STT_MODEL = "telephony"  # Optimized for phone calls

# TTS Configuration
TTS_VOICE_NAME = os.getenv('TTS_VOICE_NAME', 'en-US-Neural2-C')
TTS_VOICE_GENDER = os.getenv('TTS_VOICE_GENDER', 'NEUTRAL')
TTS_LANGUAGE_CODE = os.getenv('TTS_LANGUAGE_CODE', 'en-US')

# Performance Settings
MAX_CALL_DURATION = 3600        # 1 hour
RESPONSE_TIMEOUT = 5.0          # 5 seconds
MIN_TRANSCRIPTION_LENGTH = 2    # Minimum words

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'