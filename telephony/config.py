"""
Optimized configuration settings for telephony integration.
"""
import os
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

# Audio Configuration - Optimized for telephony
SAMPLE_RATE_TWILIO = 8000  # Keep at 8kHz
SAMPLE_RATE_AI = 8000      # Changed to 8kHz to match Twilio and reduce conversions
CHUNK_SIZE = 160           # 20ms at 8kHz
AUDIO_BUFFER_SIZE = 8000   # 1 second buffer
MAX_BUFFER_SIZE = 16000    # 2 seconds maximum buffer

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Performance Settings - Optimized for telephony
SILENCE_THRESHOLD = 0.005   # Reduced for better sensitivity
SILENCE_DURATION = 0.8      # Shorter for faster response
MAX_CALL_DURATION = 3600    # 1 hour
MAX_PROCESSING_TIME = 3.0   # Reduced processing time

# Response Settings
RESPONSE_TIMEOUT = 3.0      # Faster response timeout
MIN_TRANSCRIPTION_LENGTH = 1  # Reduced to catch single words

# STT Optimization Settings for Google Cloud - FIXED
STT_TELEPHONY_OPTIMIZED = True
STT_USE_ENHANCED_MODEL = True
STT_MODEL_NAME = "phone_call"  # Use telephony-optimized model
STT_INTERIM_RESULTS = False    # Disable for lower latency
STT_SAMPLE_RATE = 8000         # Match Twilio's rate
STT_ENCODING = "MULAW"         # Direct mulaw support (NOT ALAW)

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'