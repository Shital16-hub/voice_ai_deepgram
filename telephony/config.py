"""
Optimized configuration settings for telephony integration with improved noise handling.
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

# Audio Configuration
SAMPLE_RATE_TWILIO = 8000  # Twilio's sample rate
SAMPLE_RATE_AI = 16000     # Our AI system's sample rate
CHUNK_SIZE = 320           # 20ms at 8kHz (reduced from 320)
# Decreased buffer size for faster processing
AUDIO_BUFFER_SIZE = 8000  # 1 second buffer - reduced for faster processing
MAX_BUFFER_SIZE = 16000   # 2 seconds maximum buffer - reduced for faster processing

# WebSocket Configuration
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_MAX_MESSAGE_SIZE = 1048576  # 1MB

# Performance Settings - Optimized for troubleshooting
# Reduced silence threshold to detect more speech
SILENCE_THRESHOLD = 0.004   # Reduced from 0.008 to detect more speech
SILENCE_DURATION = 0.8      # Reduced from 1.2 to be more responsive
MAX_CALL_DURATION = 3600    # 1 hour
MAX_PROCESSING_TIME = 5.0   # Maximum time to spend processing audio (seconds)

# Response Settings
RESPONSE_TIMEOUT = 4.0      # Maximum time to wait for a response (seconds)
MIN_TRANSCRIPTION_LENGTH = 1  # Reduced from 3 to 1 during troubleshooting

# Noise Filtering Settings
HIGH_PASS_FILTER = 80       # High-pass filter cutoff frequency in Hz
NOISE_GATE_THRESHOLD = 0.01  # Reduced noise gate threshold from 0.015
ENABLE_NOISE_FILTERING = True  # Enable enhanced noise filtering

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# STT Optimization Settings
STT_INITIAL_PROMPT = "This is a clear business conversation. Transcribe the exact words spoken, ignoring background noise."
STT_NO_CONTEXT = True      # Disable context to prevent false additions in noisy environments
STT_TEMPERATURE = 0.0      # Use greedy decoding for less hallucination
STT_PRESET = "default"     # Use default preset with noise handling optimizations