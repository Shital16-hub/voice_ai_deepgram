"""
Optimized configuration settings for reduced latency.
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

# Audio Configuration - Optimized for low latency
SAMPLE_RATE_TWILIO = 8000
SAMPLE_RATE_AI = 16000
CHUNK_SIZE = 640           # Increased from 320 for better efficiency
# Reduced buffer sizes for lower latency
AUDIO_BUFFER_SIZE = 6400   # Reduced from 32000 (800ms buffer)
MAX_BUFFER_SIZE = 12800    # Reduced from 48000 (1.6s max buffer)

# WebSocket Configuration
WS_PING_INTERVAL = 10      # Reduced from 20
WS_PING_TIMEOUT = 5        # Reduced from 10
WS_MAX_MESSAGE_SIZE = 1048576

# Performance Settings - Optimized for speed
SILENCE_THRESHOLD = 0.012  # Slightly increased for noise immunity
SILENCE_DURATION = 0.8     # Reduced from 1.2 for faster response
MAX_CALL_DURATION = 3600
MAX_PROCESSING_TIME = 2.5  # Reduced from 5.0

# Response Settings - Faster timeouts
RESPONSE_TIMEOUT = 2.5     # Reduced from 4.0
MIN_TRANSCRIPTION_LENGTH = 1  # Reduced from 3 for faster response

# Noise Filtering - Balanced for speed and quality
HIGH_PASS_FILTER = 80
NOISE_GATE_THRESHOLD = 0.015
ENABLE_NOISE_FILTERING = True

# STT Optimization - Faster settings
STT_INITIAL_PROMPT = "Clear business conversation. Transcribe exactly."  # Shortened
STT_NO_CONTEXT = True
STT_TEMPERATURE = 0.0
STT_PRESET = "default"

# TTS Optimization Settings
TTS_CHUNK_SIZE = 800       # Optimized chunk size
TTS_MAX_LATENCY_OPTIMIZATION = 3  # Reduced from 4 for better quality
TTS_ENABLE_STREAMING = True
TTS_CACHE_ENABLED = True
TTS_CACHE_SIZE = 50        # Smaller cache for faster access

# Knowledge Base Optimization
KB_MAX_QUERY_TIME = 2.0    # Maximum time for KB query
KB_RESPONSE_CACHE_SIZE = 100
KB_ENABLE_PARALLEL_PROCESSING = True
KB_USE_STREAMING_RESPONSES = True

# OpenAI Settings
OPENAI_MAX_TOKENS = 150    # Reduced for faster responses
OPENAI_TEMPERATURE = 0.7   # Balanced for speed and quality

# Pinecone Settings
PINECONE_TOP_K = 3         # Reduced from 5 for faster retrieval
PINECONE_QUERY_TIMEOUT = 1.5  # Quick timeout for search

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_INTERVAL = 30  # seconds

# Connection pooling for external APIs
ENABLE_CONNECTION_POOLING = True
MAX_CONNECTIONS_PER_HOST = 10
CONNECTION_TIMEOUT = 5.0

# Parallel processing settings
MAX_WORKER_THREADS = 3
ENABLE_ASYNC_PROCESSING = True

# Buffer management
ENABLE_SMART_BUFFERING = True
BUFFER_FLUSH_INTERVAL = 0.1  # seconds

# Quality vs Speed trade-offs
PRIORITIZE_LATENCY = True    # Set to False for better quality
ADAPTIVE_QUALITY = True      # Adjust quality based on connection