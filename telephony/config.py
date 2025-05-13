"""
Optimized configuration for MULAW support and reduced latency.
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

# Audio Configuration - Optimized for MULAW and low latency
SAMPLE_RATE_TWILIO = 8000    # Native Twilio rate
SAMPLE_RATE_AI = 8000        # Changed to match Twilio (no resampling needed)
CHUNK_SIZE = 800             # Increased for MULAW (100ms at 8kHz)
AUDIO_BUFFER_SIZE = 3200     # Reduced for faster response (400ms buffer)
MAX_BUFFER_SIZE = 8000       # Reduced max buffer (1s max)

# WebSocket Configuration - Optimized
WS_PING_INTERVAL = 5         # Reduced ping interval
WS_PING_TIMEOUT = 3          # Reduced timeout
WS_MAX_MESSAGE_SIZE = 1048576

# Performance Settings - Optimized for MULAW
SILENCE_THRESHOLD = 0.015    # Adjusted for MULAW
SILENCE_DURATION = 0.6       # Faster response trigger
MAX_CALL_DURATION = 3600
MAX_PROCESSING_TIME = 2.0    # Reduced max processing time

# Response Settings - Faster
RESPONSE_TIMEOUT = 2.0       # Reduced timeout
MIN_TRANSCRIPTION_LENGTH = 1 # Allow shorter transcriptions

# Noise Filtering - Optimized for MULAW
HIGH_PASS_FILTER = 100       # Adjusted for 8kHz
NOISE_GATE_THRESHOLD = 0.02  # Adjusted for MULAW
ENABLE_NOISE_FILTERING = True

# STT Optimization - Google Cloud with MULAW
STT_LANGUAGE = "en-US"
STT_MODEL = "telephony"      # Specialized telephony model
STT_ENCODING = "MULAW"       # Direct MULAW support
STT_SAMPLE_RATE = 8000       # Match Twilio
STT_USE_ENHANCED = True      # Enhanced telephony model
STT_INTERIM_RESULTS = True   # Enable interim results
STT_SINGLE_UTTERANCE = False # Allow continuous conversation

# Speech contexts for better recognition
STT_SPEECH_CONTEXTS = [
    "pricing", "plan", "cost", "subscription", "service", "features",
    "support", "upgrade", "payment", "account", "question", "help",
    "basic", "professional", "enterprise", "dollars", "month", "year"
]

# TTS Optimization - ElevenLabs with MULAW output
TTS_VOICE_ID = os.getenv('TTS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL')
TTS_MODEL_ID = os.getenv('TTS_MODEL_ID', 'eleven_turbo_v2')
TTS_OUTPUT_FORMAT = "ulaw_8000"         # Direct MULAW output
TTS_OPTIMIZE_LATENCY = 3                # High optimization
TTS_STABILITY = 0.4                     # Optimized for clarity
TTS_CLARITY = 0.7                       # High clarity for telephony
TTS_STYLE = 0.1                         # Minimal style
TTS_ENABLE_CACHING = True               # Cache for faster response
TTS_CHUNK_SIZE = 400                    # Optimized for 8kHz

# Knowledge Base Optimization
KB_MAX_QUERY_TIME = 1.5                 # Reduced query time
KB_RESPONSE_CACHE_SIZE = 100            # Response caching
KB_ENABLE_STREAMING = True              # Enable streaming responses
KB_PARALLEL_PROCESSING = True           # Enable parallel operations
KB_MAX_CONTEXT_LENGTH = 2000            # Limit context for speed

# OpenAI Settings - Optimized
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
OPENAI_MAX_TOKENS = 200                 # Reduced for faster response
OPENAI_TEMPERATURE = 0.7                # Balanced
OPENAI_TIMEOUT = 5.0                    # API timeout

# Pinecone Settings - Optimized
PINECONE_TOP_K = 3                      # Reduced for speed
PINECONE_QUERY_TIMEOUT = 1.0            # Quick timeout
PINECONE_MIN_SCORE = 0.7                # Higher threshold

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_INTERVAL = 30           # seconds
TRACK_PROCESSING_TIMES = True
MAX_PROCESSING_HISTORY = 50             # Keep recent processing times

# Connection Pooling
ENABLE_CONNECTION_POOLING = True
MAX_CONNECTIONS_PER_HOST = 10
CONNECTION_TIMEOUT = 3.0                # Reduced timeout
TCP_KEEPALIVE = True
TCP_KEEPALIVE_IDLE = 30
TCP_KEEPALIVE_INTERVAL = 5
TCP_KEEPALIVE_PROBES = 3

# Parallel Processing
MAX_WORKER_THREADS = 2                  # Reduced for better resource usage
ENABLE_ASYNC_PROCESSING = True
THREAD_POOL_SIZE = 4

# Buffer Management - Optimized for MULAW
ENABLE_SMART_BUFFERING = True
BUFFER_FLUSH_INTERVAL = 0.05            # Faster flushing (50ms)
ADAPTIVE_BUFFERING = True
MIN_CHUNK_SIZE = 320                    # 40ms at 8kHz

# Quality vs Speed Trade-offs
PRIORITIZE_LATENCY = True               # Prioritize speed over quality
ADAPTIVE_QUALITY = True                 # Adjust based on performance
QUALITY_DEGRADATION_THRESHOLD = 2.0     # Degrade quality if processing > 2s

# MULAW Specific Settings
MULAW_COMPRESSION_ENABLED = True        # Use MULAW compression
MULAW_CHUNK_SIZE = 800                  # Optimal MULAW chunk size
MULAW_BUFFER_SIZE = 3200                # MULAW buffer size
MULAW_NOISE_THRESHOLD = 0.02            # MULAW-specific noise threshold

# Error Handling
ENABLE_FALLBACK_RESPONSES = True        # Enable fallback when errors occur
MAX_RETRIES = 2                         # Reduced retries for speed
RETRY_DELAY = 0.1                       # Quick retry
CIRCUIT_BREAKER_ENABLED = True          # Prevent cascade failures
CIRCUIT_BREAKER_THRESHOLD = 5           # Failures before circuit opens

# Security
ENABLE_CORS = os.getenv('ENABLE_CORS', 'True').lower() == 'true'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

# Health Check
HEALTH_CHECK_ENABLED = True
HEALTH_CHECK_INTERVAL = 30              # seconds
HEALTH_CHECK_TIMEOUT = 5.0              # timeout for health checks

# Feature Flags
ENABLE_ECHO_DETECTION = True            # Detect and ignore echo
ENABLE_BARGE_IN = True                  # Allow user to interrupt AI
ENABLE_VOICE_ACTIVITY_DETECTION = True  # VAD for better experience
ENABLE_NOISE_SUPPRESSION = True         # Suppress background noise

# Development/Debug Settings
SAVE_AUDIO_FILES = DEBUG                # Save audio files when debugging
AUDIO_SAVE_PATH = "./debug_audio"       # Path for saved audio files
ENABLE_METRICS_EXPORT = True            # Export metrics for monitoring
METRICS_EXPORT_INTERVAL = 60            # seconds