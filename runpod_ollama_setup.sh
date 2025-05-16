#!/bin/bash
# Ollama optimization script for Runpod (no sudo required)

echo "Optimizing Ollama for ultra low latency on Runpod..."

# Set environment variables for Ollama
export OLLAMA_NUM_THREAD=8
export OLLAMA_NUM_GPU_LAYERS=35
export OLLAMA_CONTEXT_SIZE=1024
export OLLAMA_BATCH_SIZE=512
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=24h

# Check if Ollama is running
if pgrep ollama > /dev/null; then
    echo "Stopping existing Ollama process..."
    pkill ollama
    sleep 3
fi

# Start Ollama with optimized settings in background
echo "Starting optimized Ollama..."
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 5

# Verify Ollama is running
if ! pgrep ollama > /dev/null; then
    echo "Failed to start Ollama. Check /tmp/ollama.log for errors"
    exit 1
fi

# Test connection
echo "Testing Ollama connection..."
max_retries=10
retry=0
while [ $retry -lt $max_retries ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Ollama is responding!"
        break
    else
        echo "Waiting for Ollama to respond... ($((retry + 1))/$max_retries)"
        sleep 2
        retry=$((retry + 1))
    fi
done

if [ $retry -eq $max_retries ]; then
    echo "Ollama failed to respond after $max_retries attempts"
    exit 1
fi

# Check if model exists, pull if needed
echo "Checking for model: mistral:7b-instruct-v0.2-q4_0"
if ! ollama list | grep -q "mistral:7b-instruct-v0.2-q4_0"; then
    echo "Model not found. Pulling mistral:7b-instruct-v0.2-q4_0..."
    ollama pull mistral:7b-instruct-v0.2-q4_0
fi

# Pre-warm the model for faster first response
echo "Pre-warming model for ultra low latency..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-v0.2-q4_0",
    "prompt": "Test",
    "stream": false,
    "options": {
      "num_ctx": 1024,
      "temperature": 0.1,
      "top_k": 1,
      "top_p": 0.1,
      "num_predict": 10
    }
  }' > /dev/null 2>&1

echo "Ollama optimization complete!"
echo "Model loaded and ready for ultra low latency inference"

# Optional: Show Ollama status
echo "=== Ollama Status ==="
echo "Process ID: $(pgrep ollama)"
echo "Available models:"
ollama list

# Optional: Show system resources
echo "=== System Resources ==="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
fi