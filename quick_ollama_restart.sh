#!/bin/bash
# Kill any existing Ollama process
pkill ollama
sleep 2

# Set optimized environment variables
export OLLAMA_NUM_THREAD=8
export OLLAMA_NUM_GPU_LAYERS=35
export OLLAMA_CONTEXT_SIZE=1024
export OLLAMA_BATCH_SIZE=512

# Start Ollama
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# Test
curl -s http://localhost:11434/api/tags && echo "Ollama is running!"