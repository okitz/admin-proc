#!/bin/bash

/bin/ollama serve &
pid=$!
echo "Waiting for Ollama server to start..."
sleep 5
MODEL=${OLLAMA_MODEL:-"qwen3-vl:8b"}
echo "Pulling model: $MODEL"
ollama pull $MODEL

echo "Ollama server is ready!"
wait $pid