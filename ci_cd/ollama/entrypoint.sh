#!/bin/bash
# Start Ollama server in background, pull required models, then keep server running.
#
# Two models are used:
#   OLLAMA_MODEL       (default: llama3.2)   — heavy tasks: generation, grounding
#   OLLAMA_LIGHT_MODEL (default: llama3.2:1b) — light tasks: query translation
set -e

MODEL="${OLLAMA_MODEL:-llama3.2}"
LIGHT_MODEL="${OLLAMA_LIGHT_MODEL:-llama3.2:1b}"

echo "[ollama-init] Starting Ollama server..."
ollama serve &
SERVER_PID=$!

echo "[ollama-init] Waiting for server to be ready..."
until ollama list >/dev/null 2>&1; do
    sleep 2
done

echo "[ollama-init] Pulling heavy model: ${MODEL}"
ollama pull "${MODEL}"

echo "[ollama-init] Pulling light model: ${LIGHT_MODEL}"
ollama pull "${LIGHT_MODEL}"

echo "[ollama-init] All models ready."

wait $SERVER_PID
