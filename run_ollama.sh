#!/bin/bash

# Start Ollama server in the background and keep it running
# - Checks if ollama is installed
# - Starts `ollama serve` if not already running
# - Writes logs to ./ollama.log

set -e

echo "🧠 Starting Ollama server"
echo "========================="

# Check installation
if ! command -v ollama >/dev/null 2>&1; then
  echo "❌ Ollama is not installed. Install from https://ollama.com/download"
  exit 1
fi

# Check if already running
if pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "ℹ️  Ollama server already running."
  exit 0
fi

LOG_FILE="$(pwd)/ollama.log"
echo "📄 Logging to $LOG_FILE"

# Start in the background
nohup ollama serve >> "$LOG_FILE" 2>&1 &

# Wait until it responds on default port
echo -n "⏳ Waiting for Ollama to become ready"
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "\n✅ Ollama server is ready at http://localhost:11434"
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo "\n❌ Ollama did not become ready in time. Check $LOG_FILE"
exit 1



