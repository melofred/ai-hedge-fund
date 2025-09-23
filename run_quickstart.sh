#!/bin/bash

# AI Hedge Fund Quick Start Script
# This script installs dependencies using pip and runs the AI Hedge Fund

echo "🚀 AI Hedge Fund Quick Start"
echo "=============================="

# Check if Python 3.11 is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    PIP_CMD="pip3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Install dependencies
echo "📦 Installing dependencies..."
$PIP_CMD install --user \
    langchain \
    langchain-anthropic \
    langchain-groq \
    langchain-openai \
    langchain-deepseek \
    langchain-ollama \
    langgraph \
    pandas \
    numpy \
    python-dotenv \
    matplotlib \
    tabulate \
    colorama \
    questionary \
    rich \
    langchain-google-genai \
    fastapi \
    fastapi-cli \
    pydantic \
    httpx \
    sqlalchemy \
    alembic \
    langchain-gigachat \
    langchain-xai

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ .env file created. Please edit it with your API keys."
    else
        echo "❌ .env.example file not found. Please create a .env file with your API keys."
        exit 1
    fi
fi

# Run the AI Hedge Fund
echo "🎯 Running AI Hedge Fund with AAPL, MSFT, NVDA..."
echo "=================================================="

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

$PYTHON_CMD src/main.py --ticker AAPL,MSFT,NVDA --ollama

echo "✅ AI Hedge Fund run completed!"
