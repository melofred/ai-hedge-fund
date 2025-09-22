#!/bin/bash

# AI Hedge Fund Backtester Script
# This script runs the backtester to test the AI Hedge Fund's performance

echo "📊 AI Hedge Fund Backtester"
echo "============================"

# Check if Python 3.11 is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create a .env file with your API keys."
    exit 1
fi

# Run the backtester
echo "🎯 Running AI Hedge Fund Backtester with AAPL, MSFT, NVDA..."
echo "============================================================="

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

$PYTHON_CMD src/backtester.py --tickers AAPL,MSFT,NVDA --ollama --start-date 2024-09-01

echo "✅ AI Hedge Fund Backtester completed!"

