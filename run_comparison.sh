#!/bin/bash

# Strategy Comparison Runner
# Compares agent-driven trading vs buy-and-hold strategy

echo "📊 AI Hedge Fund Strategy Comparison"
echo "===================================="

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

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default parameters (can be overridden)
TICKERS="${TICKERS:-AAPL,MSFT,NVDA}"
START_DATE="${START_DATE:-2024-11-01}"
END_DATE="${END_DATE:-2024-12-01}"
INITIAL_CASH="${INITIAL_CASH:-100000}"
# AGENTS is now optional - if not set, will prompt for selection
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
MODEL_PROVIDER="${MODEL_PROVIDER:-OpenAI}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

# Check for Ollama flag
OLLAMA_FLAG=""
if [ "$USE_OLLAMA" = "true" ] || [ "$1" = "--ollama" ]; then
    OLLAMA_FLAG="--ollama"
    MODEL_NAME="llama3.1:latest"
    MODEL_PROVIDER="Ollama"
    echo "🧠 Using Ollama for local LLM inference"
fi

echo "📈 Running strategy comparison..."
echo "Tickers: $TICKERS"
if [ -n "$AGENTS" ]; then
    echo "Agents: $AGENTS"
else
    echo "Agents: (will prompt for selection)"
fi
echo "Period: $START_DATE to $END_DATE"
echo "Initial Cash: \$$INITIAL_CASH"
echo "Model: $MODEL_NAME ($MODEL_PROVIDER)"
echo "Output: $OUTPUT_DIR/"
echo ""

# Build the command
CMD="$PYTHON_CMD src/compare_strategies.py \
    --tickers \"$TICKERS\" \
    --start-date \"$START_DATE\" \
    --end-date \"$END_DATE\" \
    --initial-cash \"$INITIAL_CASH\" \
    --model-name \"$MODEL_NAME\" \
    --model-provider \"$MODEL_PROVIDER\" \
    --output-dir \"$OUTPUT_DIR\""

# Add agents if provided
if [ -n "$AGENTS" ]; then
    CMD="$CMD --agents \"$AGENTS\""
fi

# Add Ollama flag if set
if [ -n "$OLLAMA_FLAG" ]; then
    CMD="$CMD $OLLAMA_FLAG"
fi

# Run the comparison
eval $CMD

echo ""
echo "✅ Strategy comparison completed!"
echo "📁 Check results in: $OUTPUT_DIR/"
echo "   - strategy_comparison.png (plot)"
echo "   - portfolio_comparison.csv (data)"
echo "   - performance_metrics.csv (metrics)"

