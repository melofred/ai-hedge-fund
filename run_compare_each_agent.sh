#!/bin/bash

# Individual Agent vs Buy-and-Hold Comparison Runner
# Compares each agent strategy individually against buy-and-hold baseline

echo "📊 AI Hedge Fund - Individual Agent Comparison"
echo "=============================================="

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

# Default parameters (can be overridden via environment variables)
TICKERS="${TICKERS:-AAPL,MSFT,NVDA}"
START_DATE="${START_DATE:-$(date -d '12 months ago' +%Y-%m-%d)}"
END_DATE="${END_DATE:-$(date +%Y-%m-%d)}"
INITIAL_CASH="${INITIAL_CASH:-100000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/agent_vs_bh}"

# Damodaran analysis exclusion flag (cost optimization)
INCLUDE_DAMODARAN="${INCLUDE_DAMODARAN:-false}"

# Default to Ollama (local) model - using 8B for faster inference
MODEL_NAME="${MODEL_NAME:-llama3.1:8b}"
MODEL_PROVIDER="${MODEL_PROVIDER:-Ollama}"

# Check for cloud model override or custom model settings
if [ "$USE_CLOUD" = "true" ] || [ "$1" = "--cloud" ]; then
    MODEL_NAME="gpt-4o-mini"
    MODEL_PROVIDER="OpenAI"
    echo "☁️  Using cloud-based LLM: $MODEL_NAME ($MODEL_PROVIDER)"
elif [ "$MODEL_PROVIDER" = "openai" ] || [ "$MODEL_PROVIDER" = "OpenAI" ]; then
    echo "☁️  Using OpenAI model: $MODEL_NAME ($MODEL_PROVIDER)"
elif [ "$MODEL_NAME" = "gpt-4o-mini" ] || [ "$MODEL_NAME" = "gpt-4" ] || [ "$MODEL_NAME" = "gpt-3.5-turbo" ]; then
    echo "☁️  Using cloud-based LLM: $MODEL_NAME ($MODEL_PROVIDER)"
else
    echo "🧠 Using Ollama for local LLM inference: $MODEL_NAME"
fi

echo "📈 Running individual agent comparisons..."
echo "Tickers: $TICKERS"
echo "Period: $START_DATE to $END_DATE (last 12 months)"
echo "Initial Cash: \$$INITIAL_CASH"
echo "Model: $MODEL_NAME ($MODEL_PROVIDER)"
echo "Output: $OUTPUT_DIR/"

# Damodaran analysis exclusion logging
if [ "$INCLUDE_DAMODARAN" = "false" ]; then
    echo ""
    echo "⚠️  COST OPTIMIZATION: Aswath Damodaran analysis is SKIPPED"
    echo "   💰 Reason: Damodaran agent is 6-12x more expensive than other agents"
    echo "   🚀 To include Damodaran analysis, run:"
    echo "      INCLUDE_DAMODARAN=true ./run_compare_each_agent.sh"
    echo "   📊 All other agents will run normally"
    echo ""
else
    echo ""
    echo "✅ Including Aswath Damodaran analysis (high-cost mode)"
    echo ""
fi

# Build the command with optional Damodaran exclusion
CMD="$PYTHON_CMD src/compare_each_agent.py \
    --tickers \"$TICKERS\" \
    --initial-cash \"$INITIAL_CASH\" \
    --end-date \"$END_DATE\" \
    --output-dir \"$OUTPUT_DIR\" \
    --model-name \"$MODEL_NAME\" \
    --model-provider \"$MODEL_PROVIDER\""

# Add Damodaran exclusion if not including it
if [ "$INCLUDE_DAMODARAN" = "false" ]; then
    CMD="$CMD --exclude-damodaran"
fi

# Run the comparison
eval $CMD

echo ""
echo "✅ Individual agent comparison completed!"
echo "📁 Check results in: $OUTPUT_DIR/"
echo "   - each_agent_vs_buyhold.csv (summary table)"
echo "   - {agent_key}_equity.csv (per-agent equity curves)"
echo ""
echo "💡 Usage examples:"
echo "   # Use default settings (Ollama, last 12 months, Damodaran excluded)"
echo "   ./run_compare_each_agent.sh"
echo ""
echo "   # Include Damodaran analysis (high-cost mode)"
echo "   INCLUDE_DAMODARAN=true ./run_compare_each_agent.sh"
echo ""
echo "   # Use cloud model instead"
echo "   ./run_compare_each_agent.sh --cloud"
echo ""
echo "   # Custom parameters"
echo "   TICKERS=\"AAPL,GOOGL,TSLA\" START_DATE=\"2024-01-01\" ./run_compare_each_agent.sh"
