#!/bin/bash

# Frontend Runner for AI Hedge Fund
# - Installs deps with pnpm (preferred) or npm
# - Sets VITE_API_URL (defaults to http://localhost:8000)
# - Starts Vite dev server in app/frontend

set -e

echo "🖥️  Starting Frontend (Vite)"
echo "============================"

FRONTEND_DIR="app/frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
  echo "❌ Frontend directory not found at $FRONTEND_DIR"
  exit 1
fi

# Choose package manager
if command -v pnpm >/dev/null 2>&1; then
  PM="pnpm"
elif command -v npm >/dev/null 2>&1; then
  PM="npm"
else
  echo "❌ Neither pnpm nor npm is installed. Please install one of them."
  exit 1
fi

echo "✅ Using package manager: $PM"

cd "$FRONTEND_DIR"

# Install dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  if [ "$PM" = "pnpm" ]; then
    pnpm install --frozen-lockfile || pnpm install
  else
    npm install --no-fund --no-audit --loglevel=error
  fi
  echo "✅ Dependencies installed."
else
  echo "ℹ️  node_modules exists, skipping install."
fi

# Set API URL
VITE_API_URL_DEFAULT="http://localhost:8000"
export VITE_API_URL="${VITE_API_URL:-$VITE_API_URL_DEFAULT}"
echo "🔗 VITE_API_URL=$VITE_API_URL"

# Start dev server
echo "🚀 Launching Vite dev server on http://localhost:5173"
if [ "$PM" = "pnpm" ]; then
  pnpm run dev
else
  npm run dev --silent
fi



