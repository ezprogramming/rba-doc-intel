#!/bin/bash
# Run embedding service natively on M-series Mac with MPS acceleration
# This provides better performance than Docker on Apple Silicon

set -e

echo "🚀 Starting local embedding service with M4 GPU acceleration..."

# Enable MPS fallback for operations not yet optimized for Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Model configuration
export MODEL_ID="${MODEL_ID:-nomic-ai/nomic-embed-text-v1.5}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"

# Change to embedding directory
cd "$(dirname "$0")/../docker/embedding" || exit 1

# Ensure dependencies are installed
echo "📦 Checking dependencies..."
if ! uv pip list | grep -q "sentence-transformers"; then
    echo "Installing embedding service dependencies..."
    uv pip install fastapi uvicorn sentence-transformers torch
fi

# Start the server
echo "✅ Starting embedding server on http://0.0.0.0:8000"
echo "   Model: $MODEL_ID"
echo "   Batch size: $EMBEDDING_BATCH_SIZE"
echo "   Device: MPS (Metal Performance Shaders)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uv run uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
