#!/bin/bash

# Render.com Build Script
# Phase 1 & 2 Production Build

echo "Starting build process..."

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed successfully!"

# Create necessary directories
mkdir -p static
mkdir -p outputs
mkdir -p utils

# Verify critical files exist
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found!"
    exit 1
fi

if [ ! -f "settings.py" ]; then
    echo "ERROR: settings.py not found!"
    exit 1
fi

# Check for trained model (warning only, not critical)
if [ ! -f "outputs/trained_model.pth" ]; then
    echo "WARNING: No trained model found at outputs/trained_model.pth"
    echo "Model will be created on first startup (may increase startup time)"
fi

echo "Build process completed successfully!"
echo "Phase 1 features: ✓ Pydantic settings, ✓ Structured logging, ✓ Batched inference"
echo "API will be available at /health, /analyze-video, /calibrate-threshold"
