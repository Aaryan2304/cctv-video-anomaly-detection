#!/bin/bash

# Render.com Build Script
# This script is automatically executed during deployment

echo "Starting build process..."

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed successfully!"

# Create necessary directories
mkdir -p static
mkdir -p outputs

# Optional: Run any model preparation scripts
# python prepare_deployment.py

echo "Build process completed!"
