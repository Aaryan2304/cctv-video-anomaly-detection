#!/bin/bash

# Quick Deploy Script for Render
# Run this script to commit and push changes for deployment

echo "🚀 Preparing for Render deployment..."

# Add all files
git add .

# Commit with timestamp
commit_message="Deploy: $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$commit_message"

echo "📦 Files committed with message: $commit_message"

# Push to main branch
git push origin main

echo "✅ Pushed to GitHub - Render will automatically deploy!"
echo "🔗 Check your Render dashboard for deployment progress"
echo "📱 Your API will be live at: https://your-app-name.onrender.com"
