# Deployment Scripts

This directory contains scripts for building and deploying the application to various platforms.

## Files

### Build Script

**`build.sh`**
- Render.com build script (Linux/cloud deployment)
- Installs dependencies via pip
- Creates necessary directories
- Validates critical files (app.py, settings.py)
- Checks for trained model
- Referenced by `render.yaml` in root directory

### Deployment Scripts

**`deploy.sh`** (Linux/Mac)
- Quick deployment to Render.com via git push
- Commits all changes with timestamp
- Pushes to main branch (triggers auto-deploy)
- Usage: `bash deployment/deploy.sh`

**`deploy.bat`** (Windows)
- Windows equivalent of deploy.sh
- Same functionality for Windows users
- Usage: `deployment\deploy.bat`

## Deployment Platforms

### Render.com (Recommended)
```bash
# One-time setup
# 1. Connect GitHub repo to Render.com
# 2. Render auto-detects render.yaml at root
# 3. Deployment happens automatically on git push

# To deploy
bash deployment/deploy.sh  # Linux/Mac
deployment\deploy.bat      # Windows
```

### Docker (Local or Cloud)
```bash
# Docker configs at root level:
# - Dockerfile
# - docker-compose.yml

# Build and run locally
docker-compose up --build

# Deploy to Docker registry
docker build -t yourusername/video-anomaly-detection .
docker push yourusername/video-anomaly-detection
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Configuration Notes

### Build Script Location
- `build.sh` is in `deployment/` but referenced from root via `render.yaml`
- Path in render.yaml: `./deployment/build.sh`
- Do not move build.sh without updating render.yaml

### Environment Variables
- Set in Render.com dashboard or `.env` file
- See `.env.example` for all available settings
- Production deployments should use environment variables, not .env files

### Model Files
- `outputs/trained_model.pth` should be tracked via Git LFS or uploaded separately
- build.sh checks for model existence and warns if missing
- Model is loaded at application startup

## Pre-Deployment Checklist

Before deploying to production:

1. ✅ Trained model exists at `outputs/trained_model.pth`
2. ✅ Environment variables configured (or .env file for local)
3. ✅ Dependencies up to date in `requirements.txt`
4. ✅ All tests passing (if you have tests)
5. ✅ Git committed and pushed to main branch

## Troubleshooting

**Build fails on Render:**
- Check build logs in Render dashboard
- Verify build.sh has execute permissions: `chmod +x deployment/build.sh`
- Ensure all required files are committed to git

**Application won't start:**
- Check that app.py and settings.py exist at root
- Verify model file is present or disable model loading for initial deploy
- Check Render logs for Python errors

**API not accessible:**
- Render.com provides HTTPS URL automatically
- Check health endpoint: `https://your-app.onrender.com/health`
- Verify PORT environment variable is set correctly

## Quick Reference

```bash
# Local development
docker-compose up

# Deploy to Render (auto-detects on push)
bash deployment/deploy.sh

# Manual deployment (any platform)
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) in root for comprehensive deployment instructions.
