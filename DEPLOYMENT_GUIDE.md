# Render.com Deployment Guide

## Deployment Steps

### 1. Prepare Repository
```bash
cd /path/to/project
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/anomaly-detection.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Render

1. Create account at [render.com](https://render.com)
2. New + -> Web Service
3. Connect GitHub repository
4. Configure:
   - Name: `video-anomaly-detection-api`
   - Environment: Python 3
   - Branch: `main`
   - Build Command: `./build.sh`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Create Web Service
6. API available at: `https://video-anomaly-detection-api.onrender.com`

## Alternative: Blueprint Deployment

With `render.yaml` in the repo, Render auto-configures:
- Python environment
- Build/start commands
- Health checks
- Environment variables

Click "Apply" and wait for deployment.

## Environment Configuration

### Required Environment Variables (Auto-configured):
```yaml
PYTHON_VERSION: 3.11.0
PORT: (automatically set by Render)
```

### Optional Environment Variables:
```bash
# Add these in Render dashboard if needed
MODEL_THRESHOLD=0.005069
GPU_ENABLED=false  # Render free tier is CPU-only
DEBUG=false
```

## Post-Deployment Testing

### 1. Test Health Endpoint
```bash
curl https://your-app-name.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "version": "1.0.0"
}
```

### 2. Test Web Interface
- Visit: `https://your-app-name.onrender.com`
- Upload a test video
- Verify anomaly detection works

### 3. Test API Endpoints
```bash
# Get model information
curl https://your-app-name.onrender.com/model-info

# Set threshold (example)
curl -X POST https://your-app-name.onrender.com/set-threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.008}'
```

## Troubleshooting

### Common Issues:

#### 1. **Build Fails - Missing Dependencies**
**Error**: `ModuleNotFoundError`

**Solution**: Update `requirements.txt`:
```bash
# Check if all dependencies are listed
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

#### 2. **Model Loading Fails**
**Error**: `Model file not found`

**Solution**: Ensure model files are committed:
```bash
# Check if model files are included
git add outputs/trained_model.pth
git add outputs/reconstruction_errors.npy
git commit -m "Add trained model files"
git push
```

#### 3. **Slow Cold Starts**
**Issue**: First request takes 30+ seconds

**Solution**: This is normal on Render free tier. For production:
- Upgrade to paid plan for faster cold starts
- Implement health check pinging

#### 4. **Memory Issues**
**Error**: `Out of memory`

**Solution**: Optimize for Render's memory limits:
```python
# In app.py - reduce batch processing
BATCH_SIZE = 16  # Smaller batches for CPU processing
```

## Performance Optimization for Render

### 1. **CPU Optimization** (Free Tier)
```python
# Add to app.py startup
import torch
torch.set_num_threads(2)  # Limit CPU threads
```

### 2. **Memory Management**
```python
# Process videos in smaller chunks
MAX_FRAMES_PER_BATCH = 32
```

### 3. **Caching Strategy**
```python
# Cache model loading
@lru_cache(maxsize=1)
def get_model():
    return load_model()
```

## Deployment Checklist

### Before Deployment:
- [ ] Sensitive data removed from code
- [ ] Environment variables configured
- [ ] Model files committed to repository
- [ ] requirements.txt updated
- [ ] Health check endpoint working

### After Deployment:
- [ ] Health endpoint responds
- [ ] Web interface loads
- [ ] Video analysis works
- [ ] API docs accessible at `/docs`

## Monitoring and Maintenance

### 1. **Monitor Application Logs**
- Check Render dashboard for deployment logs
- Monitor runtime errors and performance

### 2. **Set Up Alerts**
- Configure Render health check monitoring
- Set up email notifications for downtime

### 3. **Regular Updates**
```bash
# Update model or code
git add .
git commit -m "Update: description of changes"
git push

# Render automatically redeploys
```

## API Endpoints

**Base URL:** `https://your-app.onrender.com`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |
| `/analyze-video` | POST | Video analysis |
| `/set-threshold` | POST | Update threshold |
| `/calibrate-threshold` | POST | Auto-calibrate |

## Cost

**Free Tier:** 750 hours/month, CPU-only, 30-60s cold starts

**Starter ($7/month):** Faster builds, no cold starts

**Standard ($25/month):** More resources
