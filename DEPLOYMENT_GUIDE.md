# Deployment Guide

Complete guide for deploying the Video Anomaly Detection system in development and production environments.

---

## 📋 Deployment Options

### Option 1: API on Render + Dashboard Local (Recommended)
- **API:** Hosted on Render.com (auto-deploy)
- **Dashboard:** Users run locally, connects to production API
- **Cost:** Free tier
- **Best for:** Public demos, portfolio projects

### Option 2: Full Local Deployment
- **API + Dashboard:** Both run on developer machine
- **Cost:** Free
- **Best for:** Development, testing, offline use

### Option 3: Full Cloud Deployment
- **API:** Render.com
- **Dashboard:** Streamlit Cloud
- **Cost:** Free tier for both
- **Best for:** Production use, always-on access

---

## 🚀 Quick Start (Option 1 - Recommended)

### Prerequisites
```bash
# Check Python version
python --version  # Should be 3.10+

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Deploy API to Render

**Automatic Deployment:**
1. Push code to GitHub
2. Render auto-detects `render.yaml`
3. Deploys API automatically

```bash
git add .
git commit -m "deploy: production ready"
git push origin main
```

**Manual Deployment (First Time):**
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo: `Aaryan2304/cctv-video-anomaly-detection`
4. Render auto-configures from `render.yaml`
5. Click "Create Web Service"

**Your API will be available at:**
```
https://video-anomaly-detection-api.onrender.com
```

### Step 2: Run Dashboard Locally

```bash
# Set production API URL
# Windows PowerShell
$env:API_URL = "https://video-anomaly-detection-api.onrender.com"

# Linux/Mac
export API_URL="https://video-anomaly-detection-api.onrender.com"

# Launch dashboard
streamlit run dashboard.py
```

Dashboard opens at `http://localhost:8501` and connects to your production API.

---

## 🔧 Configuration Details

### Render.com Setup

**Configured via `render.yaml`:**
```yaml
services:
  - type: web
    name: video-anomaly-detection-api
    env: python
    plan: free
    buildCommand: "./build.sh"
    startCommand: "uvicorn app:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: "/health"
```

**Build Process (`build.sh`):**
1. Installs dependencies from `requirements.txt`
2. Creates output directories
3. Verifies critical files exist
4. Checks for trained model (warning if missing)

**Environment Variables (Auto-configured):**
- `PYTHON_VERSION`: 3.11.0
- `PORT`: Assigned by Render

**Optional Variables (Add in Render Dashboard):**
- `APP_DEBUG`: false
- `APP_LOG_LEVEL`: INFO
- `APP_MAX_FILE_SIZE_MB`: 100
- `APP_MAX_VIDEO_DURATION_SEC`: 300

### Dashboard Configuration

**API Connection Priority:**
1. Streamlit secrets (`.streamlit/secrets.toml`)
2. Environment variable (`API_URL`)
3. Default (`http://localhost:8000`)

**Create `.streamlit/secrets.toml`:**
```toml
API_URL = "https://video-anomaly-detection-api.onrender.com"
```

---

## 📦 Deployment Workflows

### Development Workflow

```bash
# Terminal 1: Run API locally
python app.py

# Terminal 2: Run dashboard
streamlit run dashboard.py
# Auto-connects to localhost:8000
```

### Production Workflow (Option 1)

```bash
# 1. Code changes
git add .
git commit -m "feat: your feature"
git push origin main

# 2. Render auto-deploys API (2-3 minutes)
# Monitor: Render Dashboard → Logs

# 3. Users run dashboard locally
$env:API_URL = "https://your-api.onrender.com"
streamlit run dashboard.py
```

### Testing Production Deployment

```bash
# Test API health
curl https://video-anomaly-detection-api.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "version": "2.0.0"
}

# Test API docs
# Visit: https://video-anomaly-detection-api.onrender.com/docs
```

---

## 🐳 Docker Deployment

### Build and Run API

```bash
# Build image
docker build -t anomaly-detector .

# Run container
docker run -p 8000:8000 anomaly-detector

# API available at http://localhost:8000
```

### Run Dashboard with Docker API

```bash
# Terminal 1: Docker API running on port 8000

# Terminal 2: Dashboard
streamlit run dashboard.py
# Auto-connects to localhost:8000
```

### Docker Compose (Full Stack)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - APP_DEBUG=false

  # Note: Dashboard not in compose (users run locally)
```

```bash
docker-compose up
```

---

## ☁️ Streamlit Cloud Deployment (Optional)

For fully hosted dashboard:

### Setup

1. **Fork Repository**
   - Fork `Aaryan2304/cctv-video-anomaly-detection` on GitHub

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your forked repo
   - Main file: `dashboard.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud dashboard
   - Settings → Secrets
   - Add:
     ```toml
     API_URL = "https://video-anomaly-detection-api.onrender.com"
     ```

4. **Access Dashboard**
   - URL: `https://yourapp.streamlit.app`
   - No sleep on free tier (unlike Render)

---

## 📊 Monitoring & Maintenance

### Health Monitoring

**Render.com:**
- Automatic health checks every 30 seconds
- Endpoint: `GET /health`
- Email alerts on failure (configure in dashboard)

**Manual Monitoring:**
```bash
# Check API status
curl https://your-api.onrender.com/health

# Check logs
# Render Dashboard → Service → Logs (real-time)
```

### Performance Metrics

**Free Tier Limitations:**
- **Render:** Spins down after 15 min inactivity
- **Cold start:** 30-60 seconds on first request
- **RAM:** 512 MB
- **CPU:** Shared, no GPU

**Expected Performance:**
- API startup: 30-60s (cold start), <5s (warm)
- Video processing: 0.1-2s per video (CPU)
- Dashboard load: <2s (local), ~5s (Streamlit Cloud)

### Logs

**Structured JSON Logs (Phase 1 Feature):**
```json
{
  "timestamp": "2026-01-12T10:15:26Z",
  "level": "INFO",
  "message": "POST /analyze-video -> 200 (1.2s)",
  "request_id": "5ecdd3f9",
  "status_code": 200,
  "duration_ms": 1234.5
}
```

**View Logs:**
- Render: Dashboard → Logs tab
- Local: Terminal output
- Docker: `docker logs <container_id>`

---

## 🔄 Update Procedures

### Code Updates

```bash
# 1. Make changes locally
git add .
git commit -m "fix: bug description"

# 2. Push to GitHub
git push origin main

# 3. Render auto-deploys
# Monitor progress in Render Dashboard
```

### Model Updates

**Small Models (<100MB):**
```bash
# Train new model
python main.py --mode custom --data_path your_data/

# Commit to Git
git add outputs/trained_model.pth
git commit -m "model: retrained on new dataset"
git push origin main
```

**Large Models (>100MB):**
```bash
# Use Git LFS
git lfs install
git lfs track "outputs/trained_model.pth"
git add .gitattributes outputs/trained_model.pth
git commit -m "model: add via Git LFS"
git push origin main

# Or upload manually via Render Shell
```

### Rollback

**Render Dashboard:**
1. Go to Deployments tab
2. Select previous successful deployment
3. Click "Redeploy"

**Git Rollback:**
```bash
git revert HEAD
git push origin main
```

---

## 🐛 Troubleshooting

### API Issues

**Problem: Build fails with "Module not found"**
```bash
# Solution: Update requirements.txt
pip freeze > requirements.txt
git add requirements.txt
git commit -m "fix: update dependencies"
git push
```

**Problem: API returns 500 errors**
```bash
# Check logs in Render Dashboard
# Look for Python exceptions

# Common causes:
# 1. Missing model file
# 2. Incorrect settings.py configuration
# 3. Out of memory (reduce batch size)
```

**Problem: Cold start takes >60 seconds**
```bash
# This is normal on free tier
# Solutions:
# 1. Upgrade to paid tier ($7/month for no cold starts)
# 2. Implement keep-alive ping
# 3. Accept the limitation for free hosting
```

### Dashboard Issues

**Problem: "API connection failed"**
```bash
# Check API is running
curl https://your-api.onrender.com/health

# Check API_URL environment variable
echo $API_URL  # Linux/Mac
echo $env:API_URL  # Windows PowerShell

# Verify CORS (already configured in app.py)
```

**Problem: "Cannot read video file"**
```bash
# Cause: File uploaded twice (file pointer issue)
# Fixed in dashboard.py Phase 2

# If still occurs:
# 1. Check video codec (use H.264)
# 2. Verify file size <100MB
# 3. Test with different video
```

**Problem: Streamlit shows import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Activate correct environment
conda activate gpu  # or your env name
```

### Docker Issues

**Problem: Container won't start**
```bash
# Check logs
docker logs <container_id>

# Common causes:
# 1. Port 8000 already in use
docker ps  # Check running containers
# 2. Missing files in image
docker exec -it <container_id> ls /app
```

**Problem: Out of memory in container**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory

# Or reduce batch size in settings.py
APP_BATCH_SIZE=32
```

---

## 📈 Scaling & Optimization

### Render Free Tier Optimizations

**Reduce Cold Start Time:**
```python
# In app.py, lazy load heavy dependencies
import torch  # Load on startup
# vs
def get_model():
    import torch  # Load when needed
```

**Memory Optimization:**
```python
# In settings.py
batch_size: int = 32  # Lower for free tier
max_frames: int = 1000  # Limit video length
```

**Keep-Alive (Optional):**
```bash
# Ping health endpoint every 10 minutes
# Prevents cold starts for active hours

# Use external service like UptimeRobot (free)
# Monitor: https://your-api.onrender.com/health
```

### Upgrade Paths

| Tier | Cost | RAM | Features |
|------|------|-----|----------|
| Free | $0 | 512MB | Spins down after 15min |
| Starter | $7/mo | 512MB | Always-on, faster builds |
| Standard | $25/mo | 2GB | Better performance |
| Pro | $85/mo | 4GB | Dedicated CPU |

**When to Upgrade:**
- High traffic (>100 requests/day)
- Large videos (>5 minutes)
- Need always-on availability
- Want faster cold starts

### GPU Deployment

Render doesn't support GPU. For GPU inference:

**Option 1: Google Cloud Run**
- Supports GPUs
- Pay per request
- Auto-scaling

**Option 2: AWS Lambda + GPU**
- Serverless with GPU
- Complex setup

**Option 3: Dedicated GPU Server**
- DigitalOcean GPU droplets
- $400/month for entry GPU
- Full control

---

## ✅ Production Checklist

### Before First Deployment
- [ ] All code committed to GitHub
- [ ] `.gitignore` configured (no secrets)
- [ ] `requirements.txt` updated
- [ ] `render.yaml` configured
- [ ] Model file available (or will be created on startup)
- [ ] Health endpoint tested locally

### After Deployment
- [ ] API health check returns 200
- [ ] API docs accessible at `/docs`
- [ ] Dashboard connects to API
- [ ] Video upload works end-to-end
- [ ] Logs show structured JSON format
- [ ] No errors in Render logs

### Security Checklist
- [ ] No API keys in code
- [ ] `.env` in `.gitignore`
- [ ] CORS configured correctly
- [ ] File size limits enforced
- [ ] Video duration limits enforced
- [ ] Consider adding authentication (for sensitive data)

---

## 📚 API Endpoints Reference

**Base URL:** `https://video-anomaly-detection-api.onrender.com`

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | Web interface (old HTML) | None |
| `/health` | GET | Health check + model info | None |
| `/docs` | GET | Swagger UI | None |
| `/redoc` | GET | ReDoc API docs | None |
| `/analyze-video` | POST | Video anomaly detection | None |
| `/calibrate-threshold` | POST | Auto-calibrate threshold | None |
| `/set-threshold-preset` | POST | Set preset threshold | None |

**Example API Call:**
```bash
curl -X POST "https://your-api.onrender.com/analyze-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_video.mp4"
```

---

## 🎯 Next Steps

1. **Deploy API to Render** (5 minutes)
2. **Test with dashboard locally** (2 minutes)
3. **Share API URL** with users
4. **Monitor logs** for first few days
5. **Optional:** Deploy dashboard to Streamlit Cloud
6. **Optional:** Add authentication for production use

---

**Last Updated:** January 12, 2026  
**Deployment Strategy:** Option 1 (API on Render, Dashboard local)  
**Status:** Production Ready ✅
