# Deployment Information

This system is deployed as two separate services that work together to provide video anomaly detection.

---

## 🌐 Live Services

### 1. API Service
**URL:** https://video-anomaly-detection-api.onrender.com

**What it does:**
- Processes uploaded videos and detects anomalies
- Provides REST API for programmatic access
- Hosts interactive API documentation at `/docs`

**Who should use it:**
- Developers integrating anomaly detection into their applications
- Users wanting to test API endpoints directly
- Anyone needing programmatic access to the system

### 2. Dashboard Service
**URL:** https://video-anomaly-detection-dashboard.onrender.com

**What it does:**
- User-friendly interface for uploading and analyzing videos
- Interactive timeline showing anomaly detection results
- Adjustable sensitivity controls
- Frame-by-frame viewer with export capabilities

**Who should use it:**
- Non-technical users who need to analyze surveillance footage
- Anyone preferring a visual interface over API calls
- Users wanting to explore results interactively

---

## 💡 Why Two Services?

**Separation of Concerns:**
- API handles all the heavy processing (AI model, video analysis)
- Dashboard provides the user interface
- Each can be scaled independently based on usage

**Flexibility:**
- Use the dashboard for interactive analysis
- Use the API to integrate into your own applications
- Run dashboard locally while using the cloud API

---

## 🚀 Quick Start

### Using the Live Services

**Option 1: Dashboard (Easiest)**
1. Visit https://video-anomaly-detection-dashboard.onrender.com
2. Upload your video
3. View results with interactive charts

**Option 2: API (For Developers)**
1. Visit https://video-anomaly-detection-api.onrender.com/docs
2. Try the `/analyze-video` endpoint
3. Upload a video and receive JSON results

### Running Locally

**Both Services:**
```bash
# Terminal 1: Start API
python app.py
# Runs at http://localhost:8000

# Terminal 2: Start Dashboard
streamlit run dashboard.py
# Runs at http://localhost:8501
```

**Dashboard Only (connecting to cloud API):**
```bash
# Windows
$env:API_URL = "https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py

# Linux/Mac
export API_URL="https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py
```

---

## 📝 Important Notes

**First Request Delay:**
- Free hosting services "sleep" after 15 minutes of inactivity
- First request may take 30-60 seconds while service wakes up
- Subsequent requests are fast (< 5 seconds)

**File Limitations:**
- Maximum file size: 100MB
- Maximum video duration: 5 minutes
- Supported formats: MP4, AVI, MOV

**Processing Speed:**
- Cloud (CPU only): 5-10 seconds per 10-second video
- Local with GPU: ~0.2 seconds per 10-second video
- Local without GPU: 2-5 seconds per 10-second video

---

## 🔗 Additional Resources

- **Complete Deployment Guide:** See [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for detailed setup instructions
- **Dashboard User Guide:** See [DASHBOARD_GUIDE.md](../DASHBOARD_GUIDE.md) for feature explanations
- **Project Overview:** See main [README.md](../README.md) for technical details
