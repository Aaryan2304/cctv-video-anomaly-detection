# Deployment Guide

Complete guide for using and deploying the Video Anomaly Detection system.

---

## 🎯 What This System Does

This system analyzes surveillance videos to detect unusual events (anomalies) using AI. Upload a video, and the system identifies frames where something abnormal is happening—unusual motion, unexpected objects, or patterns the model hasn't seen during training.

**Key Features:**
- **Automatic anomaly detection** in surveillance footage
- **Interactive dashboard** for visualizing results
- **Adjustable sensitivity** to match your security needs
- **Production-ready deployment** on free cloud hosting

---

## 🚀 Getting Started

You have three ways to use this system:

### Option 1: Use the Live Demo (Easiest)

**No installation required.** Access the system immediately:

1. **Visit the Dashboard:** https://video-anomaly-detection-dashboard.onrender.com
2. **Upload a video** (MP4, AVI, or MOV format)
3. **View results** instantly with interactive charts

**Note:** First analysis may take 30-60 seconds while the service wakes up (free hosting limitation).

**API Access:** If you're a developer, the REST API is available at https://video-anomaly-detection-api.onrender.com/docs

---

### Option 2: Run Locally (Full Control)

**Best for:** Testing with your own videos, offline use, or customization.

**Requirements:**
- Python 3.10 or higher
- 2GB free disk space
- Windows, Mac, or Linux

**Installation:**

```bash
# 1. Download the project
git clone https://github.com/Aaryan2304/cctv-video-anomaly-detection.git
cd cctv-video-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API backend (Terminal 1)
python app.py
# API runs at http://localhost:8000

# 4. Launch the dashboard (Terminal 2)
streamlit run dashboard.py
# Dashboard opens at http://localhost:8501
```

**What You Get:**
- Full system running on your computer
- Faster processing with GPU (if available)
- Works offline
- Complete control over all settings

---

### Option 3: Connect Local Dashboard to Cloud API

**Best for:** Using the hosted API while running the dashboard locally.

```bash
# Windows PowerShell
$env:API_URL = "https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py

# Linux/Mac
export API_URL="https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py
```

Dashboard opens at `http://localhost:8501` and sends videos to the cloud API for processing.

---

## 🔧 Configuration Details

### Render.com Setup

**Configured via `render.yaml`:**
```yaml
services:
  - type: web
    name: video-anomaly-detection-api
---

## 📖 Understanding the System

### How It Works

The system uses a **convolutional autoencoder**—a type of AI model trained to understand "normal" surveillance footage. Here's the process:

1. **Upload Video:** You provide a surveillance video (MP4, AVI, MOV)
2. **Frame Extraction:** System breaks video into individual frames
3. **AI Analysis:** Each frame is processed through the neural network
4. **Anomaly Detection:** Frames that look "different" from normal are flagged
5. **Results Display:** Interactive timeline shows exactly where anomalies occur

### What Counts as an Anomaly?

The model was trained on outdoor pedestrian footage (UCSD Ped2 dataset). It flags:
- **Unusual motion patterns** (running, erratic movement)
- **Unexpected objects** (vehicles in pedestrian zones, abandoned bags)
- **Abnormal density** (sudden crowds, empty spaces when normally busy)

**Important:** The model performs best on footage similar to its training data. For your specific cameras, you may need to retrain the model on your own footage.

### Performance Expectations

**Processing Speed:**
- **With GPU:** ~0.2 seconds per 10-second clip
- **Without GPU (CPU):** ~2-5 seconds per 10-second clip
- **Cloud (free tier):** ~5-10 seconds (first request takes 30-60 seconds while service wakes up)

**Accuracy:**
- **Precision:** 92.47% (when it flags an anomaly, it's usually correct)
- **Recall:** 83.78% (catches most real anomalies)
- **Best Use:** General surveillance, unusual event detection

---

## 🎛️ Configuration Options

### File Size and Duration Limits

**Default Limits (can be changed in settings):**
- Maximum file size: 100MB
- Maximum video duration: 5 minutes (300 seconds)
- Maximum frames: 5000

**Why These Limits?**
- Prevents server overload on free hosting
- Ensures reasonable processing times
- For longer videos, split into chunks

### Threshold Adjustment

**What is the threshold?**
The threshold determines how sensitive the system is. Higher threshold = fewer anomalies detected.

**Preset Options:**
- **Conservative (5% anomaly rate):** Low false alarms, may miss subtle events
- **Balanced (10%):** Good for general surveillance
- **Moderate (25%):** High sensitivity, more alerts
- **Sensitive (40%):** Maximum detection, expect many alerts

**Adjust in Dashboard:**
Use the slider to find the right balance for your footage without reprocessing the video.

---

## 🔧 Troubleshooting

### Common Issues

**"API connection failed" in Dashboard**
- **Cause:** API service is not running
- **Solution:** 
  - For cloud: Wait 60 seconds (service waking up)
  - For local: Check if `python app.py` is running in another terminal

**"Video processing failed"**
- **Cause:** Unsupported video format or corrupted file
- **Solution:**
  - Convert video to MP4 with H.264 codec
  - Ensure file size is under 100MB
  - Try a different video to rule out file corruption

**"High false positive rate" (many normal frames flagged)**
- **Cause:** Your footage differs significantly from training data
- **Solution:**
  - Increase threshold using the dashboard slider
  - Use "Conservative" preset
  - For production use, retrain model on your specific camera footage

**"Missing obvious anomalies"**
- **Cause:** Threshold too high
- **Solution:**
  - Decrease threshold using dashboard slider
  - Use "Sensitive" preset
  - Verify the anomaly type matches what the model was trained on

**"Slow processing on cloud"**
- **Cause:** CPU-only processing on free tier, or cold start
- **Solution:**
  - Accept 30-60 second first-request delay (free hosting limitation)
  - Run locally with GPU for faster processing
  - For production, upgrade to paid hosting with GPU

---

## 📊 Using the Results

### Interpreting the Timeline

**Blue Line:** Reconstruction error for each frame
- **Low values (0.001-0.005):** Normal frames
- **High values (0.008+):** Potential anomalies

**Red Markers:** Detected anomalies
- Click any marker to jump to that frame
- Larger markers = higher confidence anomaly

**Orange Line:** Current threshold setting
- Frames above this line are flagged as anomalies

### Exporting Data

**JSON Export:**
- Full analysis results with metadata
- Use for programmatic processing
- Integrate with other security systems

**CSV Export:**
- Frame-by-frame data for spreadsheet analysis
- Good for statistical analysis or reporting

### Integration Example

```python
import requests

# Analyze a video
with open("camera_feed.mp4", "rb") as video:
    response = requests.post(
        "https://video-anomaly-detection-api.onrender.com/analyze-video",
        files={"file": video}
    )

result = response.json()

# Check if significant anomalies detected
if result["anomaly_rate"] > 0.15:  # More than 15% anomalous frames
    print(f"⚠️ Alert: {result['anomaly_count']} anomalies detected")
    # Trigger your security protocol here
```

---

## 🆘 Getting Help

**Documentation:**
- **Dashboard Guide:** [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) - Detailed dashboard features
- **API Documentation:** Visit `/docs` on any running API instance for interactive API reference
- **Project README:** [README.md](README.md) - Technical overview and architecture

**Issues:**
- Report bugs or request features on GitHub: [Issues](https://github.com/Aaryan2304/cctv-video-anomaly-detection/issues)

**License:**
- MIT License - Free for commercial and personal use
- UCSD Ped2 dataset used for training under academic license

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
