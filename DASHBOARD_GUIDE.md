# Dashboard User Guide

## Streamlit Dashboard for Video Anomaly Detection

Production-grade UI built with Streamlit, featuring real-time analysis, interactive visualizations, and dynamic threshold adjustment.

---

## Features

### 1. Video Upload & Analysis
- Drag-and-drop interface for MP4, AVI, MOV files
- Real-time progress feedback during GPU processing
- Automatic API health check before analysis
- File validation (size, duration, format)

### 2. Interactive Anomaly Timeline
- Plotly-powered line chart showing reconstruction errors
- Red markers highlighting detected anomalies
- Orange threshold line (adjustable)
- Green vertical line tracking current frame
- Click any point to jump to that frame
- Zoom/pan controls for detailed inspection

### 3. Dynamic Threshold Adjustment
- Client-side slider for instant sensitivity control
- Recalculates anomaly flags without re-running GPU inference
- Real-time metrics update (anomaly count, rate)
- Reset button to restore original threshold
- Range: min to max reconstruction error from analysis

### 4. Frame-by-Frame Viewer
- Navigation controls: First/Previous/Next/Last buttons
- Frame slider for quick seeking
- Current frame display with anomaly highlighting (red border)
- Reconstruction error score per frame
- Status indicator (Normal vs Anomaly)

### 5. Anomaly Thumbnails
- Grid view of up to 10 detected anomalies
- Clickable thumbnails to jump to specific frames
- Automatic thumbnail generation from video frames

### 6. Export Capabilities
- **JSON Export**: Full analysis results with metadata
  - Frame count, anomaly count, anomaly rate
  - Threshold, device, processing time
  - Per-frame scores and flags
- **CSV Export**: Frame-level data for spreadsheet analysis
  - Columns: frame_number, reconstruction_error, is_anomaly
  - Compatible with Excel, Pandas, R

---

## Quick Start

### Prerequisites
1. Python 3.10+ environment
2. Dependencies installed: `pip install -r requirements.txt`
3. Trained model at `outputs/trained_model.pth`

### Step 1: Start FastAPI Backend
```bash
# Terminal 1: Start API server
python app.py

# You should see:
# INFO: Application startup complete.
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Launch Dashboard
```bash
# Terminal 2: Start Streamlit dashboard
streamlit run dashboard.py

# You should see:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### Step 3: Analyze Videos
1. Open browser to http://localhost:8501
2. Check sidebar for "✓ API Connected" status
3. Upload video file (max 100MB by default)
4. Click "🔍 Analyze Video"
5. Wait for processing (progress bar shown)
6. Explore results using interactive features

---

## Configuration

### API Connection
By default, dashboard connects to `http://localhost:8000`. To change:

**Option 1: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:API_URL = "http://your-api-host:8000"
streamlit run dashboard.py

# Linux/Mac
export API_URL="http://your-api-host:8000"
streamlit run dashboard.py
```

**Option 2: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
API_URL = "http://your-api-host:8000"
```

**Priority:** Streamlit secrets → Environment variable → Default (localhost:8000)

### Dashboard Settings
Edit page configuration in `dashboard.py`:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎥",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded",  # or "collapsed"
)
```

---

## Workflow Examples

### Example 1: Analyze CCTV Footage
1. Upload `cctv_sample.mp4`
2. Wait for analysis (~2-5 seconds for 10-sec clip on RTX 3050)
3. Review metrics: 300 frames, 8 anomalies (2.7% rate)
4. Examine timeline for anomaly spikes
5. Click spike → Jump to frame 145
6. See red border indicating anomaly
7. Export results as JSON for archival

### Example 2: Adjust Sensitivity
1. After initial analysis, notice 15% anomaly rate (too sensitive)
2. Move threshold slider right (increase threshold)
3. Watch metrics update: 15% → 5% anomaly rate
4. Inspect remaining anomalies to confirm quality
5. Export adjusted results as CSV for reporting

### Example 3: Frame-by-Frame Investigation
1. Upload suspicious video segment
2. Click largest spike in timeline (frame 87)
3. Use Previous/Next buttons to review context
4. Compare frames 85-89 for anomaly pattern
5. Screenshot anomaly frame for incident report
6. Export full analysis for security team

---

## Technical Architecture

### Data Flow
```
User Upload → Streamlit → FastAPI (/analyze-video) → GPU Inference → JSON Response
                  ↓
         Session State Storage
                  ↓
    Plotly Chart + Frame Viewer + Export
```

### Session State Management
- `analysis_result`: API response with scores/flags
- `video_path`: Temporary file path for frame extraction
- `current_threshold`: User-adjusted threshold value
- `current_frame`: Selected frame index (0-based)
- `video_frames`: Extracted RGB frames (lazy loaded)

### Client-Side Processing
- **Threshold adjustment**: Pure Python recalculation, no API call
- **Frame navigation**: Local frame array, no video re-read
- **Chart updates**: Plotly reactivity without full re-render
- **Export generation**: In-memory JSON/CSV creation

---

## Performance Tips

### For Large Videos (>5 minutes)
- API enforces 5-minute limit by default (change in `settings.py`)
- Split long videos using ffmpeg before upload:
  ```bash
  ffmpeg -i long_video.mp4 -t 300 -c copy segment1.mp4
  ```

### For Slow Networks
- Dashboard streams video frames after API analysis
- Consider reducing frame extraction quality in `extract_video_frames()`
- Or skip frame viewer entirely for analysis-only workflow

### For Multiple Users
- Run FastAPI with multiple workers: `uvicorn app:app --workers 4`
- Dashboard is stateless; each user gets isolated session
- API handles concurrent requests via ThreadPoolExecutor

---

## Troubleshooting

### "API connection failed"
- Ensure FastAPI is running: `curl http://localhost:8000/health`
- Check firewall rules blocking port 8000
- Verify `API_BASE_URL` in dashboard matches API host

### "Analysis failed: File too large"
- Default limit: 100MB (configurable in `settings.py`)
- Compress video: `ffmpeg -i input.mp4 -vcodec h264 -crf 28 output.mp4`
- Or increase `max_file_size_mb` in `.env`

### "No frames displayed"
- Check video codec compatibility (use H.264/AVC)
- Install ffmpeg: `pip install ffmpeg-python`
- Verify OpenCV can read video: `cv2.VideoCapture(path).isOpened()`

### Slider not updating metrics
- Known Streamlit behavior: drag slider fully, then release
- Click elsewhere on page to trigger update
- Or use number input instead: edit `dashboard.py` L271

---

## Advanced Customization

### Change Chart Colors
Edit `create_timeline_chart()` in `dashboard.py`:
```python
line=dict(color="purple", width=2),  # Change from blue
marker=dict(color="orange", size=8),  # Change from red
```

### Add Video Playback
Streamlit doesn't natively support synced video controls. For playback:
```python
# Add after frame viewer
st.video(st.session_state.video_path, start_time=st.session_state.current_frame / fps)
```

### Custom Export Formats
Add to `export_to_*()` functions:
```python
def export_to_xml(result: Dict) -> str:
    # Generate XML format for legacy systems
    ...
```

---

## Deployment

### Production Deployment (Docker)
```dockerfile
# Add to Dockerfile
FROM python:3.11-slim
COPY dashboard.py /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Hosting (Streamlit Cloud)
1. Push code to GitHub
2. Connect at share.streamlit.io
3. Set secrets for API_URL in dashboard settings
4. Ensure API is publicly accessible

### Reverse Proxy (Nginx)
```nginx
location /dashboard/ {
    proxy_pass http://localhost:8501/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

---

## API Reference

The dashboard communicates with these FastAPI endpoints:

### GET /health
Check API status and model info.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "2.0.0"
}
```

### POST /analyze-video
Upload video for anomaly detection.

**Request:** Multipart form with `file` field  
**Response:**
```json
{
  "frame_count": 300,
  "anomaly_count": 12,
  "anomaly_rate": 0.04,
  "anomaly_scores": [0.002, 0.003, ...],
  "anomaly_flags": [false, false, ...],
  "processing_time": 2.45,
  "model_info": {
    "device": "cuda",
    "threshold": 0.00507,
    "batch_size": 64
  }
}
```

---

## FAQ

**Q: Can I run dashboard without FastAPI?**  
A: No, dashboard is a frontend client. It requires the API for model inference.

**Q: Does threshold adjustment retrain the model?**  
A: No, it only recalculates flags from cached reconstruction errors (client-side).

**Q: Can I analyze multiple videos simultaneously?**  
A: Each dashboard session handles one video at a time. Open multiple browser tabs for parallel analysis.

**Q: How do I change the anomaly detection model?**  
A: Replace `outputs/trained_model.pth` and restart FastAPI. Dashboard is model-agnostic.

**Q: Is the dashboard mobile-friendly?**  
A: Streamlit is responsive but optimal on desktop (1920×1080+) for chart interactions.

---

**Last Updated:** January 2026  
**Dashboard Version:** 2.0.0  
**Compatible API Version:** 2.0.0+
