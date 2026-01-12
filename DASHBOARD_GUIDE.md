# Dashboard User Guide

## Streamlit Dashboard for Video Anomaly Detection

Production-grade UI for analyzing surveillance videos, identifying anomalies, and exploring reconstruction-based anomaly detection in real-time.

---

## What is This Dashboard?

This dashboard is the **visual interface** for the video anomaly detection system. It allows you to:

1. **Upload surveillance videos** and get frame-by-frame anomaly scores
2. **Visualize where anomalies occur** using interactive charts
3. **Adjust sensitivity** without re-running GPU inference
4. **Inspect specific frames** to understand what the model flagged
5. **Export results** for reports, further analysis, or archival

### How It Works (Technical Overview)

The system uses a **Convolutional Autoencoder** trained to reconstruct "normal" surveillance footage (UCSD Ped2 pedestrian videos). When it sees new video:

1. **Extracts frames** from your uploaded video
2. **Processes each frame** through the neural network
3. **Calculates reconstruction error** - how different the output is from the input
4. **Compares to threshold** - errors above threshold = anomaly

**Key Insight:** The model learned what "normal" looks like. Anything that doesn't match (unusual motion, objects, patterns) produces high reconstruction error.

### What You See in the Dashboard

#### 1. **Reconstruction Error**
- **Definition:** Pixel-level difference between original frame and reconstructed frame
- **Formula:** Mean Squared Error (MSE) across all 64×64 pixels
- **Range:** 0.0 (perfect reconstruction) to ~0.1+ (very poor reconstruction)
- **Typical Values:** 
  - Normal frames: 0.001 - 0.005
  - Anomalies: 0.008 - 0.05+

**Example:** If a frame has error 0.0032, it means the model reconstructed it almost perfectly (normal). If error is 0.0189, the model struggled (likely anomaly).

#### 2. **Threshold Line (Orange)**
- **Definition:** The cutoff value that separates normal from anomaly
- **How It's Set:** Statistical analysis of validation set (typically 95th percentile)
- **Default:** ~0.00507 for UCSD Ped2 model
- **Adjustable:** Use slider to change sensitivity

**Interpretation:** Frames with error **above** threshold are flagged as anomalies.

#### 3. **Anomaly Rate**
- **Definition:** Percentage of frames flagged as anomalies
- **Formula:** (Anomaly Count / Total Frames) × 100
- **Typical Range:** 2-10% for videos with occasional unusual events
- **Red Flag:** >50% means model isn't suited for this video type (see README_ML_PHILOSOPHY.md)

**Example:** 8 anomalies in 300 frames = 2.67% anomaly rate

#### 4. **Processing Time**
- **Definition:** GPU/CPU time to analyze all frames
- **Factors:** Video length, batch size, hardware
- **Expected:** 
  - GPU (RTX 3050): ~0.2s per 10-sec video
  - CPU (Production): ~2-5s per 10-sec video
  - Render free tier: ~5-10s per 10-sec video

---

## Dashboard Components Explained

### Sidebar: Configuration & Metrics

**API Connection Status**
```
✓ API Connected (http://localhost:8000)
```
- **Green checkmark:** Dashboard can reach FastAPI backend
- **Red X:** API unreachable (check if `python app.py` is running)
- **URL shown:** Where dashboard sends video files for analysis

**Current Threshold Value**
```
Current Threshold: 0.00507
```
- **What it means:** Frames with error >0.00507 are anomalies
- **When it changes:** After you adjust the slider
- **Impact:** Higher threshold = fewer anomalies (less sensitive)

**Key Metrics (After Analysis)**
```
📊 Analysis Results
━━━━━━━━━━━━━━━━━━━━━━━━
📹 Total Frames: 300
🚨 Anomalies Detected: 12
📈 Anomaly Rate: 4.0%
⚡ Processing Time: 2.34s
💻 Device: cuda
```

**Breakdown:**
- **Total Frames:** Video frame count (extracted by OpenCV)
- **Anomalies Detected:** Frames with error > threshold
- **Anomaly Rate:** Percentage of flagged frames
- **Processing Time:** How long GPU/CPU took to analyze
- **Device:** "cuda" (GPU) or "cpu" (processor used)

---

### Main Panel: Interactive Timeline Chart

**Chart Elements:**

1. **Blue Line:** Reconstruction error for every frame
   - **Peaks:** Frames model struggled to reconstruct
   - **Valleys:** Frames model reconstructed easily (normal)
   - **Spikes:** Sudden changes, unusual motion, anomalies

2. **Red Markers:** Detected anomalies (error > threshold)
   - **Size:** Indicates severity (bigger = higher error)
   - **Hover:** Shows exact error value and frame number
   - **Click:** Jumps to that frame in viewer below

3. **Orange Threshold Line:** Your current sensitivity setting
   - **Horizontal:** Same value across all frames
   - **Position:** Marks the anomaly cutoff
   - **Movable:** Adjust with slider, line updates instantly

4. **Green Vertical Line:** Current frame being viewed
   - **Tracks:** Which frame is displayed in viewer
   - **Moves:** When you navigate frames or click chart

**How to Use:**
- **Zoom:** Click + drag to zoom into specific time range
- **Pan:** Double-click to reset zoom, drag to pan
- **Inspect:** Hover over any point to see frame number + error value
- **Navigate:** Click any spike to jump to that frame

**What to Look For:**
- **Clusters of spikes:** Prolonged unusual activity
- **Isolated spikes:** Brief anomalies (person enters frame, object moves)
- **Flat regions:** Steady, normal footage
- **Spikes just below threshold:** Borderline anomalies (adjust slider to include/exclude)

---

### Threshold Adjustment Section

**Dynamic Threshold Slider**
```
Adjust Threshold (No Re-analysis Needed)
0.002 ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.045
        👆 Current: 0.00507
```

**How It Works:**
1. **Client-side recalculation:** Slider changes threshold instantly
2. **No API call:** Uses cached reconstruction errors from initial analysis
3. **Real-time updates:** Metrics, chart markers, and flags update immediately

**When to Adjust:**
- **Too many false positives:** Increase threshold (move right)
- **Missing obvious anomalies:** Decrease threshold (move left)
- **Experiment:** Try different values to find sweet spot for your video type

**Example Scenarios:**
- **Crowded scene:** People moving constantly → higher threshold (0.008-0.01)
- **Static camera:** Minimal motion → lower threshold (0.003-0.005)
- **Outdoor with weather:** Wind, shadows → higher threshold

**Reset Button:** Returns to original threshold from API analysis (statistical 95th percentile)

---

### Frame Viewer

**Display Area:**
- **Image:** Current frame from video (RGB, resized for display)
- **Border Color:** 
  - **Red:** Anomaly detected (error > threshold)
  - **None:** Normal frame
- **Status Indicator:** "🚨 Anomaly Detected" or "✓ Normal"

**Frame Information:**
```
Frame 145 / 300
Reconstruction Error: 0.0189
Status: 🚨 Anomaly Detected
```

**Breakdown:**
- **Frame X / Y:** Current position in video (1-indexed for humans)
- **Reconstruction Error:** MSE for this specific frame
- **Status:** Anomaly if error > threshold, Normal otherwise

**Navigation Controls:**
```
⏮ First | ◀ Previous | ▶ Next | ⏭ Last
```

**Keyboard Shortcuts (Streamlit doesn't support, but UI is clickable):**
- Click **First** → Jump to frame 0
- Click **Previous** → Go back one frame
- Click **Next** → Advance one frame
- Click **Last** → Jump to final frame

**Frame Slider:**
```
0 ━━━━━━━━●━━━━━━━━ 300
          👆 Frame 145
```
- Drag to scrub through video
- Updates frame viewer instantly

---

### Anomaly Thumbnails

**Grid Display:**
```
🚨 Detected Anomalies (Top 10)
┌──────┬──────┬──────┬──────┐
│ 45   │ 87   │ 129  │ 201  │
│ 0.012│ 0.018│ 0.009│ 0.015│
└──────┴──────┴──────┴──────┘
```

**What You See:**
- **Thumbnail:** Mini version of anomalous frame
- **Frame Number:** Click to jump to that frame
- **Error Value:** Reconstruction error for that frame
- **Sorted:** Highest error first (worst reconstructions)
- **Limit:** Top 10 anomalies max (performance)

**Use Cases:**
- **Quick scan:** See all anomalies at a glance
- **Compare:** Identify patterns across multiple anomalies
- **Prioritize:** Focus on highest-error frames first

---

### Export Section

**JSON Export:**
```json
{
  "video_info": {
    "frame_count": 300,
    "anomaly_count": 12,
    "anomaly_rate": 0.04,
    "processing_time": 2.34
  },
  "model_info": {
    "device": "cuda",
    "threshold": 0.00507,
    "batch_size": 64
  },
  "frame_data": [
    {
      "frame_number": 0,
      "reconstruction_error": 0.0032,
      "is_anomaly": false
    },
    ...
  ]
}
```

**Use Cases:**
- Programmatic analysis (Python scripts)
- Integration with other tools
- Archival with full metadata

**CSV Export:**
```csv
frame_number,reconstruction_error,is_anomaly
0,0.0032,false
1,0.0029,false
45,0.0123,true
...
```

**Use Cases:**
- Excel/Google Sheets analysis
- Statistical analysis in R
- Database import

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

### Local Development
```bash
# Terminal 1: Start FastAPI backend
python app.py

# Terminal 2: Launch dashboard
streamlit run dashboard.py
```

### Production Usage

**Scenario: API deployed on Render, Dashboard runs locally**

```bash
# Windows PowerShell
$env:API_URL = "https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py

# Linux/Mac
export API_URL="https://video-anomaly-detection-api.onrender.com"
streamlit run dashboard.py
```

The dashboard will connect to the production API automatically. No code changes needed.

**First-time API access:** The Render free tier spins down after 15 minutes of inactivity. First request may take 30-60 seconds to wake up the service.

### Alternative: Streamlit Cloud Hosting

If you want a fully hosted dashboard:

1. Fork this repo on GitHub
2. Go to https://share.streamlit.io
3. Connect your repo and select `dashboard.py`
4. Add secret in Streamlit Cloud dashboard:
   ```
   API_URL = "https://video-anomaly-detection-api.onrender.com"
   ```
5. Deploy

Your dashboard will be available at `https://yourapp.streamlit.app` and won't sleep on the free tier.

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
