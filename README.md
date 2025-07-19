# 🎥 Video Anomaly Detection System

A production-ready video anomaly detection system using unsupervised learning. Trained on UCSD Ped2 surveillance dataset with **92.47% precision** and **0.7438 AUC** performance.

## 🚀 Quick Start

### For Employers/Demo (2 minutes):
```bash
# 1. Start the API server
python app.py

# 2. Open browser to http://localhost:8000
# 3. Upload any MP4 video for instant analysis
```

### For Development:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train your own model (optional - pre-trained included)
python main.py

# 3. Test with realistic videos
python create_realistic_test_videos.py
python app.py
```

## 🎯 Real-World Capabilities

### ✅ **Works With:**
- **CCTV Camera Feeds** (MP4, AVI, MOV formats)
- **Security Surveillance Systems** (fixed camera positions)
- **Live Video Streams** (via API integration)
- **Pedestrian Areas** (sidewalks, building entrances, plazas)
- **Indoor Security** (lobbies, corridors, retail spaces)

### 📊 **Performance:**
- **Processing Speed**: ~0.2 seconds per 10-second video clip
- **GPU Acceleration**: CUDA-enabled for real-time processing
- **Accuracy**: 92.47% precision, 0.7438 AUC on validation data
- **Scalability**: Handles multiple concurrent video streams

## 🏗️ System Architecture

```
Real Camera Feed → API Endpoint → AI Model → Anomaly Detection → Alerts/Dashboard
```

### Core Components:
- **FastAPI Web Service**: Production REST API with auto-documentation
- **PyTorch Autoencoder**: Trained on real surveillance footage (UCSD Ped2)
- **Adaptive Thresholding**: Automatic calibration for different environments
- **Real-time Processing**: Optimized for live video analysis
- **Docker Deployment**: Production-ready containerization

## 📡 API Reference

### Video Analysis
```http
POST /analyze-video
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "frame_count": 60,
  "anomaly_count": 8,
  "anomaly_rate": 0.13,
  "anomaly_scores": [0.002, 0.008, 0.012, ...],
  "processing_time": 0.85,
  "model_info": {
    "device": "cuda",
    "threshold": 0.005069,
    "model_parameters": 3480897
  }
}
```

### Threshold Calibration
```http
POST /calibrate-threshold
Content-Type: application/json

{
  "target_anomaly_rate": 0.10  // 10% anomaly rate
}
```

### Preset Security Levels
```http
POST /set-threshold-preset
Content-Type: application/json

{
  "preset": "balanced"  // conservative, balanced, moderate, sensitive
}
```

## 🎛️ Deployment Configurations

### Security Level Presets:

#### 🔴 **High Security (Conservative)**
- **5% anomaly rate** - Minimal false positives
- **Use Case**: Banks, airports, critical infrastructure
- **API**: `{"preset": "conservative"}`

#### 🟡 **Balanced Security (Recommended)**
- **10% anomaly rate** - Optimal balance
- **Use Case**: Office buildings, retail stores, general surveillance
- **API**: `{"preset": "balanced"}`

#### 🟠 **High Sensitivity (Moderate)**
- **25% anomaly rate** - Catches more events
- **Use Case**: Public spaces, behavior monitoring
- **API**: `{"preset": "moderate"}`

## 🏢 Enterprise Integration Examples

### Security Management System Integration:
```python
import requests

class SecuritySystem:
    def __init__(self):
        self.api_url = "http://anomaly-api:8000"
        
    def monitor_camera(self, camera_id, video_clip):
        # Send video to anomaly detection API
        response = requests.post(
            f"{self.api_url}/analyze-video",
            files={"file": video_clip}
        )
        
        result = response.json()
        
        # Trigger security alert if anomaly rate > 15%
        if result["anomaly_rate"] > 0.15:
            self.send_security_alert(camera_id, result)
            
    def send_security_alert(self, camera_id, detection_result):
        # Integration with security dashboard
        # Automatic incident logging
        # Personnel notification
        pass
```

### Multi-Camera Processing:
```python
async def process_multiple_cameras(camera_feeds):
    """Process multiple camera feeds concurrently"""
    tasks = []
    for camera_id, feed in camera_feeds.items():
        task = analyze_camera_feed(camera_id, feed)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## 🐳 Production Deployment

### Docker (Recommended):
```bash
# Build and run
docker build -t anomaly-detector .
docker run -p 8000:8000 --gpus all anomaly-detector
```

### Cloud Deployment (Render.com):
```bash
# Push to GitHub, connect to Render
# Automatic deployment with render.yaml configuration
```

### Docker Compose (Multi-service):
```yaml
version: '3.8'
services:
  anomaly-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GPU_ENABLED=true
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## 📊 Model Training Details

### Dataset:
- **Source**: UCSD Ped2 surveillance dataset
- **Type**: Real pedestrian surveillance footage
- **Training**: 1,530 normal frames for unsupervised learning
- **Validation**: Ground truth anomaly annotations

### Architecture:
- **Model**: Convolutional Autoencoder
- **Input**: 64x64 grayscale frames
- **Latent Dimension**: 256 features
- **Parameters**: 3.48M trainable parameters

### Performance Metrics:
- **AUC Score**: 0.7438
- **Precision**: 92.47%
- **Recall**: 83.78%
- **F1-Score**: 87.91%

## 🔧 Threshold Optimization

The system includes sophisticated threshold management:

### Automatic Calibration:
```python
# Set target anomaly rate (e.g., 10%)
requests.post("/calibrate-threshold", json={"target_anomaly_rate": 0.10})
```

### Environment-Specific Tuning:
```python
# Analyze your actual camera footage
files = [open("camera1_sample.mp4", "rb"), open("camera2_sample.mp4", "rb")]
response = requests.post("/batch-analyze", files=files)

# Use suggested thresholds from your environment
thresholds = response.json()["suggested_thresholds"]
requests.post("/set-threshold", json={"threshold": thresholds["balanced_10pct"]})
```

## 📁 Project Structure

```
anomaly_detection/
├── 🎯 CORE APPLICATION
│   ├── app.py                     # FastAPI production web service
│   ├── main.py                    # Training and evaluation script
│   ├── config.py                  # Configuration settings
│   └── create_realistic_test_videos.py  # Test video generator
│
├── 🧠 MODEL COMPONENTS
│   ├── models/
│   │   ├── autoencoder.py         # Convolutional autoencoder architecture
│   │   └── detector.py            # Anomaly detection logic
│   ├── data/
│   │   ├── dataset.py             # UCSD Ped2 dataset loader
│   │   ├── preprocessing.py       # Video preprocessing pipeline
│   │   └── synthetic_data.py      # Synthetic data generation
│   ├── evaluation/
│   │   ├── metrics.py             # Performance evaluation
│   │   └── visualizer.py          # Results visualization
│   └── utils/
│       ├── file_utils.py          # File handling utilities
│       └── gpu_utils.py           # GPU optimization
│
├── 📊 TRAINED MODEL & RESULTS
│   └── outputs/
│       ├── trained_model.pth      # Production-ready model (13.9MB)
│       ├── analysis_report.txt    # Performance analysis
│       ├── roc_curve.png          # ROC curve visualization
│       └── reconstruction_errors.npy  # Training statistics
│
├── 🚀 DEPLOYMENT
│   ├── Dockerfile                 # Production container
│   ├── requirements.txt           # Dependencies
│   └── deployment/
│       ├── docker-compose.yml     # Multi-service setup
│       ├── render.yaml            # Cloud deployment config
│       └── build.sh               # Build scripts
│
├── 🧪 TESTING
│   ├── test_videos/               # Test videos for validation
│   └── cctv_samples/              # Sample CCTV footage
│
└── 📚 DATASETS
    ├── ped2/                      # UCSD Ped2 dataset
    └── UCSD_Anomaly_Dataset.v1p2/ # Full UCSD dataset
```

## 🎯 Use Cases & Success Stories

### Ideal Applications:
1. **Building Security**: Lobby and entrance monitoring
2. **Retail Loss Prevention**: Store surveillance systems
3. **Campus Safety**: University and school security
4. **Public Space Monitoring**: Parks, plazas, transportation hubs
5. **Industrial Safety**: Factory and warehouse monitoring

### Performance Optimization:
- **CPU-only**: 2-5 seconds per video clip
- **GPU-accelerated**: 0.2-0.5 seconds per video clip
- **Batch processing**: Up to 10 concurrent video streams
- **Memory efficient**: <2GB RAM usage

## 🛠️ Troubleshooting

### Common Issues:

#### High False Positive Rate:
```python
# Solution: Use conservative threshold
requests.post("/set-threshold-preset", json={"preset": "conservative"})
```

#### Missing Anomalies:
```python
# Solution: Increase sensitivity
requests.post("/set-threshold-preset", json={"preset": "sensitive"})
```

#### Slow Processing:
- Enable GPU acceleration
- Reduce video resolution
- Use batch processing

## 📞 Support & Documentation

- **API Documentation**: Available at `/docs` endpoint when running
- **Interactive Testing**: Swagger UI at `/redoc`
- **Model Architecture**: See `models/autoencoder.py`
- **Training Process**: See `main.py` for full pipeline

## 🏆 Portfolio Highlights

This project demonstrates:

✅ **Production ML Engineering**: Real-world deployment capabilities  
✅ **Computer Vision Expertise**: Video processing and anomaly detection  
✅ **API Development**: FastAPI with comprehensive documentation  
✅ **DevOps Skills**: Docker, cloud deployment, monitoring  
✅ **Performance Optimization**: GPU acceleration, efficient processing  
✅ **Security Domain Knowledge**: Surveillance and security applications  

Perfect for showcasing ML engineering skills to employers in:
- **Security Technology Companies**
- **AI/ML Startups**
- **Enterprise Software**
- **Surveillance Systems**
- **Smart City Solutions**

---

## 📄 License

This project is for educational and portfolio demonstration purposes. The UCSD Ped2 dataset is used under academic license terms.

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome via issues or pull requests.