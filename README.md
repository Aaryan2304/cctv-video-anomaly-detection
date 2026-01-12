# Video Anomaly Detection System

Video anomaly detection using convolutional autoencoders trained on UCSD Ped2 surveillance dataset. Deployed as FastAPI web service with real-time scoring.

**Metrics (UCSD Ped2):** 92.47% precision, 0.7438 AUC

## Quick Start

### Demo

```bash
python app.py
# API available at http://localhost:8000
```

### Development

```bash
pip install -r requirements.txt
python main.py                        # Train model (pre-trained included)
python create_realistic_test_videos.py # Generate test data
python app.py
```

## Capabilities

**Supported Inputs:** MP4, AVI, MOV via cv2.VideoCapture

**Performance (RTX 3050):**

- ~0.2s per 10-second clip (GPU)
- ~2-5s per clip (CPU)
- Concurrent stream support via FastAPI async

## Architecture

```text
Video → Frame Extraction → Grayscale/Resize → Autoencoder → Reconstruction Error → Threshold → Anomaly Score
```

**Components:**

- `app.py` - FastAPI service with `/analyze-video`, `/calibrate-threshold` endpoints
- `models/autoencoder.py` - Convolutional autoencoder (64×64 → 256-dim latent → 64×64)
- `models/detector.py` - Training loop, threshold calibration, inference
- `data/preprocessing.py` - Frame extraction and normalization pipeline

## API Reference

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

## Threshold Presets

| Preset | Anomaly Rate | Use Case |
|--------|--------------|----------|
| conservative | 5% | Low false-positive environments |
| balanced | 10% | General surveillance |
| moderate | 25% | High-sensitivity monitoring |
| sensitive | 40% | Maximum detection recall |

## Integration Example

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

### Multi-Camera Processing

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

## Deployment

### Docker

```bash
# Build and run
docker build -t anomaly-detector .
docker run -p 8000:8000 --gpus all anomaly-detector
```

### Cloud Deployment

```bash
# Push to GitHub, connect to Render
# Automatic deployment with render.yaml configuration
```

### Docker Compose

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

## Model Details

**Dataset:** UCSD Ped2 (1,530 normal frames, ground truth annotations)

**Architecture:** Convolutional autoencoder, 64×64 grayscale input, 256-dim latent, 3.48M parameters

**Metrics:** AUC 0.7438, Precision 92.47%, Recall 83.78%, F1 87.91%

### Threshold Calibration Example

```python
# Set target anomaly rate (e.g., 10%)
requests.post("/calibrate-threshold", json={"target_anomaly_rate": 0.10})
```

### Environment-Specific Tuning

```python
# Analyze your actual camera footage
files = [open("camera1_sample.mp4", "rb"), open("camera2_sample.mp4", "rb")]
response = requests.post("/batch-analyze", files=files)

# Use suggested thresholds from your environment
thresholds = response.json()["suggested_thresholds"]
requests.post("/set-threshold", json={"threshold": thresholds["balanced_10pct"]})
```

## Project Structure

```text
├── app.py                 # FastAPI web service
├── main.py                # Training/evaluation pipeline
├── config.py              # Configuration management
├── models/
│   ├── autoencoder.py     # ConvolutionalAutoencoder, LightweightAutoencoder
│   └── detector.py        # AnomalyDetector, EarlyStopping
├── data/
│   ├── dataset.py         # VideoDataset, SyntheticVideoDataset
│   ├── preprocessing.py   # VideoPreprocessor
│   └── synthetic_data.py  # Test data generation
├── evaluation/
│   ├── metrics.py         # PerformanceEvaluator
│   └── visualizer.py      # ResultsVisualizer
├── outputs/
│   ├── trained_model.pth  # Trained weights
│   └── *.npy              # Training artifacts
└── Dockerfile, render.yaml, requirements.txt
```

## Troubleshooting

**High false positives:** Use `{"preset": "conservative"}`

**Missing anomalies:** Use `{"preset": "sensitive"}`

**Slow processing:** Enable CUDA or reduce frame resolution

## Documentation

- API docs: `/docs` (Swagger) and `/redoc` when server is running
- Architecture: `models/autoencoder.py`
- Training: `main.py`

## License

MIT. UCSD Ped2 dataset used under academic license.
