"""
FastAPI web service for video anomaly detection.
"""

import os
import io
import tempfile
import zipfile
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import torch
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Import our models
from models.autoencoder import ConvolutionalAutoencoder
from models.detector import AnomalyDetector
from data.preprocessing import VideoPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Anomaly Detection API",
    description="Real-time video anomaly detection using unsupervised learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model
detector = None
device = None
preprocessor = None

class AnomalyResult(BaseModel):
    """Response model for anomaly detection results"""
    frame_count: int
    anomaly_count: int
    anomaly_rate: float
    anomaly_scores: List[float]
    anomaly_flags: List[bool]
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    version: str

def load_model():
    """Load the trained anomaly detection model"""
    global detector, device, preprocessor
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model
        model_path = "outputs/trained_model.pth"
        if not os.path.exists(model_path):
            logger.warning("No trained model found. Creating new model...")
            # Create a new model if no trained model exists
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
            detector = AnomalyDetector(model, device)
            # Set a default threshold based on training analysis (98th percentile for real-world use)
            detector.threshold = 0.005069
        else:
            logger.info("Loading trained model...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create detector
            detector = AnomalyDetector(model, device)
            # Ensure threshold is never None - use 98th percentile for real-world robustness
            threshold_value = checkpoint.get('threshold', 0.005069)
            detector.threshold = threshold_value if threshold_value is not None else 0.005069
            
            logger.info(f"Model loaded with threshold: {detector.threshold}")
        
        # Initialize preprocessor
        preprocessor = VideoPreprocessor(
            target_size=(64, 64),
            quality_threshold=0.001
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def process_video_frames(video_path: str) -> tuple:
    """Process video frames and detect anomalies"""
    try:
        import time
        start_time = time.time()
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            frames.append(normalized_frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames found in video")
        
        # Convert to numpy array first, then to tensor (more efficient)
        frames_array = np.array(frames)
        frames_tensor = torch.FloatTensor(frames_array).unsqueeze(1).to(device)
        
        # Detect anomalies
        detector.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for frame in frames_tensor:
                frame_batch = frame.unsqueeze(0)
                reconstructed = detector.model(frame_batch)
                error = torch.mean((frame_batch - reconstructed) ** 2).item()
                reconstruction_errors.append(error)
        
        # Apply threshold
        if detector.threshold is None or detector.threshold <= 0:
            logger.warning("Invalid threshold detected, using default value")
            detector.threshold = 0.005069
        
        anomaly_flags = [error > detector.threshold for error in reconstruction_errors]
        anomaly_count = sum(anomaly_flags)
        
        processing_time = time.time() - start_time
        
        return (
            len(frames),
            anomaly_count,
            reconstruction_errors,
            anomaly_flags,
            processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Anomaly Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .results { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .loading { display: none; }
        </style>
    </head>
    <body>
        <h1>🎥 Video Anomaly Detection</h1>
        <p>Upload a video file to detect anomalous events using our trained AI model.</p>
        
        <div class="upload-area">
            <input type="file" id="videoFile" accept="video/*" />
            <br><br>
            <button onclick="uploadVideo()">Analyze Video</button>
        </div>
        
        <div id="loading" class="loading">
            <p>Processing video... This may take a few moments.</p>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <h3>Analysis Results</h3>
            <div id="metrics"></div>
        </div>

        <script>
        async function uploadVideo() {
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/analyze-video', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResults(result);
                } else {
                    alert('Error: ' + result.detail);
                }
            } catch (error) {
                alert('Error uploading video: ' + error.message);
            }
            
            document.getElementById('loading').style.display = 'none';
        }
        
        function displayResults(result) {
            const metricsDiv = document.getElementById('metrics');
            metricsDiv.innerHTML = `
                <div class="metric">
                    <strong>Total Frames:</strong> ${result.frame_count}
                </div>
                <div class="metric">
                    <strong>Anomalies Detected:</strong> ${result.anomaly_count}
                </div>
                <div class="metric">
                    <strong>Anomaly Rate:</strong> ${(result.anomaly_rate * 100).toFixed(1)}%
                </div>
                <div class="metric">
                    <strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s
                </div>
                <div class="metric">
                    <strong>Device:</strong> ${result.model_info.device}
                </div>
            `;
            
            document.getElementById('results').style.display = 'block';
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        device=str(device) if device else "unknown",
        version="1.0.0"
    )

@app.post("/analyze-video", response_model=AnomalyResult)
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze video for anomalies
    
    Upload a video file and get anomaly detection results including:
    - Frame count
    - Number of anomalies detected
    - Anomaly rate
    - Individual frame scores
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process video
        frame_count, anomaly_count, scores, flags, processing_time = process_video_frames(temp_path)
        
        # Calculate metrics
        anomaly_rate = anomaly_count / frame_count if frame_count > 0 else 0
        
        # Get model info
        model_info = {
            "device": str(device),
            "threshold": detector.threshold,
            "model_parameters": detector.model.get_model_info()["total_parameters"]
        }
        
        return AnomalyResult(
            frame_count=frame_count,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate,
            anomaly_scores=scores,
            anomaly_flags=flags,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/analyze-image", response_model=Dict[str, Any])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze single image for anomalies
    
    Upload an image file and get anomaly detection score
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to grayscale and resize
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize and normalize
        image = image.resize((64, 64))
        frame = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.FloatTensor(frame).unsqueeze(0).unsqueeze(0).to(device)
        
        # Detect anomaly
        detector.model.eval()
        with torch.no_grad():
            reconstructed = detector.model(frame_tensor)
            error = torch.mean((frame_tensor - reconstructed) ** 2).item()
        
        # Apply threshold
        if detector.threshold is None or detector.threshold <= 0:
            logger.warning("Invalid threshold detected, using default value")
            detector.threshold = 0.005069
            
        is_anomaly = error > detector.threshold
        
        return {
            "anomaly_score": error,
            "is_anomaly": is_anomaly,
            "threshold": detector.threshold,
            "confidence": min(error / detector.threshold, 2.0) if detector.threshold > 0 else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = detector.model.get_model_info()
        # Ensure threshold is valid
        threshold = detector.threshold if detector.threshold is not None else 0.004005
        return {
            "model_architecture": info,
            "threshold": threshold,
            "device": str(device),
            "status": "loaded",
            "threshold_presets": {
                "conservative": 0.004213,  # 95th percentile - 5% anomaly rate
                "balanced": 0.004005,      # 90th percentile - 10% anomaly rate  
                "moderate": 0.003706,      # 75th percentile - 25% anomaly rate
                "sensitive": 0.003357      # 50th percentile - 50% anomaly rate
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/set-threshold")
async def set_threshold(threshold: float):
    """Update the anomaly detection threshold"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if threshold <= 0:
        raise HTTPException(status_code=400, detail="Threshold must be positive")
    
    detector.threshold = threshold
    return {"message": f"Threshold updated to {threshold}", "new_threshold": threshold}

@app.post("/set-threshold-preset")
async def set_threshold_preset(preset: str):
    """Set threshold using predefined presets"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    presets = {
        "conservative": 0.004213,  # 5% anomaly rate
        "balanced": 0.004005,      # 10% anomaly rate  
        "moderate": 0.003706,      # 25% anomaly rate
        "sensitive": 0.003357      # 50% anomaly rate
    }
    
    if preset not in presets:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from: {list(presets.keys())}")
    
    detector.threshold = presets[preset]
    return {
        "message": f"Threshold set to {preset} preset",
        "new_threshold": detector.threshold,
        "expected_anomaly_rate": {
            "conservative": "~5%",
            "balanced": "~10%", 
            "moderate": "~25%",
            "sensitive": "~50%"
        }[preset]
    }

@app.post("/calibrate-threshold")
async def calibrate_threshold(target_anomaly_rate: float):
    """
    Automatically calibrate threshold based on target anomaly rate
    using the training data reconstruction errors
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not (0.01 <= target_anomaly_rate <= 0.95):
        raise HTTPException(status_code=400, detail="Target anomaly rate must be between 1% and 95%")
    
    try:
        # Load reconstruction errors from training
        reconstruction_errors = np.load('outputs/reconstruction_errors.npy')
        
        # Calculate the threshold that would give the target anomaly rate
        target_percentile = 100 * (1 - target_anomaly_rate)
        calibrated_threshold = np.percentile(reconstruction_errors, target_percentile)
        
        # Update the detector threshold
        detector.threshold = float(calibrated_threshold)
        
        # Verify the actual rate this would produce
        actual_rate = (reconstruction_errors > calibrated_threshold).mean()
        
        return {
            "message": "Threshold calibrated successfully",
            "target_anomaly_rate": target_anomaly_rate,
            "actual_anomaly_rate": float(actual_rate),
            "new_threshold": detector.threshold,
            "percentile_used": float(target_percentile)
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Training reconstruction errors not found. Cannot auto-calibrate.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during calibration: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze_videos(files: List[UploadFile] = File(...)):
    """
    Analyze multiple videos for threshold calibration and performance assessment
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    all_errors = []
    
    for file in files:
        if not file.content_type.startswith('video/'):
            continue
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Process video
            frame_count, anomaly_count, scores, flags, processing_time = process_video_frames(temp_path)
            all_errors.extend(scores)
            
            results.append({
                "filename": file.filename,
                "frame_count": frame_count,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / frame_count if frame_count > 0 else 0,
                "avg_error": np.mean(scores),
                "max_error": max(scores),
                "min_error": min(scores)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Calculate suggested thresholds based on all videos
    if all_errors:
        all_errors = np.array(all_errors)
        suggested_thresholds = {
            "conservative_5pct": float(np.percentile(all_errors, 95)),
            "balanced_10pct": float(np.percentile(all_errors, 90)),
            "moderate_25pct": float(np.percentile(all_errors, 75)),
            "sensitive_50pct": float(np.percentile(all_errors, 50))
        }
    else:
        suggested_thresholds = {}
    
    return {
        "individual_results": results,
        "overall_stats": {
            "total_videos": len(results),
            "total_frames": sum(r.get("frame_count", 0) for r in results),
            "avg_error_across_all": float(np.mean(all_errors)) if all_errors else 0,
            "current_threshold": detector.threshold
        },
        "suggested_thresholds": suggested_thresholds
    }

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )
