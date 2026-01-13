"""
FastAPI web service for video anomaly detection.

Production-grade implementation with:
- Pydantic settings for configuration
- Structured JSON logging
- Input validation (file size, duration, format)
- Thread pool for non-blocking inference
- Batched frame processing
- Rate limiting (slowapi)
- Enhanced health checks (liveness/readiness probes)
- Prometheus metrics
- Async job queue for long videos
"""

import asyncio
import io
import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import psutil
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from data.preprocessing import VideoPreprocessor
from models.autoencoder import ConvolutionalAutoencoder
from models.detector import AnomalyDetector
from settings import Settings, get_settings
from utils.logging_utils import (
    configure_logging,
    get_logger,
    log_inference,
    log_request,
    log_response,
    set_request_id,
)


# Module-level state
_detector: Optional[AnomalyDetector] = None
_device: Optional[torch.device] = None
_preprocessor: Optional[VideoPreprocessor] = None
_executor: Optional[ThreadPoolExecutor] = None
_settings: Optional[Settings] = None
_startup_time: Optional[datetime] = None
_model_version: Optional[str] = None

logger = get_logger(__name__)

# Rate limiter (configured at startup)
limiter = Limiter(key_func=get_remote_address)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "anomaly_detection_requests_total",
    "Total number of anomaly detection requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "anomaly_detection_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
FRAMES_PROCESSED = Counter(
    "anomaly_detection_frames_processed_total",
    "Total frames processed",
)
ANOMALIES_DETECTED = Counter(
    "anomaly_detection_anomalies_total",
    "Total anomalies detected",
)
ACTIVE_JOBS = Gauge(
    "anomaly_detection_active_jobs",
    "Number of active background jobs",
)
GPU_MEMORY_USED = Gauge(
    "anomaly_detection_gpu_memory_bytes",
    "GPU memory used in bytes",
)
MODEL_INFERENCE_LATENCY = Histogram(
    "anomaly_detection_inference_latency_seconds",
    "Model inference latency per batch",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Async Job Queue
# In-memory job store (for production, use Redis or database)
_jobs: Dict[str, Dict[str, Any]] = {}
_job_results: Dict[str, Any] = {}
MAX_JOBS = 100  # Limit to prevent memory exhaustion
JOB_RETENTION_SEC = 3600  # Keep completed jobs for 1 hour


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Response models
class AnomalyResult(BaseModel):
    frame_count: int
    anomaly_count: int
    anomaly_rate: float
    anomaly_scores: List[float]
    anomaly_flags: List[bool]
    processing_time: float
    model_info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
    settings: Optional[Dict[str, Any]] = None


class LivenessResponse(BaseModel):
    """Liveness probe response - is the process running?"""
    status: str
    uptime_seconds: float
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness probe response - can the service handle requests?"""
    ready: bool
    model_loaded: bool
    checks: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None


class SystemMetrics(BaseModel):
    """System resource metrics for monitoring."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


class ValidationError(BaseModel):
    detail: str
    field: Optional[str] = None
    limit: Optional[Any] = None


class JobResponse(BaseModel):
    """Async job submission response."""
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Job status query response."""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[float] = None
    result: Optional[AnomalyResult] = None
    error: Optional[str] = None


def _get_system_metrics() -> SystemMetrics:
    """Collect system resource metrics."""
    memory = psutil.virtual_memory()
    # Use current working directory for cross-platform disk check
    disk = shutil.disk_usage(os.getcwd())
    
    gpu_used = None
    gpu_total = None
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        try:
            gpu_used = torch.cuda.memory_allocated(0) / 1024 / 1024
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        except Exception:
            pass
    
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        disk_free_gb=disk.free / 1024 / 1024 / 1024,
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
    )


def _check_model_inference_latency(timeout_ms: float = 100.0) -> tuple[bool, float]:
    """
    Quick inference latency check with dummy input.
    
    Returns (passed, latency_ms).
    """
    if _detector is None or _device is None:
        return False, 0.0
    
    try:
        dummy = torch.randn(1, 1, 64, 64, device=_device)
        _detector.model.eval()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = _detector.model(dummy)
        latency_ms = (time.perf_counter() - start) * 1000
        
        return latency_ms < timeout_ms, latency_ms
    except Exception:
        return False, 0.0


def _get_model_version_from_registry(model_path: Path) -> str:
    """
    Look up model version from model_registry.json.
    
    Returns version string or generates one from file mtime.
    """
    registry_path = model_path.parent / "model_registry.json"
    
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            
            # Find matching model by filename
            model_name = model_path.name
            for entry in registry.get("models", []):
                if entry.get("filename") == model_name and entry.get("active", False):
                    return entry.get("version", "unknown")
        except Exception:
            pass
    
    # Fallback: generate version from file modification time
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
        return f"1.0.0-{mtime.strftime('%Y%m%d')}"
    
    return "unknown"


def _load_model(settings: Settings) -> bool:
    """Load trained model. Called once at startup."""
    global _detector, _device, _preprocessor, _model_version
    
    try:
        _device = settings.torch_device
        logger.info(f"Initializing on device: {_device}")
        
        model_path = Path(settings.model_path)
        
        # Create config-compatible object for AnomalyDetector
        class ConfigCompat:
            MIXED_PRECISION = settings.mixed_precision
        
        config_obj = ConfigCompat()
        
        if not model_path.exists():
            logger.warning(f"No trained model at {model_path}, creating default model")
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=settings.latent_dim)
            _detector = AnomalyDetector(model, _device, config=config_obj)
            _detector.threshold = 0.005069
            _model_version = "default-1.0.0"
        else:
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
            
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=settings.latent_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            _detector = AnomalyDetector(model, _device, config=config_obj)
            _detector.threshold = checkpoint.get("threshold") or 0.005069
            
            # Extract model version from checkpoint or registry
            _model_version = checkpoint.get("version", _get_model_version_from_registry(model_path))
            
            logger.info(f"Model loaded, version={_model_version}, threshold={_detector.threshold:.6f}")
        
        _preprocessor = VideoPreprocessor(
            target_size=settings.frame_shape,
            quality_threshold=0.001,
        )
        
        return True
        
    except Exception as e:
        logger.exception(f"Model loading failed: {e}")
        return False


def _validate_video_file(
    filename: str,
    content_type: str,
    file_size: int,
    settings: Settings,
) -> Optional[ValidationError]:
    """Validate uploaded video file before processing."""
    
    # Check content type
    if not content_type or not content_type.startswith("video/"):
        return ValidationError(detail="File must be a video", field="content_type")
    
    # Check file extension
    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in settings.allowed_extensions_list:
        return ValidationError(
            detail=f"Unsupported format. Allowed: {settings.allowed_extensions}",
            field="extension",
            limit=settings.allowed_extensions,
        )
    
    # Check file size
    if file_size > settings.max_file_size_bytes:
        return ValidationError(
            detail=f"File exceeds {settings.max_file_size_mb}MB limit",
            field="file_size",
            limit=settings.max_file_size_mb,
        )
    
    return None


def _validate_video_metadata(video_path: str, settings: Settings) -> Optional[ValidationError]:
    """Validate video metadata after file is saved."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return ValidationError(detail="Cannot read video file. File may be corrupted.")
    
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count <= 0:
            return ValidationError(detail="Video contains no readable frames")
        
        if frame_count > settings.max_frames:
            return ValidationError(
                detail=f"Video has {frame_count} frames, max allowed: {settings.max_frames}",
                field="frame_count",
                limit=settings.max_frames,
            )
        
        if duration > settings.max_video_duration_sec:
            return ValidationError(
                detail=f"Video duration {duration:.1f}s exceeds {settings.max_video_duration_sec}s limit",
                field="duration",
                limit=settings.max_video_duration_sec,
            )
        
        return None
        
    finally:
        cap.release()


def _process_video_batched(video_path: str, settings: Settings) -> tuple:
    """
    Process video with batched inference for 2-3x speedup.
    
    Runs in thread pool to avoid blocking the event loop.
    """
    start_time = time.perf_counter()
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while len(frames) < settings.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, settings.frame_shape)
            normalized = resized.astype(np.float32) / 255.0
            frames.append(normalized)
    finally:
        cap.release()
    
    if not frames:
        raise ValueError("No frames extracted from video")
    
    # Stack and convert to tensor
    frames_array = np.stack(frames)
    frames_tensor = torch.from_numpy(frames_array).unsqueeze(1).to(_device)
    
    # Batched inference
    _detector.model.eval()
    reconstruction_errors = []
    batch_size = settings.batch_size
    
    with torch.no_grad():
        for i in range(0, len(frames_tensor), batch_size):
            batch = frames_tensor[i : i + batch_size]
            batch_start = time.perf_counter()
            
            if _detector.scaler is not None and _device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    reconstructed = _detector.model(batch)
            else:
                reconstructed = _detector.model(batch)
            
            # Track batch inference latency
            MODEL_INFERENCE_LATENCY.observe(time.perf_counter() - batch_start)
            
            errors = ((batch - reconstructed) ** 2).mean(dim=[1, 2, 3])
            reconstruction_errors.extend(errors.cpu().tolist())
    
    # Threshold application
    threshold = _detector.threshold or 0.005069
    anomaly_flags = [err > threshold for err in reconstruction_errors]
    anomaly_count = sum(anomaly_flags)
    
    processing_time = time.perf_counter() - start_time
    
    # Update Prometheus counters
    FRAMES_PROCESSED.inc(len(frames))
    ANOMALIES_DETECTED.inc(anomaly_count)
    
    log_inference(
        logger.logger,
        frames=len(frames),
        anomalies=anomaly_count,
        duration_sec=processing_time,
        batch_size=batch_size,
    )
    
    return (len(frames), anomaly_count, reconstruction_errors, anomaly_flags, processing_time)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _executor, _settings, _startup_time
    
    # Startup
    _startup_time = datetime.now(timezone.utc)
    _settings = get_settings()
    configure_logging(
        level=_settings.log_level,
        format_type=_settings.log_format,
        log_file=_settings.log_file,
    )
    
    logger.info("Starting Video Anomaly Detection API v3.0.0")
    
    _executor = ThreadPoolExecutor(
        max_workers=_settings.thread_pool_size,
        thread_name_prefix="inference",
    )
    
    if not _load_model(_settings):
        logger.error("Model loading failed during startup")
    
    # Create directories
    os.makedirs(_settings.static_dir, exist_ok=True)
    os.makedirs(_settings.temp_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    if _executor:
        _executor.shutdown(wait=True)


# Initialize FastAPI
app = FastAPI(
    title="Video Anomaly Detection API",
    description="Real-time video anomaly detection using unsupervised learning",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID and logging to all requests."""
    request_id = set_request_id()
    start_time = time.perf_counter()
    
    log_request(logger.logger, request.method, request.url.path)
    
    response = await call_next(request)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    log_response(logger.logger, request.method, request.url.path, response.status_code, duration_ms)
    
    response.headers["X-Request-ID"] = request_id
    return response


# Mount static files after app creation
@app.on_event("startup")
async def mount_static():
    settings = get_settings()
    if os.path.isdir(settings.static_dir):
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    settings = get_settings()
    return HealthResponse(
        status="healthy" if _detector else "degraded",
        model_loaded=_detector is not None,
        device=str(_device) if _device else "unknown",
        version="3.0.0",
        settings={
            "batch_size": settings.batch_size,
            "max_file_size_mb": settings.max_file_size_mb,
            "max_video_duration_sec": settings.max_video_duration_sec,
            "rate_limit_enabled": settings.rate_limit_enabled,
            "model_version": _model_version,
        },
    )


@app.get("/live", response_model=LivenessResponse)
async def liveness_probe():
    """
    Kubernetes liveness probe.
    
    Returns 200 if process is running. Used by orchestrators to detect
    hung processes that need restart.
    """
    uptime = 0.0
    if _startup_time:
        uptime = (datetime.now(timezone.utc) - _startup_time).total_seconds()
    
    return LivenessResponse(
        status="alive",
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/ready", response_model=ReadinessResponse)
async def readiness_probe():
    """
    Kubernetes readiness probe.
    
    Returns 200 only if service can handle requests:
    - Model loaded and functional
    - Sufficient disk space
    - Inference latency acceptable
    """
    settings = get_settings()
    
    # Check model loaded
    model_ok = _detector is not None and _device is not None
    
    # Check inference latency (100ms threshold)
    inference_ok, latency_ms = _check_model_inference_latency(timeout_ms=100.0)
    
    # Check disk space (>100MB free)
    try:
        # Cross-platform: use temp_dir if exists, else current working directory
        check_path = settings.temp_dir if os.path.exists(settings.temp_dir) else os.getcwd()
        disk = shutil.disk_usage(check_path)
        disk_ok = disk.free > 100 * 1024 * 1024
        disk_free_mb = disk.free / 1024 / 1024
    except Exception:
        disk_ok = True
        disk_free_mb = 0
    
    # Check memory (less than 90% used)
    memory = psutil.virtual_memory()
    memory_ok = memory.percent < 90
    
    all_checks_pass = model_ok and inference_ok and disk_ok and memory_ok
    
    response = ReadinessResponse(
        ready=all_checks_pass,
        model_loaded=model_ok,
        checks={
            "model": model_ok,
            "inference_latency": inference_ok,
            "disk_space": disk_ok,
            "memory": memory_ok,
        },
        details={
            "inference_latency_ms": round(latency_ms, 2),
            "disk_free_mb": round(disk_free_mb, 2),
            "memory_percent": memory.percent,
            "model_version": _model_version,
        },
    )
    
    if not all_checks_pass:
        # Return 503 but still include diagnostic info
        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )
    
    return response


@app.get("/metrics", response_model=SystemMetrics)
async def system_metrics():
    """
    System resource metrics for monitoring dashboards.
    
    Returns CPU, memory, disk, and GPU utilization.
    """
    return _get_system_metrics()


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus-format metrics endpoint.
    
    Exposes:
    - Request counts and latencies
    - Frames processed / anomalies detected
    - Active background jobs
    - GPU memory usage (if available)
    
    Configure Prometheus scrape target: http://host:8000/metrics/prometheus
    """
    # Update GPU memory gauge
    if torch.cuda.is_available():
        try:
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated(0))
        except Exception:
            pass
    
    # Update active jobs gauge
    active = sum(1 for j in _jobs.values() if j.get("status") in (JobStatus.PENDING, JobStatus.PROCESSING))
    ACTIVE_JOBS.set(active)
    
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# Async Job Queue Endpoints

def _cleanup_old_jobs():
    """Remove completed jobs older than retention period."""
    now = datetime.now(timezone.utc)
    expired = []
    for job_id, job in _jobs.items():
        if job.get("status") in (JobStatus.COMPLETED, JobStatus.FAILED):
            completed_at = job.get("completed_at")
            if completed_at:
                age = (now - datetime.fromisoformat(completed_at.replace("Z", "+00:00"))).total_seconds()
                if age > JOB_RETENTION_SEC:
                    expired.append(job_id)
    
    for job_id in expired:
        _jobs.pop(job_id, None)
        _job_results.pop(job_id, None)


def _process_job(job_id: str, video_path: str, settings: Settings):
    """Background job processor for async video analysis."""
    try:
        _jobs[job_id]["status"] = JobStatus.PROCESSING
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        
        # Process video
        result = _process_video_batched(video_path, settings)
        frame_count, anomaly_count, scores, flags, processing_time = result
        
        # Store result
        _job_results[job_id] = AnomalyResult(
            frame_count=frame_count,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_count / frame_count if frame_count > 0 else 0,
            anomaly_scores=scores,
            anomaly_flags=flags,
            processing_time=processing_time,
            model_info={
                "device": str(_device),
                "threshold": _detector.threshold,
                "model_parameters": _detector.model.get_model_info()["total_parameters"],
                "batch_size": settings.batch_size,
            },
        )
        
        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        REQUEST_COUNT.labels(endpoint="/jobs", status="success").inc()
        
    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        REQUEST_COUNT.labels(endpoint="/jobs", status="error").inc()
        logger.exception(f"Job {job_id} failed: {e}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass


@app.post("/jobs/submit", response_model=JobResponse)
async def submit_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Submit video for async background processing.
    
    Returns immediately with job_id. Poll /jobs/{job_id} for status.
    Useful for long videos that exceed request timeout.
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Cleanup old jobs
    _cleanup_old_jobs()
    
    # Check job limit
    active_count = sum(1 for j in _jobs.values() if j.get("status") in (JobStatus.PENDING, JobStatus.PROCESSING))
    if active_count >= MAX_JOBS:
        raise HTTPException(status_code=429, detail=f"Too many active jobs. Max: {MAX_JOBS}")
    
    settings = get_settings()
    
    # Read and validate file
    content = await file.read()
    file_size = len(content)
    
    validation_error = _validate_video_file(
        filename=file.filename or "",
        content_type=file.content_type or "",
        file_size=file_size,
        settings=settings,
    )
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error.detail)
    
    # Save to temp file
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.temp_dir) as tmp:
        tmp.write(content)
        temp_path = tmp.name
    
    # Create job
    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    
    _jobs[job_id] = {
        "status": JobStatus.PENDING,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    
    # Queue background processing
    background_tasks.add_task(_process_job, job_id, temp_path, settings)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted successfully. Poll /jobs/{job_id} for status.",
        created_at=now,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and result of async job.
    
    Returns result when status is 'completed'.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _jobs[job_id]
    result = _job_results.get(job_id)
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result=result,
        error=job.get("error"),
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs with their current status."""
    return {
        "total": len(_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
            }
            for job_id, job in _jobs.items()
        ],
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel or delete a job.
    
    Note: Cannot cancel already-processing jobs (in-memory queue limitation).
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _jobs[job_id]
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Cannot cancel job in progress")
    
    _jobs.pop(job_id, None)
    _job_results.pop(job_id, None)
    
    return {"message": f"Job {job_id} deleted"}


@app.post("/analyze-video", response_model=AnomalyResult)
@limiter.limit(lambda: f"{get_settings().rate_limit_requests}/minute" if get_settings().rate_limit_enabled else "1000/minute")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    """
    Analyze video for anomalies.
    
    Validates input, processes frames in batches, returns per-frame scores.
    Rate limited to prevent abuse.
    """
    start_time = time.perf_counter()
    
    if _detector is None:
        REQUEST_COUNT.labels(endpoint="/analyze-video", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    settings = get_settings()
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate file metadata
    validation_error = _validate_video_file(
        filename=file.filename or "",
        content_type=file.content_type or "",
        file_size=file_size,
        settings=settings,
    )
    if validation_error:
        REQUEST_COUNT.labels(endpoint="/analyze-video", status="error").inc()
        raise HTTPException(status_code=400, detail=validation_error.detail)
    
    # Save to temp file
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.temp_dir) as tmp:
        tmp.write(content)
        temp_path = tmp.name
    
    try:
        # Validate video metadata
        validation_error = _validate_video_metadata(temp_path, settings)
        if validation_error:
            REQUEST_COUNT.labels(endpoint="/analyze-video", status="error").inc()
            raise HTTPException(status_code=400, detail=validation_error.detail)
        
        # Process in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _process_video_batched,
            temp_path,
            settings,
        )
        
        frame_count, anomaly_count, scores, flags, processing_time = result
        
        # Track metrics
        REQUEST_COUNT.labels(endpoint="/analyze-video", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/analyze-video").observe(time.perf_counter() - start_time)
        
        return AnomalyResult(
            frame_count=frame_count,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_count / frame_count if frame_count > 0 else 0,
            anomaly_scores=scores,
            anomaly_flags=flags,
            processing_time=processing_time,
            model_info={
                "device": str(_device),
                "threshold": _detector.threshold,
                "model_parameters": _detector.model.get_model_info()["total_parameters"],
                "batch_size": settings.batch_size,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/analyze-image", response_model=Dict[str, Any])
async def analyze_image(file: UploadFile = File(...)):
    """Analyze single image for anomalies."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        if image.mode != "L":
            image = image.convert("L")
        
        settings = get_settings()
        image = image.resize(settings.frame_shape)
        frame = np.array(image).astype(np.float32) / 255.0
        
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(_device)
        
        _detector.model.eval()
        with torch.no_grad():
            reconstructed = _detector.model(frame_tensor)
            error = ((frame_tensor - reconstructed) ** 2).mean().item()
        
        threshold = _detector.threshold or 0.005069
        is_anomaly = error > threshold
        
        return {
            "anomaly_score": error,
            "is_anomaly": is_anomaly,
            "threshold": threshold,
            "confidence": min(error / threshold, 2.0) if threshold > 0 else 0,
        }
        
    except Exception as e:
        logger.exception(f"Image processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = _detector.model.get_model_info()
    threshold = _detector.threshold or 0.005069
    
    return {
        "model_architecture": info,
        "threshold": threshold,
        "device": str(_device),
        "status": "loaded",
        "model_version": _model_version,
        "threshold_presets": {
            "conservative": 0.004213,
            "balanced": 0.004005,
            "moderate": 0.003706,
            "sensitive": 0.003357,
        },
    }


@app.get("/model-registry")
async def get_model_registry():
    """
    Get model registry with all available model versions.
    
    Returns version history, active model, and training metrics.
    """
    registry_path = Path("outputs/model_registry.json")
    
    if not registry_path.exists():
        return {
            "schema_version": "1.0.0",
            "models": [],
            "active_model": _model_version,
            "message": "No registry file found. Using embedded version info.",
        }
    
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        
        registry["active_model"] = _model_version
        return registry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read registry: {e}")


@app.post("/model-rollback")
async def rollback_model(version: str):
    """
    Rollback to a previous model version.
    
    Requires model file to exist in outputs directory.
    This is a hot-reload operation - no restart required.
    """
    global _detector, _model_version
    
    registry_path = Path("outputs/model_registry.json")
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="Model registry not found")
    
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        
        # Find requested version
        target_model = None
        for model in registry.get("models", []):
            if model.get("version") == version:
                target_model = model
                break
        
        if not target_model:
            available = [m.get("version") for m in registry.get("models", [])]
            raise HTTPException(
                status_code=404,
                detail=f"Version {version} not found. Available: {available}",
            )
        
        # Load the model
        model_path = Path("outputs") / target_model["filename"]
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        settings = get_settings()
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=settings.latent_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        class ConfigCompat:
            MIXED_PRECISION = settings.mixed_precision
        
        _detector = AnomalyDetector(model, _device, config=ConfigCompat())
        _detector.threshold = target_model.get("training_info", {}).get("threshold", 0.005069)
        
        # Capture old version BEFORE updating
        old_version = _model_version or "unknown"
        _model_version = version
        
        # Update registry to mark new active model
        for m in registry.get("models", []):
            m["active"] = m.get("version") == version
        
        # Record rollback in history
        registry.setdefault("rollback_history", []).append({
            "from_version": old_version,
            "to_version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Model rolled back to version {version}")
        
        return {
            "message": f"Successfully rolled back to {version}",
            "active_version": version,
            "threshold": _detector.threshold,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {e}")


@app.post("/set-threshold")
async def set_threshold(threshold: float):
    """Update anomaly detection threshold."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if threshold <= 0:
        raise HTTPException(status_code=400, detail="Threshold must be positive")
    
    _detector.threshold = threshold
    logger.info(f"Threshold updated to {threshold:.6f}")
    
    return {"message": f"Threshold updated to {threshold}", "new_threshold": threshold}


@app.post("/set-threshold-preset")
async def set_threshold_preset(preset: str):
    """Set threshold using predefined presets."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    presets = {
        "conservative": 0.004213,
        "balanced": 0.004005,
        "moderate": 0.003706,
        "sensitive": 0.003357,
    }
    
    if preset not in presets:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from: {list(presets.keys())}")
    
    _detector.threshold = presets[preset]
    logger.info(f"Threshold set to {preset} preset ({presets[preset]:.6f})")
    
    return {
        "message": f"Threshold set to {preset} preset",
        "new_threshold": _detector.threshold,
        "expected_anomaly_rate": {"conservative": "~5%", "balanced": "~10%", "moderate": "~25%", "sensitive": "~50%"}[preset],
    }


@app.post("/calibrate-threshold")
async def calibrate_threshold(target_anomaly_rate: float):
    """Calibrate threshold based on target anomaly rate using training data."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not 0.01 <= target_anomaly_rate <= 0.95:
        raise HTTPException(status_code=400, detail="Target anomaly rate must be between 1% and 95%")
    
    try:
        errors_path = Path("outputs/reconstruction_errors.npy")
        if not errors_path.exists():
            raise HTTPException(status_code=503, detail="Training errors not found. Cannot auto-calibrate.")
        
        reconstruction_errors = np.load(errors_path)
        target_percentile = 100 * (1 - target_anomaly_rate)
        calibrated_threshold = float(np.percentile(reconstruction_errors, target_percentile))
        
        _detector.threshold = calibrated_threshold
        actual_rate = float((reconstruction_errors > calibrated_threshold).mean())
        
        logger.info(f"Threshold calibrated to {calibrated_threshold:.6f} for {target_anomaly_rate:.1%} target rate")
        
        return {
            "message": "Threshold calibrated successfully",
            "target_anomaly_rate": target_anomaly_rate,
            "actual_anomaly_rate": actual_rate,
            "new_threshold": calibrated_threshold,
            "percentile_used": target_percentile,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")


@app.get("/")
async def root():
    """
    Redirect to API documentation.
    
    For the interactive dashboard, visit:
    https://video-anomaly-detection-dashboard.onrender.com
    """
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
