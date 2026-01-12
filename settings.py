"""
Environment-driven configuration using pydantic-settings.

Supports .env files for local development and environment variables for production.
All settings can be overridden via environment variables with the APP_ prefix.
"""

import os
from functools import lru_cache
from typing import Optional, Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Server
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, le=32, description="Uvicorn workers")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Hardware
    device: str = Field(default="auto", description="Compute device: auto, cuda, cpu")
    batch_size: int = Field(default=64, ge=1, le=512, description="Inference batch size")
    num_workers: int = Field(default=4, ge=0, le=16, description="DataLoader workers")
    pin_memory: bool = Field(default=True, description="Pin memory for GPU transfer")
    mixed_precision: bool = Field(default=True, description="Enable AMP for inference")
    
    # Model
    model_path: str = Field(default="outputs/trained_model.pth", description="Path to trained model")
    latent_dim: int = Field(default=256, ge=32, le=1024, description="Autoencoder latent dimension")
    frame_size: int = Field(default=64, ge=32, le=256, description="Input frame size (square)")
    
    # Anomaly detection
    threshold: Optional[float] = Field(default=None, ge=0.0, description="Manual threshold override")
    threshold_factor: float = Field(default=2.5, ge=1.0, le=5.0, description="Threshold multiplier for std")
    threshold_percentile: int = Field(default=95, ge=50, le=99, description="Percentile for threshold")
    threshold_mode: Literal["statistical", "percentile"] = Field(default="statistical")
    
    # Input validation
    max_file_size_mb: int = Field(default=100, ge=1, le=500, description="Max upload size in MB")
    max_video_duration_sec: int = Field(default=300, ge=10, le=1800, description="Max video duration")
    max_frames: int = Field(default=5000, ge=100, le=50000, description="Max frames to process")
    allowed_extensions: str = Field(default=".mp4,.avi,.mov,.mkv,.webm", description="Allowed video extensions")
    
    # Concurrency
    thread_pool_size: int = Field(default=4, ge=1, le=16, description="Thread pool for inference")
    request_timeout_sec: int = Field(default=300, ge=30, le=1800, description="Request timeout")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format for production")
    log_file: Optional[str] = Field(default=None, description="Log file path (optional)")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=10, ge=1, description="Requests per window")
    rate_limit_window_sec: int = Field(default=60, ge=10, description="Rate limit window in seconds")
    
    # Paths
    output_dir: str = Field(default="outputs", description="Output directory")
    temp_dir: str = Field(default="temp", description="Temporary file directory")
    static_dir: str = Field(default="static", description="Static files directory")
    
    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v
    
    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]
    
    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.frame_size, self.frame_size)
    
    def get_device_info(self) -> dict:
        """Return device information for health checks."""
        info = {
            "device": self.device,
            "mixed_precision": self.mixed_precision,
        }
        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        return info


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    
    Call get_settings.cache_clear() to reload from environment.
    """
    return Settings()


# Backward compatibility with config.py
def settings_to_config_compat(settings: Settings) -> dict:
    """
    Convert Settings to dict matching Config class interface.
    
    Allows gradual migration from Config to Settings.
    """
    return {
        "DEVICE": settings.torch_device,
        "BATCH_SIZE": settings.batch_size,
        "NUM_WORKERS": settings.num_workers,
        "PIN_MEMORY": settings.pin_memory,
        "MIXED_PRECISION": settings.mixed_precision,
        "LATENT_DIM": settings.latent_dim,
        "FRAME_SIZE": settings.frame_shape,
        "THRESHOLD_FACTOR": settings.threshold_factor,
        "PERCENTILE_THRESHOLD": settings.threshold_percentile,
        "THRESHOLD_MODE": settings.threshold_mode,
        "OUTPUTS_DIR": settings.output_dir,
    }


if __name__ == "__main__":
    s = get_settings()
    print(f"Device: {s.device}")
    print(f"Model path: {s.model_path}")
    print(f"Max file size: {s.max_file_size_mb} MB")
    print(f"Allowed extensions: {s.allowed_extensions_list}")
    print(f"Device info: {s.get_device_info()}")
