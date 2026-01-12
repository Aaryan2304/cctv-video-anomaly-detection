"""
Structured logging utilities for production deployment.

Provides JSON-formatted logs with request tracing and configurable output.
"""

import logging
import json
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional, Any

# Context variable for request ID tracing
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    return request_id_ctx.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for current context. Generates UUID if not provided."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_ctx.set(rid)
    return rid


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add source location
        log_entry["location"] = f"{record.filename}:{record.lineno}"
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_id = get_request_id()
        rid_str = f"[{request_id}] " if request_id else ""
        
        base = f"{timestamp} | {record.levelname:8} | {rid_str}{record.name}: {record.getMessage()}"
        
        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"
        
        return base


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that includes extra context in log records."""
    
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra", {})
        extra["extra_data"] = self.extra
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: 'json' for production, 'text' for development
        log_file: Optional file path for log output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Select formatter
    formatter = JSONFormatter() if format_type == "json" else TextFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> ContextLogger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (typically __name__)
        **context: Additional context to include in all log entries
        
    Returns:
        ContextLogger instance
    """
    return ContextLogger(logging.getLogger(name), context)


# Convenience functions for structured logging
def log_request(logger: logging.Logger, method: str, path: str, **extra: Any) -> None:
    """Log incoming request."""
    logger.info(
        f"{method} {path}",
        extra={"extra_data": {"event": "request", "method": method, "path": path, **extra}}
    )


def log_response(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **extra: Any
) -> None:
    """Log outgoing response."""
    logger.info(
        f"{method} {path} -> {status_code} ({duration_ms:.1f}ms)",
        extra={"extra_data": {
            "event": "response",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **extra
        }}
    )


def log_inference(
    logger: logging.Logger,
    frames: int,
    anomalies: int,
    duration_sec: float,
    **extra: Any
) -> None:
    """Log inference result."""
    logger.info(
        f"Inference complete: {frames} frames, {anomalies} anomalies, {duration_sec:.2f}s",
        extra={"extra_data": {
            "event": "inference",
            "frames": frames,
            "anomalies": anomalies,
            "duration_sec": duration_sec,
            **extra
        }}
    )


if __name__ == "__main__":
    # Test logging configuration
    configure_logging(level="DEBUG", format_type="text")
    
    logger = get_logger(__name__, component="test")
    
    set_request_id("abc123")
    logger.info("Test message")
    logger.warning("Warning with context", extra={"extra_data": {"key": "value"}})
    
    try:
        raise ValueError("Test exception")
    except ValueError:
        logger.exception("Caught exception")
