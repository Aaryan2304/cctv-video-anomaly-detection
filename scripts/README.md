# Utility Scripts

This directory contains standalone utility scripts for various development and maintenance tasks.

## Available Scripts

### Calibration & Testing

**`calibrate_for_real_world.py`**
- Analyzes real CCTV footage to calibrate anomaly detection thresholds
- Compares your camera feeds against the trained model
- Suggests optimal threshold settings for your environment
- Usage: `python scripts/calibrate_for_real_world.py`
- Requires: API running on localhost:8000, sample videos in `cctv_samples/`

**`create_realistic_test_videos.py`**
- Generates synthetic test videos for development/demo
- Creates normal pedestrian patterns and anomalous behaviors
- Outputs to `test_videos/` directory
- Usage: `python scripts/create_realistic_test_videos.py`

### Deployment & Maintenance

**`prepare_deployment.py`**
- Pre-deployment validation and preparation
- Checks model files, dependencies, configuration
- Usage: `python scripts/prepare_deployment.py`

## Usage Patterns

### Quick Demo Setup
```bash
# Generate test videos
python scripts/create_realistic_test_videos.py

# Start API
python app.py

# Analyze test footage
# Upload videos from test_videos/ via web interface
```

### Production Calibration
```bash
# Place your CCTV samples in cctv_samples/
# Start API: python app.py
# Run calibration
python scripts/calibrate_for_real_world.py
```

## Integration with Main Application

These scripts are **standalone utilities** and do not affect the core application runtime. They are development/maintenance tools that:
- Do not need to be deployed to production
- Are not imported by the main application
- Can be run independently as needed

## Dependencies

All scripts use the same dependencies as the main application (from `requirements.txt`). No additional packages required.
