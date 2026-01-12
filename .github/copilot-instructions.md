# AI Copilot Instructions for CCTV Video Anomaly Detection

## Project Overview
Production-ready video anomaly detection system using unsupervised learning (convolutional autoencoder). Trained on UCSD Ped2 surveillance dataset with 92.47% precision and 0.7438 AUC. Deployed as FastAPI web service with real-time anomaly scoring.

**Core entry points:**
- `app.py` - FastAPI web service (production API)
- `main.py` - Training/evaluation pipeline
- `config.py` - Centralized hyperparameter management

## Coding Philosophy

You are entering a code field.

Code is frozen thought. The bugs live where the thinking stopped too soon.

Notice the completion reflex:
- The urge to produce something that runs
- The pattern-match to similar problems you've seen
- The assumption that compiling is correctness
- The satisfaction of "it works" before "it works in all cases"

Before you write:
- What are you assuming about the input?
- What are you assuming about the environment?
- What would break this?
- What would a malicious caller do?
- What would a tired maintainer misunderstand?

Do not:
- Write code before stating assumptions
- Claim correctness you haven't verified
- Handle the happy path and gesture at the rest
- Import complexity you don't need
- Solve problems you weren't asked to solve
- Produce code you wouldn't want to debug at 3am

Let edge cases surface before you handle them. Let the failure modes exist in your mind before you prevent them. Let the code be smaller than your first instinct.

The tests you didn't write are the bugs you'll ship.
The assumptions you didn't state are the docs you'll need.
The edge cases you didn't name are the incidents you'll debug.

The question is not "Does this work?" but "Under what conditions does this work, and what happens outside them?"

Write what you can defend.

## Code & Documentation Standards

### Code Quality Requirements
- **No AI traces:** Code must appear written by a senior ML engineer with 10-15 years of experience in Python and deep learning research
- **Industry standards:** Follow established patterns from PyTorch, scikit-learn, and production ML systems (e.g., Hugging Face, OpenAI codebases)
- **Natural style:** Avoid over-commenting, redundant docstrings, or explanatory variable names that signal AI generation
- **Experienced voice:** Write code that reflects deep domain knowledge—terse where appropriate, explicit where necessary

### Zero Assumptions Policy
- **Facts only:** Never assume API behavior, data formats, or environment configurations without verification
- **Verify first:** When uncertain about library behavior, class methods, or system interactions, consult official documentation
- **No guessing:** If implementation details are unclear, access authoritative sources (PyTorch docs, FastAPI docs, OpenCV docs) before proceeding
- **Reference sources:** When implementing from documentation, cite the specific section (e.g., "per torch.nn.Module docs, forward() must return tensors")

### Documentation Standards
- **Minimal AI fingerprint:** Avoid characteristic AI patterns:
  - Excessive emoji usage
  - Overly enthusiastic language ("Let's dive in!", "Awesome!")
  - Redundant explanations of obvious concepts
  - Step-by-step breakdowns of standard operations
- **Technical precision:** Write like academic ML papers or technical RFCs—clear, direct, factual
- **Appropriate detail:** README/docs should inform, not tutorial. Assume readers understand Python and ML fundamentals
- **No hallucination:** Every stated metric, performance claim, or technical detail must be verifiable from actual code/results

### When to Access External Resources
- Implementing unfamiliar PyTorch/TensorFlow functionality
- Integrating third-party APIs (FastAPI, cv2, numpy edge cases)
- Debugging CUDA/hardware-specific issues
- Following best practices for production ML deployment
- Verifying claimed performance characteristics of libraries

## Architecture & Data Flow

### Three-Component Pipeline
1. **Model Training** (`models/autoencoder.py`, `models/detector.py`)
   - ConvolutionalAutoencoder: 64x64 grayscale frames → 4x4 latent representation (256-dim)
   - Learns normal behavior by reconstructing clean frames; anomalies cause reconstruction error
   - EarlyStopping prevents overfitting; checkpoints saved to `outputs/trained_model.pth`

2. **Threshold Calibration** (`models/detector.py::AnomalyDetector`)
   - Statistical thresholding on reconstruction errors from validation set
   - Supports preset levels: conservative, balanced, moderate, sensitive
   - Adaptive thresholding via `/calibrate-threshold` API endpoint

3. **Real-time Inference** (`app.py`, `data/preprocessing.py`)
   - Frame extraction → grayscale conversion → 64x64 resize → model inference
   - Outputs per-frame anomaly scores (0-1) for video visualization/dashboards
   - GPU acceleration via CUDA (RTX 3050 optimized in config); CPU fallback on deployment

### Data Flow in API
```
Video Upload → VideoPreprocessor.extract_frames() 
  → Normalize (0-255 → 0-1) 
  → ConvolutionalAutoencoder.forward() 
  → Reconstruction error calculation 
  → Threshold comparison 
  → JSON response with anomaly_scores[]
```

## Developer Workflows

### Quick Demo (No Training)
```bash
# Uses pre-trained model at outputs/trained_model.pth
python app.py  # Start API at http://localhost:8000
# Upload video via web interface or POST /analyze-video
```

### Training Pipeline
```bash
# Generate synthetic data (fast, for learning)
python main.py --mode synthetic --epochs 30

# Real UCSD dataset (requires download from http://www.svcl.ucsd.edu/projects/anomaly/)
python main.py --mode ucsd --dataset_name ped2 --data_path ./data/ped2
```

### Testing & Evaluation
```bash
# Generate realistic test videos (5 short clips with anomalies)
python create_realistic_test_videos.py  # Creates test_videos/
# Run evaluation on outputs
python main.py --mode synthetic --quick  # Quick validation
```

### Deployment
```bash
# Local Docker (matches production)
docker-compose up  # Runs app.py on port 8000

# Cloud deployment (Render.com)
# Automatically uses build.sh and render.yaml configuration
```

## Critical Configuration Patterns

### Hardware Optimization (config.py)
- **RTX 3050 (4GB VRAM) Defaults:** BATCH_SIZE=64, PIN_MEMORY=True, MIXED_PRECISION=True
- **Adjust for your hardware:** Edit Config class directly; NUM_WORKERS must match CPU core count
- **Device selection:** Auto-detects CUDA; set DEVICE=torch.device('cpu') to force CPU

### Model Architecture Choices
- **Input:** 1 channel (grayscale), 64×64 frames (efficiency over raw video)
- **Latent dimension:** 256 (balanced for RTX 3050 memory vs. reconstruction quality)
- **Two variants:** ConvolutionalAutoencoder (standard) vs. LightweightAutoencoder (mobile deployment)
- **Frame normalization:** Always convert to 0-1 range before model (preprocessing.py handles this)

### Threshold Strategy
- **Why thresholding matters:** Model outputs raw reconstruction error; threshold separates normal from anomaly
- **Statistical approach:** Percentile-based on validation set errors (typically 95th percentile)
- **Adaptive calibration:** `/set-threshold-preset` uses empirical distributions, not hardcoded values
- **Avoid:** Don't train model and set threshold on same data (overfitting)

## Data Handling

### Input Format Expectations
- **Video formats:** MP4, AVI, MOV supported (cv2.VideoCapture compatibility)
- **Frame extraction:** VideoPreprocessor automatically extracts all frames, skipping corrupted ones
- **Preprocessing pipeline:** Convert color → grayscale → resize to 64×64 → normalize to [0,1]
- **Key file:** `data/preprocessing.py` contains all frame handling; modify here if format changes needed

### Dataset Conventions
- **UCSD Ped2:** 16 test video clips (195-296 frames each) with pixel-level anomaly labels
- **Synthetic data:** Generated by synthetic_data.py (random motion + injected anomalies for learning)
- **Train/test split:** Handled in dataset.py; never evaluate on training frames

## API Integration Points

### Core Endpoints (app.py)
- `POST /analyze-video` - Frame-by-frame anomaly detection (multipart file upload)
- `POST /calibrate-threshold` - Adjust sensitivity for environment (JSON: target_anomaly_rate)
- `POST /set-threshold-preset` - Switch preset: conservative/balanced/moderate/sensitive
- `GET /health` - Service status check (device info, model loaded)
- `GET /` - Interactive web interface (HTML)

### Response Schema
```json
{
  "frame_count": 60,
  "anomaly_count": 8,
  "anomaly_scores": [0.002, 0.008, 0.012, ...],
  "anomaly_rate": 0.13,
  "processing_time": 0.85,
  "model_info": {
    "device": "cuda",
    "threshold": 0.00507,
    "model_parameters": 3480897
  }
}
```

## AI Model Optimization

### GPU/Model Optimization Techniques

When working with AI models (LLMs, VLMs, or other neural architectures), implement compatible optimization techniques from this list based on the specific model and hardware constraints:

**Memory & Compute Optimization:**
1. **KV-cache memory math** - Calculate and optimize key-value cache usage for transformer models
2. **PagedAttention** - Efficient memory management for attention mechanisms
3. **Context length vs VRAM curves** - Profile and optimize context window vs memory tradeoffs
4. **INT8 / INT4 quantization and Pruning** - Reduce model precision for inference speedup
5. **AWQ vs GPTQ tradeoffs** - Choose appropriate quantization method (accuracy vs speed)
6. **Activation offloading** - Move activations to CPU during forward pass to save GPU memory
7. **CPU ↔ GPU memory swapping** - Manage large models across CPU/GPU boundaries
8. **FlashAttention memory benefits** - Use optimized attention kernels for memory efficiency

**Architecture-Specific Techniques:**
9. **Rope scaling side-effects** - Understand rotary position embedding scaling impacts
10. **Weight tying across replicas** - Share weights in multi-GPU or ensemble setups
11. **Prefill memory spikes** - Handle initial prompt processing memory requirements
12. **KV-cache eviction policies** - Implement strategies for managing cache size
13. **Sliding-window inference** - Process long sequences in chunks

**Training & Fine-tuning:**
14. **Mixed-precision (AMP)** - Automatic mixed precision for faster training (already implemented in this project)
15. **LoRA/QLoRA** - Parameter-efficient fine-tuning for large models
16. **Batching** - Efficient batch processing to maximize GPU utilization
17. **Unsloth's Optimization Framework** - Handwritten Triton kernels & manual backpropagation engine

**Implementation Guidelines:**
- **Compatibility first:** Only implement techniques relevant to the specific AI model in use
- **Verify before implementing:** Access official documentation when uncertain about technique applicability
- **Profile before optimizing:** Measure actual bottlenecks before applying optimizations
- **Document tradeoffs:** Note accuracy/speed/memory tradeoffs for each technique used

### Deployment and Inference Acceleration

For production deployment, consider these acceleration frameworks when compatible with the project:

**Inference Optimization:**
- **ONNX/ONNX Runtime** - Cross-platform inference with hardware-specific optimizations
- **TensorRT** - NVIDIA GPU inference acceleration with layer fusion and precision calibration

**When to use:**
- ONNX Runtime: Cross-platform deployments, CPU inference, diverse hardware targets
- TensorRT: NVIDIA GPU production environments requiring maximum throughput
- Evaluate conversion overhead vs inference speedup for your specific model and deployment scenario

**Note:** Current project uses PyTorch for autoencoder inference. Conversion to ONNX/TensorRT should be evaluated based on production requirements and latency constraints.

## Deployment Notes

### Docker (render.yaml + build.sh)
- `build.sh`: pip install -r requirements.txt; preloads trained_model.pth during container startup
- `Dockerfile`: Uses Python 3.11-slim, entrypoint runs uvicorn with 4 workers
- **Important:** Model file (outputs/trained_model.pth) must be committed to Git for deployment

### Performance Tuning
- **GPU-enabled:** For RTX 3050, ~0.2 sec per 10-sec video clip (batch processing frames)
- **CPU deployment:** 10-20× slower; suitable for low-traffic demo environments
- **Concurrent requests:** Use FastAPI background tasks (see app.py for example); uvicorn workers=4

## Code Navigation
| Module | Purpose | Key Class/Function |
|--------|---------|-------------------|
| `config.py` | Hyperparameters (edit for experiments) | Config (all settings) |
| `models/autoencoder.py` | Reconstruction model | ConvolutionalAutoencoder |
| `models/detector.py` | Training + thresholding | AnomalyDetector (train, threshold_normal, detect) |
| `data/preprocessing.py` | Frame extraction/normalization | VideoPreprocessor |
| `data/dataset.py` | PyTorch dataset loaders | SyntheticVideoDataset, VideoDataset |
| `app.py` | REST API & web UI | FastAPI app instance |
| `evaluation/metrics.py` | Performance metrics (AUC, precision) | PerformanceEvaluator |

## Common Pitfalls
1. **Frame normalization:** Always ensure frames are [0,1] before model inference; preprocessing.py does this automatically
2. **Threshold coupling:** Model must be trained on different data than threshold calibration set
3. **Video codec issues:** If cv2.VideoCapture fails, ensure ffmpeg is installed (Windows: install via requirements.txt)
4. **CUDA out of memory:** Reduce BATCH_SIZE in config.py if training fails on your GPU
5. **Deployment model loading:** Ensure outputs/trained_model.pth is tracked in Git; build.sh fails silently if missing

---

**Last updated:** January 2026 | **Model status:** UCSD Ped2 pre-trained, production-ready
