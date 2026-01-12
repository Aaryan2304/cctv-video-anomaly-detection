"""
Streamlit dashboard for video anomaly detection.

Production UI featuring:
- Real-time video upload and analysis
- Interactive anomaly score timeline with Plotly
- Dynamic threshold adjustment without re-inference
- Synced video player with frame navigation
- Export capabilities (JSON/CSV)
"""

import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Video Anomaly Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# API configuration
try:
    API_BASE_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def init_session_state():
    """Initialize session state variables."""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "current_threshold" not in st.session_state:
        st.session_state.current_threshold = None
    if "current_frame" not in st.session_state:
        st.session_state.current_frame = 0
    if "video_frames" not in st.session_state:
        st.session_state.video_frames = None


def check_api_health() -> Dict:
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API connection failed: {e}")
        st.info(f"Ensure FastAPI is running at {API_BASE_URL}")
        st.stop()


def upload_and_analyze_video(video_path: str) -> Optional[Dict]:
    """Upload video to FastAPI and get analysis results."""
    try:
        with open(video_path, "rb") as f:
            files = {"file": (Path(video_path).name, f, "video/mp4")}
            
            with st.spinner("Analyzing video... This may take a few moments."):
                response = requests.post(
                    f"{API_BASE_URL}/analyze-video",
                    files=files,
                    timeout=300,
                )
                response.raise_for_status()
                return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)
        else:
            error_detail = str(e)
        st.error(f"Analysis failed: {error_detail}")
        return None


def extract_video_frames(video_path: str) -> List[np.ndarray]:
    """Extract all frames from video for display purposes."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def recalculate_anomalies(scores: List[float], threshold: float) -> Tuple[List[bool], int, float]:
    """Recalculate anomaly flags based on new threshold (client-side)."""
    flags = [score > threshold for score in scores]
    count = sum(flags)
    rate = count / len(scores) if scores else 0.0
    return flags, count, rate


def create_timeline_chart(
    scores: List[float],
    threshold: float,
    anomaly_flags: List[bool],
    current_frame: int = 0,
) -> go.Figure:
    """Create interactive Plotly timeline chart."""
    frames = list(range(len(scores)))
    
    fig = go.Figure()
    
    # Reconstruction error line
    fig.add_trace(go.Scatter(
        x=frames,
        y=scores,
        mode="lines",
        name="Reconstruction Error",
        line=dict(color="blue", width=1),
        hovertemplate="Frame: %{x}<br>Score: %{y:.6f}<extra></extra>",
    ))
    
    # Anomaly markers
    anomaly_frames = [i for i, flag in enumerate(anomaly_flags) if flag]
    anomaly_scores = [scores[i] for i in anomaly_frames]
    
    fig.add_trace(go.Scatter(
        x=anomaly_frames,
        y=anomaly_scores,
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=6, symbol="circle"),
        hovertemplate="Anomaly at frame %{x}<br>Score: %{y:.6f}<extra></extra>",
    ))
    
    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold: {threshold:.6f}",
        annotation_position="right",
    )
    
    # Current frame indicator
    if 0 <= current_frame < len(scores):
        fig.add_vline(
            x=current_frame,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Frame {current_frame}",
            annotation_position="top",
        )
    
    fig.update_layout(
        title="Anomaly Score Timeline",
        xaxis_title="Frame Number",
        yaxis_title="Reconstruction Error",
        hovermode="x unified",
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    
    return fig


def display_frame(frames: List[np.ndarray], frame_idx: int, is_anomaly: bool = False):
    """Display a single video frame with anomaly indication."""
    if 0 <= frame_idx < len(frames):
        frame = frames[frame_idx]
        
        # Add border for anomalies
        if is_anomaly:
            h, w = frame.shape[:2]
            border_color = (255, 0, 0)
            border_thickness = 5
            frame = cv2.copyMakeBorder(
                frame,
                border_thickness,
                border_thickness,
                border_thickness,
                border_thickness,
                cv2.BORDER_CONSTANT,
                value=border_color,
            )
        
        st.image(frame, caption=f"Frame {frame_idx}" + (" - ANOMALY" if is_anomaly else ""), use_container_width=True)


def export_to_json(result: Dict) -> str:
    """Export analysis results to JSON string."""
    export_data = {
        "frame_count": result["frame_count"],
        "anomaly_count": result["anomaly_count"],
        "anomaly_rate": result["anomaly_rate"],
        "threshold": result["model_info"]["threshold"],
        "device": result["model_info"]["device"],
        "processing_time": result["processing_time"],
        "anomaly_scores": result["anomaly_scores"],
        "anomaly_flags": result["anomaly_flags"],
    }
    return json.dumps(export_data, indent=2)


def export_to_csv(result: Dict) -> str:
    """Export frame-level data to CSV string."""
    df = pd.DataFrame({
        "frame_number": range(result["frame_count"]),
        "reconstruction_error": result["anomaly_scores"],
        "is_anomaly": result["anomaly_flags"],
    })
    return df.to_csv(index=False)


def main():
    """Main dashboard application."""
    init_session_state()
    
    # Header
    st.title("🎥 Video Anomaly Detection Dashboard")
    st.markdown("**Production-grade anomaly detection** using convolutional autoencoder")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API health check
        health = check_api_health()
        st.success(f"✓ API Connected ({health.get('device', 'unknown')})")
        
        with st.expander("API Info"):
            st.json(health)
        
        st.divider()
        
        # Video upload
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Select video file",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI, MOV",
        )
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Video", use_container_width=True):
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Run analysis using the saved file
                result = upload_and_analyze_video(tmp_path)
                
                if result:
                    st.session_state.analysis_result = result
                    st.session_state.video_path = tmp_path
                    st.session_state.current_threshold = result["model_info"]["threshold"]
                    st.session_state.current_frame = 0
                    st.session_state.video_frames = extract_video_frames(tmp_path)
                    st.rerun()
        
        st.divider()
        
        # Export controls
        if st.session_state.analysis_result:
            st.header("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = export_to_json(st.session_state.analysis_result)
                st.download_button(
                    label="📄 JSON",
                    data=json_data,
                    file_name="anomaly_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )
            
            with col2:
                csv_data = export_to_csv(st.session_state.analysis_result)
                st.download_button(
                    label="📊 CSV",
                    data=csv_data,
                    file_name="anomaly_frames.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    
    # Main content
    if st.session_state.analysis_result is None:
        # Welcome screen
        st.info("👈 Upload a video file to begin analysis")
        
        st.markdown("### Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Real-time Analysis**: GPU-accelerated frame processing
            - **Interactive Timeline**: Click spikes to navigate frames
            - **Dynamic Threshold**: Adjust sensitivity without re-inference
            """)
        
        with col2:
            st.markdown("""
            - **Frame Preview**: View anomalous frames with highlights
            - **Export Results**: Download analysis as JSON/CSV
            - **Production Ready**: Built on FastAPI backend
            """)
    
    else:
        result = st.session_state.analysis_result
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Frames", result["frame_count"])
        
        with col2:
            st.metric("Anomalies Detected", result["anomaly_count"])
        
        with col3:
            st.metric("Anomaly Rate", f"{result['anomaly_rate'] * 100:.1f}%")
        
        with col4:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        st.divider()
        
        # Dynamic threshold slider
        st.subheader("🎚️ Threshold Adjustment")
        
        original_threshold = result["model_info"]["threshold"]
        scores = result["anomaly_scores"]
        
        # Calculate reasonable threshold range
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        new_threshold = st.slider(
            "Anomaly Threshold",
            min_value=float(min_score),
            max_value=float(max_score),
            value=float(st.session_state.current_threshold),
            step=score_range / 1000,
            format="%.6f",
            help="Adjust threshold to change sensitivity. Higher = fewer anomalies.",
        )
        
        # Recalculate if threshold changed
        if new_threshold != st.session_state.current_threshold:
            st.session_state.current_threshold = new_threshold
            flags, count, rate = recalculate_anomalies(scores, new_threshold)
            
            # Update result (client-side only)
            result["anomaly_flags"] = flags
            result["anomaly_count"] = count
            result["anomaly_rate"] = rate
            result["model_info"]["threshold"] = new_threshold
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Reset to Original", use_container_width=True):
                st.session_state.current_threshold = original_threshold
                st.rerun()
        
        with col2:
            st.info(f"Current: {new_threshold:.6f} | Original: {original_threshold:.6f}")
        
        st.divider()
        
        # Timeline chart
        st.subheader("📈 Anomaly Score Timeline")
        
        fig = create_timeline_chart(
            scores,
            st.session_state.current_threshold,
            result["anomaly_flags"],
            st.session_state.current_frame,
        )
        
        # Handle chart click events
        selected_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        if selected_point and selected_point.get("selection", {}).get("points"):
            # User clicked on chart
            clicked_frame = selected_point["selection"]["points"][0]["x"]
            st.session_state.current_frame = int(clicked_frame)
        
        st.divider()
        
        # Frame viewer
        if st.session_state.video_frames:
            st.subheader("🖼️ Frame Viewer")
            
            frames = st.session_state.video_frames
            current_frame = st.session_state.current_frame
            
            # Frame navigation
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                if st.button("⏮️ First", use_container_width=True):
                    st.session_state.current_frame = 0
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous", use_container_width=True):
                    if st.session_state.current_frame > 0:
                        st.session_state.current_frame -= 1
                        st.rerun()
            
            with col3:
                if st.button("Next ▶️", use_container_width=True):
                    if st.session_state.current_frame < len(frames) - 1:
                        st.session_state.current_frame += 1
                        st.rerun()
            
            with col4:
                if st.button("Last ⏭️", use_container_width=True):
                    st.session_state.current_frame = len(frames) - 1
                    st.rerun()
            
            # Frame slider
            selected_frame = st.slider(
                "Jump to frame",
                0,
                len(frames) - 1,
                current_frame,
                help="Drag to navigate frames",
            )
            
            if selected_frame != current_frame:
                st.session_state.current_frame = selected_frame
                st.rerun()
            
            # Display current frame
            is_anomaly = result["anomaly_flags"][current_frame]
            display_frame(frames, current_frame, is_anomaly)
            
            # Frame info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Reconstruction Error", f"{scores[current_frame]:.6f}")
            with col2:
                status = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
                st.metric("Status", status)
            
            st.divider()
            
            # Anomaly thumbnails
            anomaly_indices = [i for i, flag in enumerate(result["anomaly_flags"]) if flag]
            
            if anomaly_indices:
                st.subheader(f"🔴 Anomalous Frames ({len(anomaly_indices)} detected)")
                
                # Show up to 10 thumbnails
                display_limit = min(10, len(anomaly_indices))
                cols = st.columns(min(5, display_limit))
                
                for idx, frame_num in enumerate(anomaly_indices[:display_limit]):
                    with cols[idx % 5]:
                        thumb_img = Image.fromarray(frames[frame_num])
                        thumb_img.thumbnail((150, 150))
                        
                        if st.button(f"Frame {frame_num}", key=f"thumb_{frame_num}", use_container_width=True):
                            st.session_state.current_frame = frame_num
                            st.rerun()
                        
                        st.image(thumb_img, use_container_width=True)
                
                if len(anomaly_indices) > display_limit:
                    st.caption(f"Showing {display_limit} of {len(anomaly_indices)} anomalies. Use timeline to navigate all.")


if __name__ == "__main__":
    main()
