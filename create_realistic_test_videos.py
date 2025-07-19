"""
Realistic Test Video Creator
===========================

Creates more realistic test videos that mimic actual surveillance footage
for better demonstration of the anomaly detection system.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_realistic_pedestrian_video(filename, video_type="normal", duration_seconds=5, fps=10):
    """
    Create realistic pedestrian-like test videos
    
    Args:
        filename: Output video filename
        video_type: "normal", "anomaly", or "mixed"
        duration_seconds: Video duration
        fps: Frames per second
    """
    print(f"Creating realistic {video_type} test video: {filename}")
    
    # Video parameters similar to UCSD Ped2
    width, height = 320, 240
    total_frames = duration_seconds * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Background (static surveillance scene)
    background = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Add some static elements (walls, ground texture)
    cv2.rectangle(background, (0, height-50), (width, height), (70, 70, 70), -1)  # Ground
    cv2.rectangle(background, (0, 0), (width, 30), (60, 60, 60), -1)  # Top wall
    
    # Add some texture
    for i in range(0, width, 40):
        cv2.line(background, (i, height-50), (i, height), (80, 80, 80), 1)
    
    for frame_num in range(total_frames):
        frame = background.copy()
        
        if video_type == "normal":
            # Normal pedestrian movement
            create_normal_pedestrian_movement(frame, frame_num, total_frames)
            
        elif video_type == "anomaly":
            # Anomalous behavior
            create_anomalous_behavior(frame, frame_num, total_frames)
            
        elif video_type == "mixed":
            # Mix of normal and anomalous
            if frame_num < total_frames * 0.6:
                create_normal_pedestrian_movement(frame, frame_num, total_frames)
            else:
                create_anomalous_behavior(frame, frame_num, total_frames)
        
        # Add realistic noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Created {filename} with {total_frames} frames")

def create_normal_pedestrian_movement(frame, frame_num, total_frames):
    """Create normal walking patterns"""
    height, width = frame.shape[:2]
    
    # Person 1: Walking left to right
    person1_x = int((frame_num / total_frames) * width)
    person1_y = height - 80
    
    if person1_x < width - 20:
        # Draw simple person shape
        cv2.ellipse(frame, (person1_x, person1_y), (8, 15), 0, 0, 360, (120, 120, 120), -1)  # Body
        cv2.circle(frame, (person1_x, person1_y - 20), 5, (130, 130, 130), -1)  # Head
        
        # Walking motion (legs) - convert to int
        leg_offset = int(3 * np.sin(frame_num * 0.5))
        cv2.ellipse(frame, (person1_x - 3, person1_y + 8), (3, 8), 0, 0, 360, (110, 110, 110), -1)
        cv2.ellipse(frame, (person1_x + 3 + leg_offset, person1_y + 8), (3, 8), 0, 0, 360, (110, 110, 110), -1)
    
    # Person 2: Walking right to left (opposite direction)
    if frame_num > total_frames * 0.3:
        person2_x = width - int(((frame_num - total_frames * 0.3) / (total_frames * 0.7)) * width)
        person2_y = height - 90
        
        if person2_x > 20:
            cv2.ellipse(frame, (person2_x, person2_y), (8, 15), 0, 0, 360, (100, 100, 100), -1)
            cv2.circle(frame, (person2_x, person2_y - 20), 5, (110, 110, 110), -1)

def create_anomalous_behavior(frame, frame_num, total_frames):
    """Create anomalous behaviors"""
    height, width = frame.shape[:2]
    
    # Anomaly 1: Person running (faster, different gait)
    person_x = int((frame_num / total_frames) * width * 2.5) % width  # Much faster
    person_y = height - 85
    
    if person_x < width - 20:
        # Larger, more erratic shape for running
        cv2.ellipse(frame, (person_x, person_y), (10, 18), 0, 0, 360, (140, 140, 140), -1)
        cv2.circle(frame, (person_x, person_y - 25), 6, (150, 150, 150), -1)
        
        # More dramatic leg motion - convert to int
        leg_offset = int(8 * np.sin(frame_num * 1.5))
        cv2.ellipse(frame, (person_x - 5, person_y + 10), (4, 10), 0, 0, 360, (130, 130, 130), -1)
        cv2.ellipse(frame, (person_x + 5 + leg_offset, person_y + 10), (4, 10), 0, 0, 360, (130, 130, 130), -1)
    
    # Anomaly 2: Erratic movement (zigzag pattern)
    if frame_num > total_frames * 0.4:
        zigzag_x = int(width * 0.3 + 50 * np.sin(frame_num * 0.3))
        zigzag_y = int(height - 100 + 20 * np.cos(frame_num * 0.2))
        
        cv2.ellipse(frame, (zigzag_x, zigzag_y), (9, 16), 0, 0, 360, (160, 160, 160), -1)
        cv2.circle(frame, (zigzag_x, zigzag_y - 22), 5, (170, 170, 170), -1)
    
    # Anomaly 3: Large object (unusual for pedestrian area)
    if frame_num > total_frames * 0.6:
        large_obj_x = int(width * 0.7)
        large_obj_y = height - 70
        cv2.rectangle(frame, (large_obj_x, large_obj_y), (large_obj_x + 30, large_obj_y + 40), (180, 180, 180), -1)

def main():
    """Create realistic test videos"""
    output_dir = Path("test_videos")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating realistic test videos for anomaly detection...")
    
    # Create realistic videos
    create_realistic_pedestrian_video(
        output_dir / "realistic_normal.mp4", 
        "normal", 
        duration_seconds=6, 
        fps=10
    )
    
    create_realistic_pedestrian_video(
        output_dir / "realistic_anomaly.mp4", 
        "anomaly", 
        duration_seconds=6, 
        fps=10
    )
    
    create_realistic_pedestrian_video(
        output_dir / "realistic_mixed.mp4", 
        "mixed", 
        duration_seconds=8, 
        fps=10
    )
    
    print("\n✅ Realistic test videos created!")
    print("These videos simulate:")
    print("- Normal: Regular pedestrian walking patterns")
    print("- Anomaly: Running, erratic movement, large objects")
    print("- Mixed: Combination of normal and anomalous behaviors")
    print("\nThese should give more realistic anomaly detection results!")

if __name__ == "__main__":
    main()
