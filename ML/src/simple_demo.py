"""
Simple ML Traffic Control Demo - GUARANTEED TO WORK
Uses custom traffic-trained model for better vehicle detection.
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Import components
from detector import VehicleDetector
from counter import VehicleCounter
from timing import TrafficTimingCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Simple main function that guarantees video analysis works."""
    
    # Video path
    video_path = "data/videos/demo_scenarios/reference_indian_intersection.mp4"
    
    print("🚀 Starting Custom Traffic Model Demo...")
    print("=" * 60)
    
    # Initialize components with custom model
    detector = VehicleDetector(model_path="Model/model.pt", conf_threshold=0.25)
    counter = VehicleCounter("ML/rois.json")
    timing_calc = TrafficTimingCalculator()
    
    # Load custom ML model
    print("🧠 Loading custom traffic model...")
    detector.load_model()
    print("✅ Custom traffic model loaded")
    
    # Open video
    print(f"📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return
    
    print("✅ Video opened successfully")
    print("=" * 60)
    print("🎯 VIDEO ANALYSIS WINDOW WILL APPEAR:")
    print("   - Custom traffic model detection")
    print("   - ROI-based vehicle counting")
    print("   - Real-time traffic analysis")
    print("💡 Press 'q' to quit, 's' for statistics")
    print("=" * 60)
    
    frame_count = 0
    last_update = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Custom ML Detection
            detections = detector.detect_vehicles(frame)
            total_detections += len(detections)
            
            # Count vehicles per direction
            direction_counts = counter.count_vehicles_in_rois(detections)
            
            # Calculate traffic timing every 5 seconds
            current_time = time.time()
            if current_time - last_update >= 5.0:
                green_times = timing_calc.calculate_green_times(direction_counts)
                
                total_vehicles = sum(direction_counts.values())
                command = f"N={green_times.get('North', 8)};E={green_times.get('East', 8)};S={green_times.get('South', 8)};W={green_times.get('West', 8)}"
                
                print(f"🎯 Traffic Analysis Update #{frame_count//125 + 1}:")
                print(f"   🚗 Vehicles: {direction_counts} (Total: {total_vehicles})")
                print(f"   🚦 Timing: {command}")
                print("-" * 40)
                
                last_update = current_time
            
            # Create enhanced video display
            display_frame = detector.draw_detections(frame, detections)
            display_frame = counter.draw_rois(display_frame)
            
            # Add traffic analysis overlay
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Traffic Analysis: {direction_counts}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # SHOW VIDEO WINDOW
            cv2.imshow('🎯 Custom Traffic Model Analysis', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Demo stopped by user")
                break
            elif key == ord('s'):
                print(f"\n📊 STATISTICS:")
                print(f"   🎬 Frames processed: {frame_count}")
                print(f"   🚗 Total detections: {total_detections}")
                print(f"   📊 Current traffic: {direction_counts}")
                print()
    
    except KeyboardInterrupt:
        print("🛑 Demo interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Custom Traffic Model Demo completed!")
        print(f"📊 Final Stats: {frame_count} frames, {total_detections} detections")

if __name__ == "__main__":
    main()
