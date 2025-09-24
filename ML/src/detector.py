"""
Custom Traffic Vehicle Detection Module for Traffic Signal Automation.
Uses pre-trained PyTorch model specifically trained on traffic data.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    Custom traffic vehicle detector using pre-trained traffic model.
    """
    
    def __init__(self, model_path="Model/model.pt", conf_threshold=0.25):  # Higher confidence for custom model
        """
        Initialize the vehicle detector with custom traffic model.
        
        Args:
            model_path (str): Path to custom PyTorch model weights (default: Model/model.pt)
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
        # Custom model is trained specifically for traffic vehicles
        # All detections from this model are vehicles, so no class filtering needed
        self.is_custom_traffic_model = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        logger.info(f"🚗 Initialized Custom Traffic Vehicle Detector")
        logger.info(f"📁 Model Path: {self.model_path}")
        logger.info(f"🎯 Confidence Threshold: {self.conf_threshold}")
        
    def load_model(self):
        """Load custom traffic-trained PyTorch model."""
        try:
            model_full_path = Path(self.model_path)
            if not model_full_path.exists():
                logger.error(f"❌ Custom model not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"🧠 Loading Custom Traffic Model: {self.model_path}")
            logger.info(f"📊 Model Size: {model_full_path.stat().st_size / (1024*1024):.1f} MB")
            
            self.model = YOLO(self.model_path)
            
            logger.info("✅ Custom Traffic Model loaded successfully!")
            logger.info("🚗 Model is specifically trained for traffic vehicle detection")
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom model: {e}")
            logger.error("💡 Make sure Model/model.pt exists and is a valid YOLO model")
            raise
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame using custom traffic model.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            list: List of vehicle detections with enhanced information
        """
        if self.model is None:
            self.load_model()
        
        # Run inference with custom model
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Process detections from custom traffic model
        vehicle_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    
                    # Get class if available, otherwise assume vehicle
                    cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                    
                    # Calculate centroid for ROI intersection
                    centroid_x = (box[0] + box[2]) / 2
                    centroid_y = (box[1] + box[3]) / 2
                    
                    # Calculate area for size analysis
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    
                    vehicle_detections.append({
                        'bbox': box,
                        'confidence': float(conf),
                        'class': cls,
                        'centroid': (centroid_x, centroid_y),
                        'area': area,
                        'width': box[2] - box[0],
                        'height': box[3] - box[1]
                    })
        
        return vehicle_detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection results on frame with enhanced visualization.
        
        Args:
            frame (np.ndarray): Input frame
            detections (list): List of detection dictionaries
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Enhanced colors for better visibility
        colors = [
            (0, 255, 255),    # Bright Cyan
            (255, 0, 255),    # Bright Magenta  
            (0, 255, 0),      # Bright Green
            (255, 255, 0),    # Bright Yellow
            (255, 128, 0),    # Orange
            (128, 255, 0),    # Lime
            (255, 0, 128),    # Pink
            (0, 128, 255)     # Light Blue
        ]
        
        for i, det in enumerate(detections):
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            centroid = det['centroid']
            
            # Use cycling colors for better distinction
            color = colors[i % len(colors)]
            
            # Draw bounding box with thick border
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            # Draw centroid with larger circle
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 8, color, -1)
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 8, (255, 255, 255), 2)
            
            # Enhanced label with confidence
            label = f"Vehicle: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 15), 
                         (bbox[0] + label_size[0] + 10, bbox[1]), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (bbox[0] + 5, bbox[1] - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add performance info
        self.update_fps()
        
        # FPS counter with background
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(annotated_frame, (10, 10), (20 + fps_size[0], 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, fps_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Detection count with background
        count_text = f"Vehicles: {len(detections)}"
        count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(annotated_frame, (10, 50), (20 + count_size[0], 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, count_text, (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Model info
        model_text = "Custom Traffic Model"
        model_size = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_frame, (10, 90), (20 + model_size[0], 115), (0, 0, 0), -1)
        cv2.putText(annotated_frame, model_text, (15, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated_frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file with vehicle detection.
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save output video
            display (bool): Whether to display video window
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect vehicles
                detections = self.detect_vehicles(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Vehicle Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            logger.info(f"Processing complete. Processed {frame_count} frames")


def main():
    """Main function for testing the detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detector')
    parser.add_argument('--video', type=str, default='data/videos/sample.mp4',
                       help='Path to input video file')
    parser.add_argument('--model', type=str, default='Model/model.pt',
                       help='Path to custom PyTorch model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    
    args = parser.parse_args()
    
    # Create detector
    detector = VehicleDetector(model_path=args.model, conf_threshold=args.conf)
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
