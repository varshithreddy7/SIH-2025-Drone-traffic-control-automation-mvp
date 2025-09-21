"""
YOLOv8-based vehicle detection module for traffic signal automation.
Handles video input, runs inference, and displays annotated results.
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
    YOLOv8-based vehicle detector for traffic monitoring.
    """
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.1):  # Lower for Indian traffic
        """
        Initialize the vehicle detector.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_model(self):
        """Load YOLOv8 model."""
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (detections, annotated_frame)
        """
        if self.model is None:
            self.load_model()
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Filter for vehicle classes only
        vehicle_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, cls in enumerate(boxes.cls):
                    if int(cls) in self.vehicle_classes:
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        vehicle_detections.append({
                            'bbox': box,
                            'confidence': conf,
                            'class': int(cls),
                            'centroid': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        })
        
        # Create annotated frame
        annotated_frame = self.annotate_frame(frame.copy(), vehicle_detections)
        
        return vehicle_detections, annotated_frame
    
    def annotate_frame(self, frame, detections):
        """
        Annotate frame with detection results.
        
        Args:
            frame (np.ndarray): Input frame
            detections (list): List of detection dictionaries
            
        Returns:
            np.ndarray: Annotated frame
        """
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        colors = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 0, 255), 7: (255, 255, 0)}
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            cls = det['class']
            centroid = det['centroid']
            
            # Draw bounding box
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw centroid
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, color, -1)
            
            # Draw label
            label = f"{class_names.get(cls, 'vehicle')}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add FPS counter
        self.update_fps()
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection count
        cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
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
                detections, annotated_frame = self.detect_vehicles(frame)
                
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
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.1,
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
