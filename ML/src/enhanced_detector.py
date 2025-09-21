"""
Enhanced Vehicle Detector for Traffic Signal Control
Detects both moving and stationary vehicles for accurate traffic density measurement.
Designed specifically for traffic signal timing optimization.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVehicleDetector:
    """
    Enhanced vehicle detector that tracks both moving and stationary vehicles.
    Uses temporal tracking to maintain vehicle counts for traffic signal control.
    """
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.1):
        """
        Initialize the enhanced vehicle detector.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
        
        # Vehicle tracking for stationary detection
        self.vehicle_tracks = {}  # Track vehicles across frames
        self.next_track_id = 0
        self.max_track_age = 30  # Keep tracks for 30 frames
        self.stationary_threshold = 10.0  # Pixels movement threshold
        self.min_track_length = 5  # Minimum frames to consider a track valid
        
        # Frame history for temporal analysis
        self.frame_history = deque(maxlen=10)  # Keep last 10 frames
        self.detection_history = deque(maxlen=10)  # Keep last 10 detection sets
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Statistics
        self.total_vehicles = 0
        self.moving_vehicles = 0
        self.stationary_vehicles = 0
        
    def load_model(self):
        """Load YOLOv8 model."""
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Enhanced vehicle detector model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Calculate intersection
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update_vehicle_tracks(self, detections):
        """
        Update vehicle tracks to identify stationary vs moving vehicles.
        
        Args:
            detections (list): Current frame detections
            
        Returns:
            dict: Updated tracking information
        """
        current_time = time.time()
        
        # Match current detections with existing tracks
        matched_tracks = set()
        new_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            centroid = detection['centroid']
            best_match_id = None
            best_iou = 0.0
            
            # Find best matching existing track
            for track_id, track_info in self.vehicle_tracks.items():
                if track_info['active']:
                    track_bbox = track_info['last_bbox']
                    iou = self.calculate_iou(bbox, track_bbox)
                    
                    if iou > 0.3 and iou > best_iou:  # IoU threshold for matching
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.vehicle_tracks[best_match_id]
                
                # Calculate movement
                last_centroid = track['centroids'][-1] if track['centroids'] else centroid
                movement = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                                 (centroid[1] - last_centroid[1])**2)
                
                # Update track information
                track['last_bbox'] = bbox
                track['centroids'].append(centroid)
                track['movements'].append(movement)
                track['last_seen'] = current_time
                track['age'] += 1
                
                # Keep only recent history
                if len(track['centroids']) > 20:
                    track['centroids'] = track['centroids'][-20:]
                    track['movements'] = track['movements'][-20:]
                
                # Determine if vehicle is stationary
                if len(track['movements']) >= self.min_track_length:
                    avg_movement = np.mean(track['movements'][-self.min_track_length:])
                    track['is_stationary'] = avg_movement < self.stationary_threshold
                else:
                    track['is_stationary'] = False
                
                detection['track_id'] = best_match_id
                detection['is_stationary'] = track['is_stationary']
                detection['track_age'] = track['age']
                
                matched_tracks.add(best_match_id)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.vehicle_tracks[track_id] = {
                    'id': track_id,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_bbox': bbox,
                    'centroids': [centroid],
                    'movements': [0.0],
                    'is_stationary': False,
                    'active': True,
                    'age': 1,
                    'class': detection['class'],
                    'confidence': detection['confidence']
                }
                
                detection['track_id'] = track_id
                detection['is_stationary'] = False
                detection['track_age'] = 1
        
        # Mark unmatched tracks as inactive
        for track_id, track_info in self.vehicle_tracks.items():
            if track_id not in matched_tracks:
                track_info['age'] += 1
                if track_info['age'] > self.max_track_age:
                    track_info['active'] = False
        
        # Clean up old inactive tracks
        active_tracks = {k: v for k, v in self.vehicle_tracks.items() 
                        if v['active'] or v['age'] <= self.max_track_age * 2}
        self.vehicle_tracks = active_tracks
        
        return detections
    
    def detect_vehicles_enhanced(self, frame):
        """
        Enhanced vehicle detection with stationary vehicle tracking.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (detections_with_tracking, annotated_frame, statistics)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Store frame in history
        self.frame_history.append(frame.copy())
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if cls in self.vehicle_classes:
                        x1, y1, x2, y2 = box.astype(int)
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        detection = {
                            'bbox': np.array([x1, y1, x2, y2]),
                            'confidence': float(conf),
                            'class': int(cls),
                            'centroid': centroid
                        }
                        detections.append(detection)
        
        # Update vehicle tracking
        detections_with_tracking = self.update_vehicle_tracks(detections)
        
        # Store detection history
        self.detection_history.append(detections_with_tracking)
        
        # Calculate statistics
        self.total_vehicles = len(detections_with_tracking)
        self.moving_vehicles = sum(1 for d in detections_with_tracking if not d.get('is_stationary', False))
        self.stationary_vehicles = sum(1 for d in detections_with_tracking if d.get('is_stationary', False))
        
        # Create annotated frame
        annotated_frame = self.draw_enhanced_annotations(frame, detections_with_tracking)
        
        # Update FPS
        self.update_fps()
        
        processing_time = (time.time() - start_time) * 1000
        
        statistics = {
            'total_vehicles': self.total_vehicles,
            'moving_vehicles': self.moving_vehicles,
            'stationary_vehicles': self.stationary_vehicles,
            'active_tracks': len([t for t in self.vehicle_tracks.values() if t['active']]),
            'processing_time_ms': processing_time,
            'fps': self.current_fps
        }
        
        return detections_with_tracking, annotated_frame, statistics
    
    def draw_enhanced_annotations(self, frame, detections):
        """
        Draw enhanced annotations showing both moving and stationary vehicles.
        
        Args:
            frame (np.ndarray): Input frame
            detections (list): Detections with tracking info
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color scheme
        colors = {
            2: (255, 0, 0),    # car - blue
            3: (0, 255, 0),    # motorcycle - green
            5: (0, 0, 255),    # bus - red
            7: (255, 255, 0)   # truck - cyan
        }
        
        class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            conf = detection['confidence']
            cls = detection['class']
            centroid = detection['centroid']
            is_stationary = detection.get('is_stationary', False)
            track_id = detection.get('track_id', -1)
            track_age = detection.get('track_age', 0)
            
            # Choose color based on movement status
            base_color = colors.get(cls, (255, 255, 255))
            if is_stationary:
                # Red border for stationary vehicles
                border_color = (0, 0, 255)
                thickness = 3
            else:
                # Green border for moving vehicles
                border_color = (0, 255, 0)
                thickness = 2
            
            # Draw bounding box with status-based color
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), border_color, thickness)
            
            # Draw inner box with vehicle type color
            cv2.rectangle(annotated_frame, (bbox[0]+2, bbox[1]+2), (bbox[2]-2, bbox[3]-2), base_color, 1)
            
            # Draw centroid
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 4, border_color, -1)
            
            # Create label with enhanced information
            status = "STOP" if is_stationary else "MOVE"
            label = f"{class_names.get(cls, 'vehicle')} {status} ID:{track_id} ({conf:.2f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), border_color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add enhanced statistics
        self.update_fps()
        
        # Statistics panel
        stats_y = 30
        cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        stats_y += 25
        cv2.putText(annotated_frame, f"Total Vehicles: {self.total_vehicles}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        stats_y += 25
        cv2.putText(annotated_frame, f"Moving: {self.moving_vehicles}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        stats_y += 25
        cv2.putText(annotated_frame, f"Stationary: {self.stationary_vehicles}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        stats_y += 25
        active_tracks = len([t for t in self.vehicle_tracks.values() if t['active']])
        cv2.putText(annotated_frame, f"Active Tracks: {active_tracks}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated_frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_vehicle_summary(self):
        """Get summary of vehicle tracking statistics."""
        active_tracks = [t for t in self.vehicle_tracks.values() if t['active']]
        
        summary = {
            'total_active_tracks': len(active_tracks),
            'stationary_count': len([t for t in active_tracks if t.get('is_stationary', False)]),
            'moving_count': len([t for t in active_tracks if not t.get('is_stationary', False)]),
            'vehicle_types': {},
            'long_term_stationary': len([t for t in active_tracks 
                                       if t.get('is_stationary', False) and t['age'] > 15])
        }
        
        # Count by vehicle type
        for track in active_tracks:
            cls = track['class']
            class_name = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}.get(cls, 'unknown')
            summary['vehicle_types'][class_name] = summary['vehicle_types'].get(class_name, 0) + 1
        
        return summary


def main():
    """Test the enhanced vehicle detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Vehicle Detector with Stationary Detection')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EnhancedVehicleDetector(model_path=args.model, conf_threshold=args.conf)
    detector.load_model()
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if needed
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    logger.info(f"Processing video: {width}x{height} @ {fps}FPS")
    logger.info("Press 'q' to quit, 's' to show summary")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhanced detection
            detections, annotated_frame, stats = detector.detect_vehicles_enhanced(frame)
            
            # Save frame if writer available
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Enhanced Vehicle Detection - Moving & Stationary', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                summary = detector.get_vehicle_summary()
                logger.info("=== Vehicle Summary ===")
                for key, value in summary.items():
                    logger.info(f"{key}: {value}")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final summary
        summary = detector.get_vehicle_summary()
        logger.info("=== Final Vehicle Summary ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
