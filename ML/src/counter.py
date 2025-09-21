"""
Vehicle counter module for traffic signal automation.
Counts vehicles per direction based on ROI intersections with YOLO detections.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleCounter:
    """
    Vehicle counter that uses ROIs to count vehicles per traffic direction.
    """
    
    def __init__(self, roi_config_path: str = "ML/rois.json"):
        """
        Initialize the vehicle counter.
        
        Args:
            roi_config_path (str): Path to ROI configuration file
        """
        self.roi_config_path = roi_config_path
        self.rois = {}
        self.directions = ['North', 'East', 'South', 'West']
        self.colors = {
            'North': (0, 255, 0),    # Green
            'East': (255, 0, 0),     # Blue
            'South': (0, 0, 255),    # Red
            'West': (255, 255, 0)    # Cyan
        }
        self.counts = {direction: 0 for direction in self.directions}
        self.load_rois()
    
    def load_rois(self) -> bool:
        """
        Load ROI configuration from file.
        
        Returns:
            bool: Success status
        """
        try:
            if not Path(self.roi_config_path).exists():
                logger.error(f"ROI configuration file not found: {self.roi_config_path}")
                return False
            
            with open(self.roi_config_path, 'r') as f:
                roi_data = json.load(f)
            
            self.rois = roi_data.get('rois', {})
            
            if len(self.rois) != 4:
                logger.error(f"Expected 4 ROIs, found {len(self.rois)}")
                return False
            
            logger.info(f"Loaded {len(self.rois)} ROIs: {list(self.rois.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ROI configuration: {e}")
            return False
    
    def point_in_roi(self, point: Tuple[float, float], roi: Dict) -> bool:
        """
        Check if a point is inside an ROI rectangle.
        
        Args:
            point (tuple): (x, y) coordinates
            roi (dict): ROI configuration with x1, y1, x2, y2
            
        Returns:
            bool: True if point is inside ROI
        """
        x, y = point
        return (roi['x1'] <= x <= roi['x2'] and 
                roi['y1'] <= y <= roi['y2'])
    
    def count_vehicles_in_frame(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles per direction based on detections and ROIs.
        
        Args:
            detections (list): List of vehicle detection dictionaries
            
        Returns:
            dict: Vehicle counts per direction
        """
        # Reset counts
        self.counts = {direction: 0 for direction in self.directions}
        
        if not self.rois:
            logger.warning("No ROIs loaded. Cannot count vehicles.")
            return self.counts
        
        # Count vehicles in each ROI
        for detection in detections:
            centroid = detection['centroid']
            
            # Check which ROI contains this vehicle
            for direction, roi in self.rois.items():
                if self.point_in_roi(centroid, roi):
                    self.counts[direction] += 1
                    break  # Vehicle can only be in one ROI
        
        return self.counts.copy()
    
    def draw_rois_and_counts(self, frame: np.ndarray, detections: List[Dict] = None) -> np.ndarray:
        """
        Draw ROIs and vehicle counts on frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (list, optional): Vehicle detections to visualize
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        if not self.rois:
            # Draw warning if no ROIs
            cv2.putText(annotated_frame, "No ROIs loaded!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        # Draw ROIs
        for direction, roi in self.rois.items():
            color = self.colors.get(direction, (255, 255, 255))
            
            # Draw ROI rectangle
            cv2.rectangle(annotated_frame, (roi['x1'], roi['y1']), 
                         (roi['x2'], roi['y2']), color, 2)
            
            # Draw direction label
            label_pos = (roi['x1'], roi['y1'] - 10)
            cv2.putText(annotated_frame, direction, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw vehicle count
            count = self.counts.get(direction, 0)
            count_pos = (roi['x1'], roi['y2'] + 25)
            cv2.putText(annotated_frame, f"Count: {count}", count_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw vehicle centroids if detections provided
        if detections:
            for detection in detections:
                centroid = detection['centroid']
                
                # Determine which ROI this vehicle belongs to
                vehicle_color = (255, 255, 255)  # Default white
                for direction, roi in self.rois.items():
                    if self.point_in_roi(centroid, roi):
                        vehicle_color = self.colors.get(direction, (255, 255, 255))
                        break
                
                # Draw centroid
                cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 
                          8, vehicle_color, -1)
                cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 
                          10, (0, 0, 0), 2)
        
        # Draw summary statistics
        self.draw_summary_stats(annotated_frame)
        
        return annotated_frame
    
    def draw_summary_stats(self, frame: np.ndarray):
        """
        Draw summary statistics on frame.
        
        Args:
            frame (np.ndarray): Frame to draw on
        """
        # Calculate total vehicles
        total_vehicles = sum(self.counts.values())
        
        # Draw background for stats
        stats_bg_height = 120
        cv2.rectangle(frame, (frame.shape[1] - 250, 10), 
                     (frame.shape[1] - 10, stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (frame.shape[1] - 250, 10), 
                     (frame.shape[1] - 10, stats_bg_height), (255, 255, 255), 2)
        
        # Draw stats text
        stats_text = [
            "Vehicle Counts:",
            f"North: {self.counts['North']}",
            f"East: {self.counts['East']}",
            f"South: {self.counts['South']}",
            f"West: {self.counts['West']}",
            f"Total: {total_vehicles}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 15
            color = (255, 255, 255) if i == 0 or i == len(stats_text) - 1 else (200, 200, 200)
            cv2.putText(frame, text, (frame.shape[1] - 240, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def get_traffic_density_ratio(self) -> Dict[str, float]:
        """
        Calculate traffic density ratio for each direction.
        
        Returns:
            dict: Density ratios (0.0 to 1.0) per direction
        """
        total_vehicles = sum(self.counts.values())
        
        if total_vehicles == 0:
            return {direction: 0.25 for direction in self.directions}  # Equal distribution
        
        ratios = {}
        for direction in self.directions:
            ratios[direction] = self.counts[direction] / total_vehicles
        
        return ratios
    
    def get_priority_order(self) -> List[str]:
        """
        Get directions ordered by vehicle count (highest first).
        
        Returns:
            list: Directions ordered by priority
        """
        return sorted(self.directions, key=lambda d: self.counts[d], reverse=True)
    
    def reset_counts(self):
        """Reset all vehicle counts to zero."""
        self.counts = {direction: 0 for direction in self.directions}
    
    def get_counts_summary(self) -> Dict:
        """
        Get comprehensive summary of current counts.
        
        Returns:
            dict: Summary including counts, ratios, and priorities
        """
        total = sum(self.counts.values())
        ratios = self.get_traffic_density_ratio()
        priority_order = self.get_priority_order()
        
        return {
            'counts': self.counts.copy(),
            'total': total,
            'ratios': ratios,
            'priority_order': priority_order,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }


def main():
    """Main function for testing the counter."""
    import argparse
    from detector import VehicleDetector
    
    parser = argparse.ArgumentParser(description='Vehicle Counter Test')
    parser.add_argument('--video', type=str, default='data/videos/sample.mp4',
                       help='Path to input video file')
    parser.add_argument('--rois', type=str, default='ML/rois.json',
                       help='Path to ROI configuration file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model weights')
    
    args = parser.parse_args()
    
    # Check if ROI file exists
    if not Path(args.rois).exists():
        logger.error(f"ROI file not found: {args.rois}")
        logger.info("Please run roi_selector.py first to create ROI configuration.")
        return
    
    # Initialize components
    detector = VehicleDetector(model_path=args.model)
    counter = VehicleCounter(roi_config_path=args.rois)
    
    if not counter.rois:
        logger.error("Failed to load ROIs. Exiting.")
        return
    
    # Process video
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return
    
    logger.info("Starting vehicle counting demo. Press 'q' to quit.")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect vehicles
            detections, _ = detector.detect_vehicles(frame)
            
            # Count vehicles per direction
            counts = counter.count_vehicles_in_frame(detections)
            
            # Draw ROIs and counts
            annotated_frame = counter.draw_rois_and_counts(frame, detections)
            
            # Display frame
            cv2.imshow('Vehicle Counter Demo', annotated_frame)
            
            # Print counts every 30 frames
            if frame_count % 30 == 0:
                summary = counter.get_counts_summary()
                logger.info(f"Frame {frame_count}: {summary['counts']}, Total: {summary['total']}")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Demo completed. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
