"""
Interactive ROI (Region of Interest) selector for traffic directions.
Allows user to click and define rectangular regions for each traffic approach.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ROISelector:
    """
    Interactive ROI selector for traffic monitoring.
    Allows selection of rectangular regions for each traffic direction.
    """
    
    def __init__(self):
        """Initialize the ROI selector."""
        self.directions = ['North', 'East', 'South', 'West']
        self.colors = {
            'North': (0, 255, 0),    # Green
            'East': (255, 0, 0),     # Blue
            'South': (0, 0, 255),    # Red
            'West': (255, 255, 0)    # Cyan
        }
        self.rois = {}
        self.current_direction = 0
        self.current_roi = []
        self.drawing = False
        self.frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for ROI selection.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Mouse flags
            param: Additional parameters
        """
        if self.current_direction >= len(self.directions):
            return
            
        direction = self.directions[self.current_direction]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.drawing = True
            self.current_roi = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update rectangle while drawing
                self.frame = self.original_frame.copy()
                self.draw_existing_rois()
                
                # Draw current rectangle being drawn
                color = self.colors[direction]
                cv2.rectangle(self.frame, self.current_roi[0], (x, y), color, 2)
                
                # Add instruction text
                self.add_instruction_text()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.drawing = False
            if len(self.current_roi) == 1:
                self.current_roi.append((x, y))
                
                # Ensure rectangle has positive dimensions
                x1, y1 = self.current_roi[0]
                x2, y2 = self.current_roi[1]
                
                # Normalize coordinates (top-left, bottom-right)
                roi = {
                    'x1': min(x1, x2),
                    'y1': min(y1, y2),
                    'x2': max(x1, x2),
                    'y2': max(y1, y2),
                    'direction': direction,
                    'color': self.colors[direction]
                }
                
                self.rois[direction] = roi
                logger.info(f"ROI selected for {direction}: {roi}")
                
                # Move to next direction
                self.current_direction += 1
                self.current_roi = []
                
                # Redraw frame
                self.frame = self.original_frame.copy()
                self.draw_existing_rois()
                self.add_instruction_text()
    
    def draw_existing_rois(self):
        """Draw all existing ROIs on the frame."""
        for direction, roi in self.rois.items():
            color = roi['color']
            cv2.rectangle(self.frame, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), color, 2)
            
            # Add direction label
            label_pos = (roi['x1'], roi['y1'] - 10)
            cv2.putText(self.frame, direction, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
    
    def add_instruction_text(self):
        """Add instruction text to the frame."""
        instructions = [
            "ROI Selection for Traffic Signal Control",
            "",
            "Instructions:",
            "1. Click and drag to select rectangular ROI",
            "2. Select ROIs in order: North -> East -> South -> West",
            "3. Press 'r' to reset current selection",
            "4. Press 's' to save and exit",
            "5. Press 'q' to quit without saving",
            ""
        ]
        
        if self.current_direction < len(self.directions):
            current_dir = self.directions[self.current_direction]
            instructions.append(f"Current: {current_dir} ({self.current_direction + 1}/4)")
            instructions.append(f"Color: {current_dir}")
        else:
            instructions.append("All ROIs selected! Press 's' to save.")
        
        # Add background for text
        text_bg_height = len(instructions) * 25 + 20
        cv2.rectangle(self.frame, (10, 10), (400, text_bg_height), (0, 0, 0), -1)
        cv2.rectangle(self.frame, (10, 10), (400, text_bg_height), (255, 255, 255), 2)
        
        # Add text
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 20
            color = (255, 255, 255)
            
            # Highlight current direction
            if "Current:" in instruction and self.current_direction < len(self.directions):
                color = self.colors[self.directions[self.current_direction]]
            
            cv2.putText(self.frame, instruction, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def select_rois_from_video(self, video_path, output_path="ML/rois.json"):
        """
        Select ROIs from a video file.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save ROI configuration
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from video")
            cap.release()
            return False
        
        cap.release()
        
        return self.select_rois_from_frame(frame, output_path)
    
    def select_rois_from_frame(self, frame, output_path="ML/rois.json"):
        """
        Select ROIs from a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            output_path (str): Path to save ROI configuration
        """
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('ROI Selection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('ROI Selection', self.mouse_callback)
        
        # Initial display
        self.add_instruction_text()
        
        logger.info("Starting ROI selection. Follow on-screen instructions.")
        
        while True:
            cv2.imshow('ROI Selection', self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("ROI selection cancelled by user")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                # Reset current direction
                if self.current_direction > 0:
                    self.current_direction -= 1
                    direction = self.directions[self.current_direction]
                    if direction in self.rois:
                        del self.rois[direction]
                    logger.info(f"Reset ROI for {direction}")
                    
                    # Redraw frame
                    self.frame = self.original_frame.copy()
                    self.draw_existing_rois()
                    self.add_instruction_text()
            
            elif key == ord('s'):
                # Save ROIs
                if len(self.rois) == 4:
                    success = self.save_rois(output_path)
                    cv2.destroyAllWindows()
                    return success
                else:
                    logger.warning(f"Only {len(self.rois)}/4 ROIs selected. Complete all selections before saving.")
            
            # Auto-save when all ROIs are selected
            if len(self.rois) == 4 and self.current_direction >= len(self.directions):
                logger.info("All ROIs selected. Press 's' to save or 'r' to reset last ROI.")
    
    def save_rois(self, output_path):
        """
        Save ROIs to JSON file.
        
        Args:
            output_path (str): Path to save ROI configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for JSON serialization
            roi_data = {
                'rois': self.rois,
                'directions': self.directions,
                'metadata': {
                    'total_rois': len(self.rois),
                    'frame_shape': self.original_frame.shape if self.original_frame is not None else None
                }
            }
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            logger.info(f"ROIs saved successfully to {output_path}")
            logger.info(f"Saved {len(self.rois)} ROIs: {list(self.rois.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ROIs: {e}")
            return False
    
    @staticmethod
    def load_rois(roi_path="ML/rois.json"):
        """
        Load ROIs from JSON file.
        
        Args:
            roi_path (str): Path to ROI configuration file
            
        Returns:
            dict: ROI configuration or None if failed
        """
        try:
            if not Path(roi_path).exists():
                logger.error(f"ROI file not found: {roi_path}")
                return None
            
            with open(roi_path, 'r') as f:
                roi_data = json.load(f)
            
            logger.info(f"ROIs loaded successfully from {roi_path}")
            return roi_data
            
        except Exception as e:
            logger.error(f"Failed to load ROIs: {e}")
            return None


def main():
    """Main function for ROI selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive ROI Selector for Traffic Monitoring')
    parser.add_argument('--video', type=str, default='data/videos/sample.mp4',
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='ML/rois.json',
                       help='Path to save ROI configuration')
    
    args = parser.parse_args()
    
    # Create ROI selector
    selector = ROISelector()
    
    # Select ROIs from video
    success = selector.select_rois_from_video(args.video, args.output)
    
    if success:
        logger.info("ROI selection completed successfully!")
        logger.info(f"ROI configuration saved to: {args.output}")
        logger.info("You can now run the main control system.")
    else:
        logger.error("ROI selection failed or was cancelled.")


if __name__ == "__main__":
    main()
