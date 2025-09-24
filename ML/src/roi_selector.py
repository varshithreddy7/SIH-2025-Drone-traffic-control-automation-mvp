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
    
    def __init__(self, video_path: str, output_path: str = "ML/rois.json"):
        """
        Initialize ROI selector.
        
        Args:
            video_path: Path to video file
            output_path: Path to save ROI configuration
        """
        self.video_path = video_path
        self.output_path = output_path
        self.cap = None
        self.frame = None
        self.original_frame = None
        
        # ROI selection state
        self.current_roi = []
        self.rois = {}
        self.directions = ['North', 'East', 'South', 'West']
        self.current_direction_index = 0
        self.is_selecting = False
        
        # Colors for each direction (bright and distinct)
        self.colors = {
            'North': (0, 255, 255),    # Bright Cyan
            'East': (255, 0, 255),     # Bright Magenta
            'South': (0, 255, 0),      # Bright Green
            'West': (255, 255, 0)      # Bright Yellow
        }
        
        # Moveable instruction panel settings
        self.panel_pos = [20, 20]  # [x, y] position - can be moved
        self.panel_size = [350, 160]  # [width, height] - compact size
        self.panel_dragging = False
        self.drag_offset = [0, 0]
        
        logger.info(f"ROI Selector initialized for video: {video_path}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for ROI selection.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Mouse flags
            param: Additional parameters
        """
        if self.current_direction_index >= len(self.directions):
            return
            
        direction = self.directions[self.current_direction_index]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.is_selecting = True
            self.current_roi = [(x, y)]
            
            # Check if panel is being dragged
            if (self.panel_pos[0] < x < self.panel_pos[0] + self.panel_size[0] and
                self.panel_pos[1] < y < self.panel_pos[1] + self.panel_size[1]):
                self.panel_dragging = True
                self.drag_offset = [x - self.panel_pos[0], y - self.panel_pos[1]]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_selecting:
                # Update rectangle while drawing
                self.frame = self.original_frame.copy()
                self.draw_existing_rois()
                
                # Draw current rectangle being drawn
                color = self.colors[direction]
                cv2.rectangle(self.frame, self.current_roi[0], (x, y), color, 2)
                
                # Add instruction text
                self.add_instruction_text()
                
            elif self.panel_dragging:
                # Move panel
                self.panel_pos = [x - self.drag_offset[0], y - self.drag_offset[1]]
                self.add_instruction_text()
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.is_selecting = False
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
                self.current_direction_index += 1
                self.current_roi = []
                
                # Redraw frame
                self.frame = self.original_frame.copy()
                self.draw_existing_rois()
                self.add_instruction_text()
            
            # Stop panel dragging
            self.panel_dragging = False
    
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
        """Add instruction text overlay to the frame."""
        instructions = [
            "ROI Selection for Traffic Signal Control",
            "",
            "Instructions:",
            "1. Click and drag to select rectangular ROI",
            "2. Select ROIs in order: North -> East -> South -> West",
            "3. Press 'r' to reset current selection",
            "4. Press 's' to save and exit",
            "5. Press 'q' to quit without saving",
            "6. Drag this panel to move it",
            ""
        ]
        
        if self.current_direction_index < len(self.directions):
            current_dir = self.directions[self.current_direction_index]
            instructions.append(f"Current: {current_dir} ({self.current_direction_index + 1}/4)")
            instructions.append(f"Color: {current_dir}")
        else:
            instructions.append("All ROIs selected! Press 's' to save.")
        
        # Ensure panel stays within frame bounds
        frame_height, frame_width = self.frame.shape[:2]
        self.panel_pos[0] = max(0, min(self.panel_pos[0], frame_width - self.panel_size[0]))
        self.panel_pos[1] = max(0, min(self.panel_pos[1], frame_height - self.panel_size[1]))
        
        # Add semi-transparent background for better visibility
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (self.panel_pos[0], self.panel_pos[1]), 
                      (self.panel_pos[0] + self.panel_size[0], self.panel_pos[1] + self.panel_size[1]), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, self.frame, 0.2, 0, self.frame)
        
        # Add border with drag indicator
        border_color = (0, 255, 255) if self.panel_dragging else (255, 255, 255)
        cv2.rectangle(self.frame, (self.panel_pos[0], self.panel_pos[1]), 
                      (self.panel_pos[0] + self.panel_size[0], self.panel_pos[1] + self.panel_size[1]), 
                      border_color, 2)
        
        # Add drag handle indicator
        cv2.rectangle(self.frame, (self.panel_pos[0] + self.panel_size[0] - 20, self.panel_pos[1]), 
                      (self.panel_pos[0] + self.panel_size[0], self.panel_pos[1] + 20), 
                      (100, 100, 100), -1)
        cv2.putText(self.frame, "≡", (self.panel_pos[0] + self.panel_size[0] - 15, self.panel_pos[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add text with better formatting
        for i, instruction in enumerate(instructions):
            y_pos = self.panel_pos[1] + 20 + i * 15
            color = (255, 255, 255)
            font_size = 0.4
            
            # Title formatting
            if i == 0:
                color = (0, 255, 255)
                font_size = 0.5
            
            # Highlight current direction
            if "Current:" in instruction and self.current_direction_index < len(self.directions):
                color = self.colors[self.directions[self.current_direction_index]]
                font_size = 0.5
            
            # Color indicator for current direction
            if "Color:" in instruction and self.current_direction_index < len(self.directions):
                color = self.colors[self.directions[self.current_direction_index]]
                # Draw color box
                cv2.rectangle(self.frame, 
                             (self.panel_pos[0] + 200, y_pos - 10), 
                             (self.panel_pos[0] + 220, y_pos + 5), 
                             color, -1)
                cv2.rectangle(self.frame, 
                             (self.panel_pos[0] + 200, y_pos - 10), 
                             (self.panel_pos[0] + 220, y_pos + 5), 
                             (255, 255, 255), 1)
            
            cv2.putText(self.frame, instruction, (self.panel_pos[0] + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
        
        # Add progress indicator
        progress_text = f"Progress: {len(self.rois)}/4 ROIs selected"
        cv2.putText(self.frame, progress_text, 
                   (self.panel_pos[0] + 10, self.panel_pos[1] + self.panel_size[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def select_rois_from_video(self):
        """
        Select ROIs from a video file.
        """
        if not Path(self.video_path).exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from video")
            self.cap.release()
            return False
        
        self.cap.release()
        
        return self.select_rois_from_frame(frame)
    
    def select_rois_from_frame(self, frame):
        """
        Select ROIs from a single frame.
        
        Args:
            frame (np.ndarray): Input frame
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
                if self.current_direction_index > 0:
                    self.current_direction_index -= 1
                    direction = self.directions[self.current_direction_index]
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
                    success = self.save_rois()
                    cv2.destroyAllWindows()
                    return success
                else:
                    logger.warning(f"Only {len(self.rois)}/4 ROIs selected. Complete all selections before saving.")
            
            # Auto-save when all ROIs are selected
            if len(self.rois) == 4 and self.current_direction_index >= len(self.directions):
                logger.info("All ROIs selected. Press 's' to save or 'r' to reset last ROI.")
    
    def save_rois(self):
        """
        Save ROIs to JSON file.
        
        Returns:
            bool: Success status
        """
        try:
            # Ensure output directory exists
            output_dir = Path(self.output_path).parent
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
            with open(self.output_path, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            logger.info(f"ROIs saved successfully to {self.output_path}")
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
    selector = ROISelector(args.video, args.output)
    
    # Select ROIs from video
    success = selector.select_rois_from_video()
    
    if success:
        logger.info("ROI selection completed successfully!")
        logger.info(f"ROI configuration saved to: {args.output}")
        logger.info("You can now run the main control system.")
    else:
        logger.error("ROI selection failed or was cancelled.")


if __name__ == "__main__":
    main()
