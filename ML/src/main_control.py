"""
Main control system for drone-based AI traffic signal automation.
Integrates detection, counting, timing, and command generation.
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import argparse
from typing import Dict, Optional

from detector import VehicleDetector
from counter import VehicleCounter
from timing import TrafficTimingCalculator, TimingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficControlSystem:
    """
    Main traffic control system that coordinates all components.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 roi_config_path: str = "ML/rois.json",
                 timing_config: TimingConfig = None,
                 update_interval: int = 5):
        """
        Initialize the traffic control system.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            roi_config_path (str): Path to ROI configuration
            timing_config (TimingConfig): Timing configuration
            update_interval (int): Update interval in seconds
        """
        self.model_path = model_path
        self.roi_config_path = roi_config_path
        self.update_interval = update_interval
        
        # Initialize components
        self.detector = VehicleDetector(model_path=model_path)
        self.counter = VehicleCounter(roi_config_path=roi_config_path)
        self.timing_calculator = TrafficTimingCalculator(config=timing_config)
        
        # Control variables
        self.running = False
        self.last_update_time = 0
        self.frame_count = 0
        self.total_detections = 0
        
        # Performance tracking
        self.fps_history = []
        self.processing_times = []
        
        # Validate initialization
        self.validate_setup()
    
    def validate_setup(self) -> bool:
        """
        Validate that all components are properly initialized.
        
        Returns:
            bool: True if setup is valid
        """
        if not self.counter.rois:
            logger.error("ROI configuration not loaded. Please run roi_selector.py first.")
            return False
        
        logger.info("Traffic control system initialized successfully")
        logger.info(f"Loaded {len(self.counter.rois)} ROIs: {list(self.counter.rois.keys())}")
        return True
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (annotated_frame, vehicle_counts, should_update_timing)
        """
        start_time = time.time()
        
        # Detect vehicles
        detections, _ = self.detector.detect_vehicles(frame)
        self.total_detections += len(detections)
        
        # Count vehicles per direction
        vehicle_counts = self.counter.count_vehicles_in_frame(detections)
        
        # Draw ROIs and counts
        annotated_frame = self.counter.draw_rois_and_counts(frame, detections)
        
        # Add system information
        self.add_system_info(annotated_frame)
        
        # Check if timing should be updated
        current_time = time.time()
        should_update_timing = (current_time - self.last_update_time) >= self.update_interval
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return annotated_frame, vehicle_counts, should_update_timing
    
    def update_traffic_timing(self, vehicle_counts: Dict[str, int]) -> str:
        """
        Update traffic signal timing based on vehicle counts.
        
        Args:
            vehicle_counts (dict): Current vehicle counts
            
        Returns:
            str: Arduino command string
        """
        # Compute green times
        green_times = self.timing_calculator.compute_green_times(vehicle_counts)
        
        # Format command for Arduino
        command = self.timing_calculator.format_command(green_times)
        
        # Update last update time
        self.last_update_time = time.time()
        
        # Log the update
        total_vehicles = sum(vehicle_counts.values())
        logger.info(f"Timing Update - Vehicles: {vehicle_counts}, Total: {total_vehicles}")
        logger.info(f"Green Times: {green_times}")
        logger.info(f"Arduino Command: {command}")
        
        return command
    
    def add_system_info(self, frame: np.ndarray):
        """
        Add system information overlay to frame.
        
        Args:
            frame (np.ndarray): Frame to annotate
        """
        # Calculate current FPS
        if hasattr(self.detector, 'current_fps'):
            current_fps = self.detector.current_fps
        else:
            current_fps = 0
        
        # System info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"FPS: {current_fps:.1f}",
            f"Total Detections: {self.total_detections}",
            f"Update Interval: {self.update_interval}s",
            f"Next Update: {max(0, self.update_interval - (time.time() - self.last_update_time)):.1f}s"
        ]
        
        # Draw background
        info_bg_height = len(info_lines) * 20 + 20
        cv2.rectangle(frame, (10, frame.shape[0] - info_bg_height - 10), 
                     (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, frame.shape[0] - info_bg_height - 10), 
                     (300, frame.shape[0] - 10), (255, 255, 255), 2)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y_pos = frame.shape[0] - info_bg_height + 15 + i * 20
            cv2.putText(frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_video_processing(self, video_path: str, display: bool = True, save_output: bool = False):
        """
        Run the traffic control system on a video file.
        
        Args:
            video_path (str): Path to input video
            display (bool): Whether to display video window
            save_output (bool): Whether to save annotated output
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
        logger.info(f"Update interval: {self.update_interval} seconds")
        
        # Setup video writer if saving output
        writer = None
        if save_output:
            output_path = f"output_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        self.running = True
        self.last_update_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                self.frame_count += 1
                
                # Process frame
                annotated_frame, vehicle_counts, should_update = self.process_frame(frame)
                
                # Update timing if needed
                if should_update:
                    command = self.update_traffic_timing(vehicle_counts)
                    print(f"\n[ARDUINO COMMAND] {command}\n")
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Control System', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord(' '):
                        # Pause/resume
                        logger.info("Paused - press any key to continue")
                        cv2.waitKey(0)
                
                # Progress logging
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    avg_processing_time = np.mean(self.processing_times[-50:]) if self.processing_times else 0
                    logger.info(f"Progress: {self.frame_count}/{total_frames} ({progress:.1f}%) - "
                              f"Avg processing: {avg_processing_time*1000:.1f}ms")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            self.running = False
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            self.print_final_statistics()
    
    def run_camera_processing(self, camera_id: int = 0, display: bool = True):
        """
        Run the traffic control system on camera feed.
        
        Args:
            camera_id (int): Camera device ID
            display (bool): Whether to display video window
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Starting camera processing (Camera {camera_id})")
        logger.info(f"Update interval: {self.update_interval} seconds")
        logger.info("Press 'q' to quit, 'space' to pause")
        
        self.running = True
        self.last_update_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                self.frame_count += 1
                
                # Process frame
                annotated_frame, vehicle_counts, should_update = self.process_frame(frame)
                
                # Update timing if needed
                if should_update:
                    command = self.update_traffic_timing(vehicle_counts)
                    print(f"\n[ARDUINO COMMAND] {command}\n")
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Control System - Live', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord(' '):
                        # Pause/resume
                        logger.info("Paused - press any key to continue")
                        cv2.waitKey(0)
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            self.running = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final processing statistics."""
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            max_processing_time = max(self.processing_times)
            min_processing_time = min(self.processing_times)
            
            logger.info("="*50)
            logger.info("FINAL STATISTICS")
            logger.info("="*50)
            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Total vehicle detections: {self.total_detections}")
            logger.info(f"Average processing time: {avg_processing_time*1000:.1f}ms")
            logger.info(f"Min processing time: {min_processing_time*1000:.1f}ms")
            logger.info(f"Max processing time: {max_processing_time*1000:.1f}ms")
            
            if avg_processing_time > 0:
                theoretical_fps = 1.0 / avg_processing_time
                logger.info(f"Theoretical max FPS: {theoretical_fps:.1f}")
            
            # Timing summary
            timing_summary = self.timing_calculator.get_timing_summary()
            if timing_summary['current_timings']:
                logger.info(f"Last timing update: {timing_summary['current_timings']}")
                logger.info(f"Last cycle time: {timing_summary['current_cycle_time']}s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Traffic Control System')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file (if not specified, uses camera)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model weights')
    parser.add_argument('--rois', type=str, default='ML/rois.json',
                       help='Path to ROI configuration file')
    parser.add_argument('--update-interval', type=int, default=5,
                       help='Timing update interval in seconds')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    parser.add_argument('--save-output', action='store_true',
                       help='Save annotated output video')
    
    # Timing configuration arguments
    parser.add_argument('--min-green', type=int, default=6,
                       help='Minimum green time in seconds')
    parser.add_argument('--max-green', type=int, default=25,
                       help='Maximum green time in seconds')
    parser.add_argument('--base-green', type=int, default=6,
                       help='Base green time when no vehicles')
    parser.add_argument('--time-per-vehicle', type=int, default=2,
                       help='Additional seconds per vehicle')
    
    args = parser.parse_args()
    
    # Check if ROI file exists
    if not Path(args.rois).exists():
        logger.error(f"ROI configuration file not found: {args.rois}")
        logger.info("Please run roi_selector.py first to create ROI configuration:")
        logger.info(f"python ML\\src\\roi_selector.py --video {args.video or 'data/videos/sample.mp4'}")
        return
    
    # Create timing configuration
    timing_config = TimingConfig(
        min_green_time=args.min_green,
        max_green_time=args.max_green,
        base_green_time=args.base_green,
        time_per_vehicle=args.time_per_vehicle
    )
    
    # Initialize traffic control system
    control_system = TrafficControlSystem(
        model_path=args.model,
        roi_config_path=args.rois,
        timing_config=timing_config,
        update_interval=args.update_interval
    )
    
    # Validate setup
    if not control_system.validate_setup():
        return
    
    # Run processing
    if args.video:
        logger.info(f"Processing video: {args.video}")
        control_system.run_video_processing(
            video_path=args.video,
            display=not args.no_display,
            save_output=args.save_output
        )
    else:
        logger.info(f"Processing camera feed: {args.camera}")
        control_system.run_camera_processing(
            camera_id=args.camera,
            display=not args.no_display
        )


if __name__ == "__main__":
    main()
