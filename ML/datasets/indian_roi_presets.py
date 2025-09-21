"""
Indian Traffic Intersection ROI Presets
Pre-configured ROI templates for Indian 4-way intersections from drone perspective.
Calibrated for mixed vehicle types (cars, bikes, rickshaws, buses) typical in Indian traffic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianROIPresets:
    """
    ROI preset manager for Indian traffic intersections.
    Provides pre-calibrated regions for different video formats and intersection types.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize ROI preset manager."""
        self.data_dir = Path(data_dir)
        self.rois_dir = self.data_dir / "rois"
        self.indian_presets_dir = self.rois_dir / "indian_presets"
        
        # Create directories
        self.indian_presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Indian intersection ROI templates
        self.roi_templates = {
            'indian_4way_720p': {
                'description': 'Standard Indian 4-way intersection 720p drone view',
                'resolution': (1280, 720),
                'intersection_center': (640, 360),
                'approach_length': 200,  # pixels from center
                'lane_width': 80,        # pixels per lane
                'regions': {
                    'North': {
                        'entry': [(580, 50), (700, 300)],    # Top approach (vehicles entering)
                        'exit': [(580, 420), (700, 670)],    # Bottom exit
                        'direction': 'south_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw']
                    },
                    'East': {
                        'entry': [(780, 280), (1230, 400)],  # Right approach
                        'exit': [(50, 280), (500, 400)],     # Left exit
                        'direction': 'west_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw']
                    },
                    'South': {
                        'entry': [(580, 420), (700, 670)],   # Bottom approach
                        'exit': [(580, 50), (700, 300)],     # Top exit
                        'direction': 'north_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw']
                    },
                    'West': {
                        'entry': [(50, 280), (500, 400)],    # Left approach
                        'exit': [(780, 280), (1230, 400)],   # Right exit
                        'direction': 'east_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw']
                    }
                },
                'intersection_zone': [(500, 250), (780, 470)],  # Central intersection area
                'calibration_notes': 'Optimized for dense Indian traffic with high motorcycle density'
            },
            
            'indian_dense_1080p': {
                'description': '1080p dense Indian traffic intersection',
                'resolution': (1920, 1080),
                'intersection_center': (960, 540),
                'approach_length': 300,
                'lane_width': 120,
                'regions': {
                    'North': {
                        'entry': [(840, 75), (1080, 450)],
                        'exit': [(840, 630), (1080, 1005)],
                        'direction': 'south_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck']
                    },
                    'East': {
                        'entry': [(1170, 420), (1845, 600)],
                        'exit': [(75, 420), (750, 600)],
                        'direction': 'west_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck']
                    },
                    'South': {
                        'entry': [(840, 630), (1080, 1005)],
                        'exit': [(840, 75), (1080, 450)],
                        'direction': 'north_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck']
                    },
                    'West': {
                        'entry': [(75, 420), (750, 600)],
                        'exit': [(1170, 420), (1845, 600)],
                        'direction': 'east_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck']
                    }
                },
                'intersection_zone': [(750, 375), (1170, 705)],
                'calibration_notes': 'High-resolution preset for detailed vehicle tracking in dense traffic'
            },
            
            'indian_compact_480p': {
                'description': '480p compact Indian intersection (mobile/low-res drone)',
                'resolution': (854, 480),
                'intersection_center': (427, 240),
                'approach_length': 120,
                'lane_width': 50,
                'regions': {
                    'North': {
                        'entry': [(377, 30), (477, 180)],
                        'exit': [(377, 300), (477, 450)],
                        'direction': 'south_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw']
                    },
                    'East': {
                        'entry': [(547, 190), (824, 290)],
                        'exit': [(30, 190), (307, 290)],
                        'direction': 'west_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw']
                    },
                    'South': {
                        'entry': [(377, 300), (477, 450)],
                        'exit': [(377, 30), (477, 180)],
                        'direction': 'north_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw']
                    },
                    'West': {
                        'entry': [(30, 190), (307, 290)],
                        'exit': [(547, 190), (824, 290)],
                        'direction': 'east_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw']
                    }
                },
                'intersection_zone': [(307, 150), (547, 330)],
                'calibration_notes': 'Compact preset for lower resolution drone footage'
            },
            
            'indian_unbalanced_720p': {
                'description': 'Unbalanced Indian intersection (heavy traffic one direction)',
                'resolution': (1280, 720),
                'intersection_center': (640, 360),
                'approach_length': 250,
                'lane_width': 100,
                'regions': {
                    'North': {
                        'entry': [(540, 30), (740, 280)],    # Heavy traffic direction
                        'exit': [(540, 440), (740, 690)],
                        'direction': 'south_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck'],
                        'traffic_weight': 3.0  # Higher weight for timing calculation
                    },
                    'East': {
                        'entry': [(740, 260), (1250, 380)],
                        'exit': [(30, 260), (540, 380)],
                        'direction': 'west_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw'],
                        'traffic_weight': 1.0
                    },
                    'South': {
                        'entry': [(540, 440), (740, 690)],   # Heavy traffic direction
                        'exit': [(540, 30), (740, 280)],
                        'direction': 'north_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'bus', 'rickshaw', 'truck'],
                        'traffic_weight': 3.0
                    },
                    'West': {
                        'entry': [(30, 260), (540, 380)],
                        'exit': [(740, 260), (1250, 380)],
                        'direction': 'east_bound',
                        'typical_vehicles': ['car', 'motorcycle', 'rickshaw'],
                        'traffic_weight': 1.0
                    }
                },
                'intersection_zone': [(480, 220), (800, 500)],
                'calibration_notes': 'Preset for unbalanced traffic flow (office/market areas)'
            }
        }
        
        # Vehicle type mappings for Indian traffic
        self.indian_vehicle_types = {
            'motorcycle': {'yolo_class': 3, 'size_range': (20, 80), 'priority': 0.5},
            'car': {'yolo_class': 2, 'size_range': (60, 150), 'priority': 1.0},
            'bus': {'yolo_class': 5, 'size_range': (120, 300), 'priority': 2.0},
            'truck': {'yolo_class': 7, 'size_range': (100, 280), 'priority': 1.8},
            'rickshaw': {'yolo_class': 2, 'size_range': (40, 100), 'priority': 0.7}  # Often detected as car
        }
    
    def create_preset(self, preset_name: str, output_path: Optional[Path] = None) -> bool:
        """
        Create and save an ROI preset.
        
        Args:
            preset_name (str): Name of preset to create
            output_path (Optional[Path]): Custom output path
            
        Returns:
            bool: Success status
        """
        if preset_name not in self.roi_templates:
            logger.error(f"Unknown preset: {preset_name}")
            return False
        
        try:
            preset_data = self.roi_templates[preset_name].copy()
            
            # Add metadata
            preset_data['preset_name'] = preset_name
            preset_data['created_for'] = 'Indian traffic intersections'
            preset_data['vehicle_types'] = self.indian_vehicle_types
            preset_data['usage_notes'] = [
                'Optimized for mixed Indian vehicle types',
                'Handles high motorcycle density',
                'Accounts for lane discipline variations',
                'Suitable for drone altitude 50-100 feet'
            ]
            
            # Determine output path
            if output_path is None:
                output_path = self.indian_presets_dir / f"{preset_name}.json"
            
            # Save preset
            with open(output_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            logger.info(f"✓ Created ROI preset: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to create preset {preset_name}: {e}")
            return False
    
    def create_all_presets(self) -> Dict[str, bool]:
        """
        Create all Indian ROI presets.
        
        Returns:
            Dict[str, bool]: Creation results
        """
        logger.info("Creating all Indian ROI presets...")
        
        results = {}
        for preset_name in self.roi_templates.keys():
            success = self.create_preset(preset_name)
            results[preset_name] = success
        
        successful = sum(results.values())
        logger.info(f"Created {successful}/{len(results)} ROI presets")
        return results
    
    def load_preset(self, preset_name: str) -> Optional[Dict]:
        """
        Load an ROI preset.
        
        Args:
            preset_name (str): Name of preset to load
            
        Returns:
            Optional[Dict]: Preset data or None if not found
        """
        preset_file = self.indian_presets_dir / f"{preset_name}.json"
        
        if not preset_file.exists():
            logger.warning(f"Preset file not found: {preset_file}")
            return None
        
        try:
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
            
            logger.info(f"✓ Loaded ROI preset: {preset_name}")
            return preset_data
        
        except Exception as e:
            logger.error(f"✗ Failed to load preset {preset_name}: {e}")
            return None
    
    def auto_select_preset(self, video_path: Path) -> Optional[str]:
        """
        Automatically select best ROI preset for a video.
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            Optional[str]: Best preset name or None
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            resolution = (width, height)
            logger.info(f"Video resolution: {width}x{height}")
            
            # Find best matching preset
            best_preset = None
            min_diff = float('inf')
            
            for preset_name, preset_data in self.roi_templates.items():
                preset_res = preset_data['resolution']
                
                # Calculate resolution difference
                width_diff = abs(preset_res[0] - width)
                height_diff = abs(preset_res[1] - height)
                total_diff = width_diff + height_diff
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_preset = preset_name
            
            logger.info(f"Auto-selected preset: {best_preset}")
            return best_preset
        
        except Exception as e:
            logger.error(f"Auto-selection failed: {e}")
            return None
    
    def visualize_preset(self, preset_name: str, video_path: Path, 
                        output_path: Optional[Path] = None) -> bool:
        """
        Create visualization of ROI preset on sample frame.
        
        Args:
            preset_name (str): Name of preset
            video_path (Path): Path to video file
            output_path (Optional[Path]): Output image path
            
        Returns:
            bool: Success status
        """
        try:
            # Load preset
            preset_data = self.load_preset(preset_name)
            if preset_data is None:
                return False
            
            # Open video and get first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Cannot read frame from video")
                return False
            
            # Draw ROI regions
            colors = {
                'North': (0, 255, 0),    # Green
                'East': (255, 0, 0),     # Blue  
                'South': (0, 255, 255),  # Yellow
                'West': (255, 0, 255)    # Magenta
            }
            
            for direction, region_data in preset_data['regions'].items():
                color = colors.get(direction, (255, 255, 255))
                
                # Draw entry region
                entry_pts = np.array(region_data['entry'], np.int32)
                cv2.rectangle(frame, tuple(entry_pts[0]), tuple(entry_pts[1]), color, 2)
                
                # Add label
                label_pos = (entry_pts[0][0], entry_pts[0][1] - 10)
                cv2.putText(frame, f"{direction} Entry", label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw intersection zone
            if 'intersection_zone' in preset_data:
                zone_pts = np.array(preset_data['intersection_zone'], np.int32)
                cv2.rectangle(frame, tuple(zone_pts[0]), tuple(zone_pts[1]), 
                             (0, 0, 255), 3)  # Red for intersection
                cv2.putText(frame, "Intersection", (zone_pts[0][0], zone_pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add title
            cv2.putText(frame, f"ROI Preset: {preset_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Save visualization
            if output_path is None:
                output_path = self.indian_presets_dir / f"{preset_name}_visualization.jpg"
            
            cv2.imwrite(str(output_path), frame)
            logger.info(f"✓ ROI visualization saved: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"✗ Visualization failed: {e}")
            return False
    
    def get_available_presets(self) -> List[str]:
        """
        Get list of available ROI presets.
        
        Returns:
            List[str]: Available preset names
        """
        return list(self.roi_templates.keys())
    
    def get_preset_info(self, preset_name: str) -> Optional[Dict]:
        """
        Get information about a specific preset.
        
        Args:
            preset_name (str): Name of preset
            
        Returns:
            Optional[Dict]: Preset information
        """
        if preset_name not in self.roi_templates:
            return None
        
        preset_data = self.roi_templates[preset_name]
        return {
            'name': preset_name,
            'description': preset_data['description'],
            'resolution': preset_data['resolution'],
            'regions_count': len(preset_data['regions']),
            'calibration_notes': preset_data['calibration_notes']
        }


def main():
    """Main function for ROI preset management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Indian ROI Preset Manager')
    parser.add_argument('--create-all', action='store_true',
                       help='Create all ROI presets')
    parser.add_argument('--create', type=str,
                       help='Create specific preset')
    parser.add_argument('--list', action='store_true',
                       help='List available presets')
    parser.add_argument('--info', type=str,
                       help='Show preset information')
    parser.add_argument('--auto-select', type=str,
                       help='Auto-select preset for video file')
    parser.add_argument('--visualize', type=str, nargs=2,
                       metavar=('PRESET', 'VIDEO'),
                       help='Create ROI visualization')
    
    args = parser.parse_args()
    
    roi_manager = IndianROIPresets()
    
    if args.create_all:
        results = roi_manager.create_all_presets()
        successful = sum(results.values())
        print(f"Created {successful}/{len(results)} ROI presets")
        return
    
    if args.create:
        success = roi_manager.create_preset(args.create)
        print(f"✓ Created preset: {args.create}" if success else f"✗ Failed to create: {args.create}")
        return
    
    if args.list:
        presets = roi_manager.get_available_presets()
        print("\nAvailable Indian ROI Presets:")
        print("="*40)
        for preset in presets:
            info = roi_manager.get_preset_info(preset)
            print(f"{preset}: {info['description']}")
            print(f"  Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
            print(f"  Regions: {info['regions_count']}")
            print()
        return
    
    if args.info:
        info = roi_manager.get_preset_info(args.info)
        if info:
            print(f"\nPreset: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
            print(f"Regions: {info['regions_count']}")
            print(f"Notes: {info['calibration_notes']}")
        else:
            print(f"Preset not found: {args.info}")
        return
    
    if args.auto_select:
        video_path = Path(args.auto_select)
        if video_path.exists():
            preset = roi_manager.auto_select_preset(video_path)
            print(f"Recommended preset: {preset}")
        else:
            print(f"Video file not found: {args.auto_select}")
        return
    
    if args.visualize:
        preset_name, video_path = args.visualize
        success = roi_manager.visualize_preset(preset_name, Path(video_path))
        print(f"✓ Visualization created" if success else f"✗ Visualization failed")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
