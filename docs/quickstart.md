# Drone-Based AI Traffic Signal Automation - Quick Start Guide

## Overview

This project implements an MVP for drone-based AI traffic signal automation using YOLOv8 for vehicle detection and direct microcontroller control. The system detects and counts vehicles per direction, calculates optimal green light timings, and outputs Arduino-compatible commands.

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **Camera**: Optional (for live processing)

## Quick Setup (Windows)

### 1. Environment Setup

```powershell
# Navigate to project directory
cd "C:\Users\HP\varshithreddy\SIH-2025\SIH-2025-Drone-traffic-control-automation-mvp"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r ML\requirements.txt

# Verify YOLO installation
yolo --help
```

### 2. Prepare Sample Video

Place your traffic video file in the `data/videos/` directory:
```
data/videos/sample.mp4
```

### 3. Configure ROIs (Regions of Interest)

Run the interactive ROI selector to define traffic directions:

```powershell
python ML\src\roi_selector.py --video data\videos\sample.mp4
```

**Instructions:**
1. Click and drag to select rectangular ROIs
2. Select in order: North → East → South → West
3. Press 's' to save, 'r' to reset, 'q' to quit
4. ROIs are saved to `ML/rois.json`

### 4. Run the Traffic Control System

```powershell
# Process video file
python ML\src\main_control.py --video data\videos\sample.mp4

# Use live camera (default camera 0)
python ML\src\main_control.py

# Process with custom settings
python ML\src\main_control.py --video data\videos\sample.mp4 --update-interval 3 --min-green 8
```

## Command Line Options

### Main Control System (`main_control.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | None | Path to video file (uses camera if not specified) |
| `--camera` | 0 | Camera device ID |
| `--model` | yolov8n.pt | YOLOv8 model weights path |
| `--rois` | ML/rois.json | ROI configuration file |
| `--update-interval` | 5 | Timing update interval (seconds) |
| `--min-green` | 6 | Minimum green time (seconds) |
| `--max-green` | 25 | Maximum green time (seconds) |
| `--base-green` | 6 | Base green time when no vehicles |
| `--time-per-vehicle` | 2 | Additional seconds per vehicle |
| `--no-display` | False | Disable video window |
| `--save-output` | False | Save annotated video |

### ROI Selector (`roi_selector.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | data/videos/sample.mp4 | Input video for ROI selection |
| `--output` | ML/rois.json | Output ROI configuration file |

### Individual Component Testing

```powershell
# Test vehicle detector
python ML\src\detector.py --video data\videos\sample.mp4

# Test vehicle counter (requires ROIs)
python ML\src\counter.py --video data\videos\sample.mp4

# Test timing calculator
python ML\src\timing.py --test-counts "{\"North\": 5, \"East\": 3, \"South\": 8, \"West\": 2}"
```

## Expected Output

### Console Output
The system outputs Arduino commands every 5 seconds (configurable):

```
[ARDUINO COMMAND] N=12;E=8;S=10;W=6

2024-01-15 14:30:15 - INFO - Timing Update - Vehicles: {'North': 5, 'East': 3, 'South': 4, 'West': 2}, Total: 14
2024-01-15 14:30:15 - INFO - Green Times: {'North': 16, 'East': 12, 'South': 14, 'West': 10}
2024-01-15 14:30:15 - INFO - Arduino Command: N=16;E=12;S=14;W=10
```

### Visual Display
- **Vehicle Detection**: Bounding boxes around detected vehicles
- **ROI Visualization**: Colored rectangles for each direction
- **Vehicle Counts**: Real-time counts per direction
- **System Stats**: FPS, frame count, next update timer

## Performance Targets

- **FPS**: ≥15 FPS with YOLOv8n on CPU
- **Accuracy**: Vehicle detection confidence ≥50%
- **Latency**: <100ms processing time per frame
- **Update Rate**: Traffic timing updates every 3-5 seconds

## Troubleshooting

### Common Issues

**1. "ROI file not found" Error**
```powershell
# Solution: Run ROI selector first
python ML\src\roi_selector.py --video data\videos\sample.mp4
```

**2. Low FPS Performance**
```powershell
# Use smaller resolution or reduce update frequency
python ML\src\main_control.py --video data\videos\sample.mp4 --update-interval 10
```

**3. OpenCV Window Issues**
- Run in PowerShell or VS Code terminal
- Ensure `cv2.waitKey()` is in the processing loop

**4. YOLO Model Download**
- First run automatically downloads `yolov8n.pt` (~6MB)
- Ensure internet connection for initial setup

### Performance Optimization

**For CPU-only systems:**
```powershell
# Use nano model for better speed
python ML\src\main_control.py --model yolov8n.pt --update-interval 5
```

**For systems with GPU:**
```powershell
# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## File Structure

```
SIH-2025-Drone-traffic-control-automation-mvp/
├── ML/
│   ├── src/
│   │   ├── detector.py          # YOLOv8 vehicle detection
│   │   ├── roi_selector.py      # Interactive ROI selection
│   │   ├── counter.py           # Vehicle counting per direction
│   │   ├── timing.py            # Traffic timing calculations
│   │   └── main_control.py      # Main control system
│   ├── requirements.txt         # Python dependencies
│   └── rois.json               # ROI configuration (generated)
├── data/
│   └── videos/                 # Traffic video files
├── docs/
│   └── quickstart.md           # This guide
└── .gitattributes              # Git LFS configuration
```

## Arduino Integration (Future Phase)

The system outputs commands in the format: `N=12;E=8;S=10;W=6`

**Command Format:**
- `N` = North direction green time (seconds)
- `E` = East direction green time (seconds)  
- `S` = South direction green time (seconds)
- `W` = West direction green time (seconds)

**Arduino Implementation Notes:**
- Parse semicolon-separated values
- Implement safe state machine with yellow/all-red transitions
- Use default timings if no ML updates received
- Validate timing values before applying

## Next Steps

1. **Test with your traffic videos**: Place videos in `data/videos/` and run the system
2. **Optimize ROIs**: Fine-tune ROI positions for better counting accuracy
3. **Adjust timing parameters**: Modify min/max green times based on traffic patterns
4. **Hardware integration**: Connect to Arduino/ESP32 for physical signal control

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure ROI configuration is properly set up
4. Test individual components before running the full system

## Performance Monitoring

The system provides detailed performance metrics:
- Real-time FPS display
- Processing time statistics
- Vehicle detection counts
- Timing calculation history

Monitor these metrics to ensure optimal performance for your specific use case.
