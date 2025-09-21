# Traffic Video Dataset Management Guide

## Overview

This guide covers the complete traffic video dataset infrastructure for the drone-based AI traffic signal automation system. The dataset management system provides automated acquisition, validation, and organization of traffic videos from multiple sources.

## Quick Start

### 1. Automated Setup

Run the automated setup script to configure everything:

```powershell
# Complete setup with all features
python setup_datasets.py

# Quick setup (essential components only)
python setup_datasets.py --quick

# Skip interactive Kaggle setup
python setup_datasets.py --skip-kaggle

# Skip starter dataset download
python setup_datasets.py --skip-download
```

### 2. Manual Setup (if needed)

```powershell
# Install additional dependencies
pip install -r ML/requirements.txt

# Setup Kaggle API (optional)
python -m ML.datasets.kaggle_downloader --setup

# Download starter dataset
python -m ML.datasets.manager --download starter
```

## Dataset Sources

### 1. Kaggle Datasets

**Available Datasets:**
- `highway`: Road traffic monitoring videos (11MB)
- `highway-multi`: Multiple highway traffic clips (Variable size)
- `hd-traffic`: 720p traffic videos for object detection (144MB)

**Usage:**
```powershell
# Setup Kaggle API credentials
python -m ML.datasets.kaggle_downloader --setup

# List available datasets
python -m ML.datasets.kaggle_downloader --list

# Download specific dataset
python -m ML.datasets.kaggle_downloader --download highway

# Download all datasets
python -m ML.datasets.kaggle_downloader --download-all
```

### 2. Stock Videos

**Sources:**
- **Coverr.co**: Free commercial use videos
- **Vecteezy**: Free videos with attribution

**Usage:**
```powershell
# Download from specific source
python -m ML.datasets.stock_downloader --source coverr --max-videos 5

# Download from all sources
python -m ML.datasets.stock_downloader --source all --max-videos 3

# Generate attribution file
python -m ML.datasets.stock_downloader --generate-attribution
```

### 3. YouTube Videos

**Categories:**
- `intersection`: 4-way intersections, traffic lights
- `highway`: Highway and freeway traffic
- `city`: Urban street traffic

**Usage:**
```powershell
# Download curated videos (recommended)
python -m ML.datasets.youtube_downloader --curated-only

# Download by category
python -m ML.datasets.youtube_downloader --category intersection --max-videos 2

# Download from all categories
python -m ML.datasets.youtube_downloader --category all --max-videos 2

# Clean up failed downloads
python -m ML.datasets.youtube_downloader --cleanup
```

## Unified Dataset Management

### Dataset Manager Commands

```powershell
# List all available datasets
python -m ML.datasets.manager --list

# Download starter dataset (recommended first step)
python -m ML.datasets.manager --download starter

# Download from specific source
python -m ML.datasets.manager --download kaggle --dataset-key highway
python -m ML.datasets.manager --download youtube --dataset-key curated
python -m ML.datasets.manager --download stock

# Validate downloaded videos
python -m ML.datasets.manager --validate

# Generate sample clips
python -m ML.datasets.manager --generate-samples

# Check dataset status
python -m ML.datasets.manager --status
```

## Directory Structure

After setup, your directory structure will look like:

```
data/
├── videos/
│   ├── intersection/          # 4-way intersection videos
│   ├── highway/              # Highway and freeway videos
│   ├── city/                 # Urban street videos
│   ├── demo/                 # Curated demo videos
│   └── samples/              # Short sample clips
├── rois/
│   ├── intersection_720p.json
│   ├── highway_1080p.json
│   └── city_standard.json
└── metadata/
    ├── dataset_catalog.json
    ├── download_log.json
    └── quality_report.json
```

## Video Validation

The system automatically validates videos for:

- **Duration**: Minimum 10 seconds
- **Resolution**: At least 480x360
- **Vehicle presence**: Basic detection check
- **File integrity**: Readable by OpenCV

**Manual validation:**
```powershell
python -m ML.datasets.manager --validate --min-duration 15 --min-vehicles 5
```

## Integration with ML Pipeline

### 1. ROI Configuration

Create ROI presets for different video types:

```powershell
# Interactive ROI selection
python ML\src\roi_selector.py --video data\videos\intersection\sample.mp4 --output data\rois\intersection_custom.json

# Use with main control system
python ML\src\main_control.py --video data\videos\intersection\sample.mp4 --rois data\rois\intersection_custom.json
```

### 2. Batch Processing

Process multiple videos:

```powershell
# Process all videos in a directory
python ML\src\main_control.py --batch-process data\videos\intersection\

# Process with specific ROI preset
python ML\src\main_control.py --batch-process data\videos\highway\ --rois data\rois\highway_1080p.json
```

### 3. Demo Mode

Run with curated demo videos:

```powershell
python ML\src\main_control.py --demo-mode --video-set demo
```

## Performance Optimization

### Video Preprocessing

```powershell
# Generate 30-second samples from long videos
python -m ML.datasets.manager --generate-samples --duration 30 --max-samples 5

# Resize videos to 720p (if needed)
# This would be done automatically during download
```

### Quality Control

```powershell
# Validate all videos with strict criteria
python -m ML.datasets.manager --validate --min-duration 20 --min-vehicles 5

# Check validation report
cat data\metadata\quality_report.json
```

## Troubleshooting

### Common Issues

**1. Kaggle API Authentication**
```powershell
# Check credentials
python -c "import kaggle; print('Kaggle API working')"

# Reconfigure
python -m ML.datasets.kaggle_downloader --setup
```

**2. YouTube Download Failures**
```powershell
# Update yt-dlp
pip install --upgrade yt-dlp

# Clean up failed downloads
python -m ML.datasets.youtube_downloader --cleanup
```

**3. Low Video Quality**
```powershell
# Check validation report
python -m ML.datasets.manager --status

# Re-validate with lower thresholds
python -m ML.datasets.manager --validate --min-duration 5 --min-vehicles 1
```

**4. Network Issues**
```powershell
# Test with smaller downloads first
python -m ML.datasets.manager --download youtube --dataset-key curated

# Check download log
cat data\metadata\download_log.json
```

### Performance Issues

**Slow Downloads:**
- Use `--max-videos` parameter to limit downloads
- Download during off-peak hours
- Check internet connection stability

**Storage Issues:**
- Monitor disk space: `python -m ML.datasets.manager --status`
- Generate samples instead of keeping full videos
- Use video compression if needed

## Advanced Usage

### Custom Video Sources

Add your own videos to the appropriate category directories:

```powershell
# Copy videos to appropriate directories
copy my_traffic_video.mp4 data\videos\intersection\
copy highway_footage.mp4 data\videos\highway\

# Update catalog
python -m ML.datasets.manager --status
```

### Batch ROI Creation

Create ROI presets for multiple video formats:

```python
# Example: Create ROI preset for 720p intersection videos
from ML.datasets.roi_batch import create_roi_preset

create_roi_preset(
    video_path="data/videos/intersection/sample_720p.mp4",
    output_path="data/rois/intersection_720p.json",
    preset_name="intersection_720p"
)
```

### Demo Preparation

```powershell
# Select best videos for demo
python -m ML.datasets.demo_prep --scenarios light,heavy,unbalanced --duration 30

# Create demo playlist
python -m ML.datasets.demo_prep --create-playlist --output demo_playlist.json
```

## API Reference

### DatasetManager Class

```python
from ML.datasets.manager import DatasetManager

manager = DatasetManager()

# List available datasets
datasets = manager.list_available_datasets()

# Download starter dataset
success = manager.download_starter_dataset(max_size_mb=100)

# Validate videos
results = manager.validate_videos(min_duration=10, min_vehicles=3)

# Get status
status = manager.get_status()
```

### Individual Downloaders

```python
# Kaggle downloader
from ML.datasets.kaggle_downloader import KaggleDownloader
kaggle = KaggleDownloader()
kaggle.download_dataset('highway')

# YouTube downloader
from ML.datasets.youtube_downloader import YouTubeDownloader
youtube = YouTubeDownloader()
youtube.download_curated_videos(max_videos=3)

# Stock downloader
from ML.datasets.stock_downloader import StockDownloader
stock = StockDownloader()
stock.download_all_sources(max_videos_per_source=2)
```

## Best Practices

### 1. Start Small
- Begin with starter dataset
- Test ML pipeline with 2-3 videos
- Expand dataset gradually

### 2. Quality Over Quantity
- Validate videos before processing
- Remove low-quality or corrupted files
- Focus on diverse traffic scenarios

### 3. Organize Systematically
- Use consistent naming conventions
- Maintain proper directory structure
- Keep metadata files updated

### 4. Monitor Resources
- Check disk space regularly
- Monitor download logs
- Clean up failed downloads

### 5. Backup Important Data
- Save ROI configurations
- Export successful video lists
- Keep validation reports

## Integration Examples

### Complete Workflow

```powershell
# 1. Setup infrastructure
python setup_datasets.py

# 2. Download starter dataset
python -m ML.datasets.manager --download starter

# 3. Validate videos
python -m ML.datasets.manager --validate

# 4. Create ROI configuration
python ML\src\roi_selector.py --video data\videos\intersection\sample.mp4

# 5. Run ML pipeline
python ML\src\main_control.py --video data\videos\intersection\sample.mp4

# 6. Process additional videos
python ML\src\main_control.py --batch-process data\videos\intersection\
```

### Automated Demo Setup

```powershell
# Download and prepare demo dataset
python -m ML.datasets.manager --download youtube --dataset-key curated
python -m ML.datasets.manager --generate-samples --duration 30
python ML\src\roi_selector.py --video data\videos\samples\sample_1.mp4
python ML\src\main_control.py --demo-mode
```

This comprehensive dataset infrastructure ensures you have high-quality traffic videos for testing and demonstrating your drone-based AI traffic signal automation system!
