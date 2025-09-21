"""
Unified dataset manager for traffic video acquisition.
Coordinates Kaggle, stock video, and YouTube downloaders.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
from datetime import datetime

from .kaggle_downloader import KaggleDownloader
from .stock_downloader import StockDownloader
from .youtube_downloader import YouTubeDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Unified dataset manager that coordinates all video acquisition sources.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.metadata_dir = self.data_dir / "metadata"
        self.rois_dir = self.data_dir / "rois"
        
        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.rois_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloaders
        self.kaggle_downloader = KaggleDownloader(data_dir)
        self.stock_downloader = StockDownloader(data_dir)
        self.youtube_downloader = YouTubeDownloader(data_dir)
        
        # Dataset catalog
        self.catalog_file = self.metadata_dir / "dataset_catalog.json"
        self.download_log_file = self.metadata_dir / "download_log.json"
        
        # Available sources
        self.sources = {
            'kaggle': {
                'name': 'Kaggle Datasets',
                'downloader': self.kaggle_downloader,
                'description': 'Curated traffic datasets from Kaggle',
                'size_estimate': '50-200MB',
                'video_count_estimate': '5-15'
            },
            'stock': {
                'name': 'Stock Videos',
                'downloader': self.stock_downloader,
                'description': 'Free stock videos from Coverr and Vecteezy',
                'size_estimate': '20-100MB',
                'video_count_estimate': '3-10'
            },
            'youtube': {
                'name': 'YouTube Videos',
                'downloader': self.youtube_downloader,
                'description': 'Traffic surveillance videos from YouTube',
                'size_estimate': '30-150MB',
                'video_count_estimate': '5-12'
            }
        }
    
    def list_available_datasets(self) -> Dict:
        """
        List all available datasets from all sources.
        
        Returns:
            dict: Available datasets organized by source
        """
        available = {}
        
        # Kaggle datasets
        try:
            kaggle_datasets = self.kaggle_downloader.list_available_datasets()
            available['kaggle'] = {
                'source_info': self.sources['kaggle'],
                'datasets': kaggle_datasets
            }
        except Exception as e:
            logger.warning(f"Failed to list Kaggle datasets: {e}")
            available['kaggle'] = {'source_info': self.sources['kaggle'], 'datasets': {}, 'error': str(e)}
        
        # Stock video sources
        try:
            stock_sources = self.stock_downloader.sources
            available['stock'] = {
                'source_info': self.sources['stock'],
                'sources': stock_sources
            }
        except Exception as e:
            logger.warning(f"Failed to list stock sources: {e}")
            available['stock'] = {'source_info': self.sources['stock'], 'sources': {}, 'error': str(e)}
        
        # YouTube categories
        try:
            youtube_queries = self.youtube_downloader.search_queries
            available['youtube'] = {
                'source_info': self.sources['youtube'],
                'categories': youtube_queries,
                'curated_videos': self.youtube_downloader.curated_videos
            }
        except Exception as e:
            logger.warning(f"Failed to list YouTube categories: {e}")
            available['youtube'] = {'source_info': self.sources['youtube'], 'categories': {}, 'error': str(e)}
        
        return available
    
    def download_dataset(self, source_name: str, dataset_key: str = None, **kwargs) -> bool:
        """
        Download dataset from specified source.
        
        Args:
            source_name (str): Source name ('kaggle', 'stock', 'youtube')
            dataset_key (str): Specific dataset/category key
            **kwargs: Additional arguments for downloaders
            
        Returns:
            bool: Success status
        """
        if source_name not in self.sources:
            logger.error(f"Unknown source: {source_name}")
            return False
        
        logger.info(f"Starting download from {source_name}")
        start_time = time.time()
        
        try:
            if source_name == 'kaggle':
                if dataset_key:
                    success = self.kaggle_downloader.download_dataset(dataset_key, **kwargs)
                else:
                    results = self.kaggle_downloader.download_all_datasets(**kwargs)
                    success = any(results.values())
            
            elif source_name == 'stock':
                if dataset_key:
                    results = self.stock_downloader.download_videos_from_source(dataset_key, **kwargs)
                    success = any(r['success'] for r in results)
                else:
                    results = self.stock_downloader.download_all_sources(**kwargs)
                    success = any(any(r['success'] for r in source_results) 
                                for source_results in results.values())
            
            elif source_name == 'youtube':
                if dataset_key:
                    if dataset_key == 'curated':
                        results = self.youtube_downloader.download_curated_videos(**kwargs)
                        success = any(results.values())
                    else:
                        results = self.youtube_downloader.search_and_download(dataset_key, **kwargs)
                        success = any(results.values())
                else:
                    results = self.youtube_downloader.download_all_categories(**kwargs)
                    success = any(any(category_results.values()) 
                                for category_results in results.values())
            
            else:
                success = False
            
            # Log download attempt
            self._log_download_attempt(source_name, dataset_key, success, time.time() - start_time)
            
            if success:
                logger.info(f"✓ Successfully downloaded from {source_name}")
                self._update_catalog()
            else:
                logger.error(f"✗ Failed to download from {source_name}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error downloading from {source_name}: {e}")
            self._log_download_attempt(source_name, dataset_key, False, time.time() - start_time, str(e))
            return False
    
    def download_starter_dataset(self, max_size_mb: int = 50) -> bool:
        """
        Download a small starter dataset for quick testing.
        
        Args:
            max_size_mb (int): Maximum total size in MB
            
        Returns:
            bool: Success status
        """
        logger.info("Downloading starter dataset for quick testing...")
        
        # Try to download small datasets from each source
        success_count = 0
        
        # 1. Try Kaggle highway dataset (smallest)
        try:
            if self.kaggle_downloader.download_dataset('highway'):
                success_count += 1
                logger.info("✓ Downloaded Kaggle highway dataset")
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
        
        # 2. Try YouTube curated videos (limited)
        try:
            results = self.youtube_downloader.download_curated_videos(max_videos=2)
            if any(results.values()):
                success_count += 1
                logger.info("✓ Downloaded YouTube curated videos")
        except Exception as e:
            logger.warning(f"YouTube download failed: {e}")
        
        # 3. Try stock videos (limited)
        try:
            results = self.stock_downloader.download_videos_from_source('coverr', max_videos=2)
            if any(r['success'] for r in results):
                success_count += 1
                logger.info("✓ Downloaded stock videos")
        except Exception as e:
            logger.warning(f"Stock video download failed: {e}")
        
        if success_count > 0:
            logger.info(f"✓ Starter dataset ready! Downloaded from {success_count} sources")
            self._update_catalog()
            return True
        else:
            logger.error("✗ Failed to download starter dataset from any source")
            return False
    
    def validate_videos(self, min_duration: int = 10, min_vehicles: int = 3) -> Dict:
        """
        Validate downloaded videos for quality and ML compatibility.
        
        Args:
            min_duration (int): Minimum duration in seconds
            min_vehicles (int): Minimum vehicles detected in sample frames
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating downloaded videos...")
        
        validation_results = {
            'total_videos': 0,
            'valid_videos': 0,
            'invalid_videos': 0,
            'details': []
        }
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.videos_dir.rglob(f'*{ext}'))
        
        validation_results['total_videos'] = len(video_files)
        
        if not video_files:
            logger.warning("No video files found to validate")
            return validation_results
        
        # Validate each video
        for video_file in video_files:
            try:
                result = self._validate_single_video(video_file, min_duration, min_vehicles)
                validation_results['details'].append(result)
                
                if result['valid']:
                    validation_results['valid_videos'] += 1
                else:
                    validation_results['invalid_videos'] += 1
            
            except Exception as e:
                logger.warning(f"Failed to validate {video_file.name}: {e}")
                validation_results['details'].append({
                    'file': str(video_file.relative_to(self.data_dir)),
                    'valid': False,
                    'error': str(e)
                })
                validation_results['invalid_videos'] += 1
        
        # Save validation report
        report_file = self.metadata_dir / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation complete: {validation_results['valid_videos']}/{validation_results['total_videos']} videos valid")
        return validation_results
    
    def _validate_single_video(self, video_file: Path, min_duration: int, min_vehicles: int) -> Dict:
        """
        Validate a single video file.
        
        Args:
            video_file (Path): Path to video file
            min_duration (int): Minimum duration
            min_vehicles (int): Minimum vehicles
            
        Returns:
            dict: Validation result
        """
        import cv2
        
        result = {
            'file': str(video_file.relative_to(self.data_dir)),
            'valid': False,
            'duration': 0,
            'fps': 0,
            'resolution': None,
            'sample_vehicles': 0,
            'issues': []
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                result['issues'].append('Cannot open video file')
                return result
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            result['fps'] = fps
            result['resolution'] = f"{width}x{height}"
            
            if fps > 0:
                result['duration'] = frame_count / fps
            
            # Check duration
            if result['duration'] < min_duration:
                result['issues'].append(f'Duration too short: {result["duration"]:.1f}s < {min_duration}s')
            
            # Check resolution
            if width < 480 or height < 360:
                result['issues'].append(f'Resolution too low: {width}x{height}')
            
            # Sample frames for vehicle detection (simplified check)
            vehicle_count = 0
            sample_frames = min(5, int(frame_count // 10))  # Sample every 10% of video
            
            for i in range(sample_frames):
                frame_pos = int((i + 1) * frame_count / (sample_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    # Simple vehicle detection using basic image analysis
                    # This is a placeholder - in practice, you'd use YOLO here
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Count potential vehicle-like objects (simplified)
                    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
                    vehicle_count += len(large_contours)
            
            result['sample_vehicles'] = vehicle_count // sample_frames if sample_frames > 0 else 0
            
            # Check vehicle presence
            if result['sample_vehicles'] < min_vehicles:
                result['issues'].append(f'Too few vehicles detected: {result["sample_vehicles"]} < {min_vehicles}')
            
            cap.release()
            
            # Determine if valid
            result['valid'] = len(result['issues']) == 0
            
        except Exception as e:
            result['issues'].append(f'Validation error: {str(e)}')
        
        return result
    
    def generate_samples(self, duration: int = 30, max_samples: int = 3) -> bool:
        """
        Generate short sample clips from longer videos.
        
        Args:
            duration (int): Sample duration in seconds
            max_samples (int): Maximum samples per video
            
        Returns:
            bool: Success status
        """
        logger.info(f"Generating {duration}s sample clips...")
        
        samples_dir = self.videos_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Find videos longer than sample duration
        video_files = list(self.videos_dir.rglob('*.mp4'))
        samples_created = 0
        
        for video_file in video_files:
            if 'samples' in str(video_file):
                continue  # Skip existing samples
            
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_file))
                
                if not cap.isOpened():
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                if video_duration > duration * 2:  # Only sample if video is significantly longer
                    # Create sample using ffmpeg (if available) or moviepy
                    sample_name = f"sample_{video_file.stem}_{duration}s.mp4"
                    sample_path = samples_dir / sample_name
                    
                    if self._create_video_sample(video_file, sample_path, duration):
                        samples_created += 1
                        logger.info(f"✓ Created sample: {sample_name}")
                    
                    if samples_created >= max_samples:
                        break
            
            except Exception as e:
                logger.warning(f"Failed to process {video_file.name}: {e}")
        
        logger.info(f"Created {samples_created} sample clips")
        return samples_created > 0
    
    def _create_video_sample(self, input_path: Path, output_path: Path, duration: int) -> bool:
        """
        Create a video sample using moviepy.
        
        Args:
            input_path (Path): Input video path
            output_path (Path): Output sample path
            duration (int): Sample duration
            
        Returns:
            bool: Success status
        """
        try:
            from moviepy.editor import VideoFileClip
            
            with VideoFileClip(str(input_path)) as video:
                # Take sample from middle of video
                start_time = max(0, (video.duration - duration) / 2)
                end_time = min(video.duration, start_time + duration)
                
                sample = video.subclip(start_time, end_time)
                sample.write_videofile(str(output_path), verbose=False, logger=None)
            
            return True
        
        except ImportError:
            logger.warning("moviepy not available for sample creation")
            return False
        except Exception as e:
            logger.warning(f"Failed to create sample: {e}")
            return False
    
    def organize_by_scenario(self) -> bool:
        """
        Organize videos by traffic scenario (light/medium/heavy).
        
        Returns:
            bool: Success status
        """
        logger.info("Organizing videos by traffic scenario...")
        
        # This would analyze videos and categorize them
        # For now, create the directory structure
        scenarios = ['light', 'medium', 'heavy', 'unbalanced']
        
        for scenario in scenarios:
            scenario_dir = self.videos_dir / scenario
            scenario_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Scenario directories created")
        return True
    
    def _update_catalog(self):
        """Update the dataset catalog with current status."""
        catalog = {
            'last_updated': datetime.now().isoformat(),
            'sources': {},
            'total_videos': 0,
            'total_size_mb': 0
        }
        
        # Count videos and calculate sizes
        for source in self.sources:
            source_info = {
                'videos': 0,
                'size_mb': 0,
                'categories': {}
            }
            
            # Count by category
            for category_dir in self.videos_dir.iterdir():
                if category_dir.is_dir():
                    category_videos = list(category_dir.glob('*.mp4'))
                    category_size = sum(f.stat().st_size for f in category_videos) / (1024 * 1024)
                    
                    source_info['categories'][category_dir.name] = {
                        'videos': len(category_videos),
                        'size_mb': round(category_size, 2)
                    }
                    
                    source_info['videos'] += len(category_videos)
                    source_info['size_mb'] += category_size
            
            catalog['sources'][source] = source_info
            catalog['total_videos'] += source_info['videos']
            catalog['total_size_mb'] += source_info['size_mb']
        
        # Save catalog
        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
    
    def _log_download_attempt(self, source: str, dataset_key: str, success: bool, 
                            duration: float, error: str = None):
        """Log download attempt."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'dataset_key': dataset_key,
            'success': success,
            'duration_seconds': round(duration, 2),
            'error': error
        }
        
        # Load existing log
        log_data = []
        if self.download_log_file.exists():
            try:
                with open(self.download_log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                pass
        
        # Add new entry
        log_data.append(log_entry)
        
        # Keep only last 100 entries
        log_data = log_data[-100:]
        
        # Save log
        with open(self.download_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_status(self) -> Dict:
        """
        Get comprehensive dataset status.
        
        Returns:
            dict: Dataset status information
        """
        status = {
            'catalog': {},
            'validation': {},
            'download_log': [],
            'disk_usage': {}
        }
        
        # Load catalog
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    status['catalog'] = json.load(f)
            except:
                pass
        
        # Load validation report
        quality_report_file = self.metadata_dir / "quality_report.json"
        if quality_report_file.exists():
            try:
                with open(quality_report_file, 'r') as f:
                    status['validation'] = json.load(f)
            except:
                pass
        
        # Load download log
        if self.download_log_file.exists():
            try:
                with open(self.download_log_file, 'r') as f:
                    status['download_log'] = json.load(f)[-10:]  # Last 10 entries
            except:
                pass
        
        # Calculate disk usage
        try:
            total_size = sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file())
            status['disk_usage'] = {
                'total_mb': round(total_size / (1024 * 1024), 2),
                'videos_mb': round(sum(f.stat().st_size for f in self.videos_dir.rglob('*') if f.is_file()) / (1024 * 1024), 2)
            }
        except:
            status['disk_usage'] = {'total_mb': 0, 'videos_mb': 0}
        
        return status


def main():
    """Main function for testing dataset manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Manager')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--download', type=str,
                       help='Download from source (kaggle/stock/youtube/starter)')
    parser.add_argument('--dataset-key', type=str,
                       help='Specific dataset key to download')
    parser.add_argument('--validate', action='store_true',
                       help='Validate downloaded videos')
    parser.add_argument('--generate-samples', action='store_true',
                       help='Generate sample clips')
    parser.add_argument('--status', action='store_true',
                       help='Show dataset status')
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.list:
        available = manager.list_available_datasets()
        print("\nAvailable Datasets:")
        print("="*50)
        for source, info in available.items():
            print(f"\n{source.upper()}: {info['source_info']['name']}")
            print(f"  Description: {info['source_info']['description']}")
            print(f"  Estimated size: {info['source_info']['size_estimate']}")
            print(f"  Estimated videos: {info['source_info']['video_count_estimate']}")
        return
    
    if args.download:
        if args.download == 'starter':
            success = manager.download_starter_dataset()
        else:
            success = manager.download_dataset(args.download, args.dataset_key)
        
        if success:
            print(f"✓ Successfully downloaded from {args.download}")
        else:
            print(f"✗ Failed to download from {args.download}")
        return
    
    if args.validate:
        results = manager.validate_videos()
        print(f"\nValidation Results:")
        print(f"Valid videos: {results['valid_videos']}/{results['total_videos']}")
        return
    
    if args.generate_samples:
        success = manager.generate_samples()
        if success:
            print("✓ Sample clips generated")
        else:
            print("✗ Failed to generate samples")
        return
    
    if args.status:
        status = manager.get_status()
        print("\nDataset Status:")
        print("="*50)
        if status['catalog']:
            print(f"Total videos: {status['catalog']['total_videos']}")
            print(f"Total size: {status['catalog']['total_size_mb']:.1f} MB")
        if status['disk_usage']:
            print(f"Disk usage: {status['disk_usage']['total_mb']:.1f} MB")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
