"""
Indian Traffic Intersection Video Downloader
Specialized for drone-based AI traffic signal automation in Indian conditions.
Focuses on overhead/aerial views of 4-way intersections with mixed Indian vehicles.
"""

import os
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yt_dlp
import cv2
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianTrafficDownloader:
    """
    Specialized downloader for Indian traffic intersection videos.
    Focuses on drone/aerial footage of 4-way intersections.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize Indian traffic downloader."""
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.indian_dir = self.videos_dir / "indian_intersections"
        self.dense_dir = self.videos_dir / "indian_dense_traffic"
        self.demo_dir = self.videos_dir / "demo_scenarios"
        
        # Create directories
        for directory in [self.indian_dir, self.dense_dir, self.demo_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Indian-specific search terms for YouTube
        self.indian_search_terms = {
            'aerial_intersections': [
                "drone footage 4 way intersection India",
                "aerial view traffic signal India top down",
                "overhead drone Indian traffic junction",
                "bird eye view traffic intersection India",
                "drone shot traffic signal Delhi Mumbai Bangalore",
                "quadcopter traffic intersection aerial India",
                "Indian traffic junction overhead view",
                "drone video traffic light India four way",
                "aerial traffic signal junction India",
                "overhead view Indian intersection traffic"
            ],
            'city_specific': [
                "drone traffic signal Delhi overhead",
                "aerial view Mumbai traffic junction",
                "Bangalore traffic intersection drone",
                "Hyderabad traffic signal aerial view",
                "Chennai traffic junction overhead",
                "Pune traffic intersection drone footage",
                "Kolkata traffic signal bird eye view",
                "Ahmedabad intersection aerial drone"
            ],
            'dense_traffic': [
                "heavy traffic India intersection aerial",
                "dense Indian traffic overhead drone",
                "rush hour traffic India aerial view",
                "crowded intersection India drone footage",
                "peak hour traffic signal India overhead"
            ]
        }
        
        # Reference videos (known good examples)
        self.reference_videos = {
            'primary_reference': {
                'url': 'https://www.youtube.com/watch?v=a_7adqgcXeQ',
                'description': 'Perfect example of Indian intersection drone view',
                'category': 'reference'
            },
            'curated_indian': [
                {
                    'search': 'Indian traffic intersection aerial view drone',
                    'duration_range': (30, 300),  # 30 seconds to 5 minutes
                    'min_resolution': 720
                },
                {
                    'search': 'overhead traffic signal India four way junction',
                    'duration_range': (20, 180),
                    'min_resolution': 480
                },
                {
                    'search': 'bird eye view traffic intersection Delhi Mumbai',
                    'duration_range': (30, 240),
                    'min_resolution': 720
                }
            ]
        }
        
        # yt-dlp configuration for Indian traffic videos
        self.ydl_opts = {
            'format': 'best[height<=720][ext=mp4]',  # Prefer 720p MP4
            'outtmpl': str(self.indian_dir / 'indian_%(title)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': False,
            'extractflat': False,
            'playlistend': 5,  # Limit playlist downloads
        }
    
    def download_reference_video(self) -> bool:
        """
        Download the primary reference video for testing.
        
        Returns:
            bool: Success status
        """
        logger.info("Downloading primary reference video...")
        
        try:
            reference_url = self.reference_videos['primary_reference']['url']
            
            # Special config for reference video
            ref_opts = self.ydl_opts.copy()
            ref_opts['outtmpl'] = str(self.demo_dir / 'reference_indian_intersection.%(ext)s')
            
            with yt_dlp.YoutubeDL(ref_opts) as ydl:
                ydl.download([reference_url])
            
            logger.info("✓ Reference video downloaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to download reference video: {e}")
            return False
    
    def search_indian_intersections(self, category: str = 'aerial_intersections', 
                                  max_videos: int = 5) -> List[Dict]:
        """
        Search for Indian intersection videos.
        
        Args:
            category (str): Search category
            max_videos (int): Maximum videos to find
            
        Returns:
            List[Dict]: Found video information
        """
        if category not in self.indian_search_terms:
            logger.error(f"Unknown category: {category}")
            return []
        
        found_videos = []
        search_terms = self.indian_search_terms[category]
        
        for search_term in search_terms[:3]:  # Limit to first 3 terms
            logger.info(f"Searching: {search_term}")
            
            try:
                search_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'playlistend': 3,  # Get top 3 results per search
                }
                
                with yt_dlp.YoutubeDL(search_opts) as ydl:
                    search_results = ydl.extract_info(
                        f"ytsearch3:{search_term}",
                        download=False
                    )
                    
                    if search_results and 'entries' in search_results:
                        for entry in search_results['entries']:
                            if len(found_videos) >= max_videos:
                                break
                            
                            video_info = {
                                'url': entry.get('url', ''),
                                'title': entry.get('title', 'Unknown'),
                                'duration': entry.get('duration', 0),
                                'view_count': entry.get('view_count', 0),
                                'search_term': search_term,
                                'category': category
                            }
                            
                            # Basic filtering
                            if self._is_suitable_indian_video(video_info):
                                found_videos.append(video_info)
            
            except Exception as e:
                logger.warning(f"Search failed for '{search_term}': {e}")
                continue
            
            if len(found_videos) >= max_videos:
                break
        
        logger.info(f"Found {len(found_videos)} suitable Indian intersection videos")
        return found_videos
    
    def _is_suitable_indian_video(self, video_info: Dict) -> bool:
        """
        Check if video is suitable for Indian traffic analysis.
        
        Args:
            video_info (Dict): Video information
            
        Returns:
            bool: True if suitable
        """
        title = video_info.get('title', '').lower()
        duration = video_info.get('duration', 0)
        
        # Duration check (15 seconds to 10 minutes)
        if not (15 <= duration <= 600):
            return False
        
        # Title must contain Indian traffic keywords
        indian_keywords = [
            'india', 'indian', 'delhi', 'mumbai', 'bangalore', 'hyderabad',
            'chennai', 'pune', 'kolkata', 'ahmedabad', 'traffic', 'intersection',
            'junction', 'signal', 'aerial', 'drone', 'overhead', 'bird'
        ]
        
        keyword_count = sum(1 for keyword in indian_keywords if keyword in title)
        if keyword_count < 2:  # Must have at least 2 relevant keywords
            return False
        
        # Exclude unwanted content
        exclude_keywords = [
            'accident', 'crash', 'police', 'protest', 'construction',
            'highway', 'expressway', 'flyover', 'underpass', 'tunnel'
        ]
        
        if any(keyword in title for keyword in exclude_keywords):
            return False
        
        return True
    
    def download_indian_intersections(self, category: str = 'aerial_intersections',
                                    max_videos: int = 3) -> Dict[str, bool]:
        """
        Download Indian intersection videos.
        
        Args:
            category (str): Video category
            max_videos (int): Maximum videos to download
            
        Returns:
            Dict[str, bool]: Download results
        """
        logger.info(f"Downloading Indian intersection videos - Category: {category}")
        
        # Search for videos
        found_videos = self.search_indian_intersections(category, max_videos * 2)
        
        if not found_videos:
            logger.warning("No suitable Indian intersection videos found")
            return {}
        
        results = {}
        downloaded = 0
        
        for video_info in found_videos:
            if downloaded >= max_videos:
                break
            
            try:
                logger.info(f"Downloading: {video_info['title']}")
                
                # Configure output directory based on category
                if 'dense' in category:
                    output_dir = self.dense_dir
                else:
                    output_dir = self.indian_dir
                
                download_opts = self.ydl_opts.copy()
                download_opts['outtmpl'] = str(output_dir / f"indian_{category}_%(title)s.%(ext)s")
                
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_info['url']])
                
                results[video_info['title']] = True
                downloaded += 1
                logger.info(f"✓ Downloaded: {video_info['title']}")
                
                # Brief pause between downloads
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"✗ Failed to download {video_info['title']}: {e}")
                results[video_info['title']] = False
        
        logger.info(f"Downloaded {downloaded}/{max_videos} Indian intersection videos")
        return results
    
    def download_all_categories(self, max_per_category: int = 2) -> Dict[str, Dict]:
        """
        Download videos from all Indian traffic categories.
        
        Args:
            max_per_category (int): Maximum videos per category
            
        Returns:
            Dict[str, Dict]: Results by category
        """
        logger.info("Downloading from all Indian traffic categories...")
        
        all_results = {}
        
        # Download reference video first
        ref_success = self.download_reference_video()
        all_results['reference'] = {'reference_video': ref_success}
        
        # Download from each category
        for category in self.indian_search_terms.keys():
            logger.info(f"\n--- Downloading {category} ---")
            category_results = self.download_indian_intersections(category, max_per_category)
            all_results[category] = category_results
            
            # Brief pause between categories
            time.sleep(3)
        
        return all_results
    
    def validate_indian_video(self, video_path: Path) -> Dict:
        """
        Validate if video is suitable for Indian traffic analysis.
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            Dict: Validation results
        """
        result = {
            'file': str(video_path.name),
            'valid': False,
            'perspective': 'unknown',
            'intersection_type': 'unknown',
            'traffic_density': 0,
            'vehicle_diversity': 0,
            'issues': []
        }
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                result['issues'].append('Cannot open video file')
                return result
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check basic requirements
            if duration < 15:
                result['issues'].append(f'Too short: {duration:.1f}s < 15s')
            
            if width < 480 or height < 360:
                result['issues'].append(f'Resolution too low: {width}x{height}')
            
            # Sample frames for analysis
            sample_frames = min(5, int(frame_count // 10))
            vehicle_detections = 0
            
            for i in range(sample_frames):
                frame_pos = int((i + 1) * frame_count / (sample_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    # Basic vehicle detection (simplified)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Look for vehicle-like objects
                    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count potential vehicles (medium-sized moving objects)
                    vehicles_in_frame = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 500 < area < 5000:  # Vehicle-sized objects
                            vehicles_in_frame += 1
                    
                    vehicle_detections += vehicles_in_frame
            
            avg_vehicles = vehicle_detections / sample_frames if sample_frames > 0 else 0
            result['traffic_density'] = avg_vehicles
            
            # Check for Indian traffic characteristics
            if avg_vehicles >= 5:  # Dense traffic expected in Indian intersections
                result['vehicle_diversity'] = min(100, avg_vehicles * 10)
            else:
                result['issues'].append(f'Low traffic density: {avg_vehicles:.1f} vehicles/frame')
            
            # Perspective check (simplified - based on aspect ratio and content)
            aspect_ratio = width / height
            if 1.2 <= aspect_ratio <= 2.0:  # Typical drone footage aspect ratio
                result['perspective'] = 'aerial_likely'
            else:
                result['perspective'] = 'ground_level_likely'
                result['issues'].append('May not be aerial/drone footage')
            
            # Overall validation
            result['valid'] = len(result['issues']) == 0 and avg_vehicles >= 3
            
            cap.release()
        
        except Exception as e:
            result['issues'].append(f'Validation error: {str(e)}')
        
        return result
    
    def validate_all_videos(self) -> Dict:
        """
        Validate all downloaded Indian traffic videos.
        
        Returns:
            Dict: Validation results
        """
        logger.info("Validating Indian traffic videos...")
        
        validation_results = {
            'total_videos': 0,
            'valid_videos': 0,
            'invalid_videos': 0,
            'details': []
        }
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for directory in [self.indian_dir, self.dense_dir, self.demo_dir]:
            for ext in video_extensions:
                video_files.extend(directory.glob(f'*{ext}'))
        
        validation_results['total_videos'] = len(video_files)
        
        if not video_files:
            logger.warning("No Indian traffic videos found to validate")
            return validation_results
        
        # Validate each video
        for video_file in video_files:
            try:
                result = self.validate_indian_video(video_file)
                validation_results['details'].append(result)
                
                if result['valid']:
                    validation_results['valid_videos'] += 1
                    logger.info(f"✓ Valid: {video_file.name}")
                else:
                    validation_results['invalid_videos'] += 1
                    logger.warning(f"⚠ Issues with {video_file.name}: {result['issues']}")
            
            except Exception as e:
                logger.error(f"Failed to validate {video_file.name}: {e}")
                validation_results['invalid_videos'] += 1
        
        logger.info(f"Validation complete: {validation_results['valid_videos']}/{validation_results['total_videos']} videos valid for Indian traffic analysis")
        return validation_results
    
    def get_download_status(self) -> Dict:
        """
        Get status of Indian traffic video downloads.
        
        Returns:
            Dict: Download status
        """
        status = {
            'directories': {},
            'total_videos': 0,
            'total_size_mb': 0
        }
        
        for directory in [self.indian_dir, self.dense_dir, self.demo_dir]:
            if directory.exists():
                video_files = list(directory.glob('*.mp4'))
                total_size = sum(f.stat().st_size for f in video_files) / (1024 * 1024)
                
                status['directories'][directory.name] = {
                    'videos': len(video_files),
                    'size_mb': round(total_size, 2),
                    'files': [f.name for f in video_files]
                }
                
                status['total_videos'] += len(video_files)
                status['total_size_mb'] += total_size
        
        status['total_size_mb'] = round(status['total_size_mb'], 2)
        return status


def main():
    """Main function for testing Indian traffic downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Indian Traffic Video Downloader')
    parser.add_argument('--download-reference', action='store_true',
                       help='Download reference video')
    parser.add_argument('--category', type=str, default='aerial_intersections',
                       help='Video category to download')
    parser.add_argument('--max-videos', type=int, default=3,
                       help='Maximum videos to download')
    parser.add_argument('--download-all', action='store_true',
                       help='Download from all categories')
    parser.add_argument('--validate', action='store_true',
                       help='Validate downloaded videos')
    parser.add_argument('--status', action='store_true',
                       help='Show download status')
    
    args = parser.parse_args()
    
    downloader = IndianTrafficDownloader()
    
    if args.download_reference:
        success = downloader.download_reference_video()
        print("✓ Reference video downloaded" if success else "✗ Reference download failed")
        return
    
    if args.download_all:
        results = downloader.download_all_categories(args.max_videos)
        total_success = sum(sum(category.values()) for category in results.values() if isinstance(category, dict))
        print(f"Downloaded videos from all categories. Total successful: {total_success}")
        return
    
    if args.validate:
        results = downloader.validate_all_videos()
        print(f"Validation Results: {results['valid_videos']}/{results['total_videos']} videos suitable for Indian traffic analysis")
        return
    
    if args.status:
        status = downloader.get_download_status()
        print("\nIndian Traffic Video Status:")
        print("="*50)
        for dir_name, info in status['directories'].items():
            print(f"{dir_name}: {info['videos']} videos ({info['size_mb']} MB)")
        print(f"\nTotal: {status['total_videos']} videos ({status['total_size_mb']} MB)")
        return
    
    # Default: download from specified category
    results = downloader.download_indian_intersections(args.category, args.max_videos)
    successful = sum(results.values())
    print(f"Downloaded {successful}/{len(results)} videos from {args.category}")


if __name__ == "__main__":
    main()
