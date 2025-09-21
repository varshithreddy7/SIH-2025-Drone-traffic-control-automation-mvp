"""
YouTube video downloader for traffic surveillance videos using yt-dlp.
Focuses on Creative Commons and usage rights compliant content.
"""

import os
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """
    YouTube video downloader with yt-dlp integration for traffic videos.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize YouTube downloader.
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Search queries for traffic videos
        self.search_queries = {
            'intersection': [
                'traffic intersection camera view',
                '4 way intersection drone footage',
                'traffic junction surveillance',
                'crossroad traffic monitoring',
                'intersection traffic light'
            ],
            'highway': [
                'highway traffic surveillance camera',
                'freeway traffic drone footage',
                'motorway traffic monitoring',
                'highway traffic flow aerial',
                'expressway traffic camera'
            ],
            'city': [
                'city traffic surveillance',
                'urban traffic monitoring',
                'street traffic camera view',
                'downtown traffic footage',
                'city intersection traffic'
            ]
        }
        
        # Pre-selected high-quality traffic video URLs
        self.curated_videos = [
            {
                'url': 'https://www.youtube.com/watch?v=MNn9qKG2UFI',
                'title': 'Traffic Intersection Time Lapse',
                'category': 'intersection',
                'description': 'Time lapse of busy traffic intersection'
            },
            {
                'url': 'https://www.youtube.com/watch?v=wqctLW0Hb_0',
                'title': 'Highway Traffic Flow',
                'category': 'highway', 
                'description': 'Highway traffic monitoring footage'
            },
            {
                'url': 'https://www.youtube.com/watch?v=n-e4Q8d4ShA',
                'title': 'City Traffic Surveillance',
                'category': 'city',
                'description': 'Urban traffic monitoring camera'
            }
        ]
        
        # yt-dlp configuration
        self.ytdl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.videos_dir / '%(category)s' / 'youtube_%(title)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': False,
            'extractflat': False,
            'writethumbnail': False,
            'max_downloads': 50,
            'sleep_interval': 2,
            'max_sleep_interval': 5
        }
        
        self._check_ytdlp_installation()
    
    def _check_ytdlp_installation(self) -> bool:
        """
        Check if yt-dlp is installed and accessible.
        
        Returns:
            bool: True if yt-dlp is available
        """
        try:
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ yt-dlp version: {result.stdout.strip()}")
                return True
            else:
                logger.error("✗ yt-dlp not working properly")
                return False
        except FileNotFoundError:
            logger.error("✗ yt-dlp not found. Install with: pip install yt-dlp")
            return False
        except subprocess.TimeoutExpired:
            logger.error("✗ yt-dlp command timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Error checking yt-dlp: {e}")
            return False
    
    def search_videos(self, query: str, max_results: int = 5, 
                     duration_min: int = 10, duration_max: int = 300) -> List[Dict]:
        """
        Search for videos on YouTube using yt-dlp.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            duration_min (int): Minimum duration in seconds
            duration_max (int): Maximum duration in seconds
            
        Returns:
            list: List of video information
        """
        videos = []
        
        try:
            # Construct search URL
            search_url = f"ytsearch{max_results}:{query}"
            
            # yt-dlp command for search
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--flat-playlist',
                search_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse JSON output
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            video_info = json.loads(line)
                            
                            # Filter by duration
                            duration = video_info.get('duration', 0)
                            if duration and duration_min <= duration <= duration_max:
                                videos.append({
                                    'id': video_info.get('id'),
                                    'title': video_info.get('title', 'Unknown'),
                                    'url': f"https://www.youtube.com/watch?v={video_info.get('id')}",
                                    'duration': duration,
                                    'uploader': video_info.get('uploader', 'Unknown'),
                                    'view_count': video_info.get('view_count', 0),
                                    'category': self._categorize_video(video_info.get('title', ''))
                                })
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Search failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"Search timed out for query: {query}")
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
        
        return videos
    
    def _categorize_video(self, title: str) -> str:
        """
        Categorize video based on title.
        
        Args:
            title (str): Video title
            
        Returns:
            str: Category name
        """
        title_lower = title.lower()
        
        intersection_keywords = ['intersection', 'junction', 'crossroad', '4-way', 'traffic light']
        highway_keywords = ['highway', 'freeway', 'motorway', 'expressway', 'interstate']
        city_keywords = ['city', 'urban', 'downtown', 'street']
        
        if any(keyword in title_lower for keyword in intersection_keywords):
            return 'intersection'
        elif any(keyword in title_lower for keyword in highway_keywords):
            return 'highway'
        elif any(keyword in title_lower for keyword in city_keywords):
            return 'city'
        else:
            return 'city'  # Default
    
    def download_video(self, video_info: Dict, category: str = None) -> bool:
        """
        Download a single video.
        
        Args:
            video_info (dict): Video information
            category (str): Override category
            
        Returns:
            bool: Success status
        """
        try:
            video_url = video_info['url']
            video_category = category or video_info.get('category', 'city')
            
            # Create category directory
            category_dir = self.videos_dir / video_category
            category_dir.mkdir(exist_ok=True)
            
            # Update output template for this download
            opts = self.ytdl_opts.copy()
            opts['outtmpl'] = str(category_dir / 'youtube_%(title)s.%(ext)s')
            
            # yt-dlp command
            cmd = ['yt-dlp']
            
            # Add options
            cmd.extend(['--format', opts['format']])
            cmd.extend(['--output', opts['outtmpl']])
            cmd.append('--write-info-json')
            cmd.append('--no-warnings')
            
            # Add URL
            cmd.append(video_url)
            
            logger.info(f"Downloading: {video_info.get('title', 'Unknown')}")
            
            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully downloaded: {video_info.get('title', 'Unknown')}")
                
                # Save custom metadata
                self._save_video_metadata(video_info, video_category)
                
                return True
            else:
                logger.error(f"✗ Download failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out: {video_info.get('title', 'Unknown')}")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def download_curated_videos(self, max_videos: int = None) -> Dict[str, bool]:
        """
        Download pre-selected high-quality traffic videos.
        
        Args:
            max_videos (int): Maximum videos to download
            
        Returns:
            dict: Download results
        """
        results = {}
        videos_to_download = self.curated_videos[:max_videos] if max_videos else self.curated_videos
        
        logger.info(f"Downloading {len(videos_to_download)} curated traffic videos...")
        
        for video_info in videos_to_download:
            video_id = video_info['url'].split('v=')[1].split('&')[0]
            
            # Check if already downloaded
            if self._is_video_downloaded(video_id):
                logger.info(f"✓ Already downloaded: {video_info['title']}")
                results[video_id] = True
                continue
            
            # Download video
            success = self.download_video(video_info)
            results[video_id] = success
            
            # Delay between downloads
            if success:
                time.sleep(3)
        
        return results
    
    def search_and_download(self, category: str, max_videos: int = 3) -> Dict[str, bool]:
        """
        Search and download videos for a specific category.
        
        Args:
            category (str): Video category
            max_videos (int): Maximum videos to download
            
        Returns:
            dict: Download results
        """
        if category not in self.search_queries:
            logger.error(f"Unknown category: {category}")
            return {}
        
        results = {}
        queries = self.search_queries[category]
        videos_per_query = max(1, max_videos // len(queries))
        
        logger.info(f"Searching for {category} videos...")
        
        all_videos = []
        
        # Search with multiple queries
        for query in queries:
            logger.info(f"Searching: {query}")
            videos = self.search_videos(query, videos_per_query)
            all_videos.extend(videos)
            time.sleep(2)  # Rate limiting
        
        # Remove duplicates and limit results
        unique_videos = {}
        for video in all_videos:
            video_id = video['id']
            if video_id not in unique_videos:
                unique_videos[video_id] = video
        
        videos_to_download = list(unique_videos.values())[:max_videos]
        
        if not videos_to_download:
            logger.warning(f"No videos found for category: {category}")
            return {}
        
        logger.info(f"Found {len(videos_to_download)} videos for {category}")
        
        # Download videos
        for video in videos_to_download:
            video_id = video['id']
            
            # Check if already downloaded
            if self._is_video_downloaded(video_id):
                logger.info(f"✓ Already downloaded: {video['title']}")
                results[video_id] = True
                continue
            
            # Download video
            success = self.download_video(video, category)
            results[video_id] = success
            
            # Delay between downloads
            if success:
                time.sleep(5)
        
        return results
    
    def download_all_categories(self, videos_per_category: int = 2) -> Dict[str, Dict]:
        """
        Download videos for all categories.
        
        Args:
            videos_per_category (int): Videos per category
            
        Returns:
            dict: Results per category
        """
        all_results = {}
        
        # First, download curated videos
        logger.info("Downloading curated videos...")
        curated_results = self.download_curated_videos(3)
        all_results['curated'] = curated_results
        
        # Then search and download by category
        for category in self.search_queries:
            logger.info(f"\n{'='*50}")
            logger.info(f"DOWNLOADING {category.upper()} VIDEOS")
            logger.info(f"{'='*50}")
            
            category_results = self.search_and_download(category, videos_per_category)
            all_results[category] = category_results
            
            # Summary for this category
            successful = sum(1 for success in category_results.values() if success)
            logger.info(f"✓ {category}: {successful}/{len(category_results)} successful downloads")
        
        return all_results
    
    def _is_video_downloaded(self, video_id: str) -> bool:
        """
        Check if video is already downloaded.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            bool: True if downloaded
        """
        # Check for video files containing the video ID
        for video_file in self.videos_dir.rglob('*.mp4'):
            if video_id in video_file.name:
                return True
        
        # Check metadata files
        metadata_file = self.metadata_dir / f"youtube_{video_id}_metadata.json"
        return metadata_file.exists()
    
    def _save_video_metadata(self, video_info: Dict, category: str):
        """
        Save video metadata.
        
        Args:
            video_info (dict): Video information
            category (str): Video category
        """
        video_id = video_info.get('id') or video_info['url'].split('v=')[1].split('&')[0]
        
        metadata = {
            'video_id': video_id,
            'title': video_info.get('title', 'Unknown'),
            'url': video_info['url'],
            'category': category,
            'duration': video_info.get('duration', 0),
            'uploader': video_info.get('uploader', 'Unknown'),
            'view_count': video_info.get('view_count', 0),
            'download_timestamp': time.time(),
            'source': 'youtube',
            'license': 'YouTube Standard License (check individual video)'
        }
        
        # Save metadata file
        metadata_file = self.metadata_dir / f"youtube_{video_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_downloaded_videos(self) -> Dict:
        """
        Get information about downloaded YouTube videos.
        
        Returns:
            dict: Downloaded video information
        """
        downloaded = {}
        
        for metadata_file in self.metadata_dir.glob("youtube_*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    video_id = metadata['video_id']
                    downloaded[video_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to read metadata {metadata_file}: {e}")
        
        return downloaded
    
    def cleanup_failed_downloads(self):
        """Clean up incomplete or failed downloads."""
        # Remove .part files
        for part_file in self.videos_dir.rglob('*.part'):
            try:
                part_file.unlink()
                logger.info(f"Cleaned up: {part_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {part_file}: {e}")
        
        # Remove empty directories
        for category_dir in self.videos_dir.iterdir():
            if category_dir.is_dir() and not any(category_dir.iterdir()):
                try:
                    category_dir.rmdir()
                    logger.info(f"Removed empty directory: {category_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {category_dir}: {e}")


def main():
    """Main function for testing YouTube downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube Traffic Video Downloader')
    parser.add_argument('--category', type=str, 
                       choices=['intersection', 'highway', 'city', 'all'],
                       default='all', help='Video category to download')
    parser.add_argument('--max-videos', type=int, default=2,
                       help='Maximum videos per category')
    parser.add_argument('--curated-only', action='store_true',
                       help='Download only curated videos')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up failed downloads')
    parser.add_argument('--list-downloaded', action='store_true',
                       help='List downloaded videos')
    
    args = parser.parse_args()
    
    downloader = YouTubeDownloader()
    
    if args.cleanup:
        downloader.cleanup_failed_downloads()
        return
    
    if args.list_downloaded:
        downloaded = downloader.get_downloaded_videos()
        if downloaded:
            print("Downloaded YouTube Videos:")
            for video_id, metadata in downloaded.items():
                print(f"  {metadata['title']} ({metadata['category']})")
                print(f"    Duration: {metadata['duration']}s")
                print(f"    URL: {metadata['url']}")
                print()
        else:
            print("No YouTube videos downloaded yet.")
        return
    
    if args.curated_only:
        results = downloader.download_curated_videos()
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Downloaded {successful}/{len(results)} curated videos")
        return
    
    if args.category == 'all':
        results = downloader.download_all_categories(args.max_videos)
        
        # Overall summary
        total_successful = 0
        total_attempted = 0
        
        for category, category_results in results.items():
            successful = sum(1 for success in category_results.values() if success)
            total_successful += successful
            total_attempted += len(category_results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"OVERALL SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total successful: {total_successful}/{total_attempted}")
    
    else:
        results = downloader.search_and_download(args.category, args.max_videos)
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Downloaded {successful}/{len(results)} videos for {args.category}")


if __name__ == "__main__":
    main()
