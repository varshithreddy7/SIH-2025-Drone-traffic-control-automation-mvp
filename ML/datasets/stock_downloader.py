"""
Stock video downloader for free traffic videos from Coverr.co and Vecteezy.
Implements web scraping and automated download with proper attribution.
"""

import os
import json
import time
import asyncio
import aiohttp
import aiofiles
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDownloader:
    """
    Free stock video downloader with web scraping capabilities.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize stock video downloader.
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Stock video sources configuration
        self.sources = {
            'coverr': {
                'name': 'Coverr.co',
                'base_url': 'https://coverr.co',
                'search_queries': [
                    'traffic',
                    'highway',
                    'intersection',
                    'cars',
                    'road'
                ],
                'license': 'Free for commercial use',
                'attribution_required': False
            },
            'vecteezy': {
                'name': 'Vecteezy',
                'base_url': 'https://www.vecteezy.com',
                'search_queries': [
                    'traffic-videos',
                    'highway-videos',
                    'intersection-videos',
                    'road-videos'
                ],
                'license': 'Free with attribution',
                'attribution_required': True
            }
        }
        
        # Video categories
        self.categories = {
            'intersection': ['intersection', '4-way', 'crossroad', 'junction'],
            'highway': ['highway', 'freeway', 'motorway', 'expressway'],
            'city': ['city', 'urban', 'street', 'downtown']
        }
        
        # Request headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def search_coverr_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for videos on Coverr.co.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            list: List of video information dictionaries
        """
        videos = []
        
        try:
            search_url = f"https://coverr.co/search?q={query}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find video elements (this may need adjustment based on site structure)
            video_elements = soup.find_all('div', class_='video-item')[:max_results]
            
            for element in video_elements:
                try:
                    # Extract video information
                    title_elem = element.find('h3') or element.find('a')
                    title = title_elem.get_text(strip=True) if title_elem else f"Traffic Video {len(videos)+1}"
                    
                    # Find video link
                    link_elem = element.find('a')
                    if link_elem and link_elem.get('href'):
                        video_url = urljoin('https://coverr.co', link_elem['href'])
                        
                        # Get direct download link
                        download_url = self._get_coverr_download_url(video_url)
                        
                        if download_url:
                            videos.append({
                                'title': title,
                                'url': video_url,
                                'download_url': download_url,
                                'source': 'coverr',
                                'category': self._categorize_video(title),
                                'license': 'Free for commercial use'
                            })
                
                except Exception as e:
                    logger.warning(f"Failed to parse video element: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to search Coverr for '{query}': {e}")
        
        return videos
    
    def _get_coverr_download_url(self, video_page_url: str) -> Optional[str]:
        """
        Extract direct download URL from Coverr video page.
        
        Args:
            video_page_url (str): URL of video page
            
        Returns:
            str: Direct download URL or None
        """
        try:
            response = requests.get(video_page_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for download button or direct video link
            download_btn = soup.find('a', {'class': 'download-btn'}) or soup.find('a', string=re.compile('Download', re.I))
            
            if download_btn and download_btn.get('href'):
                return urljoin('https://coverr.co', download_btn['href'])
            
            # Alternative: look for video element
            video_elem = soup.find('video')
            if video_elem and video_elem.get('src'):
                return urljoin('https://coverr.co', video_elem['src'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get download URL from {video_page_url}: {e}")
            return None
    
    def search_vecteezy_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for videos on Vecteezy (simplified implementation).
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            
        Returns:
            list: List of video information dictionaries
        """
        videos = []
        
        # Note: Vecteezy has more complex anti-bot measures
        # This is a simplified implementation
        try:
            search_url = f"https://www.vecteezy.com/free-videos/{query}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # This would need to be adapted based on actual site structure
                video_elements = soup.find_all('div', class_='asset-item')[:max_results]
                
                for i, element in enumerate(video_elements):
                    # Simplified video info
                    videos.append({
                        'title': f"Vecteezy Traffic Video {i+1}",
                        'url': f"https://www.vecteezy.com/video/{query}-{i+1}",
                        'download_url': None,  # Would need to be extracted
                        'source': 'vecteezy',
                        'category': self._categorize_video(query),
                        'license': 'Free with attribution required'
                    })
        
        except Exception as e:
            logger.error(f"Failed to search Vecteezy for '{query}': {e}")
        
        return videos
    
    def _categorize_video(self, title_or_query: str) -> str:
        """
        Categorize video based on title or query.
        
        Args:
            title_or_query (str): Video title or search query
            
        Returns:
            str: Category name
        """
        text = title_or_query.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'city'  # Default category
    
    async def download_video_async(self, video_info: Dict, session: aiohttp.ClientSession) -> bool:
        """
        Download a single video asynchronously.
        
        Args:
            video_info (dict): Video information
            session (aiohttp.ClientSession): HTTP session
            
        Returns:
            bool: Success status
        """
        if not video_info.get('download_url'):
            logger.warning(f"No download URL for {video_info['title']}")
            return False
        
        try:
            # Create category directory
            category = video_info['category']
            category_dir = self.videos_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Generate filename
            safe_title = re.sub(r'[^\w\s-]', '', video_info['title'])
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{video_info['source']}_{safe_title}.mp4"
            file_path = category_dir / filename
            
            # Skip if already exists
            if file_path.exists():
                logger.info(f"✓ Already exists: {filename}")
                return True
            
            # Download video
            async with session.get(video_info['download_url']) as response:
                if response.status == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded += len(chunk)
                    
                    logger.info(f"✓ Downloaded: {filename} ({downloaded/1024/1024:.1f}MB)")
                    
                    # Save video metadata
                    self._save_video_metadata(video_info, file_path)
                    
                    return True
                else:
                    logger.error(f"✗ Failed to download {filename}: HTTP {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"✗ Error downloading {video_info['title']}: {e}")
            return False
    
    def download_videos_from_source(self, source: str, max_videos: int = 10) -> List[Dict]:
        """
        Download videos from a specific source.
        
        Args:
            source (str): Source name ('coverr' or 'vecteezy')
            max_videos (int): Maximum videos to download
            
        Returns:
            list: List of download results
        """
        if source not in self.sources:
            logger.error(f"Unknown source: {source}")
            return []
        
        logger.info(f"Searching {self.sources[source]['name']} for traffic videos...")
        
        all_videos = []
        queries = self.sources[source]['search_queries']
        videos_per_query = max(1, max_videos // len(queries))
        
        # Search for videos
        for query in queries:
            if source == 'coverr':
                videos = self.search_coverr_videos(query, videos_per_query)
            elif source == 'vecteezy':
                videos = self.search_vecteezy_videos(query, videos_per_query)
            else:
                continue
            
            all_videos.extend(videos)
            time.sleep(1)  # Be respectful to the server
        
        # Limit total videos
        all_videos = all_videos[:max_videos]
        
        if not all_videos:
            logger.warning(f"No videos found from {source}")
            return []
        
        logger.info(f"Found {len(all_videos)} videos from {source}")
        
        # Download videos asynchronously
        return asyncio.run(self._download_videos_batch(all_videos))
    
    async def _download_videos_batch(self, videos: List[Dict]) -> List[Dict]:
        """
        Download multiple videos in batch.
        
        Args:
            videos (list): List of video information
            
        Returns:
            list: Download results
        """
        results = []
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            
            for video in videos:
                task = self.download_video_async(video, session)
                tasks.append(task)
            
            # Execute downloads with some delay between requests
            for i, task in enumerate(tasks):
                if i > 0:
                    await asyncio.sleep(2)  # Delay between downloads
                
                success = await task
                results.append({
                    'video': video,
                    'success': success
                })
        
        return results
    
    def download_all_sources(self, max_videos_per_source: int = 5) -> Dict[str, List]:
        """
        Download videos from all sources.
        
        Args:
            max_videos_per_source (int): Maximum videos per source
            
        Returns:
            dict: Results per source
        """
        all_results = {}
        
        for source in self.sources:
            logger.info(f"\n{'='*50}")
            logger.info(f"DOWNLOADING FROM {source.upper()}")
            logger.info(f"{'='*50}")
            
            results = self.download_videos_from_source(source, max_videos_per_source)
            all_results[source] = results
            
            # Summary for this source
            successful = sum(1 for r in results if r['success'])
            logger.info(f"✓ {source}: {successful}/{len(results)} successful downloads")
        
        return all_results
    
    def _save_video_metadata(self, video_info: Dict, file_path: Path):
        """
        Save video metadata including attribution.
        
        Args:
            video_info (dict): Video information
            file_path (Path): Path to downloaded file
        """
        metadata = {
            'title': video_info['title'],
            'source': video_info['source'],
            'source_url': video_info['url'],
            'category': video_info['category'],
            'license': video_info['license'],
            'file_path': str(file_path.relative_to(self.data_dir)),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'download_timestamp': time.time()
        }
        
        # Save metadata file
        metadata_file = self.metadata_dir / f"stock_{file_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_attribution_file(self):
        """Generate attribution file for videos requiring attribution."""
        attribution_content = []
        attribution_content.append("# Video Attribution\n")
        attribution_content.append("This project uses the following stock videos:\n")
        
        # Read all metadata files
        for metadata_file in self.metadata_dir.glob("stock_*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if 'attribution' in metadata.get('license', '').lower():
                    attribution_content.append(f"- **{metadata['title']}**")
                    attribution_content.append(f"  - Source: {metadata['source']}")
                    attribution_content.append(f"  - URL: {metadata['source_url']}")
                    attribution_content.append(f"  - License: {metadata['license']}")
                    attribution_content.append("")
            
            except Exception as e:
                logger.warning(f"Failed to read metadata {metadata_file}: {e}")
        
        # Save attribution file
        attribution_file = self.data_dir / "ATTRIBUTION.md"
        with open(attribution_file, 'w') as f:
            f.write('\n'.join(attribution_content))
        
        logger.info(f"✓ Attribution file saved: {attribution_file}")


def main():
    """Main function for testing stock downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Video Downloader')
    parser.add_argument('--source', type=str, choices=['coverr', 'vecteezy', 'all'],
                       default='all', help='Video source to download from')
    parser.add_argument('--max-videos', type=int, default=5,
                       help='Maximum videos per source')
    parser.add_argument('--generate-attribution', action='store_true',
                       help='Generate attribution file')
    
    args = parser.parse_args()
    
    downloader = StockDownloader()
    
    if args.generate_attribution:
        downloader.generate_attribution_file()
        return
    
    if args.source == 'all':
        results = downloader.download_all_sources(args.max_videos)
        
        # Overall summary
        total_successful = 0
        total_attempted = 0
        
        for source, source_results in results.items():
            successful = sum(1 for r in source_results if r['success'])
            total_successful += successful
            total_attempted += len(source_results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"OVERALL SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total successful: {total_successful}/{total_attempted}")
        
        # Generate attribution file
        downloader.generate_attribution_file()
    
    else:
        results = downloader.download_videos_from_source(args.source, args.max_videos)
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Downloaded {successful}/{len(results)} videos from {args.source}")


if __name__ == "__main__":
    main()
