"""
Kaggle dataset downloader for traffic video datasets.
Handles authentication, download, extraction, and organization.
"""

import os
import json
import zipfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDownloader:
    """
    Automated Kaggle dataset downloader with authentication and error handling.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Kaggle downloader.
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Kaggle datasets configuration
        self.datasets = {
            'highway': {
                'name': 'shawon10/road-traffic-video-monitoring',
                'size': '11MB',
                'description': 'Highway traffic monitoring videos',
                'category': 'highway',
                'files': ['*.mp4', '*.avi']
            },
            'highway-multi': {
                'name': 'aryashah2k/highway-traffic-videos-dataset', 
                'size': 'Variable',
                'description': 'Multiple highway traffic video clips',
                'category': 'highway',
                'files': ['*.mp4', '*.mov']
            },
            'hd-traffic': {
                'name': 'chicicecream/720p-road-and-traffic-video-for-object-detection',
                'size': '144MB', 
                'description': 'HD traffic videos for object detection',
                'category': 'city',
                'files': ['*.mp4']
            }
        }
        
        self.kaggle_api = None
        self._setup_kaggle_api()
    
    def _setup_kaggle_api(self) -> bool:
        """
        Setup Kaggle API with authentication.
        
        Returns:
            bool: Success status
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            logger.info("✓ Kaggle API authenticated successfully")
            return True
            
        except ImportError:
            logger.error("✗ Kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"✗ Kaggle authentication failed: {e}")
            logger.info("Please ensure kaggle.json is in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY")
            return False
    
    def setup_kaggle_credentials(self) -> bool:
        """
        Interactive setup for Kaggle credentials.
        
        Returns:
            bool: Success status
        """
        logger.info("Setting up Kaggle API credentials...")
        
        # Check if credentials already exist
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if kaggle_json.exists():
            logger.info("✓ Kaggle credentials already exist")
            return self._setup_kaggle_api()
        
        # Interactive credential setup
        print("\n" + "="*50)
        print("KAGGLE API SETUP")
        print("="*50)
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json file")
        print("4. Enter your credentials below:")
        print()
        
        try:
            username = input("Kaggle Username: ").strip()
            key = input("Kaggle API Key: ").strip()
            
            if not username or not key:
                logger.error("Username and API key are required")
                return False
            
            # Create kaggle directory and credentials file
            kaggle_dir.mkdir(exist_ok=True)
            
            credentials = {
                "username": username,
                "key": key
            }
            
            with open(kaggle_json, 'w') as f:
                json.dump(credentials, f)
            
            # Set proper permissions (important for security)
            if os.name != 'nt':  # Not Windows
                os.chmod(kaggle_json, 0o600)
            
            logger.info(f"✓ Credentials saved to {kaggle_json}")
            return self._setup_kaggle_api()
            
        except KeyboardInterrupt:
            logger.info("Setup cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Failed to setup credentials: {e}")
            return False
    
    def list_available_datasets(self) -> Dict:
        """
        List all available Kaggle datasets.
        
        Returns:
            dict: Dataset information
        """
        return self.datasets.copy()
    
    def check_dataset_exists(self, dataset_key: str) -> bool:
        """
        Check if dataset exists on Kaggle.
        
        Args:
            dataset_key (str): Dataset key from self.datasets
            
        Returns:
            bool: True if dataset exists
        """
        if not self.kaggle_api:
            return False
        
        if dataset_key not in self.datasets:
            return False
        
        try:
            dataset_name = self.datasets[dataset_key]['name']
            # Try to get dataset info
            dataset_info = self.kaggle_api.dataset_view(dataset_name)
            return True
        except Exception as e:
            logger.warning(f"Dataset {dataset_key} not accessible: {e}")
            return False
    
    def download_dataset(self, dataset_key: str, force_redownload: bool = False) -> bool:
        """
        Download and extract a Kaggle dataset.
        
        Args:
            dataset_key (str): Dataset key from self.datasets
            force_redownload (bool): Force redownload even if exists
            
        Returns:
            bool: Success status
        """
        if not self.kaggle_api:
            logger.error("Kaggle API not initialized")
            return False
        
        if dataset_key not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = self.datasets[dataset_key]
        dataset_name = dataset_info['name']
        category = dataset_info['category']
        
        # Create category directory
        category_dir = self.videos_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if not force_redownload and self._is_dataset_downloaded(dataset_key):
            logger.info(f"✓ Dataset {dataset_key} already downloaded")
            return True
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"Size: {dataset_info['size']}")
        logger.info(f"Category: {category}")
        
        try:
            # Download to temporary directory
            temp_dir = self.data_dir / "temp" / dataset_key
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            self.kaggle_api.dataset_download_files(
                dataset_name,
                path=str(temp_dir),
                unzip=True
            )
            
            # Move video files to appropriate category
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            moved_files = []
            
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    # Create unique filename
                    new_name = f"{dataset_key}_{file_path.name}"
                    dest_path = category_dir / new_name
                    
                    # Move file
                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append(dest_path)
                    logger.info(f"✓ Moved: {dest_path.name}")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Save metadata
            self._save_dataset_metadata(dataset_key, moved_files)
            
            logger.info(f"✓ Successfully downloaded {len(moved_files)} videos from {dataset_key}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_key}: {e}")
            # Clean up on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download all available datasets.
        
        Args:
            force_redownload (bool): Force redownload existing datasets
            
        Returns:
            dict: Download results for each dataset
        """
        results = {}
        
        logger.info("Starting bulk dataset download...")
        
        for dataset_key in self.datasets:
            logger.info(f"\nProcessing dataset: {dataset_key}")
            results[dataset_key] = self.download_dataset(dataset_key, force_redownload)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"DOWNLOAD SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Successful: {successful}/{total}")
        logger.info(f"Failed: {total - successful}/{total}")
        
        return results
    
    def _is_dataset_downloaded(self, dataset_key: str) -> bool:
        """
        Check if dataset is already downloaded.
        
        Args:
            dataset_key (str): Dataset key
            
        Returns:
            bool: True if downloaded
        """
        metadata_file = self.metadata_dir / f"{dataset_key}_metadata.json"
        return metadata_file.exists()
    
    def _save_dataset_metadata(self, dataset_key: str, file_paths: List[Path]):
        """
        Save dataset metadata.
        
        Args:
            dataset_key (str): Dataset key
            file_paths (list): List of downloaded file paths
        """
        metadata = {
            'dataset_key': dataset_key,
            'dataset_info': self.datasets[dataset_key],
            'download_timestamp': pd.Timestamp.now().isoformat(),
            'files': [str(p.relative_to(self.data_dir)) for p in file_paths],
            'file_count': len(file_paths)
        }
        
        metadata_file = self.metadata_dir / f"{dataset_key}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_downloaded_datasets(self) -> Dict:
        """
        Get information about downloaded datasets.
        
        Returns:
            dict: Downloaded dataset information
        """
        downloaded = {}
        
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    dataset_key = metadata['dataset_key']
                    downloaded[dataset_key] = metadata
            except Exception as e:
                logger.warning(f"Failed to read metadata {metadata_file}: {e}")
        
        return downloaded


def main():
    """Main function for testing Kaggle downloader."""
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Kaggle Dataset Downloader')
    parser.add_argument('--setup', action='store_true',
                       help='Setup Kaggle API credentials')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--download', type=str,
                       help='Download specific dataset')
    parser.add_argument('--download-all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--force', action='store_true',
                       help='Force redownload existing datasets')
    
    args = parser.parse_args()
    
    downloader = KaggleDownloader()
    
    if args.setup:
        success = downloader.setup_kaggle_credentials()
        if success:
            print("✓ Kaggle API setup complete!")
        else:
            print("✗ Kaggle API setup failed")
        return
    
    if args.list:
        datasets = downloader.list_available_datasets()
        print("\nAvailable Kaggle Datasets:")
        print("="*50)
        for key, info in datasets.items():
            print(f"Key: {key}")
            print(f"  Name: {info['name']}")
            print(f"  Size: {info['size']}")
            print(f"  Category: {info['category']}")
            print(f"  Description: {info['description']}")
            print()
        return
    
    if args.download:
        success = downloader.download_dataset(args.download, args.force)
        if success:
            print(f"✓ Successfully downloaded {args.download}")
        else:
            print(f"✗ Failed to download {args.download}")
        return
    
    if args.download_all:
        results = downloader.download_all_datasets(args.force)
        print(f"\nDownload completed. Results: {results}")
        return
    
    # Default: show status
    downloaded = downloader.get_downloaded_datasets()
    if downloaded:
        print("Downloaded Datasets:")
        for key, metadata in downloaded.items():
            print(f"  {key}: {metadata['file_count']} files")
    else:
        print("No datasets downloaded yet.")
        print("Use --setup to configure Kaggle API")
        print("Use --list to see available datasets")


if __name__ == "__main__":
    main()
