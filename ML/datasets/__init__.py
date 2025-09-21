"""
Dataset management package for traffic video acquisition and processing.
Provides automated download, validation, and organization of traffic videos.
"""

__version__ = "1.0.0"
__author__ = "SIH 2025 Team"

# Package imports
from .manager import DatasetManager
from .kaggle_downloader import KaggleDownloader
from .stock_downloader import StockDownloader
from .youtube_downloader import YouTubeDownloader
from .validator import VideoValidator
from .demo_prep import DemoPreparator

__all__ = [
    'DatasetManager',
    'KaggleDownloader', 
    'StockDownloader',
    'YouTubeDownloader',
    'VideoValidator',
    'DemoPreparator'
]
