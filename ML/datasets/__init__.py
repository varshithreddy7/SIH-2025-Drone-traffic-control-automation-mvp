"""
Traffic video dataset management package.
Provides automated acquisition from multiple sources with focus on Indian traffic.
"""

from .kaggle_downloader import KaggleDownloader
from .stock_downloader import StockDownloader  
from .youtube_downloader import YouTubeDownloader
from .manager import DatasetManager
from .indian_traffic_downloader import IndianTrafficDownloader
from .indian_roi_presets import IndianROIPresets

__version__ = "1.0.0"
__author__ = "SIH 2025 Team"

__all__ = [
    "KaggleDownloader",
    "StockDownloader", 
    "YouTubeDownloader",
    "DatasetManager",
    "IndianTrafficDownloader",
    "IndianROIPresets"
]
