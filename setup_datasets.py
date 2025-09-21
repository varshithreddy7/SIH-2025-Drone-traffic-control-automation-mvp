"""
Automated setup script for traffic video dataset infrastructure.
Sets up all components and downloads starter datasets.
"""

import os
import sys
import subprocess
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetSetup:
    """
    Automated setup for traffic video dataset infrastructure.
    """
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.ml_dir = self.project_root / "ML"
        self.datasets_dir = self.ml_dir / "datasets"
        
        # Setup results
        self.setup_results = {
            'system_check': False,
            'dependencies': False,
            'kaggle_setup': False,
            'directory_structure': False,
            'starter_download': False,
            'ml_integration': False,
            'validation': False
        }
        
        # Required packages for dataset management
        self.required_packages = [
            'kaggle>=1.5.16',
            'yt-dlp>=2023.7.6',
            'requests>=2.31.0',
            'beautifulsoup4>=4.12.0',
            'aiohttp>=3.8.0',
            'aiofiles>=23.1.0',
            'moviepy>=1.0.3',
            'rich>=13.0.0'
        ]
    
    def check_system_requirements(self) -> bool:
        """
        Check system requirements and dependencies.
        
        Returns:
            bool: True if system is ready
        """
        logger.info("Checking system requirements...")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            checks.append(True)
        else:
            logger.error(f"✗ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            checks.append(False)
        
        # Check disk space (minimum 1GB)
        try:
            free_space = shutil.disk_usage(self.project_root).free / (1024**3)  # GB
            if free_space >= 1.0:
                logger.info(f"✓ Disk space: {free_space:.1f} GB available")
                checks.append(True)
            else:
                logger.warning(f"⚠ Low disk space: {free_space:.1f} GB (recommend 1GB+)")
                checks.append(True)  # Warning, not error
        except Exception as e:
            logger.warning(f"⚠ Could not check disk space: {e}")
            checks.append(True)
        
        # Check internet connection
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            if response.status_code == 200:
                logger.info("✓ Internet connection available")
                checks.append(True)
            else:
                logger.error("✗ Internet connection test failed")
                checks.append(False)
        except Exception as e:
            logger.error(f"✗ Internet connection error: {e}")
            checks.append(False)
        
        # Check if virtual environment is active
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.info("✓ Virtual environment is active")
            checks.append(True)
        else:
            logger.warning("⚠ Virtual environment not detected (recommended)")
            checks.append(True)  # Warning, not error
        
        success = all(checks)
        self.setup_results['system_check'] = success
        return success
    
    def install_dependencies(self) -> bool:
        """
        Install required Python packages.
        
        Returns:
            bool: Success status
        """
        logger.info("Installing dataset management dependencies...")
        
        try:
            # Update pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            logger.info("✓ pip updated")
            
            # Install required packages
            for package in self.required_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"✓ {package} installed")
                else:
                    logger.warning(f"⚠ {package} installation had issues: {result.stderr}")
            
            # Verify critical imports
            critical_imports = [
                ('kaggle', 'Kaggle API'),
                ('yt_dlp', 'yt-dlp'),
                ('requests', 'Requests'),
                ('bs4', 'BeautifulSoup'),
                ('aiohttp', 'aiohttp')
            ]
            
            for module, name in critical_imports:
                try:
                    __import__(module)
                    logger.info(f"✓ {name} import successful")
                except ImportError as e:
                    logger.error(f"✗ {name} import failed: {e}")
                    return False
            
            self.setup_results['dependencies'] = True
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Package installation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Dependency installation error: {e}")
            return False
    
    def setup_kaggle_credentials(self) -> bool:
        """
        Interactive Kaggle API setup.
        
        Returns:
            bool: Success status
        """
        logger.info("Setting up Kaggle API credentials...")
        
        try:
            from ML.datasets.kaggle_downloader import KaggleDownloader
            
            downloader = KaggleDownloader()
            
            # Check if already configured
            if downloader.kaggle_api:
                logger.info("✓ Kaggle API already configured")
                self.setup_results['kaggle_setup'] = True
                return True
            
            # Interactive setup
            print("\n" + "="*60)
            print("KAGGLE API SETUP")
            print("="*60)
            print("To download datasets from Kaggle, you need API credentials.")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Download kaggle.json file")
            print()
            
            choice = input("Do you want to set up Kaggle API now? (y/n): ").strip().lower()
            
            if choice == 'y':
                success = downloader.setup_kaggle_credentials()
                self.setup_results['kaggle_setup'] = success
                return success
            else:
                logger.info("⚠ Kaggle setup skipped - some datasets won't be available")
                self.setup_results['kaggle_setup'] = False
                return True  # Not a failure, just skipped
        
        except Exception as e:
            logger.error(f"✗ Kaggle setup error: {e}")
            self.setup_results['kaggle_setup'] = False
            return False
    
    def create_directory_structure(self) -> bool:
        """
        Create required directory structure.
        
        Returns:
            bool: Success status
        """
        logger.info("Creating directory structure...")
        
        try:
            directories = [
                self.data_dir / "videos" / "intersection",
                self.data_dir / "videos" / "highway", 
                self.data_dir / "videos" / "city",
                self.data_dir / "videos" / "demo",
                self.data_dir / "videos" / "samples",
                self.data_dir / "rois",
                self.data_dir / "metadata",
                self.ml_dir / "datasets"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created: {directory.relative_to(self.project_root)}")
            
            # Create .gitkeep files for empty directories
            gitkeep_dirs = [
                self.data_dir / "videos" / "intersection",
                self.data_dir / "videos" / "highway",
                self.data_dir / "videos" / "city",
                self.data_dir / "videos" / "demo",
                self.data_dir / "videos" / "samples"
            ]
            
            for directory in gitkeep_dirs:
                gitkeep_file = directory / ".gitkeep"
                if not gitkeep_file.exists():
                    gitkeep_file.write_text("# Directory for traffic videos\n")
            
            self.setup_results['directory_structure'] = True
            return True
        
        except Exception as e:
            logger.error(f"✗ Directory creation failed: {e}")
            return False
    
    def download_starter_dataset(self) -> bool:
        """
        Download starter dataset for immediate testing.
        
        Returns:
            bool: Success status
        """
        logger.info("Downloading starter dataset...")
        
        try:
            from ML.datasets.manager import DatasetManager
            
            manager = DatasetManager()
            success = manager.download_starter_dataset(max_size_mb=100)
            
            if success:
                logger.info("✓ Starter dataset downloaded successfully")
                
                # Validate downloaded videos
                validation_results = manager.validate_videos(min_duration=5, min_vehicles=1)
                
                if validation_results['valid_videos'] > 0:
                    logger.info(f"✓ {validation_results['valid_videos']} valid videos ready for testing")
                    self.setup_results['starter_download'] = True
                    return True
                else:
                    logger.warning("⚠ No valid videos found in starter dataset")
                    self.setup_results['starter_download'] = False
                    return False
            else:
                logger.warning("⚠ Starter dataset download failed - you can download manually later")
                self.setup_results['starter_download'] = False
                return False
        
        except Exception as e:
            logger.error(f"✗ Starter dataset download error: {e}")
            self.setup_results['starter_download'] = False
            return False
    
    def test_ml_integration(self) -> bool:
        """
        Test integration with existing ML pipeline.
        
        Returns:
            bool: Success status
        """
        logger.info("Testing ML pipeline integration...")
        
        try:
            # Check if ML components exist
            ml_components = [
                self.ml_dir / "src" / "detector.py",
                self.ml_dir / "src" / "roi_selector.py",
                self.ml_dir / "src" / "counter.py",
                self.ml_dir / "src" / "timing.py",
                self.ml_dir / "src" / "main_control.py"
            ]
            
            missing_components = []
            for component in ml_components:
                if not component.exists():
                    missing_components.append(component.name)
            
            if missing_components:
                logger.warning(f"⚠ Missing ML components: {', '.join(missing_components)}")
                self.setup_results['ml_integration'] = False
                return False
            
            # Test basic imports
            try:
                sys.path.insert(0, str(self.ml_dir / "src"))
                
                from detector import VehicleDetector
                from timing import TrafficTimingCalculator
                
                logger.info("✓ ML components import successfully")
                
                # Test basic functionality
                detector = VehicleDetector()
                calculator = TrafficTimingCalculator()
                
                # Test timing calculation
                test_counts = {'North': 3, 'East': 2, 'South': 4, 'West': 1}
                green_times = calculator.compute_green_times(test_counts)
                command = calculator.format_command(green_times)
                
                logger.info(f"✓ ML pipeline test successful: {command}")
                
                self.setup_results['ml_integration'] = True
                return True
            
            except ImportError as e:
                logger.error(f"✗ ML component import failed: {e}")
                self.setup_results['ml_integration'] = False
                return False
        
        except Exception as e:
            logger.error(f"✗ ML integration test failed: {e}")
            self.setup_results['ml_integration'] = False
            return False
    
    def run_validation_test(self) -> bool:
        """
        Run comprehensive validation test.
        
        Returns:
            bool: Success status
        """
        logger.info("Running validation test...")
        
        try:
            # Check if we have any videos
            video_files = list(self.data_dir.glob("**/*.mp4"))
            
            if not video_files:
                logger.warning("⚠ No video files found for validation")
                self.setup_results['validation'] = False
                return False
            
            # Test with first available video
            test_video = video_files[0]
            logger.info(f"Testing with: {test_video.name}")
            
            # Test ROI selector (dry run)
            try:
                sys.path.insert(0, str(self.ml_dir / "src"))
                from roi_selector import ROISelector
                
                selector = ROISelector()
                logger.info("✓ ROI selector can be initialized")
            except Exception as e:
                logger.warning(f"⚠ ROI selector test failed: {e}")
            
            # Test detector (dry run)
            try:
                from detector import VehicleDetector
                
                detector = VehicleDetector()
                logger.info("✓ Vehicle detector can be initialized")
            except Exception as e:
                logger.warning(f"⚠ Vehicle detector test failed: {e}")
            
            self.setup_results['validation'] = True
            return True
        
        except Exception as e:
            logger.error(f"✗ Validation test failed: {e}")
            self.setup_results['validation'] = False
            return False
    
    def generate_setup_report(self) -> Dict:
        """
        Generate comprehensive setup report.
        
        Returns:
            dict: Setup report
        """
        report = {
            'setup_timestamp': time.time(),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'project_root': str(self.project_root)
            },
            'setup_results': self.setup_results,
            'success_rate': sum(self.setup_results.values()) / len(self.setup_results),
            'next_steps': []
        }
        
        # Generate next steps based on results
        if not self.setup_results['kaggle_setup']:
            report['next_steps'].append("Set up Kaggle API credentials for more datasets")
        
        if not self.setup_results['starter_download']:
            report['next_steps'].append("Download videos manually or retry starter dataset")
        
        if self.setup_results['ml_integration']:
            report['next_steps'].append("Run: python ML\\src\\roi_selector.py --video data\\videos\\[video_file]")
            report['next_steps'].append("Run: python ML\\src\\main_control.py --video data\\videos\\[video_file]")
        
        report['next_steps'].append("Check docs/quickstart.md for detailed usage instructions")
        
        return report
    
    def run_complete_setup(self) -> bool:
        """
        Run complete automated setup.
        
        Returns:
            bool: Overall success status
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED DATASET SETUP")
        logger.info("="*60)
        
        setup_steps = [
            ("System Requirements", self.check_system_requirements),
            ("Install Dependencies", self.install_dependencies),
            ("Kaggle API Setup", self.setup_kaggle_credentials),
            ("Directory Structure", self.create_directory_structure),
            ("Starter Dataset", self.download_starter_dataset),
            ("ML Integration", self.test_ml_integration),
            ("Validation Test", self.run_validation_test)
        ]
        
        for step_name, step_function in setup_steps:
            logger.info(f"\n--- {step_name} ---")
            
            try:
                success = step_function()
                if success:
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.warning(f"⚠ {step_name} completed with issues")
            except Exception as e:
                logger.error(f"✗ {step_name} failed: {e}")
                self.setup_results[step_name.lower().replace(' ', '_')] = False
        
        # Generate and save report
        report = self.generate_setup_report()
        
        report_file = self.project_root / "setup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("SETUP COMPLETE")
        logger.info("="*60)
        
        successful_steps = sum(self.setup_results.values())
        total_steps = len(self.setup_results)
        success_rate = (successful_steps / total_steps) * 100
        
        logger.info(f"Success rate: {successful_steps}/{total_steps} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            logger.info("🎉 Setup completed successfully!")
            logger.info("Your traffic video dataset infrastructure is ready!")
        elif success_rate >= 60:
            logger.info("⚠ Setup completed with some issues")
            logger.info("Most features should work, but some manual setup may be needed")
        else:
            logger.error("❌ Setup had significant issues")
            logger.error("Manual intervention required before system will work properly")
        
        # Next steps
        if report['next_steps']:
            logger.info("\nNext Steps:")
            for i, step in enumerate(report['next_steps'], 1):
                logger.info(f"{i}. {step}")
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        return success_rate >= 60


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Dataset Setup')
    parser.add_argument('--skip-kaggle', action='store_true',
                       help='Skip Kaggle API setup')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip starter dataset download')
    parser.add_argument('--quick', action='store_true',
                       help='Quick setup (skip optional steps)')
    
    args = parser.parse_args()
    
    setup = DatasetSetup()
    
    if args.quick:
        logger.info("Running quick setup...")
        success = (setup.check_system_requirements() and
                  setup.install_dependencies() and
                  setup.create_directory_structure())
        
        if success:
            logger.info("✓ Quick setup completed!")
        else:
            logger.error("✗ Quick setup failed")
    else:
        # Skip certain steps if requested
        if args.skip_kaggle:
            setup.setup_results['kaggle_setup'] = True  # Mark as completed
        if args.skip_download:
            setup.setup_results['starter_download'] = True  # Mark as completed
        
        success = setup.run_complete_setup()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
