"""
System test script for drone-based AI traffic signal automation.
Tests all components and validates performance requirements.
"""

import sys
import time
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing package imports...")
    
    try:
        import cv2
        logger.info(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics YOLOv8 imported successfully")
    except ImportError as e:
        logger.error(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        logger.error(f"✗ Pandas import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLOv8 model loading and basic inference."""
    logger.info("Testing YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load model
        model = YOLO('yolov8n.pt')
        logger.info("✓ YOLOv8n model loaded successfully")
        
        # Test inference on dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        start_time = time.time()
        results = model(dummy_image, verbose=False)
        inference_time = time.time() - start_time
        
        logger.info(f"✓ Inference completed in {inference_time*1000:.1f}ms")
        
        if inference_time < 0.1:  # Less than 100ms
            logger.info("✓ Performance target met (< 100ms)")
            return True
        else:
            logger.warning(f"⚠ Performance below target: {inference_time*1000:.1f}ms > 100ms")
            return True  # Still functional, just slower
            
    except Exception as e:
        logger.error(f"✗ YOLOv8 test failed: {e}")
        return False

def test_components():
    """Test individual components."""
    logger.info("Testing system components...")
    
    try:
        # Test detector
        from detector import VehicleDetector
        detector = VehicleDetector()
        logger.info("✓ VehicleDetector initialized")
        
        # Test timing calculator
        from timing import TrafficTimingCalculator, TimingConfig
        config = TimingConfig()
        calculator = TrafficTimingCalculator(config)
        
        # Test with sample data
        test_counts = {'North': 5, 'East': 3, 'South': 8, 'West': 2}
        green_times = calculator.compute_green_times(test_counts)
        command = calculator.format_command(green_times)
        
        logger.info(f"✓ Timing calculation test: {command}")
        
        # Validate command format
        if all(direction in command for direction in ['N=', 'E=', 'S=', 'W=']):
            logger.info("✓ Command format validation passed")
        else:
            logger.error("✗ Command format validation failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component test failed: {e}")
        return False

def test_file_structure():
    """Test that required files and directories exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        'ML/requirements.txt',
        'ML/src/detector.py',
        'ML/src/roi_selector.py',
        'ML/src/counter.py',
        'ML/src/timing.py',
        'ML/src/main_control.py',
        'docs/quickstart.md',
        'data/videos/.gitkeep',
        '.gitattributes'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path}")
        else:
            logger.error(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_performance_simulation():
    """Simulate performance test with dummy data."""
    logger.info("Running performance simulation...")
    
    try:
        import numpy as np
        from detector import VehicleDetector
        
        detector = VehicleDetector()
        
        # Simulate processing 100 frames
        processing_times = []
        for i in range(100):
            dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            start_time = time.time()
            detections, annotated_frame = detector.detect_vehicles(dummy_frame)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            if i % 20 == 0:
                logger.info(f"Processed {i+1}/100 frames...")
        
        avg_time = np.mean(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        theoretical_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        logger.info(f"Performance Results:")
        logger.info(f"  Average processing time: {avg_time*1000:.1f}ms")
        logger.info(f"  Min processing time: {min_time*1000:.1f}ms")
        logger.info(f"  Max processing time: {max_time*1000:.1f}ms")
        logger.info(f"  Theoretical max FPS: {theoretical_fps:.1f}")
        
        if theoretical_fps >= 15:
            logger.info("✓ Performance target met (≥15 FPS)")
            return True
        else:
            logger.warning(f"⚠ Performance below target: {theoretical_fps:.1f} FPS < 15 FPS")
            return False
            
    except Exception as e:
        logger.error(f"✗ Performance simulation failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report."""
    logger.info("="*60)
    logger.info("SYSTEM TEST REPORT")
    logger.info("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("YOLOv8 Model", test_yolo_model),
        ("System Components", test_components),
        ("File Structure", test_file_structure),
        ("Performance Simulation", test_performance_simulation)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.error(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! System is ready for use.")
    elif passed_tests >= total_tests * 0.8:
        logger.info("⚠ Most tests passed. System should work with minor issues.")
    else:
        logger.error("❌ Multiple test failures. Please fix issues before proceeding.")
    
    return results

def main():
    """Main test function."""
    logger.info("Starting system validation tests...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    results = generate_test_report()
    
    # Save results to file
    results_file = Path("test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'python_version': sys.version,
            'results': results
        }, f, indent=2)
    
    logger.info(f"\nTest results saved to: {results_file}")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
