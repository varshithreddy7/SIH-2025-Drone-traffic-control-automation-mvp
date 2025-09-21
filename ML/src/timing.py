"""
Traffic signal timing calculation module.
Computes green light durations based on vehicle counts and formats commands for Arduino.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimingConfig:
    """Configuration for timing calculations."""
    min_green_time: int = 6      # Minimum green time in seconds
    max_green_time: int = 25     # Maximum green time in seconds
    base_green_time: int = 6     # Base green time when no vehicles
    time_per_vehicle: int = 2    # Additional seconds per vehicle
    yellow_time: int = 3         # Yellow light duration
    all_red_time: int = 1        # All-red clearance time
    cycle_time_limit: int = 120  # Maximum total cycle time


class TrafficTimingCalculator:
    """
    Calculates optimal traffic signal timing based on vehicle counts.
    """
    
    def __init__(self, config: TimingConfig = None):
        """
        Initialize the timing calculator.
        
        Args:
            config (TimingConfig, optional): Timing configuration
        """
        self.config = config or TimingConfig()
        self.directions = ['North', 'East', 'South', 'West']
        self.last_timings = {}
        self.timing_history = []
        
    def compute_green_times(self, vehicle_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Compute green light durations for each direction based on vehicle counts.
        
        Args:
            vehicle_counts (dict): Vehicle counts per direction
            
        Returns:
            dict: Green light durations per direction
        """
        green_times = {}
        total_vehicles = sum(vehicle_counts.values())
        
        # Handle case with no vehicles
        if total_vehicles == 0:
            logger.info("No vehicles detected, using minimum green times")
            for direction in self.directions:
                green_times[direction] = self.config.base_green_time
            return green_times
        
        # Calculate raw green times based on vehicle counts
        raw_times = {}
        for direction in self.directions:
            count = vehicle_counts.get(direction, 0)
            raw_time = self.config.base_green_time + (count * self.config.time_per_vehicle)
            raw_times[direction] = raw_time
        
        # Apply min/max constraints
        for direction in self.directions:
            green_time = max(self.config.min_green_time, 
                           min(self.config.max_green_time, raw_times[direction]))
            green_times[direction] = green_time
        
        # Check total cycle time and adjust if necessary
        total_cycle_time = self.calculate_total_cycle_time(green_times)
        
        if total_cycle_time > self.config.cycle_time_limit:
            logger.warning(f"Cycle time {total_cycle_time}s exceeds limit {self.config.cycle_time_limit}s")
            green_times = self.adjust_for_cycle_limit(green_times, vehicle_counts)
        
        # Store timing for history
        self.last_timings = green_times.copy()
        self.timing_history.append({
            'timestamp': time.time(),
            'vehicle_counts': vehicle_counts.copy(),
            'green_times': green_times.copy(),
            'total_cycle_time': self.calculate_total_cycle_time(green_times)
        })
        
        # Keep only last 100 entries
        if len(self.timing_history) > 100:
            self.timing_history = self.timing_history[-100:]
        
        logger.info(f"Computed green times: {green_times}")
        return green_times
    
    def calculate_total_cycle_time(self, green_times: Dict[str, int]) -> int:
        """
        Calculate total cycle time including yellow and all-red phases.
        
        Args:
            green_times (dict): Green light durations per direction
            
        Returns:
            int: Total cycle time in seconds
        """
        total_green = sum(green_times.values())
        # Each direction has yellow + all-red after green
        total_transition = len(self.directions) * (self.config.yellow_time + self.config.all_red_time)
        return total_green + total_transition
    
    def adjust_for_cycle_limit(self, green_times: Dict[str, int], 
                              vehicle_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Adjust green times to fit within cycle time limit while maintaining proportionality.
        
        Args:
            green_times (dict): Original green times
            vehicle_counts (dict): Vehicle counts for proportional adjustment
            
        Returns:
            dict: Adjusted green times
        """
        # Calculate available time for green phases
        transition_time = len(self.directions) * (self.config.yellow_time + self.config.all_red_time)
        available_green_time = self.config.cycle_time_limit - transition_time
        
        # Ensure minimum green times can be accommodated
        min_total_green = len(self.directions) * self.config.min_green_time
        if available_green_time < min_total_green:
            logger.error(f"Cannot fit minimum green times in cycle limit")
            # Return minimum times anyway
            return {direction: self.config.min_green_time for direction in self.directions}
        
        # Calculate proportional adjustment
        total_vehicles = sum(vehicle_counts.values())
        adjusted_times = {}
        
        if total_vehicles > 0:
            # Distribute available time proportionally based on vehicle counts
            remaining_time = available_green_time
            
            # First, assign minimum times
            for direction in self.directions:
                adjusted_times[direction] = self.config.min_green_time
                remaining_time -= self.config.min_green_time
            
            # Distribute remaining time proportionally
            for direction in self.directions:
                if remaining_time > 0:
                    proportion = vehicle_counts.get(direction, 0) / total_vehicles
                    additional_time = int(remaining_time * proportion)
                    adjusted_times[direction] += additional_time
        else:
            # Equal distribution when no vehicles
            time_per_direction = available_green_time // len(self.directions)
            for direction in self.directions:
                adjusted_times[direction] = max(self.config.min_green_time, time_per_direction)
        
        logger.info(f"Adjusted green times for cycle limit: {adjusted_times}")
        return adjusted_times
    
    def format_command(self, green_times: Dict[str, int]) -> str:
        """
        Format timing command for Arduino/ESP32 controller.
        
        Args:
            green_times (dict): Green light durations per direction
            
        Returns:
            str: Formatted command string
        """
        # Format: "N=12;E=8;S=10;W=6"
        command_parts = []
        direction_codes = {'North': 'N', 'East': 'E', 'South': 'S', 'West': 'W'}
        
        for direction in self.directions:
            code = direction_codes[direction]
            duration = green_times.get(direction, self.config.base_green_time)
            command_parts.append(f"{code}={duration}")
        
        command = ";".join(command_parts)
        logger.info(f"Formatted command: {command}")
        return command
    
    def get_timing_summary(self) -> Dict:
        """
        Get comprehensive timing summary.
        
        Returns:
            dict: Timing summary with current and historical data
        """
        current_cycle_time = 0
        if self.last_timings:
            current_cycle_time = self.calculate_total_cycle_time(self.last_timings)
        
        return {
            'current_timings': self.last_timings.copy(),
            'current_cycle_time': current_cycle_time,
            'config': {
                'min_green': self.config.min_green_time,
                'max_green': self.config.max_green_time,
                'base_green': self.config.base_green_time,
                'time_per_vehicle': self.config.time_per_vehicle,
                'yellow_time': self.config.yellow_time,
                'all_red_time': self.config.all_red_time,
                'cycle_limit': self.config.cycle_time_limit
            },
            'history_count': len(self.timing_history)
        }
    
    def get_optimization_suggestions(self, vehicle_counts: Dict[str, int]) -> List[str]:
        """
        Get optimization suggestions based on current traffic patterns.
        
        Args:
            vehicle_counts (dict): Current vehicle counts
            
        Returns:
            list: List of optimization suggestions
        """
        suggestions = []
        total_vehicles = sum(vehicle_counts.values())
        
        if total_vehicles == 0:
            suggestions.append("No vehicles detected - consider reducing cycle time")
            return suggestions
        
        # Analyze traffic distribution
        max_count = max(vehicle_counts.values())
        min_count = min(vehicle_counts.values())
        
        if max_count > min_count * 3:
            heavy_directions = [d for d, c in vehicle_counts.items() if c == max_count]
            suggestions.append(f"Heavy traffic in {', '.join(heavy_directions)} - consider longer green times")
        
        # Check for very low traffic
        low_traffic_dirs = [d for d, c in vehicle_counts.items() if c <= 1]
        if len(low_traffic_dirs) >= 2:
            suggestions.append(f"Low traffic in {', '.join(low_traffic_dirs)} - consider minimum green times")
        
        # Check cycle time efficiency
        green_times = self.compute_green_times(vehicle_counts)
        cycle_time = self.calculate_total_cycle_time(green_times)
        
        if cycle_time > self.config.cycle_time_limit * 0.8:
            suggestions.append("Cycle time approaching limit - consider traffic management")
        
        return suggestions


def main():
    """Main function for testing the timing calculator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Timing Calculator Test')
    parser.add_argument('--test-counts', type=str, 
                       default='{"North": 5, "East": 3, "South": 8, "West": 2}',
                       help='Test vehicle counts as JSON string')
    
    args = parser.parse_args()
    
    # Parse test counts
    try:
        test_counts = json.loads(args.test_counts)
    except json.JSONDecodeError:
        logger.error("Invalid JSON format for test counts")
        return
    
    # Create timing calculator
    config = TimingConfig()
    calculator = TrafficTimingCalculator(config)
    
    logger.info("Testing Traffic Timing Calculator")
    logger.info(f"Configuration: {config}")
    logger.info(f"Test vehicle counts: {test_counts}")
    
    # Compute green times
    green_times = calculator.compute_green_times(test_counts)
    
    # Format command
    command = calculator.format_command(green_times)
    
    # Get summary
    summary = calculator.get_timing_summary()
    
    # Get suggestions
    suggestions = calculator.get_optimization_suggestions(test_counts)
    
    # Display results
    print("\n" + "="*50)
    print("TRAFFIC TIMING RESULTS")
    print("="*50)
    print(f"Vehicle Counts: {test_counts}")
    print(f"Green Times: {green_times}")
    print(f"Arduino Command: {command}")
    print(f"Total Cycle Time: {summary['current_cycle_time']} seconds")
    
    if suggestions:
        print("\nOptimization Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    print("\nConfiguration:")
    for key, value in summary['config'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
