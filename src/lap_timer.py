"""
Lap timing system for tracking lap times and detecting lap completion.

This module provides comprehensive lap timing functionality including:
- Current lap time tracking
- Last and best lap time recording
- Lap completion detection based on track segments
- Time formatting for display
"""

import time
import math
from typing import Optional, Tuple
from .track_generator import Track
from .constants import LAP_DETECTION_POSITION_TOLERANCE


class LapTimer:
    """Manages lap timing and lap completion detection"""
    
    def __init__(self, track: Optional[Track] = None, car_id: str = "Unknown"):
        """
        Initialize lap timer.
        
        Args:
            track: Track instance for lap detection, can be None for time-only mode
            car_id: Identifier for the car (for debug messages)
        """
        self.track = track
        self.car_id = car_id
        
        # Timing state
        self.current_lap_start_time = None
        self.current_lap_time = 0.0
        self.last_lap_time = None
        self.best_lap_time = None
        
        # Lap detection state
        self.has_crossed_startline = False
        self.last_car_position = None
        self.lap_count = 0
        self.startline_segment = None
        
        # Lap validation constants
        self.minimum_lap_time = 10.0  # Minimum realistic lap time in seconds
        self.minimum_lap_distance_percent = 0.95  # Require 95% of track length
        self.minimum_lap_distance = 100.0  # Fallback minimum distance if track length unavailable
        self.total_distance_traveled = 0.0  # Track total distance for validation
        
        # Lap completion cooldown to prevent rapid re-triggering
        self.lap_completion_cooldown = 2.0  # Seconds to wait after completing a lap
        self.last_lap_completion_time = None  # Time when last lap was completed
        
        # Calculate dynamic minimum distance based on track length
        if self.track:
            track_length = self.track.get_total_track_length()
            if track_length > 0:
                self.minimum_lap_distance = track_length * self.minimum_lap_distance_percent
                print(f"LapTimer: Track length {track_length:.1f}m, minimum lap distance: {self.minimum_lap_distance:.1f}m ({self.minimum_lap_distance_percent*100:.0f}%)")
        
        # Find startline segment if track is provided
        if self.track:
            self._find_startline_segment()
    
    def _find_startline_segment(self) -> None:
        """Find the STARTLINE segment in the track"""
        if not self.track or not self.track.segments:
            return
            
        for segment in self.track.segments:
            if segment.segment_type == "STARTLINE":
                self.startline_segment = segment
                break
    
    def start_timing(self, simulation_time: float = None) -> None:
        """Start timing a new lap"""
        if simulation_time is not None:
            self.current_lap_start_time = simulation_time
        else:
            # Fallback to wall-clock time for backward compatibility
            self.current_lap_start_time = time.time()
        self.current_lap_time = 0.0
    
    def update(self, car_position: Optional[Tuple[float, float]] = None, simulation_time: float = None) -> bool:
        """
        Update lap timer for one time step.
        
        Args:
            car_position: Current car position (x, y) for lap detection
            simulation_time: Current simulation time (use instead of wall-clock time)
            
        Returns:
            True if a lap was completed this update, False otherwise
        """
        # Update current lap time if timing is active
        if self.current_lap_start_time is not None:
            if simulation_time is not None:
                self.current_lap_time = simulation_time - self.current_lap_start_time
            else:
                # Fallback to wall-clock time for backward compatibility
                self.current_lap_time = time.time() - self.current_lap_start_time
        
        # Update distance traveled if we have position data
        if car_position and self.last_car_position:
            self._update_distance_traveled(car_position)
        
        # Check for lap completion if we have position and track data
        lap_completed = False
        if car_position and self.startline_segment:
            lap_completed = self._check_lap_completion(car_position, simulation_time)
        
        # Store position for next update
        self.last_car_position = car_position
        
        return lap_completed
    
    def _update_distance_traveled(self, current_position: Tuple[float, float]) -> None:
        """
        Update the total distance traveled by the car.
        
        Args:
            current_position: Current car position (x, y)
        """
        if self.last_car_position:
            dx = current_position[0] - self.last_car_position[0]
            dy = current_position[1] - self.last_car_position[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Only add realistic distance increments (prevents teleportation from inflating distance)
            if distance < 50.0:  # Maximum realistic distance per update (at 60fps, this is ~3000 m/s max speed)
                self.total_distance_traveled += distance
    
    def _check_lap_completion(self, car_position: Tuple[float, float], simulation_time: float = None) -> bool:
        """
        Check if car has completed a lap by crossing the startline.
        
        Args:
            car_position: Current car position (x, y)
            simulation_time: Current simulation time
            
        Returns:
            True if lap was completed, False otherwise
        """
        if not self.startline_segment or self.last_car_position is None:
            return False
        
        # Check if car has crossed the startline segment
        crossed_now = self._is_position_on_startline(car_position)
        crossed_before = self._is_position_on_startline(self.last_car_position)
        
        # Lap completion occurs when:
        # 1. Car crosses into startline area (crossed_now and not crossed_before)
        # 2. Car has already crossed startline at least once (to avoid counting start as lap)
        # 3. We're currently timing a lap
        # 4. Sufficient time has passed for a realistic lap (minimum time validation)
        # 5. Sufficient distance has been traveled (minimum distance validation)
        
        if crossed_now and not crossed_before:
            if self.has_crossed_startline and self.current_lap_start_time is not None:
                # Check cooldown period to prevent rapid re-triggering
                current_time = time.time()
                if self.last_lap_completion_time is not None:
                    time_since_last_lap = current_time - self.last_lap_completion_time
                    if time_since_last_lap < self.lap_completion_cooldown:
                        # Still in cooldown period, ignore this crossing
                        return False
                
                # Validate minimum lap time
                if self.current_lap_time < self.minimum_lap_time:
                    print(f"⚠️  {self.car_id} Lap completion rejected: time too short ({self.current_lap_time:.3f}s < {self.minimum_lap_time}s)")
                    return False
                
                # Validate minimum distance traveled
                if self.total_distance_traveled < self.minimum_lap_distance:
                    percent_completed = (self.total_distance_traveled / self.minimum_lap_distance) * self.minimum_lap_distance_percent * 100
                    print(f"⚠️  {self.car_id} Lap completion rejected: distance too short")
                    print(f"     Distance traveled: {self.total_distance_traveled:.1f}m")
                    print(f"     Required distance: {self.minimum_lap_distance:.1f}m ({self.minimum_lap_distance_percent*100:.0f}% of track)")
                    print(f"     Track completion: {percent_completed:.1f}%")
                    return False
                
                # Complete the current lap
                self._complete_lap(simulation_time)
                return True
            elif not self.has_crossed_startline:
                # First crossing - start timing
                self.has_crossed_startline = True
                self.start_timing(simulation_time)
                # Reset distance tracking for new lap
                self.total_distance_traveled = 0.0
        
        return False
    
    def _is_position_on_startline(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is within the startline segment area.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if position is on startline, False otherwise
        """
        if not self.startline_segment:
            return False
        
        car_x, car_y = position
        start_x, start_y = self.startline_segment.start_position
        end_x, end_y = self.startline_segment.end_position
        width = self.startline_segment.width
        
        # Calculate distance from car to startline centerline
        # Use point-to-line-segment distance calculation
        
        # Vector from start to end of segment
        dx = end_x - start_x
        dy = end_y - start_y
        segment_length_sq = dx * dx + dy * dy
        
        if segment_length_sq < 1e-6:  # Very short segment
            # Distance to start point
            dist_to_line = math.sqrt((car_x - start_x)**2 + (car_y - start_y)**2)
        else:
            # Project car position onto line segment
            t = max(0, min(1, ((car_x - start_x) * dx + (car_y - start_y) * dy) / segment_length_sq))
            closest_x = start_x + t * dx
            closest_y = start_y + t * dy
            dist_to_line = math.sqrt((car_x - closest_x)**2 + (car_y - closest_y)**2)
        
        # Car is on startline if within half track width of the centerline
        return dist_to_line <= (width / 2.0)
    
    def _complete_lap(self, simulation_time: float = None) -> None:
        """Complete the current lap and update timing records"""
        if self.current_lap_start_time is None:
            return
        
        # Record the completed lap time
        completed_time = self.current_lap_time
        self.last_lap_time = completed_time
        
        # Update best time if this is a new best
        if self.best_lap_time is None or completed_time < self.best_lap_time:
            self.best_lap_time = completed_time
        
        # Record completion time for cooldown (use simulation time if available)
        if simulation_time is not None:
            self.last_lap_completion_time = simulation_time
        else:
            self.last_lap_completion_time = time.time()
        
        # Start timing the next lap
        self.start_timing(simulation_time)
        
        # Increment lap count and reset distance tracking for next lap (consecutive operations)
        self.lap_count += 1
        self.total_distance_traveled = 0.0
    
    
    def reset(self) -> None:
        """Reset lap timer to initial state"""
        self.current_lap_start_time = None
        self.current_lap_time = 0.0
        self.last_lap_time = None
        self.best_lap_time = None
        self.has_crossed_startline = False
        self.last_car_position = None
        self.lap_count = 0
        self.total_distance_traveled = 0.0
        self.last_lap_completion_time = None
    
    def get_current_lap_time(self) -> float:
        """Get current lap time in seconds"""
        return self.current_lap_time
    
    def get_last_lap_time(self) -> Optional[float]:
        """Get last completed lap time in seconds, None if no laps completed"""
        return self.last_lap_time
    
    def get_best_lap_time(self) -> Optional[float]:
        """Get best lap time in seconds, None if no laps completed"""
        return self.best_lap_time
    
    def get_lap_count(self) -> int:
        """Get number of completed laps"""
        return self.lap_count
    
    def is_timing(self) -> bool:
        """Check if lap timing is currently active"""
        return self.current_lap_start_time is not None
    
    @staticmethod
    def format_time(time_seconds: Optional[float]) -> str:
        """
        Format time in seconds to MM:SS.mmm display format.
        
        Args:
            time_seconds: Time in seconds, None for no time
            
        Returns:
            Formatted time string, "--:--.---" if None
        """
        if time_seconds is None:
            return "--:--.---"
        
        # Handle negative times (shouldn't happen but be safe)
        if time_seconds < 0:
            return "--:--.---"
        
        # Convert to minutes, seconds, milliseconds
        total_seconds = int(time_seconds)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        milliseconds = int(round((time_seconds - total_seconds) * 1000))
        
        # Format as MM:SS.mmm
        return f"{minutes:2d}:{seconds:02d}.{milliseconds:03d}"
    
    def get_timing_info(self) -> dict:
        """
        Get comprehensive timing information for display/logging.
        
        Returns:
            Dictionary with timing information
        """
        return {
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "lap_count": self.lap_count,
            "is_timing": self.is_timing(),
            "has_crossed_startline": self.has_crossed_startline,
            "total_distance_traveled": self.total_distance_traveled,
            "formatted_current": self.format_time(self.current_lap_time if self.is_timing() else None),
            "formatted_last": self.format_time(self.last_lap_time),
            "formatted_best": self.format_time(self.best_lap_time)
        }
    
    def __str__(self) -> str:
        """String representation of lap timer state"""
        info = self.get_timing_info()
        return (f"LapTimer: Current: {info['formatted_current']}, "
                f"Last: {info['formatted_last']}, "
                f"Best: {info['formatted_best']}, "
                f"Laps: {self.lap_count}")