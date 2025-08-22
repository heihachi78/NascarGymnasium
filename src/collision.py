"""
Collision detection and reporting system.

This module provides comprehensive collision tracking, force calculation,
and direction reporting for car-track interactions.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from .constants import (
    COLLISION_FORCE_THRESHOLD,
    MAX_COLLISION_HISTORY,
    COLLISION_COOLDOWN_TIME,
    COLLISION_MAX_FORCE,
    TWO_PI
)


class CollisionEvent:
    """Represents a single collision event with full details"""
    
    def __init__(self, 
                 position: Tuple[float, float],
                 impulse: float,
                 normal: Tuple[float, float],
                 car_angle: float,
                 timestamp: float):
        """
        Initialize collision event.
        
        Args:
            position: World collision point (x, y)
            impulse: Collision impulse magnitude (mass * delta_velocity)
            normal: Collision surface normal vector (x, y)
            car_angle: Car orientation at time of collision in radians
            timestamp: Simulation time when collision occurred
        """
        self.position = position
        self.impulse = impulse
        self.normal = normal
        self.car_angle = car_angle
        self.timestamp = timestamp
        
        # Calculate derived properties
        self.normal_angle = math.atan2(normal[1], normal[0])
        self.relative_angle = self._calculate_relative_angle()
        
    def _calculate_relative_angle(self) -> float:
        """Calculate collision angle relative to car orientation"""
        relative = self.normal_angle - self.car_angle
        
        # Normalize to [-π, π]
        while relative > math.pi:
            relative -= TWO_PI
        while relative < -math.pi:
            relative += TWO_PI
            
        return relative
        
            
    def get_direction_description(self) -> str:
        """Get human-readable collision direction"""
        angle_deg = math.degrees(self.relative_angle)
        
        # Convert to compass-like directions relative to car
        if -22.5 <= angle_deg <= 22.5:
            return "front"
        elif 22.5 < angle_deg <= 67.5:
            return "front-right"
        elif 67.5 < angle_deg <= 112.5:
            return "right"
        elif 112.5 < angle_deg <= 157.5:
            return "rear-right"
        elif 157.5 < angle_deg or angle_deg <= -157.5:
            return "rear"
        elif -157.5 < angle_deg <= -112.5:
            return "rear-left"
        elif -112.5 < angle_deg <= -67.5:
            return "left"
        else:  # -67.5 < angle_deg <= -22.5
            return "front-left"
            
    def __str__(self) -> str:
        return (f"Collision: {self.get_direction_description()} "
                f"({self.impulse:.0f}N⋅s, "
                f"t={self.timestamp:.1f}s)")


class CollisionReporter:
    """Manages collision detection, tracking, and reporting"""
    
    def __init__(self):
        """Initialize collision reporter"""
        self.collision_history: List[CollisionEvent] = []
        self.current_simulation_time = 0.0
        self.total_collisions = 0
        self.last_collision_time = -float('inf')
        
        # Per-car collision tracking to prevent spam
        self.per_car_last_collision_time = {}  # car_id -> last collision time
        self.collision_rate_limiter = {}  # car_id -> collision count in current window
        self.rate_limit_window_start = {}  # car_id -> window start time
        
        # Collision impulse statistics (no severity classification)
        self.total_impulse = 0.0
        self.max_impulse = 0.0
        
        # Direction statistics
        self.direction_stats = {
            "front": 0, "front-right": 0, "right": 0, "rear-right": 0,
            "rear": 0, "rear-left": 0, "left": 0, "front-left": 0
        }
        
    def update_time(self, simulation_time: float) -> None:
        """Update current simulation time"""
        self.current_simulation_time = simulation_time
        
    def report_collision(self,
                        position: Tuple[float, float],
                        impulse: float,
                        normal: Tuple[float, float], 
                        car_angle: float,
                        car_id: Optional[str] = None) -> bool:
        """
        Report a new collision event.
        
        Args:
            position: World collision point
            impulse: Collision impulse magnitude
            normal: Collision surface normal
            car_angle: Car orientation in radians
            car_id: Optional car identifier for per-car rate limiting
            
        Returns:
            True if collision was recorded (above threshold and cooldown)
        """
        # Check impulse threshold
        if impulse < COLLISION_FORCE_THRESHOLD:
            return False
            
        # Check global cooldown period (backward compatibility)
        if self.current_simulation_time - self.last_collision_time < COLLISION_COOLDOWN_TIME:
            return False
            
        # Check per-car cooldown period to prevent spam from individual cars
        if car_id is not None:
            last_car_collision_time = self.per_car_last_collision_time.get(car_id, -float('inf'))
            if self.current_simulation_time - last_car_collision_time < COLLISION_COOLDOWN_TIME:
                return False
                
            # Implement rate limiting per car (max 5 collisions per second per car)
            window_duration = 1.0  # 1 second window
            max_collisions_per_window = 5
            
            # Reset window if enough time has passed
            window_start = self.rate_limit_window_start.get(car_id, self.current_simulation_time)
            if self.current_simulation_time - window_start >= window_duration:
                self.rate_limit_window_start[car_id] = self.current_simulation_time
                self.collision_rate_limiter[car_id] = 0
            
            # Check if we've exceeded the rate limit
            current_count = self.collision_rate_limiter.get(car_id, 0)
            if current_count >= max_collisions_per_window:
                return False
                
            # Update rate limiter
            self.collision_rate_limiter[car_id] = current_count + 1
            self.per_car_last_collision_time[car_id] = self.current_simulation_time
            
        # Create collision event
        collision = CollisionEvent(
            position=position,
            impulse=impulse,
            normal=normal,
            car_angle=car_angle,
            timestamp=self.current_simulation_time
        )
        
        # Print collision force to console
        #print(f"Collision Force: {impulse:.1f} N⋅s")
        
        # Add to history
        self.collision_history.append(collision)
        self.last_collision_time = self.current_simulation_time
        self.total_collisions += 1
        
        # Update statistics
        self.total_impulse += collision.impulse
        self.max_impulse = max(self.max_impulse, collision.impulse)
        self.direction_stats[collision.get_direction_description()] += 1
        
        # Maintain history size limit
        if len(self.collision_history) > MAX_COLLISION_HISTORY:
            removed = self.collision_history.pop(0)
            # Update stats for removed collision
            self.total_impulse -= removed.impulse
            self.direction_stats[removed.get_direction_description()] -= 1
            
        return True
        
    def get_latest_collision(self) -> Optional[CollisionEvent]:
        """Get the most recent collision event"""
        if not self.collision_history:
            return None
        return self.collision_history[-1]
        
    def get_collision_for_observation(self) -> Tuple[float, float]:
        """
        Get collision data formatted for environment observation.
        
        Returns:
            Tuple of (collision_impulse, collision_angle_relative_to_car)
        """
        recent_collision = self.get_recent_collision()
        if not recent_collision:
            return (0.0, 0.0)
            
        return (recent_collision.impulse, recent_collision.relative_angle)
        
    def get_recent_collision(self, max_age: float = 1.0) -> Optional[CollisionEvent]:
        """
        Get most recent collision within specified time window.
        
        Args:
            max_age: Maximum age of collision in seconds
            
        Returns:
            Most recent collision within time window, or None
        """
        if not self.collision_history:
            return None
            
        # Find most recent collision within time window
        cutoff_time = self.current_simulation_time - max_age
        recent_collisions = [c for c in self.collision_history if c.timestamp > cutoff_time]
        
        if not recent_collisions:
            return None
            
        # Return most severe recent collision
        return max(recent_collisions, key=lambda c: c.impulse)
        
    def get_collisions_in_timeframe(self, 
                                   start_time: float, 
                                   end_time: float) -> List[CollisionEvent]:
        """Get all collisions within specified time range"""
        return [c for c in self.collision_history 
                if start_time <= c.timestamp <= end_time]
                
    def clear_old_collisions(self, max_age: float = 10.0) -> None:
        """Remove collisions older than specified age"""
        cutoff_time = self.current_simulation_time - max_age
        old_count = len(self.collision_history)
        
        # Filter out old collisions
        self.collision_history = [c for c in self.collision_history 
                                if c.timestamp > cutoff_time]
        
        removed_count = old_count - len(self.collision_history)
        if removed_count > 0:
            # Recalculate statistics (simplified approach)
            self._recalculate_statistics()
            
    def _recalculate_statistics(self) -> None:
        """Recalculate statistics from current collision history"""
        # Reset stats
        self.total_impulse = 0.0
        self.max_impulse = 0.0
        for key in self.direction_stats:
            self.direction_stats[key] = 0
            
        # Recalculate from current history
        for collision in self.collision_history:
            self.total_impulse += collision.impulse
            self.max_impulse = max(self.max_impulse, collision.impulse)
            self.direction_stats[collision.get_direction_description()] += 1
            
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collision statistics"""
        if not self.collision_history:
            return {
                "total_collisions": 0,
                "recent_collisions": 0,
                "direction_distribution": self.direction_stats.copy(),
                "average_impulse": 0.0,
                "max_impulse": 0.0,
                "collision_rate": 0.0,
                "total_impulse": 0.0
            }
            
        # Calculate statistics
        recent_collisions = len([c for c in self.collision_history 
                               if c.timestamp > self.current_simulation_time - 60.0])
        
        impulses = [c.impulse for c in self.collision_history]
        # Validate impulses to ensure realistic values
        valid_impulses = [i for i in impulses if 0 <= i <= COLLISION_MAX_FORCE]
        if valid_impulses:
            avg_impulse = sum(valid_impulses) / len(valid_impulses)
            max_impulse = max(valid_impulses)
        else:
            avg_impulse = 0.0
            max_impulse = 0.0
        
        # Collision rate (per minute)
        time_span = max(1.0, self.current_simulation_time)
        collision_rate = len(self.collision_history) / (time_span / 60.0)
        
        return {
            "total_collisions": len(self.collision_history),
            "recent_collisions": recent_collisions,
            "direction_distribution": self.direction_stats.copy(),
            "average_impulse": avg_impulse,
            "max_impulse": max_impulse,
            "collision_rate": collision_rate,
            "total_impulse": self.total_impulse
        }
        
    def has_recent_collision(self, max_age: float = 0.5) -> bool:
        """Check if there was a recent collision"""
        return self.get_recent_collision(max_age) is not None
        
    def reset(self) -> None:
        """Reset all collision data"""
        self.collision_history.clear()
        self.current_simulation_time = 0.0
        self.total_collisions = 0
        self.last_collision_time = -float('inf')
        
        # Reset per-car tracking
        self.per_car_last_collision_time.clear()
        self.collision_rate_limiter.clear()
        self.rate_limit_window_start.clear()
        
        # Reset statistics
        self.total_impulse = 0.0
        self.max_impulse = 0.0
        for key in self.direction_stats:
            self.direction_stats[key] = 0
            
    def get_debug_info(self) -> str:
        """Get debug information string"""
        recent = self.get_recent_collision()
        if recent:
            return (f"Recent collision: {recent.get_direction_description()} "
                   f"({recent.impulse:.0f}N⋅s)")
        return f"No recent collisions (Total: {self.total_collisions})"
        
    def __str__(self) -> str:
        """String representation of collision reporter state"""
        stats = self.get_collision_statistics()
        return (f"CollisionReporter: {stats['total_collisions']} total, "
                f"{stats['recent_collisions']} recent, "
                f"rate: {stats['collision_rate']:.1f}/min")