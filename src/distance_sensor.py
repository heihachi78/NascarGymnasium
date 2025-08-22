"""
Distance sensor implementation for 8-directional boundary detection.

This module provides the DistanceSensor class which uses Box2D raycasting
to detect distances from the car center to track boundaries in 8 directions
relative to the car's orientation.
"""

import math
import numpy as np
import Box2D
from typing import Tuple
from .constants import (
    SENSOR_NUM_DIRECTIONS,
    SENSOR_MAX_DISTANCE,
    SENSOR_ANGLE_STEP
)


class DistanceSensorCallback(Box2D.b2RayCastCallback):
    """Ray cast callback for distance sensor detection"""
    
    def __init__(self):
        super().__init__()
        self.hit_distance = SENSOR_MAX_DISTANCE
        self.hit_point = None
        
    def ReportFixture(self, fixture, point, normal, fraction):
        """Called when raycast hits a fixture"""
        # Only consider track walls (ignore car body)
        if fixture.userData is None or fixture.userData.get('type') != 'track_wall':
            return 1  # Continue ray
            
        # Record the hit (normal parameter unused but required by Box2D callback)
        self.hit_distance = fraction * SENSOR_MAX_DISTANCE
        self.hit_point = point
        return fraction  # Stop ray at this point
        
    def reset(self):
        """Reset callback for next raycast"""
        self.hit_distance = SENSOR_MAX_DISTANCE
        self.hit_point = None


class DistanceSensor:
    """8-directional distance sensor for track boundary detection"""
    
    def __init__(self):
        """Initialize distance sensor"""
        self.callback = DistanceSensorCallback()
        
    def get_sensor_distances(self, world: Box2D.b2World, 
                           car_position: Tuple[float, float], 
                           car_angle: float) -> np.ndarray:
        """
        Get distances to track boundaries in 8 directions relative to car orientation.
        
        Args:
            world: Box2D world containing track walls
            car_position: Car center position (x, y) in meters
            car_angle: Car orientation in radians
            
        Returns:
            numpy array of 8 distances in meters, starting from car front clockwise
        """
        if world is None:
            # No track - return max distances
            return np.full(SENSOR_NUM_DIRECTIONS, SENSOR_MAX_DISTANCE, dtype=np.float32)
            
        distances = np.zeros(SENSOR_NUM_DIRECTIONS, dtype=np.float32)
        
        # Calculate sensor directions relative to car orientation
        for i in range(SENSOR_NUM_DIRECTIONS):
            # Sensor angle in degrees (0 = front, clockwise)
            sensor_angle_deg = i * SENSOR_ANGLE_STEP
            # Convert to radians and add car angle (negative for clockwise to counterclockwise)
            sensor_angle_rad = -math.radians(sensor_angle_deg) + car_angle
            
            # Calculate ray direction
            ray_direction = (
                math.cos(sensor_angle_rad),
                math.sin(sensor_angle_rad)
            )
            
            # Calculate ray end point
            ray_end = (
                car_position[0] + ray_direction[0] * SENSOR_MAX_DISTANCE,
                car_position[1] + ray_direction[1] * SENSOR_MAX_DISTANCE
            )
            
            # Perform raycast
            self.callback.reset()
            world.RayCast(self.callback, car_position, ray_end)
            
            distances[i] = self.callback.hit_distance
            
        return distances
    
    def get_sensor_angles(self, car_angle: float) -> np.ndarray:
        """
        Get absolute world angles for all 8 sensor directions.
        
        Args:
            car_angle: Car orientation in radians
            
        Returns:
            numpy array of 8 absolute angles in radians
        """
        angles = np.zeros(SENSOR_NUM_DIRECTIONS, dtype=np.float32)
        
        for i in range(SENSOR_NUM_DIRECTIONS):
            # Sensor angle in degrees (0 = front, clockwise)  
            sensor_angle_deg = i * SENSOR_ANGLE_STEP
            # Convert to radians and add car angle (negative for clockwise to counterclockwise)
            sensor_angle_rad = -math.radians(sensor_angle_deg) + car_angle
            # Normalize to [0, 2π]
            sensor_angle_rad = sensor_angle_rad % (2 * math.pi)
            if sensor_angle_rad < 0:
                sensor_angle_rad += 2 * math.pi
            angles[i] = sensor_angle_rad
            
        return angles
    
    def get_sensor_end_points(self, car_position: Tuple[float, float],
                            distances: np.ndarray, angles: np.ndarray) -> list:
        """
        Calculate sensor ray end points for visualization.
        
        Args:
            car_position: Car center position (x, y)
            distances: Array of sensor distances
            angles: Array of sensor angles in radians
            
        Returns:
            List of (x, y) end points for each sensor ray
        """
        end_points = []
        
        for i in range(SENSOR_NUM_DIRECTIONS):
            distance = distances[i]
            angle = angles[i]
            
            # Calculate end point
            end_x = car_position[0] + distance * math.cos(angle)
            end_y = car_position[1] + distance * math.sin(angle)
            end_points.append((end_x, end_y))
            
        return end_points