"""
Distance sensor implementation for multi-directional boundary detection.

This module provides the DistanceSensor class which uses Box2D raycasting
to detect distances from the car center to track boundaries in multiple directions
relative to the car's orientation (configurable via SENSOR_NUM_DIRECTIONS constant).
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
    """Ray cast callback for distance sensor detection.

    This class is used with Box2D's RayCast method to determine the distance
    to the nearest track wall.
    """
    
    def __init__(self):
        """Initializes the DistanceSensorCallback."""
        super().__init__()
        self.hit_distance = SENSOR_MAX_DISTANCE
        self.hit_point = None
        
    def ReportFixture(self, fixture, point, normal, fraction):
        """Called by Box2D when a raycast hits a fixture.

        Args:
            fixture: The fixture that was hit.
            point: The point of impact.
            normal: The normal vector of the surface at the point of impact.
            fraction: The fraction of the ray's length at which the impact occurred.

        Returns:
            The fraction of the ray's length to continue the raycast.
        """
        # Only consider track walls (ignore car body)
        if fixture.userData is None or fixture.userData.get('type') != 'track_wall':
            return 1  # Continue ray
            
        # Record the hit (normal parameter unused but required by Box2D callback)
        self.hit_distance = fraction * SENSOR_MAX_DISTANCE
        self.hit_point = point
        return fraction  # Stop ray at this point
        
    def reset(self):
        """Resets the callback for the next raycast."""
        self.hit_distance = SENSOR_MAX_DISTANCE
        self.hit_point = None


class DistanceSensor:
    """Multi-directional distance sensor for track boundary detection.

    This class uses Box2D raycasting to measure the distance to track walls
    in multiple directions relative to the car's orientation.
    """
    
    def __init__(self):
        """Initializes the DistanceSensor."""
        self.callback = DistanceSensorCallback()
        
    def get_sensor_distances(self, world: Box2D.b2World, 
                           car_position: Tuple[float, float], 
                           car_angle: float) -> np.ndarray:
        """
        Gets the distances to track boundaries in multiple directions.
        
        Args:
            world (Box2D.b2World): The Box2D world containing the track walls.
            car_position (Tuple[float, float]): The car's center position (x, y) in meters.
            car_angle (float): The car's orientation in radians.
            
        Returns:
            A numpy array of distances in meters, starting from the car's front and
            proceeding clockwise.
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
        Gets the absolute world angles for all sensor directions.
        
        Args:
            car_angle (float): The car's orientation in radians.
            
        Returns:
            A numpy array of absolute angles in radians.
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
        Calculates the end points of the sensor rays for visualization.
        
        Args:
            car_position (Tuple[float, float]): The car's center position (x, y).
            distances (np.ndarray): An array of sensor distances.
            angles (np.ndarray): An array of sensor angles in radians.
            
        Returns:
            A list of (x, y) end points for each sensor ray.
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