"""
Base controller class that provides common functionality for all controllers.

This module provides a base class with fallback control logic that can be
shared by TD3, PPO, and other controller implementations.
"""

import numpy as np


class BaseController:
    """
    Base controller class with common fallback control logic.
    
    This class provides:
    - Per-instance control state management
    - Fallback control logic using the default rule-based algorithm
    - Common initialization for derived controllers
    """
    
    def __init__(self, name=None):
        """
        Initialize the base controller.
        
        Args:
            name: Optional name for this controller (for logging)
        """
        self.name = name or "BaseController"
        self.use_fallback = False
        
        # Per-instance control state (avoids global state issues)
        self.control_state = {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0
        }
    
    def _fallback_control(self, observation):
        """
        Rule-based fallback control logic.
        
        This is the fallback control algorithm used when
        the model is not available or fails. It maintains
        per-instance state for multi-car scenarios.
        
        Args:
            observation: numpy array of shape (38,) containing:
                - Position (x, y): indices 0-1
                - Velocity (x, y, magnitude): indices 2-4
                - Orientation and angular velocity: indices 5-6
                - Tire loads (4): indices 7-10
                - Tire temperatures (4): indices 11-14
                - Tire wear (4): indices 15-18
                - Collision data (impulse, angle): indices 19-20
                - Cumulative impact percentage: index 21
                - Distance sensor readings (16 directions): indices 22-37
        
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
            - throttle_brake: -1.0 (full brake) to 1.0 (full throttle)
            - steering: -1.0 to 1.0 (left/right)
        """
        # Extract sensor data (16 sensors from index 22-37)
        sensors = observation[22:38]   # All 16 sensor distances
        forward = sensors[0]           # Forward sensor (0°) - used for speed control
        current_speed = observation[4] # Speed from observation
        
        # Calculate speed limit based on forward distance
        speed_limit = forward * 500 / 3.6
        
        # Throttle control - accumulate changes
        if current_speed * 200 < speed_limit:
            self.control_state['throttle'] += 0.1
        if current_speed * 200 > speed_limit:
            self.control_state['throttle'] -= 0.1
        
        # Brake control - accumulate changes
        if current_speed * 200 < speed_limit:
            self.control_state['brake'] -= 0.01
        if current_speed * 200 > speed_limit:
            self.control_state['brake'] += 0.01
        
        # Steering control based on sensor readings (improved with 16 sensors)
        # Compare left vs right sensor groups for better steering decisions
        right_sensors = (sensors[1] + sensors[2] + sensors[3] + sensors[4]) / 4  # 22.5° to 90°
        left_sensors = (sensors[13] + sensors[14] + sensors[15] + sensors[12]) / 4  # 270° to 337.5°
        
        if right_sensors > left_sensors:
            self.control_state['steering'] = min(0.5, (right_sensors - left_sensors) * 2.0)
        elif left_sensors > right_sensors:
            self.control_state['steering'] = max(-0.5, -(left_sensors - right_sensors) * 2.0)
        else:
            self.control_state['steering'] *= 0.9  # Gradual return to center
        
        # Apply limits and adjustments
        self.control_state['brake'] = max(min(self.control_state['brake'], 1), 0)
        self.control_state['steering'] = max(min(self.control_state['steering'], 1), -1)
        
        # Reduce throttle when steering hard
        if abs(self.control_state['steering']) > 0.1:
            self.control_state['throttle'] -= 0.05
        
        self.control_state['throttle'] = max(min(self.control_state['throttle'], 1), 0)
        
        # Convert separate throttle/brake to combined throttle_brake axis
        throttle_brake = self.control_state['throttle'] - self.control_state['brake']
        
        return np.array([
            throttle_brake,
            self.control_state['steering']
        ], dtype=np.float32)
    
    def control(self, observation):
        """
        Default control method that uses fallback control.
        
        Derived classes should override this method to implement
        their specific control logic and fall back to this when needed.
        
        Args:
            observation: numpy array of shape (38,) containing car state
        
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
        """
        return self._fallback_control(observation)
    
    def get_info(self):
        """
        Get information about this controller.
        
        Derived classes should override this to add model-specific info.
        
        Returns:
            dict: Controller information
        """
        return {
            'name': self.name,
            'using_fallback': self.use_fallback
        }