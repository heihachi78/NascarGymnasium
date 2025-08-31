"""
Observation validation utilities for reinforcement learning environment.

This module provides utilities to validate observation vectors, check bounds,
and ensure consistency between observation space definitions and actual data.
"""

import numpy as np
from typing import Dict, Any
from .constants import (
    CAR_OBSERVATION_SHAPE,
    CAR_OBSERVATION_LOW,
    CAR_OBSERVATION_HIGH,
    SENSOR_NUM_DIRECTIONS,
    NORM_MAX_POSITION,
    NORM_MAX_VELOCITY,
    NORM_MAX_ANGULAR_VEL,
    NORM_MAX_TYRE_LOAD,
    NORM_MAX_TYRE_TEMP,
    NORM_MAX_TYRE_WEAR,
    INSTANT_DISABLE_IMPACT_THRESHOLD,
    CUMULATIVE_DISABLE_IMPACT_THRESHOLD,
    SENSOR_MAX_DISTANCE
)



class ObservationValidator:
    """Validates observation vectors for consistency and bounds.

    This class provides methods to validate the shape, bounds, and logical
    consistency of observation vectors from the car environment.
    """
    
    # Expected observation structure (22 base elements + sensors)
    OBSERVATION_STRUCTURE = {
        'pos_x': (0, 'position'),
        'pos_y': (1, 'position'), 
        'vel_x': (2, 'velocity'),
        'vel_y': (3, 'velocity'),
        'speed_magnitude': (4, 'speed'),
        'orientation': (5, 'orientation'),
        'angular_vel': (6, 'angular_velocity'),
        'tyre_load_fl': (7, 'tyre_load'),
        'tyre_load_fr': (8, 'tyre_load'),
        'tyre_load_rl': (9, 'tyre_load'),
        'tyre_load_rr': (10, 'tyre_load'),
        'tyre_temp_fl': (11, 'tyre_temp'),
        'tyre_temp_fr': (12, 'tyre_temp'),
        'tyre_temp_rl': (13, 'tyre_temp'),
        'tyre_temp_rr': (14, 'tyre_temp'),
        'tyre_wear_fl': (15, 'tyre_wear'),
        'tyre_wear_fr': (16, 'tyre_wear'),
        'tyre_wear_rl': (17, 'tyre_wear'),
        'tyre_wear_rr': (18, 'tyre_wear'),
        'collision_impulse': (19, 'collision_impulse'),
        'collision_angle': (20, 'collision_angle'),
        'cumulative_impact': (21, 'cumulative_impact'),
        # Sensor distances start at index 22
        'sensors': (22, 'sensors')
    }
    
    def __init__(self):
        """Initializes the ObservationValidator."""
        self.validation_errors = []
        self.warnings = []
        
    def validate_observation_shape(self, observation: np.ndarray) -> bool:
        """
        Validates that the observation has the correct shape.
        
        Args:
            observation (np.ndarray): The observation vector to validate.
            
        Returns:
            bool: True if the shape is correct, False otherwise.
        """
        expected_shape = CAR_OBSERVATION_SHAPE[0]
        actual_shape = observation.shape
        
        if len(actual_shape) != 1:
            self.validation_errors.append(f"Expected 1D array, got shape {actual_shape}")
            return False
            
        if actual_shape[0] != expected_shape:
            self.validation_errors.append(
                f"Expected {expected_shape} elements, got {actual_shape[0]}"
            )
            return False
            
        return True
    
    def validate_observation_bounds(self, observation: np.ndarray) -> bool:
        """
        Validates that all observation elements are within the expected bounds.
        
        Args:
            observation (np.ndarray): The observation vector to validate.
            
        Returns:
            bool: True if all elements are within bounds, False otherwise.
        """
        if not self.validate_observation_shape(observation):
            return False
            
        bounds_valid = True
        
        for i, (value, low_bound, high_bound) in enumerate(
            zip(observation, CAR_OBSERVATION_LOW, CAR_OBSERVATION_HIGH)
        ):
            if value < low_bound or value > high_bound:
                element_name = self._get_element_name(i)
                self.validation_errors.append(
                    f"Element {i} ({element_name}): {value} not in bounds [{low_bound}, {high_bound}]"
                )
                bounds_valid = False
                
        return bounds_valid
    
    def validate_observation_consistency(self, observation: np.ndarray) -> bool:
        """
        Validates the logical consistency within the observation vector.
        
        Args:
            observation (np.ndarray): The observation vector to validate.
            
        Returns:
            bool: True if the observation is logically consistent, False otherwise.
        """
        if not self.validate_observation_shape(observation):
            return False
            
        consistency_valid = True
        
        # Check speed magnitude consistency with velocity components
        vel_x = observation[self.OBSERVATION_STRUCTURE['vel_x'][0]]
        vel_y = observation[self.OBSERVATION_STRUCTURE['vel_y'][0]]
        speed_magnitude = observation[self.OBSERVATION_STRUCTURE['speed_magnitude'][0]]
        
        # Convert normalized values back to m/s for comparison
        actual_vel_x = vel_x * NORM_MAX_VELOCITY
        actual_vel_y = vel_y * NORM_MAX_VELOCITY
        actual_speed = speed_magnitude * NORM_MAX_VELOCITY
        expected_speed = np.sqrt(actual_vel_x**2 + actual_vel_y**2)
        
        if abs(actual_speed - expected_speed) > 0.1:  # 0.1 m/s tolerance
            self.validation_errors.append(
                f"Speed inconsistency: magnitude {actual_speed:.3f} vs calculated {expected_speed:.3f}"
            )
            consistency_valid = False
            
        # Check tyre loads are positive (normalized, so should be >= 0)
        for i in range(7, 11):  # Tyre load indices
            if observation[i] < 0:
                tyre_name = self._get_element_name(i)
                self.validation_errors.append(f"{tyre_name} is negative: {observation[i]}")
                consistency_valid = False
                
        # Check sensor distances are in valid range
        sensor_start_idx = self.OBSERVATION_STRUCTURE['sensors'][0]
        for i in range(sensor_start_idx, sensor_start_idx + SENSOR_NUM_DIRECTIONS):
            if i < len(observation) and (observation[i] < 0 or observation[i] > 1):
                self.validation_errors.append(
                    f"Sensor {i - sensor_start_idx} distance out of range [0,1]: {observation[i]}"
                )
                consistency_valid = False
                
        return consistency_valid
    
    def validate_multi_car_observations(self, observations: np.ndarray, num_cars: int) -> bool:
        """
        Validates a multi-car observation array.
        
        Args:
            observations (np.ndarray): The multi-car observation array.
            num_cars (int): The expected number of cars.
            
        Returns:
            bool: True if all observations are valid, False otherwise.
        """
        if observations.shape[0] != num_cars:
            self.validation_errors.append(
                f"Expected {num_cars} car observations, got {observations.shape[0]}"
            )
            return False
            
        all_valid = True
        for car_idx in range(num_cars):
            car_obs = observations[car_idx]
            if not self.validate_full_observation(car_obs):
                self.validation_errors.append(f"Car {car_idx} observation invalid")
                all_valid = False
                
        return all_valid
    
    def validate_full_observation(self, observation: np.ndarray) -> bool:
        """
        Runs the full validation suite on an observation.
        
        Args:
            observation (np.ndarray): The observation vector to validate.
            
        Returns:
            bool: True if the observation passes all checks, False otherwise.
        """
        self.reset_validation_state()
        
        shape_valid = self.validate_observation_shape(observation)
        bounds_valid = self.validate_observation_bounds(observation)
        consistency_valid = self.validate_observation_consistency(observation)
        
        return shape_valid and bounds_valid and consistency_valid
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Gets a detailed validation report.
        
        Returns:
            A dictionary containing validation errors and warnings.
        """
        return {
            'errors': self.validation_errors.copy(),
            'warnings': self.warnings.copy(),
            'has_errors': len(self.validation_errors) > 0,
            'has_warnings': len(self.warnings) > 0
        }
    
    def reset_validation_state(self) -> None:
        """Resets the validation error and warning lists."""
        self.validation_errors.clear()
        self.warnings.clear()
    
    def _get_element_name(self, index: int) -> str:
        """Gets a human-readable name for an observation element at a given index."""
        for name, (idx, _) in self.OBSERVATION_STRUCTURE.items():
            if idx == index:
                return name
            elif name == 'sensors' and index >= idx:
                sensor_num = index - idx
                if sensor_num < SENSOR_NUM_DIRECTIONS:
                    return f"sensor_{sensor_num}"
                    
        return f"unknown_element_{index}"
    
    @staticmethod
    def create_observation_summary(observation: np.ndarray) -> Dict[str, Any]:
        """
        Creates a human-readable summary of observation values.
        
        Args:
            observation (np.ndarray): The observation vector to summarize.
            
        Returns:
            A dictionary with categorized observation values.
        """
        if len(observation) < 22:
            return {'error': 'Observation too short for summary'}
            
        summary = {
            'position': {
                'x': observation[0] * NORM_MAX_POSITION,
                'y': observation[1] * NORM_MAX_POSITION
            },
            'velocity': {
                'x': observation[2] * NORM_MAX_VELOCITY,
                'y': observation[3] * NORM_MAX_VELOCITY,
                'magnitude': observation[4] * NORM_MAX_VELOCITY
            },
            'orientation': {
                'angle_rad': observation[5] * np.pi,
                'angular_vel': observation[6] * NORM_MAX_ANGULAR_VEL
            },
            'tyres': {
                'loads': [observation[i] * NORM_MAX_TYRE_LOAD for i in range(7, 11)],
                'temperatures': [observation[i] * NORM_MAX_TYRE_TEMP for i in range(11, 15)],
                'wear': [observation[i] * NORM_MAX_TYRE_WEAR for i in range(15, 19)]
            },
            'collision': {
                'impulse': observation[19] * INSTANT_DISABLE_IMPACT_THRESHOLD,
                'angle': observation[20] * np.pi,
                'cumulative_impact': observation[21] * CUMULATIVE_DISABLE_IMPACT_THRESHOLD
            }
        }
        
        # Add sensor distances if available
        if len(observation) >= 22 + SENSOR_NUM_DIRECTIONS:
            sensor_distances = []
            for i in range(22, 22 + SENSOR_NUM_DIRECTIONS):
                sensor_distances.append(observation[i] * SENSOR_MAX_DISTANCE)
            summary['sensors'] = {
                'distances': sensor_distances,
                'count': SENSOR_NUM_DIRECTIONS
            }
            
        return summary


def validate_environment_observations(env, num_steps: int = 10) -> Dict[str, Any]:
    """
    Runs validation tests on the environment's observations.
    
    Args:
        env: The environment instance to test.
        num_steps (int): The number of environment steps to test.
        
    Returns:
        A dictionary containing the validation report.
    """
    validator = ObservationValidator()
    report = {
        'total_steps': 0,
        'valid_steps': 0,
        'errors_by_step': [],
        'summary': {}
    }
    
    try:
        # Reset environment and validate initial observation
        obs, _ = env.reset()
        
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:  # Single car
                is_valid = validator.validate_full_observation(obs)
            else:  # Multi-car
                is_valid = validator.validate_multi_car_observations(obs, env.num_cars)
                
            if is_valid:
                report['valid_steps'] += 1
            else:
                report['errors_by_step'].append({
                    'step': 0,
                    'type': 'reset',
                    'report': validator.get_validation_report()
                })
        
        report['total_steps'] += 1
        
        # Test random actions for num_steps
        for step in range(1, num_steps + 1):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            
            validator.reset_validation_state()
            if isinstance(obs, np.ndarray):
                if obs.ndim == 1:  # Single car
                    is_valid = validator.validate_full_observation(obs)
                else:  # Multi-car
                    is_valid = validator.validate_multi_car_observations(obs, env.num_cars)
                    
                if is_valid:
                    report['valid_steps'] += 1
                else:
                    report['errors_by_step'].append({
                        'step': step,
                        'type': 'step',
                        'report': validator.get_validation_report()
                    })
            
            report['total_steps'] += 1
            
            if terminated or truncated:
                break
                
    except Exception as e:
        report['validation_exception'] = str(e)
        
    # Create summary
    report['summary'] = {
        'success_rate': report['valid_steps'] / report['total_steps'] if report['total_steps'] > 0 else 0,
        'total_errors': len(report['errors_by_step']),
        'validation_passed': len(report['errors_by_step']) == 0
    }
    
    return report