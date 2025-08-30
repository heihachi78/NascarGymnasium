import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from .constants import (
    CAR_ACTION_SHAPE,
    CAR_ACTION_LOW,
    CAR_ACTION_HIGH,
    CAR_ACTION_SHAPE_INTERNAL,
    CAR_OBSERVATION_SHAPE,
    CAR_OBSERVATION_LOW,
    CAR_OBSERVATION_HIGH,
    CAR_MASS,
    GRAVITY_MS2,
    TYRE_START_TEMPERATURE,
    INITIAL_ELAPSED_TIME,
    DEFAULT_RENDER_FPS,
    RENDER_MODE_HUMAN,
    DEFAULT_REWARD,
    DEFAULT_TERMINATED,
    DEFAULT_TRUNCATED,
    # Normalization constants
    NORM_MAX_TYRE_LOAD,
    NORM_MAX_TYRE_TEMP,
    # Sensor constants
    SENSOR_NUM_DIRECTIONS
)


class BaseEnv(gym.Env):
    """Base environment for car simulation with continuous or discrete action space"""
    metadata = {"render_modes": [RENDER_MODE_HUMAN], "render_fps": DEFAULT_RENDER_FPS}
    
    def __init__(self, discrete_action_space=False, num_cars=1):
        super().__init__()
        
        self.discrete_action_space = discrete_action_space
        self.num_cars = num_cars
        
        if discrete_action_space:
            # Discrete action space: 5 actions per car
            if num_cars == 1:
                # Single-car: scalar discrete action
                self.action_space = spaces.Discrete(5)
            else:
                # Multi-car: array of discrete actions
                self.action_space = spaces.MultiDiscrete([5] * num_cars)
        else:
            # Continuous action space: [throttle_brake, steering] per car
            if num_cars == 1:
                # Single-car: scalar continuous action
                self.action_space = spaces.Box(
                    low=CAR_ACTION_LOW,
                    high=CAR_ACTION_HIGH,
                    shape=(2,),
                    dtype=np.float32
                )
            else:
                # Multi-car: array of continuous actions
                self.action_space = spaces.Box(
                    low=np.tile(CAR_ACTION_LOW, (num_cars, 1)),
                    high=np.tile(CAR_ACTION_HIGH, (num_cars, 1)),
                    shape=(num_cars, 2),
                    dtype=np.float32
                )
        
        # Observation space
        if num_cars == 1:
            # Single-car: scalar observation
            self.observation_space = spaces.Box(
                low=CAR_OBSERVATION_LOW,
                high=CAR_OBSERVATION_HIGH,
                shape=(CAR_OBSERVATION_SHAPE[0],),
                dtype=np.float32
            )
        else:
            # Multi-car: array of observations
            self.observation_space = spaces.Box(
                low=np.tile(CAR_OBSERVATION_LOW, (num_cars, 1)),
                high=np.tile(CAR_OBSERVATION_HIGH, (num_cars, 1)),
                shape=(num_cars, CAR_OBSERVATION_SHAPE[0]),
                dtype=np.float32
            )
        
        self.start_time = None
        self.elapsed_time = INITIAL_ELAPSED_TIME
        self.last_action = np.zeros(CAR_ACTION_SHAPE_INTERNAL, dtype=np.float32)  # Internal 3-element format
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.start_time = time.time()
        self.elapsed_time = INITIAL_ELAPSED_TIME
        self.last_action = np.zeros(CAR_ACTION_SHAPE_INTERNAL, dtype=np.float32)  # Internal 3-element format
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Validate action
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Convert discrete action to continuous if needed
        if self.discrete_action_space:
            # Convert discrete to 2-element continuous, then to 3-element internal
            continuous_action_2d = self._discrete_to_continuous(action)
            internal_action = self._convert_to_internal_action(continuous_action_2d)
            self.last_action = np.array(internal_action, dtype=np.float32)
        else:
            # Convert 2-element action to internal 3-element format
            internal_action = self._convert_to_internal_action(action)
            self.last_action = np.array(internal_action, dtype=np.float32)
        
        current_time = time.time()
        self.elapsed_time = current_time - self.start_time
        
        observation = self._get_obs()
        reward = DEFAULT_REWARD
        terminated = DEFAULT_TERMINATED
        truncated = DEFAULT_TRUNCATED
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        # Normalized car state observation vector
        # All values are normalized to [-1, 1] or [0, 1] ranges
        # [pos_x, pos_y, vel_x, vel_y, speed_magnitude, orientation, angular_vel,
        #  tyre_load_fl, tyre_load_fr, tyre_load_rl, tyre_load_rr,
        #  tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, 
        #  tyre_wear_fl, tyre_wear_fr, tyre_wear_rl, tyre_wear_rr,
        #  collision_impulse, collision_angle_relative, cumulative_impact_percentage,
        #  sensor_dist_0, sensor_dist_1, ..., sensor_dist_15]
        
        # Initialize with normalized default values (will be replaced by actual car physics)
        static_load = CAR_MASS * GRAVITY_MS2 / 4.0  # Equal weight distribution
        normalized_static_load = static_load / NORM_MAX_TYRE_LOAD
        normalized_start_temp = TYRE_START_TEMPERATURE / NORM_MAX_TYRE_TEMP
        
        observation = np.array([
            0.0, 0.0,  # car position (x, y) - normalized to [-1, 1], at origin initially
            0.0, 0.0,  # car velocity (vx, vy) - normalized to [-1, 1], stationary initially
            0.0,       # speed magnitude - normalized to [0, 1], stationary initially 
            0.0, 0.0,  # orientation, angular velocity - normalized to [-1, 1], facing forward, not rotating
            normalized_static_load, normalized_static_load, 
            normalized_static_load, normalized_static_load,  # normalized equal tyre loads
            normalized_start_temp, normalized_start_temp, 
            normalized_start_temp, normalized_start_temp,  # normalized start temperatures
            0.0, 0.0, 0.0, 0.0,  # no tyre wear initially (normalized to [0, 1])
            0.0, 0.0,   # no collision initially (normalized)
            0.0,        # no cumulative impact initially (normalized to [0, 1])
            *[1.0] * SENSOR_NUM_DIRECTIONS  # normalized sensor distances (1.0 = max range)
        ], dtype=np.float32)
        
        return observation
    
    def _convert_to_internal_action(self, action):
        """Convert 2-element action to internal 3-element format.
        
        Args:
            action: 2-element action [throttle_brake, steering]
                    throttle_brake: -1.0 (full brake) to 1.0 (full throttle)
                    steering: -1.0 (left) to 1.0 (right)
            
        Returns:
            3-element action [throttle, brake, steering]
        """
        throttle_brake = action[0]
        steering = action[1]
        
        # Convert combined throttle/brake to separate values
        if throttle_brake >= 0:
            # Positive values map to throttle
            throttle = throttle_brake
            brake = 0.0
        else:
            # Negative values map to brake
            throttle = 0.0
            brake = -throttle_brake  # Convert negative to positive brake value
        
        return [throttle, brake, steering]
    
    def _discrete_to_continuous(self, action):
        """Convert discrete action to 2-element continuous action values.
        
        Args:
            action: Discrete action (0-4)
            
        Returns:
            2-element continuous action [throttle_brake, steering]
        """
        if action == 0:
            # Do nothing
            return [0.0, 0.0]
        elif action == 1:
            # Accelerate
            return [1.0, 0.0]  # full throttle, no steering
        elif action == 2:
            # Brake
            return [-1.0, 0.0]  # full brake, no steering
        elif action == 3:
            # Turn left
            return [0.0, -1.0]  # no throttle/brake, full left
        elif action == 4:
            # Turn right
            return [0.0, 1.0]   # no throttle/brake, full right
        else:
            raise ValueError(f"Invalid discrete action: {action}")
    
    def _get_info(self):
        return {
            "elapsed_time": self.elapsed_time,
            "last_action": self.last_action.tolist(),
            "throttle": float(self.last_action[0]),
            "brake": float(self.last_action[1]),
            "steering": float(self.last_action[2])
        }