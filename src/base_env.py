import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from .constants import (
    CAR_ACTION_SHAPE,
    CAR_ACTION_LOW,
    CAR_ACTION_HIGH,
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
    NORM_MAX_TYRE_TEMP
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
            # 0: do nothing, 1: accelerate, 2: brake, 3: turn left, 4: turn right
            if num_cars == 1:
                self.action_space = spaces.Discrete(5)
            else:
                # Multi-agent discrete: array of discrete actions
                self.action_space = spaces.MultiDiscrete([5] * num_cars)
        else:
            # Continuous action space: [throttle, brake, steering] per car
            if num_cars == 1:
                self.action_space = spaces.Box(
                    low=CAR_ACTION_LOW,
                    high=CAR_ACTION_HIGH,
                    shape=CAR_ACTION_SHAPE,
                    dtype=np.float32
                )
            else:
                # Multi-agent continuous: (num_cars, 3) shape
                self.action_space = spaces.Box(
                    low=np.tile(CAR_ACTION_LOW, (num_cars, 1)),
                    high=np.tile(CAR_ACTION_HIGH, (num_cars, 1)),
                    shape=(num_cars, 3),
                    dtype=np.float32
                )
        
        # Observation space: single car or multi-car
        if num_cars == 1:
            self.observation_space = spaces.Box(
                low=CAR_OBSERVATION_LOW,
                high=CAR_OBSERVATION_HIGH,
                shape=CAR_OBSERVATION_SHAPE,
                dtype=np.float32
            )
        else:
            # Multi-agent observations: (num_cars, observation_dim) shape
            self.observation_space = spaces.Box(
                low=np.tile(CAR_OBSERVATION_LOW, (num_cars, 1)),
                high=np.tile(CAR_OBSERVATION_HIGH, (num_cars, 1)),
                shape=(num_cars, CAR_OBSERVATION_SHAPE[0]),
                dtype=np.float32
            )
        
        self.start_time = None
        self.elapsed_time = INITIAL_ELAPSED_TIME
        self.last_action = np.zeros(CAR_ACTION_SHAPE, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.start_time = time.time()
        self.elapsed_time = INITIAL_ELAPSED_TIME
        self.last_action = np.zeros(CAR_ACTION_SHAPE, dtype=np.float32)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Validate action
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Convert discrete action to continuous if needed
        if self.discrete_action_space:
            continuous_action = self._discrete_to_continuous(action)
            self.last_action = np.array(continuous_action, dtype=np.float32)
        else:
            self.last_action = np.array(action, dtype=np.float32)
        
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
        #  collision_impulse, collision_angle_relative,
        #  sensor_dist_0, sensor_dist_1, ..., sensor_dist_7]
        
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
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # normalized sensor distances (1.0 = max range)
        ], dtype=np.float32)
        
        return observation
    
    def _discrete_to_continuous(self, action):
        """Convert discrete action to continuous action values.
        
        Args:
            action: Discrete action (0-4)
            
        Returns:
            Continuous action [throttle, brake, steering]
        """
        if action == 0:
            # Do nothing
            return [0.0, 0.0, 0.0]
        elif action == 1:
            # Accelerate
            return [1.0, 0.0, 0.0]
        elif action == 2:
            # Brake
            return [0.0, 1.0, 0.0]
        elif action == 3:
            # Turn left
            return [0.0, 0.0, -1.0]
        elif action == 4:
            # Turn right
            return [0.0, 0.0, 1.0]
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