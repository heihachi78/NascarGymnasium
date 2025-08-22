# Sensors and Observations Constants

import numpy as np
from .control import MAX_TYRE_LOAD
from .tyre import TYRE_MAX_WEAR
from .collision import COLLISION_MAX_FORCE

# Distance Sensor Constants
SENSOR_NUM_DIRECTIONS = 8  # Number of sensor directions around the car
SENSOR_MAX_DISTANCE = 250.0  # Maximum sensor range in meters
SENSOR_ANGLE_STEP = 45.0  # Degrees between sensors (360/8)

# Car Observation Space Constants (Comprehensive Car State)
# Observation vector: [pos_x, pos_y, vel_x, vel_y, speed_magnitude, orientation, angular_vel, 
#                      tyre_load_fl, tyre_load_fr, tyre_load_rl, tyre_load_rr,
#                      tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr,
#                      tyre_wear_fl, tyre_wear_fr, tyre_wear_rl, tyre_wear_rr,
#                      collision_impulse, collision_angle_relative,
#                      sensor_dist_0, sensor_dist_1, ..., sensor_dist_7]
CAR_OBSERVATION_SHAPE = (29,)  # 29 total observation elements (21 + 8 sensor distances)

# Observation bounds
MAX_POSITION_VALUE = 10000.0  # maximum world position coordinate
MAX_VELOCITY_VALUE = 200.0    # maximum velocity (m/s, well above max car speed)  
MAX_ANGULAR_VELOCITY = 10.0   # maximum angular velocity (rad/s)
MAX_COLLISION_FORCE = 100000.0  # maximum collision force (N)

# Normalization factors for observation space
# These values are used to normalize observations to [-1, 1] or [0, 1] ranges
NORM_MAX_POSITION = MAX_POSITION_VALUE  # Use same as max position
NORM_MAX_VELOCITY = MAX_VELOCITY_VALUE  # Use same as max velocity
NORM_MAX_ANGULAR_VEL = MAX_ANGULAR_VELOCITY  # Use same as max angular velocity
NORM_MAX_TYRE_TEMP = 200.0  # Maximum realistic tyre temperature in Celsius
NORM_MAX_TYRE_WEAR = TYRE_MAX_WEAR  # 100.0 - maximum wear percentage
NORM_MAX_COLLISION_IMPULSE = MAX_COLLISION_FORCE  # Use same as max collision force
NORM_MAX_TYRE_LOAD = MAX_TYRE_LOAD  # Use same as max tyre load

# Normalized Observation Space Arrays
# All values are normalized to [-1, 1] or [0, 1] ranges for better neural network training
CAR_OBSERVATION_LOW = np.array([
    -1.0, -1.0,         # pos_x, pos_y (normalized to [-1, 1])
    -1.0, -1.0,         # vel_x, vel_y (normalized to [-1, 1])
    0.0,                # speed_magnitude (normalized to [0, 1])
    -1.0, -1.0,         # orientation, angular_vel (normalized to [-1, 1])
    0.0, 0.0, 0.0, 0.0, # tyre loads (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0, # tyre temperatures (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0, # tyre wear (normalized to [0, 1])
    0.0, -1.0,          # collision impulse (normalized to [0, 1]), angle (normalized to [-1, 1])
    0.0, 0.0, 0.0, 0.0, # sensor distances (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0  # sensor distances (normalized to [0, 1])
], dtype=np.float32)

CAR_OBSERVATION_HIGH = np.array([
    1.0, 1.0,           # pos_x, pos_y (normalized to [-1, 1])
    1.0, 1.0,           # vel_x, vel_y (normalized to [-1, 1])
    1.0,                # speed_magnitude (normalized to [0, 1])
    1.0, 1.0,           # orientation, angular_vel (normalized to [-1, 1])
    1.0, 1.0, 1.0, 1.0, # tyre loads (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0, # tyre temperatures (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0, # tyre wear (normalized to [0, 1])
    1.0, 1.0,           # collision impulse (normalized to [0, 1]), angle (normalized to [-1, 1])
    1.0, 1.0, 1.0, 1.0, # sensor distances (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0  # sensor distances (normalized to [0, 1])
], dtype=np.float32)