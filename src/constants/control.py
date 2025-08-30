# Car Control and Input Constants

import numpy as np
from .car_specs import CAR_MASS

# Define gravity
GRAVITY_MS2 = 9.81  # Gravitational acceleration (m/s²)

# Action Space Constants (for continuous control)
THROTTLE_MIN = 0.0  # minimum throttle input
THROTTLE_MAX = 1.0  # maximum throttle input
BRAKE_MIN = 0.0  # minimum brake input  
BRAKE_MAX = 1.0  # maximum brake input
STEERING_MIN = -1.0  # full left steering
STEERING_MAX = 1.0  # full right steering

# Combined throttle/brake axis constants
THROTTLE_BRAKE_MIN = -1.0  # full brake (negative values)
THROTTLE_BRAKE_MAX = 1.0   # full throttle (positive values)

# Control Response Constants
MAX_STEERING_ANGLE = 45.0  # degrees maximum wheel turn angle (increased for tighter turns)

# Car Action Space Constants (Continuous Control)
CAR_ACTION_SHAPE = (2,)  # [throttle_brake, steering] - merged throttle/brake axis
CAR_ACTION_LOW = np.array([THROTTLE_BRAKE_MIN, STEERING_MIN], dtype=np.float32)
CAR_ACTION_HIGH = np.array([THROTTLE_BRAKE_MAX, STEERING_MAX], dtype=np.float32)

# Legacy 3-element action shape for internal use
CAR_ACTION_SHAPE_INTERNAL = (3,)  # [throttle, brake, steering]
CAR_ACTION_LOW_INTERNAL = np.array([THROTTLE_MIN, BRAKE_MIN, STEERING_MIN], dtype=np.float32)
CAR_ACTION_HIGH_INTERNAL = np.array([THROTTLE_MAX, BRAKE_MAX, STEERING_MAX], dtype=np.float32)

# Force Application Constants
# RWD_GRIP_FACTOR removed - rely directly on tire physics model for grip calculation

# Friction Circle Constants (grip sharing between longitudinal and lateral forces)
FRICTION_CIRCLE_STEERING_REDUCTION_MAX = 0.4  # Maximum grip reduction when steering (40% reduction at full lock - racing tires)
FRICTION_CIRCLE_STEERING_FACTOR = 1.5  # Multiplier for steering angle effect on grip (reduced for racing tires)
FRICTION_CIRCLE_BRAKE_REDUCTION_MAX = 0.3  # Maximum brake force reduction when steering (30% reduction - racing tires)
FRICTION_CIRCLE_BRAKE_FACTOR = 1.5  # Multiplier for steering angle effect on braking

# Tyre load constants
MAX_TYRE_LOAD = CAR_MASS * GRAVITY_MS2 * 2.0  # maximum load on single tyre (2x static)
STATIC_LOAD_PER_TYRE = CAR_MASS * GRAVITY_MS2 / 4.0  # static load per tyre with equal distribution (N)

# Rolling resistance
from .car_specs import CAR_ROLLING_RESISTANCE_COEFFICIENT
ROLLING_RESISTANCE_FORCE = CAR_ROLLING_RESISTANCE_COEFFICIENT * CAR_MASS * GRAVITY_MS2  # total rolling resistance force (N)

# Braking Constants
MAX_BRAKE_DECELERATION_G = 14.0  # Maximum braking deceleration in m/s² (~1.4g)
BRAKE_FORCE_DISTRIBUTION_WHEELS = 4.0  # Number of wheels for force distribution
BRAKE_FRICTION_SPEED_THRESHOLD = 1.0  # Speed (m/s) below which brake friction for heating is reduced
BRAKE_FRICTION_MIN_SPEED_FACTOR = 0.05  # Minimum fraction of brake friction when stationary (for residual heating)

# Speed and Control Thresholds
MINIMUM_SPEED_FOR_DRAG = 0.1  # Minimum speed (m/s) to apply aerodynamic drag
MINIMUM_SPEED_FOR_BRAKE = 0.1  # Minimum speed (m/s) to apply braking force
MINIMUM_SPEED_FOR_STEERING = 0.1  # Minimum speed (m/s) to apply steering forces (reduced from 0.5 to help escape walls)
MINIMUM_THROTTLE_THRESHOLD = 0.01  # Minimum throttle to consider "on throttle"
MINIMUM_BRAKE_THRESHOLD = 0.01  # Minimum brake to consider "braking"
MINIMUM_STEERING_THRESHOLD = 0.01  # Minimum steering angle to consider "steering"

# Steering Force Constants
STEERING_TORQUE_MULTIPLIER = 0.8  # Steering torque multiplier factor (reduced for smoother turns)
STEERING_ANGULAR_DAMPING = 4.0  # Damping factor for angular velocity (increased to reduce over-rotation)
LATERAL_FORCE_SPEED_THRESHOLD = 0.05  # Minimum speed (m/s) for lateral tyre forces (reduced from 0.2 to enable steering when stuck)
MAX_LATERAL_FORCE = 30000.0  # Maximum lateral force from steering (N) - increased from 15000 to help escape walls
LATERAL_FORCE_SPEED_MULTIPLIER = 40.0  # Multiplier for speed-dependent lateral force (increased multiplier)
LATERAL_FORCE_STEERING_MULTIPLIER = 10.0  # Additional multiplier for steering force calculation (increased multiplier)
VELOCITY_ALIGNMENT_FORCE_FACTOR = 5.0  # Force multiplier for aligning velocity with car orientation (increased from 2.5 for stronger correction)

# Friction Force Constants (for tyre heating)
MAX_FRICTION_FORCE_CAP = 2000.0  # Maximum friction force per tyre (N)
REAR_WHEEL_COUNT = 2.0  # Number of rear wheels for RWD force distribution

# Acceleration Limits (realistic vehicle dynamics)
MAX_LONGITUDINAL_ACCELERATION = 12.0  # m/s² maximum acceleration/braking (0.8g - realistic road car limit)
MAX_LATERAL_ACCELERATION = 12.0  # m/s² maximum cornering acceleration (1.2g - original limit)
ACCELERATION_SANITY_CHECK_THRESHOLD = 14.0  # m/s² threshold for detecting unrealistic accelerations
ACCELERATION_SANITY_DAMPENING = 0.5  # Dampening factor applied when acceleration exceeds sanity threshold
ACCELERATION_HISTORY_SIZE = 10  # number of acceleration samples to average for smoothing
