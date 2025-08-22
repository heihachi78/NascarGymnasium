# Engine and Drivetrain Constants

import math

# Engine RPM Constants
ENGINE_IDLE_RPM = 1000.0  # RPM at idle
ENGINE_PEAK_TORQUE_RPM = 5500.0  # RPM at peak torque output
ENGINE_MAX_RPM = 9000.0  # Maximum RPM (redline)
ENGINE_MIN_RPM = 600.0  # Minimum stable RPM
ENGINE_MAX_RPM_LIMIT = 9500.0  # Hard RPM limit
ENGINE_REDLINE_RPM_RANGE = 800.0  # RPM range from idle to redline
ENGINE_RPM_RESPONSE_RATE = 3000.0  # RPM per second response rate
ENGINE_RPM_RESPONSE_EPSILON = 0.1  # Small value to prevent division by zero

# Torque Curve Constants
ENGINE_TORQUE_CURVE_LOW_FACTOR = 0.7  # Torque factor at idle RPM
ENGINE_TORQUE_CURVE_HIGH_FACTOR = 0.3  # Additional torque factor at peak RPM
ENGINE_TORQUE_CURVE_FALLOFF_FACTOR = 0.6  # Torque falloff factor after peak RPM

# Drivetrain Constants
FINAL_DRIVE_RATIO = 7.5  # Final drive ratio (gearbox + differential)
WHEEL_RADIUS = 0.35  # Wheel radius in meters
WHEEL_CIRCUMFERENCE_TO_RPM = 60.0  # Conversion factor for wheel RPM calculation

# Power Limiting Constants  
POWER_LIMIT_TRANSITION_START = 12.0  # Speed (m/s) where power limiting begins to blend (delayed transition)
POWER_LIMIT_TRANSITION_END = 25.0  # Speed (m/s) where power limiting fully active (much higher threshold)
POWER_LIMIT_TRANSITION_RANGE = POWER_LIMIT_TRANSITION_END - POWER_LIMIT_TRANSITION_START

# Smooth Transition Curve Constants
POWER_TRANSITION_CURVE_SHARPNESS = 2.0  # Exponential curve sharpness (gentler transition)
POWER_TRANSITION_MIN_BLEND = 0.05  # Minimum blend factor to avoid division issues
POWER_TRANSITION_MAX_BLEND = 0.75  # Allow more torque contribution at high speeds

# Control Response Constants
ENGINE_RESPONSE_TIME = 0.1  # seconds for engine to respond to throttle
BRAKE_RESPONSE_TIME = 0.05  # seconds for brakes to respond

# Pre-computed wheel circumference (eliminates trigonometric calculation)
WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS