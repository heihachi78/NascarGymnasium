# Car Specifications and Properties Constants

import math

# Car Specifications
CAR_MASS = 1500.0  # kg
CAR_HORSEPOWER = 670.0  # hp
CAR_MAX_TORQUE = 820.0  # Nm (increased to maintain performance after smooth transition fix)
CAR_MAX_SPEED_MPH = 200.0  # mph
CAR_ACCELERATION_0_100_KMH = 3.0  # seconds
CAR_WEIGHT_DISTRIBUTION_FRONT = 0.50  # 50% front
CAR_WEIGHT_DISTRIBUTION_REAR = 0.50  # 50% rear
CAR_DRAG_COEFFICIENT = 0.38  # dimensionless (high-performance aerodynamics)
CAR_DRIVE_TYPE = "RWD"  # Rear Wheel Drive

# Car Dimensions (in meters, converted from mm)
CAR_LENGTH = 5.042  # meters (5042 mm)
CAR_WIDTH = 1.996  # meters (1996 mm)
CAR_WHEELBASE = 2.794  # meters (2794 mm)

# Physics Conversion Factors
HP_TO_WATTS = 745.7  # 1 HP = 745.7 Watts
MPH_TO_MS = 0.44704  # 1 MPH = 0.44704 m/s
KMH_TO_MS = 0.277778  # 1 km/h = 0.277778 m/s
MS_TO_KMH = 3.6  # 1 m/s = 3.6 km/h

# Derived Car Constants
CAR_MAX_POWER = CAR_HORSEPOWER * HP_TO_WATTS  # Watts
CAR_MAX_SPEED_MS = CAR_MAX_SPEED_MPH * MPH_TO_MS  # m/s
CAR_TARGET_100KMH_MS = 100.0 * KMH_TO_MS  # 27.78 m/s (100 km/h in m/s)

# Car Physics Properties
CAR_FRONTAL_AREA = 2.5  # m² (estimated frontal area for drag calculation)
AIR_DENSITY = 1.225  # kg/m³ (air density at sea level)
CAR_ROLLING_RESISTANCE_COEFFICIENT = 0.015  # typical for racing tires on asphalt

# Car Box2D Physics Properties  
CAR_DENSITY = CAR_MASS / (CAR_LENGTH * CAR_WIDTH)  # kg/m²
CAR_FRICTION = 0.7  # friction coefficient with track
CAR_RESTITUTION = 0.1  # low restitution for realistic collisions
CAR_MOMENT_OF_INERTIA_FACTOR = 0.5  # factor for calculating rotational inertia (increased for realistic car body rotation)

# Car rendering constants
CAR_COLOR = (255, 0, 0)  # Red color for car (legacy single-car)
CAR_OUTLINE_COLOR = (0, 0, 0)  # Black outline for car
CAR_VISUAL_LENGTH = 5.0  # Visual length in meters (same as CAR_LENGTH)
CAR_VISUAL_WIDTH = 2.0   # Visual width in meters (same as CAR_WIDTH)

# Multi-car constants
MAX_CARS = 10  # Maximum number of cars allowed
MULTI_CAR_COLORS = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue  
    (0, 255, 0),    # Green
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 192, 203), # Pink
    (128, 128, 128) # Gray
]

# Pre-computed moment of inertia (eliminates complex calculation)
CAR_MOMENT_OF_INERTIA = CAR_MASS * (CAR_LENGTH**2 + CAR_WIDTH**2) * CAR_MOMENT_OF_INERTIA_FACTOR / 12.0

# Aerodynamic downforce constants for high-speed physics
AERODYNAMIC_DOWNFORCE_COEFFICIENT = 0.12  # downforce coefficient for high speeds (realistic sports car)
AERODYNAMIC_DOWNFORCE_REAR_BIAS = 0.6  # 60% of downforce goes to rear axle (more balanced)
AERODYNAMIC_DOWNFORCE_SPEED_THRESHOLD = 50.0  # m/s - speed above which downforce becomes significant
MAX_DOWNFORCE_MULTIPLIER = 1.5  # Maximum downforce as multiple of car weight (realistic limit)

# Weight Transfer Physics
CAR_CENTER_OF_GRAVITY_HEIGHT = 0.35  # meters above ground (realistic for sports car, balanced between 0.15m and 0.50m)
TRACK_WIDTH = CAR_WIDTH * 0.8  # effective track width for weight transfer calculation

# Effective Weight Transfer Factors (combines physics and rigidity)
# These combine the geometric factors (CoG height / distance) with suspension rigidity
EFFECTIVE_LONGITUDINAL_TRANSFER_FACTOR = 0.02  # (CoG_height / wheelbase) * longitudinal_rigidity (original)
EFFECTIVE_LATERAL_TRANSFER_FACTOR = 0.01      # (CoG_height / track_width) * lateral_rigidity (original)

# Weight Transfer Limits
MIN_TYRE_LOAD = 200.0  # Minimum load per tyre in Newtons (prevents complete unloading)
MAX_LONGITUDINAL_TRANSFER_RATIO = 0.95  # Maximum fraction of axle load that can transfer (95%)
MAX_LATERAL_TRANSFER_RATIO = 0.6  # Maximum fraction of axle load that can transfer laterally (60%)

# Physical constants (eliminates magic numbers)
TYRES_PER_AXLE = 2.0  # Number of tyres per axle
MIN_TYRE_LOAD_CONSTRAINT = 50.0  # Minimum load to prevent complete tyre unloading (Newtons)

# Coordinate System Constants
WORLD_FORWARD_VECTOR = (1.0, 0.0)  # Forward direction in world coordinates  
WORLD_RIGHT_VECTOR = (0.0, 1.0)  # Right direction in world coordinates
COORDINATE_ZERO = (0.0, 0.0)  # Zero coordinate pair
BODY_CENTER_OFFSET = (0.0, 0.0)  # Center of mass offset from body center
RESET_ANGULAR_VELOCITY = 0.0  # Angular velocity value for reset
AERODYNAMIC_DRAG_FACTOR = 0.5  # Aerodynamic drag equation factor (½ in ½ρCdAv²)

# Pre-computed drag calculation constant (eliminates 4 multiplications per physics step)
DRAG_CONSTANT = AERODYNAMIC_DRAG_FACTOR * AIR_DENSITY * CAR_DRAG_COEFFICIENT * CAR_FRONTAL_AREA

# Pre-computed angle constants (eliminates repeated trigonometric calculations)
PERPENDICULAR_ANGLE_OFFSET = math.pi / 2  # 90 degrees in radians
TWO_PI = 2 * math.pi  # Full circle in radians