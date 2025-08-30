# Box2D Physics Engine Constants

from .rendering import DEFAULT_RENDER_FPS

# Collision Filtering Constants
COLLISION_CATEGORY_TRACK_WALLS = 0x0001   # Category bit for track walls
COLLISION_CATEGORY_CARS = 0x0002          # Category bit for cars
COLLISION_MASK_TRACK_WALLS = 0xFFFF       # Track walls collide with everything
COLLISION_MASK_CARS = 0x0001              # Cars only collide with track walls (not other cars)

# Box2D Physics Constants
BOX2D_WALL_DENSITY = 0.0  # Density for track wall fixtures
BOX2D_WALL_FRICTION = 0.333  # Friction for track wall fixtures (reduced from 0.8 to prevent sticking)
BOX2D_WALL_RESTITUTION = 0.25  # Restitution (bounciness) for track wall fixtures (increased from 0.2 for better bounce-off)
BOX2D_TIME_STEP = 1.0/DEFAULT_RENDER_FPS  # Physics simulation time step (auto-calculated)
BOX2D_VELOCITY_ITERATIONS = 6  # Box2D velocity constraint solver iterations
BOX2D_POSITION_ITERATIONS = 4  # Box2D position constraint solver iterations (increased for better stability)

# Box2D Physics Detail Constants
PHYSICS_CURVE_DEGREES_PER_SEGMENT = 1.0  # Degrees per physics wall segment (lower = more detail)
PHYSICS_CURVE_MIN_SEGMENTS = 8  # Minimum segments per curved track section
PHYSICS_CURVE_MAX_SEGMENTS = 180  # Maximum segments to prevent excessive detail

# Physics Timing Constants
MAX_PHYSICS_TIMESTEP = 1.0 / 30.0  # Maximum timestep to prevent instability (33.33ms)
MIN_PHYSICS_TIMESTEP = 1.0 / 1000.0  # Minimum timestep to prevent division issues (1ms)

# Banking Physics Constants
BANKING_LATERAL_ASSIST_FACTOR = 0.3        # Fraction of normal force that becomes lateral assist (30%)
BANKING_MINIMUM_ANGLE_THRESHOLD = 0.1      # Degrees - minimum banking angle to apply forces
BANKING_MINIMUM_SPEED_THRESHOLD = 1.0      # m/s - minimum speed for banking effects
BANKING_LATERAL_SPEED_THRESHOLD = 5.0      # m/s - minimum speed for lateral banking forces
GRAVITY_ACCELERATION = 9.81                # m/s² - Earth's gravitational acceleration