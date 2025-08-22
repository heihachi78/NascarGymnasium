# Box2D Physics Engine Constants

from .rendering import DEFAULT_RENDER_FPS

# Collision Filtering Constants
COLLISION_CATEGORY_TRACK_WALLS = 0x0001   # Category bit for track walls
COLLISION_CATEGORY_CARS = 0x0002          # Category bit for cars
COLLISION_MASK_TRACK_WALLS = 0xFFFF       # Track walls collide with everything
COLLISION_MASK_CARS = 0x0001              # Cars only collide with track walls (not other cars)

# Box2D Physics Constants
BOX2D_WALL_DENSITY = 0.0  # Density for track wall fixtures
BOX2D_WALL_FRICTION = 0.8  # Friction for track wall fixtures  
BOX2D_WALL_RESTITUTION = 0.2  # Restitution (bounciness) for track wall fixtures
BOX2D_TIME_STEP = 1.0/DEFAULT_RENDER_FPS  # Physics simulation time step (auto-calculated)
BOX2D_VELOCITY_ITERATIONS = 6  # Box2D velocity constraint solver iterations
BOX2D_POSITION_ITERATIONS = 2  # Box2D position constraint solver iterations

# Box2D Physics Detail Constants
PHYSICS_CURVE_DEGREES_PER_SEGMENT = 1.0  # Degrees per physics wall segment (lower = more detail)
PHYSICS_CURVE_MIN_SEGMENTS = 8  # Minimum segments per curved track section
PHYSICS_CURVE_MAX_SEGMENTS = 180  # Maximum segments to prevent excessive detail