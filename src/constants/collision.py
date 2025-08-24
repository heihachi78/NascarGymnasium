# Collision Detection and Damage Constants

# Car Collision Properties
CAR_COLLISION_SHAPES = 4  # number of collision shapes for car (corners)
CAR_COLLISION_RADIUS = 0.3  # radius of corner collision shapes in meters
COLLISION_FORCE_THRESHOLD = 100.0  # minimum force to register as significant collision (increased to ignore light touches)
COLLISION_DAMAGE_FACTOR = 0.001  # damage per unit collision force

# Collision Force Threshold
# Note: Severity classification has been removed - now only track if collision is significant

# Stuck Detection Thresholds
STUCK_SPEED_THRESHOLD = 0.5  # m/s - speed below which car might be considered stuck
STUCK_TIME_THRESHOLD = 10.0  # seconds - minimum time before checking movement distance
STUCK_DISTANCE_THRESHOLD = 1.0  # meters - if car moves less than this in STUCK_TIME_THRESHOLD, it's stuck
STUCK_EXTENDED_TIME_THRESHOLD = 15.0  # seconds - after this time, disable car regardless of movement
STUCK_INPUT_THRESHOLD = 0.1  # minimum input magnitude to consider as "active input"
STUCK_RECENT_INPUT_TIME = 3.0  # seconds - time window to check for recent meaningful inputs

# Backward movement detection
BACKWARD_MOVEMENT_THRESHOLD = 25.0  # meters - distance backward before penalty starts
BACKWARD_DISABLE_THRESHOLD = 200.0  # meters - distance backward before car is disabled

# Impact-based disabling thresholds
INSTANT_DISABLE_IMPACT_THRESHOLD = 50_000.0  # N⋅s - single impact to instantly disable car
CUMULATIVE_DISABLE_IMPACT_THRESHOLD = 100_000.0  # N⋅s - total accumulated impacts to disable car
