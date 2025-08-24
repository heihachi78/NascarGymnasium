# Environment and Episode Management Constants

# Environment Constants  
INITIAL_ELAPSED_TIME = 0.0
DEFAULT_REWARD = 0.0
DEFAULT_TERMINATED = False
DEFAULT_TRUNCATED = False

# Termination Condition Constants
TERMINATION_MIN_REWARD = -1000.0  # Terminate if cumulative reward drops below this
TERMINATION_MAX_TIME = 60.0  # Terminate after this many seconds
TRUNCATION_MAX_TIME = 300.0  # Truncate (hard limit) after this many seconds

# Logging Constants
DEFAULT_LOG_LEVEL = "INFO"  # Default logging level
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Default log format

# Performance Validation Constants
MIN_REALISTIC_FPS = 5.0  # Minimum realistic FPS value
MAX_REALISTIC_FPS = 300.0  # Maximum realistic FPS value

# Performance History
VELOCITY_HISTORY_SIZE = 600  # History size at 60 FPS (10 seconds * 60 FPS)
PERFORMANCE_VALIDATION_MIN_SAMPLES = 10  # Minimum samples for performance validation
PERFORMANCE_SPEED_TOLERANCE = 0.95  # Speed tolerance (95% of target)
PERFORMANCE_TIME_TOLERANCE = 1.1  # Time tolerance (110% of target)