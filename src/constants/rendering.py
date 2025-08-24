# Rendering and Display System Constants

# Rendering Constants
DEFAULT_RENDER_FPS = 30
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768
DEFAULT_WINDOW_SIZE = (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
RENDER_MODE_HUMAN = "human"

# Colors (RGB)
BACKGROUND_COLOR = (0, 100, 0)  # Green
FPS_COLOR_LOW = (255, 0, 0)     # Red
FPS_COLOR_NORMAL = (0, 255, 0)  # Green

# Window Configuration
WINDOW_CAPTION = "Nascar Racing Environment"
FPS_THRESHOLD = 60

# Physics Constants (rendering related)
PIXELS_PER_METER = 3  # Default conversion factor for rendering (now dynamic)

# Rendering/Display Constants  
DISPLAY_RESET_DELAY = 0.1  # Seconds to wait during display reset
WINDOW_CREATION_MAX_ATTEMPTS = 3  # Maximum attempts for robust window creation
WINDOW_CREATION_STEP_DELAY = 0.05  # Seconds between multi-step window creation
TEMPORARY_WINDOW_SIZE = (100, 100)  # Small window size for intermediate steps
MINIMUM_LINE_WIDTH = 1  # Minimum line width for drawing
SEGMENT_RECT_HEIGHT_DIVISOR = 10  # Divisor for zero-length segment height

# Camera Constants
CAMERA_MARGIN_FACTOR = 0.1  # 10% margin around track when auto-fitting
MIN_ZOOM_FACTOR = 0.01  # Minimum pixels per meter (maximum zoom out) - allows very long tracks
MAX_ZOOM_FACTOR = 50.0  # Maximum pixels per meter (maximum zoom in)
FULLSCREEN_TOGGLE_KEY = 'f'  # Key to toggle fullscreen mode

# Camera Mode Constants
CAMERA_MODE_TOGGLE_KEY = 'c'  # Key to toggle camera mode (track view vs car follow)
CAR_FOLLOW_ZOOM_FACTOR = 6.0  # Pixels per meter when following car
CAMERA_MODE_TRACK_VIEW = "track_view"  # Show entire track (default mode)
CAMERA_MODE_CAR_FOLLOW = "car_follow"  # Follow car with zoom
DEFAULT_ZOOM = 1.0  # Default camera zoom level
DEFAULT_TRACK_WIDTH_FALLBACK = 100.0  # Default width when track has zero width
DEFAULT_TRACK_HEIGHT_FALLBACK = 100.0  # Default height when track has zero height

# Track Polygon Rendering Constants
POLYGON_CENTERLINE_DEFAULT_SPACING = 3.5  # Default centerline spacing for polygon rendering (meters) - high detail mode
POLYGON_POSITION_MATCHING_TOLERANCE = 5.0  # Tolerance for position matching (world units)
POLYGON_MIN_CENTERLINE_SPACING = 0.5  # Minimum allowed centerline spacing (meters)