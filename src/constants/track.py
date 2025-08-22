# Track Geometry and Properties Constants

# Track Constants
DEFAULT_TRACK_WIDTH = 20.0  # meters
DEFAULT_GRID_LENGTH = 100.0  # meters
STARTLINE_LENGTH = 5.0  # meters
FINISHLINE_LENGTH = 5.0  # meters

# Track Colors (RGB)
GRID_COLOR = (192, 192, 192)  # Light gray
TRACK_COLOR = (128, 128, 128)  # Dark gray
STARTLINE_COLOR = (255, 255, 0)  # Yellow
FINISHLINE_COLOR = (255, 255, 0)  # Yellow

# Track Wall Properties
TRACK_WALL_THICKNESS = 1.0  # meters

# Centerline Generation Constants
CENTERLINE_DEFAULT_SPACING = 2.0  # Default target spacing between centerline points (meters)
CENTERLINE_MIN_CURVE_POINTS = 8  # Minimum number of points for curved segments
CENTERLINE_TIGHT_CURVE_THRESHOLD = 50.0  # Radius threshold for tight curve detection (meters)
CENTERLINE_ADAPTIVE_MIN_FACTOR = 1.0  # Minimum adaptive sampling factor
CENTERLINE_SMOOTHING_FACTOR = 0.1  # Default smoothing factor for centerline smoothing

# Track Boundary Constants  
BOUNDARY_SMOOTHING_MAX_ANGLE = 120.0  # Maximum angle before corner smoothing kicks in (degrees)
BOUNDARY_SMOOTHING_MAX_FACTOR = 0.3  # Maximum smoothing amount (30%)
BOUNDARY_POINTS_EQUAL_TOLERANCE = 1e-6  # Tolerance for point equality checks
BOUNDARY_MIN_POLYGON_AREA = 1.0  # Minimum polygon area for validation (square units)

# Lap Detection Constants
LAP_DETECTION_POSITION_TOLERANCE = 2.0  # Meters tolerance for position-based detection