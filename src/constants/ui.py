# User Interface Elements Constants

import pygame

# UI Constants
FONT_SIZE = 36
FPS_TEXT_TOP_MARGIN = 10

# Action Bar UI Constants
ACTION_BAR_WIDTH = 200  # pixels width of each action bar
ACTION_BAR_HEIGHT = 15  # pixels height of each action bar
ACTION_BAR_SPACING = 8  # pixels spacing between bars
ACTION_BAR_TOP_MARGIN = 50  # pixels from top of screen (below FPS)
ACTION_BAR_PADDING = 4  # pixels padding inside bars
ACTION_BAR_LABEL_FONT_SIZE = 18  # font size for action labels

# Action Bar Colors (RGB)
THROTTLE_BAR_COLOR = (0, 200, 0)  # Green for throttle
THROTTLE_BAR_BG_COLOR = (50, 100, 50)  # Dark green background
BRAKE_BAR_COLOR = (200, 0, 0)  # Red for brake  
BRAKE_BAR_BG_COLOR = (100, 50, 50)  # Dark red background
STEERING_BAR_COLOR = (0, 100, 200)  # Blue for steering
STEERING_BAR_BG_COLOR = (50, 50, 100)  # Dark blue background
ACTION_BAR_BORDER_COLOR = (255, 255, 255)  # White borders
ACTION_BAR_TEXT_COLOR = (255, 255, 255)  # White text

# Race Tables UI Constants
RACE_TABLES_FONT_SIZE = 20  # Font size for race tables
RACE_TABLES_WIDTH = 200  # Width of each table in pixels
RACE_TABLES_HEIGHT = 300  # Height of each table in pixels
RACE_TABLES_PADDING = 10  # Padding inside tables in pixels
RACE_TABLES_SPACING = 20  # Space between left and right tables in pixels
RACE_TABLES_LINE_HEIGHT = 22  # Height per text line in pixels

# Race Tables Colors (RGB)
RACE_TABLES_BG_COLOR = (0, 0, 0)  # Black background
RACE_TABLES_BG_ALPHA = 180  # Semi-transparent background
RACE_TABLES_HEADER_COLOR = (255, 255, 0)  # Yellow headers
RACE_TABLES_POSITION_COLOR = (100, 255, 100)  # Green for positions
RACE_TABLES_LAP_TIME_COLOR = (255, 200, 100)  # Orange for lap times

# Track Info Display Constants
TRACK_INFO_TOGGLE_KEY = 'i'  # Key to toggle track info display
TRACK_INFO_TEXT_COLOR = (255, 255, 255)  # White text
TRACK_INFO_BG_COLOR = (0, 0, 0)  # Black background
TRACK_INFO_BG_ALPHA = 128  # Semi-transparent background
TRACK_INFO_PADDING = 20  # Padding around text in info box
TRACK_INFO_MARGIN = 20  # Margin from screen edge
TRACK_INFO_FONT_SIZE = 24  # Font size for track info (smaller than main FONT_SIZE)

# Debug Visualization Constants
DEBUG_TOGGLE_KEY = 'd'  # Key to toggle debug visualization
DEBUG_CENTERLINE_COLOR = (255, 0, 0)  # Red color for centerline
DEBUG_CENTERLINE_WIDTH = 1  # Thin line width for centerline

# Debug Info Panel Constants
DEBUG_INFO_PANEL_X = 20  # Left margin for debug info panel (pixels)
DEBUG_INFO_PANEL_Y = 120  # Top margin for debug info panel (pixels)
DEBUG_INFO_PANEL_WIDTH = 400  # Width of debug info panel (pixels)
DEBUG_INFO_PANEL_HEIGHT = 800  # Height of debug info panel (pixels)
DEBUG_INFO_PANEL_BG_COLOR = (0, 0, 0)  # Black background for debug panel
DEBUG_INFO_PANEL_BG_ALPHA = 200  # Semi-transparent background (0-255)
DEBUG_INFO_PANEL_TEXT_COLOR = (255, 255, 255)  # White text color
DEBUG_INFO_PANEL_PADDING = 12  # Inner padding for debug panel (pixels)
DEBUG_INFO_TEXT_FONT_SIZE = 11  # Font size for debug text
DEBUG_INFO_LINE_HEIGHT = 22  # Line spacing for debug text (pixels)

# Debug Vector Rendering Constants
DEBUG_VELOCITY_VECTOR_COLOR = (0, 255, 0)  # Green for velocity vector
DEBUG_ACCELERATION_VECTOR_COLOR = (255, 255, 0)  # Yellow for acceleration vector
DEBUG_STEERING_INPUT_VECTOR_COLOR = (0, 0, 255)  # Blue for steering input vector
DEBUG_STEERING_ACTUAL_VECTOR_COLOR = (255, 0, 255)  # Magenta for actual steering vector
DEBUG_VECTOR_VELOCITY_SCALE_FACTOR = 2.0  # Scale factor for velocity vectors (pixels per m/s)
DEBUG_VECTOR_ACCELERATION_SCALE_FACTOR = 10.0  # Scale factor for acceleration vectors (pixels per m/s²) - increased for visibility
DEBUG_VECTOR_STEERING_SCALE_FACTOR = 50.0  # Scale factor for steering vectors (pixels per unit) - increased for visibility
DEBUG_VECTOR_MAX_LENGTH = 80  # Maximum vector length to prevent screen overflow (pixels)
DEBUG_VECTOR_ARROW_WIDTH = 3  # Width of vector arrows (pixels)
DEBUG_VECTOR_ARROW_HEAD_SIZE = 8  # Size of vector arrow heads (pixels)
DEBUG_VECTOR_MIN_LENGTH = 5  # Minimum vector length for rendering (pixels)

# Distance Sensor Debug Constants  
DEBUG_SENSOR_VECTOR_COLOR = (255, 0, 255)  # Magenta color for distance sensor vectors

# Lap Timer Display Constants
LAP_TIMER_FONT_SIZE = 28  # Font size for lap time display
LAP_TIMER_BOTTOM_MARGIN = 15  # Pixels from bottom of screen
LAP_TIMER_LINE_SPACING = 25  # Pixels between timer lines

# Lap Timer Colors (RGB)
LAP_TIMER_CURRENT_COLOR = (255, 255, 255)  # White for current lap time
LAP_TIMER_LAST_COLOR = (255, 255, 100)     # Yellow for last lap time
LAP_TIMER_BEST_COLOR = (100, 255, 100)     # Green for best lap time
LAP_TIMER_BG_COLOR = (0, 0, 0)             # Black background
LAP_TIMER_BG_ALPHA = 180                   # Semi-transparent background

# Countdown Clock Display Constants
COUNTDOWN_FONT_SIZE = 32                     # Font size for countdown clock
COUNTDOWN_TOP_MARGIN = 15                    # Pixels from top of screen
COUNTDOWN_LEFT_MARGIN = 15                   # Pixels from left of screen
COUNTDOWN_BG_PADDING = 10                    # Padding around text background
COUNTDOWN_BG_ALPHA = 200                     # Semi-transparent background

# Countdown Clock Colors
COUNTDOWN_TIME_SAFE_COLOR = (100, 255, 100)     # Green when plenty of time
COUNTDOWN_TIME_WARNING_COLOR = (255, 255, 100)  # Yellow when time getting low
COUNTDOWN_TIME_DANGER_COLOR = (255, 100, 100)   # Red when time almost up
COUNTDOWN_BG_COLOR = (0, 0, 0)                  # Black background
COUNTDOWN_WARNING_THRESHOLD = 30.0              # Seconds - switch to yellow
COUNTDOWN_DANGER_THRESHOLD = 10.0               # Seconds - switch to red

# Reward display toggle
REWARD_TOGGLE_KEY = 'r'  # Key to toggle reward display

# Car switching key mappings (pygame keys)
CAR_SELECT_KEYS = [
    pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9
]

# Observation Visualization Constants
OBSERVATION_TOGGLE_KEY = 'o'  # Key to toggle observation display
OBSERVATION_HISTORY_LENGTH = 300  # Number of data points to keep in rolling history
OBSERVATION_GRAPH_MARGIN = 20  # Margin around graphs in pixels
OBSERVATION_GRAPH_SPACING = 10  # Spacing between graphs in pixels
OBSERVATION_OVERLAY_BG_COLOR = (0, 0, 0)  # Black background for overlay
OBSERVATION_OVERLAY_BG_ALPHA = 220  # Semi-transparent background for overlay
OBSERVATION_GRAPH_COLS = 2  # Number of graph columns in layout
OBSERVATION_GRAPH_ROWS = 3  # Number of graph rows in layout