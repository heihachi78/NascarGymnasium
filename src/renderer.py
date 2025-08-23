import pygame
import os
import time
import platform
import logging
from typing import Optional
from .constants import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_RENDER_FPS,
    UNLIMITED_FPS_CAP,
    BACKGROUND_COLOR,
    FPS_COLOR_LOW,
    FPS_COLOR_NORMAL,
    FONT_SIZE,
    FPS_TEXT_TOP_MARGIN,
    FPS_THRESHOLD,
    WINDOW_CAPTION,
    FULLSCREEN_TOGGLE_KEY,
    TRACK_INFO_TOGGLE_KEY,
    TRACK_INFO_TEXT_COLOR,
    CAMERA_MODE_CAR_FOLLOW,
    TRACK_INFO_BG_COLOR,
    TRACK_INFO_BG_ALPHA,
    TRACK_INFO_PADDING,
    TRACK_INFO_MARGIN,
    TRACK_INFO_FONT_SIZE,
    DISPLAY_RESET_DELAY,
    WINDOW_CREATION_MAX_ATTEMPTS,
    WINDOW_CREATION_STEP_DELAY,
    TEMPORARY_WINDOW_SIZE,
    DEBUG_TOGGLE_KEY,
    CAR_COLOR,
    CAR_OUTLINE_COLOR,
    CAR_VISUAL_LENGTH,
    CAR_VISUAL_WIDTH,
    CAMERA_MODE_TOGGLE_KEY,
    ACTION_BAR_WIDTH,
    ACTION_BAR_HEIGHT,
    ACTION_BAR_SPACING,
    ACTION_BAR_TOP_MARGIN,
    ACTION_BAR_PADDING,
    ACTION_BAR_LABEL_FONT_SIZE,
    THROTTLE_BAR_COLOR,
    THROTTLE_BAR_BG_COLOR,
    BRAKE_BAR_COLOR,
    BRAKE_BAR_BG_COLOR,
    STEERING_BAR_COLOR,
    STEERING_BAR_BG_COLOR,
    ACTION_BAR_BORDER_COLOR,
    ACTION_BAR_TEXT_COLOR,
    LAP_TIMER_FONT_SIZE,
    LAP_TIMER_BOTTOM_MARGIN,
    LAP_TIMER_LINE_SPACING,
    LAP_TIMER_LABEL_WIDTH,
    LAP_TIMER_CURRENT_COLOR,
    LAP_TIMER_LAST_COLOR,
    LAP_TIMER_BEST_COLOR,
    LAP_TIMER_BG_COLOR,
    LAP_TIMER_BG_ALPHA,
    # Race Tables Constants
    RACE_TABLES_FONT_SIZE,
    RACE_TABLES_WIDTH,
    RACE_TABLES_HEIGHT,
    RACE_TABLES_PADDING,
    RACE_TABLES_SPACING,
    RACE_TABLES_LINE_HEIGHT,
    RACE_TABLES_BG_COLOR,
    RACE_TABLES_BG_ALPHA,
    RACE_TABLES_TEXT_COLOR,
    RACE_TABLES_HEADER_COLOR,
    RACE_TABLES_POSITION_COLOR,
    RACE_TABLES_LAP_TIME_COLOR,
    # Countdown Clock Constants
    COUNTDOWN_FONT_SIZE,
    COUNTDOWN_TOP_MARGIN,
    COUNTDOWN_LEFT_MARGIN,
    COUNTDOWN_BG_PADDING,
    COUNTDOWN_BG_ALPHA,
    COUNTDOWN_TIME_SAFE_COLOR,
    COUNTDOWN_TIME_WARNING_COLOR,
    COUNTDOWN_TIME_DANGER_COLOR,
    COUNTDOWN_BG_COLOR,
    COUNTDOWN_WARNING_THRESHOLD,
    COUNTDOWN_DANGER_THRESHOLD
)
from .track_generator import Track
from .camera import Camera
from .track_polygon_renderer import TrackPolygonRenderer
from .debug_info_renderer import DebugInfoRenderer

# Setup module logger
logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, render_fps=DEFAULT_RENDER_FPS, track: Optional[Track] = None, 
                 use_antialiasing: bool = True, enable_fps_limit: bool = True):
        self.window_size = window_size
        self.render_fps = render_fps
        self.enable_fps_limit = enable_fps_limit
        self.window = None
        self.clock = None
        
        # Manual FPS calculation for uncapped rendering
        self.frame_times = []
        self.last_frame_time = None
        self._initialized_pygame = False
        self.font = None
        self.track = track
        self.camera = Camera(window_size)
        self.is_fullscreen = False
        self.original_window_size = window_size
        self.show_track_info = False
        self.show_debug = False
        self.debug_centerline = False
        
        # Polygon rendering system with performance settings
        self.polygon_renderer = TrackPolygonRenderer(self.camera)
        self.polygon_renderer.set_antialiasing(use_antialiasing)
        
        # Debug info rendering system
        self.debug_info_renderer = DebugInfoRenderer(self.camera)
        
        # Use default spacing (no quality modes)
        
        if track:
            self.camera.set_track(track)
        
        # Platform detection for fullscreen handling
        self.is_wsl = self._is_wsl()
        self.is_linux = platform.system() == "Linux"
        
    
    def set_performance_mode(self, use_antialiasing: bool = None):
        """Dynamically adjust performance settings"""
        if use_antialiasing is not None:
            self.polygon_renderer.set_antialiasing(use_antialiasing)
    
    def _is_wsl(self):
        """Detect if running under WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except (FileNotFoundError, PermissionError, OSError):
            return False
    
    def _set_sdl_hints(self):
        """Set SDL environment variables for better window behavior"""
        # Center the window
        os.environ['SDL_VIDEO_WINDOW_POS'] = 'centered'
        
        # For WSL/Linux, force specific video driver behavior
        if self.is_wsl or self.is_linux:
            os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = '0'
    
    def _force_display_reset(self):
        """Completely reset pygame display system - Phase 1"""
        logger.debug("Forcing complete display reset...")
        
        # Store current font for restoration
        font_size = FONT_SIZE if self.font is None else FONT_SIZE
        
        # Complete shutdown
        pygame.display.quit()
        
        # Set SDL hints before reinitializing
        self._set_sdl_hints()
        
        # Small delay to let window manager process
        time.sleep(DISPLAY_RESET_DELAY)
        
        # Reinitialize display
        pygame.display.init()
        
        # Restore font
        self.font = pygame.font.Font(None, font_size)
        
        logger.debug("Display reset complete")
    
    def _create_windowed_window_robust(self, target_size, max_attempts=WINDOW_CREATION_MAX_ATTEMPTS):
        """Robustly create windowed window with multiple strategies - Phase 3"""
        
        for attempt in range(max_attempts):
            logger.debug(f"Window creation attempt {attempt + 1}/{max_attempts}")
            
            try:
                if attempt == 0:
                    # Method 1: Direct creation
                    logger.debug(f"Method 1: Direct window creation {target_size}")
                    window = pygame.display.set_mode(target_size, 0)
                    
                elif attempt == 1:
                    # Method 2: Multi-step resize
                    logger.debug("Method 2: Multi-step window creation")
                    # Create small window first
                    window = pygame.display.set_mode(TEMPORARY_WINDOW_SIZE, 0)
                    time.sleep(WINDOW_CREATION_STEP_DELAY)  # Let window manager process
                    # Resize to target
                    window = pygame.display.set_mode(target_size, 0)
                    
                elif attempt == 2:
                    # Method 3: Force reset + creation
                    logger.debug("Method 3: Reset + creation")
                    self._force_display_reset()
                    window = pygame.display.set_mode(target_size, 0)
                
                # Validate the window size
                actual_size = window.get_size()
                logger.debug(f"Requested: {target_size}, Got: {actual_size}")
                
                if actual_size == target_size:
                    logger.debug(f"Window creation successful with method {attempt + 1}")
                    return window
                else:
                    logger.warning(f"Method {attempt + 1} failed - wrong size")
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        logger.warning("All methods failed, using what we got")
                        return window
                        
            except pygame.error as e:
                logger.warning(f"Method {attempt + 1} failed with error: {e}")
                if attempt == max_attempts - 1:
                    raise
                continue
        
        # Should never reach here, but fallback
        return pygame.display.set_mode(target_size, 0)
    
    def init_pygame(self):
        if not self._initialized_pygame:
            # Only initialize if not already initialized
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
            if not pygame.font.get_init():
                pygame.font.init()
            self._initialized_pygame = True
            
    def render_frame(self, car_position=None, car_angle=None, debug_data=None, current_action=None, lap_timing_info=None, reward_info=None, cars_data=None, followed_car_index=0, race_positions_data=None, best_lap_times_data=None, countdown_info=None):
        self.init_pygame()
        
        if self.window is None:
            if self.is_fullscreen:
                self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption(WINDOW_CAPTION)
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            self.font = pygame.font.Font(None, FONT_SIZE)
        
        # Handle events for fullscreen toggle
        self._handle_events()
        
        # Update camera for car follow mode - follow the followed car
        if cars_data and followed_car_index < len(cars_data):
            followed_car_data = cars_data[followed_car_index]
            if self.camera.get_camera_mode() == "car_follow":
                self.camera.update_car_follow_camera(followed_car_data['position'])
        elif car_position is not None and self.camera.get_camera_mode() == "car_follow":
            # Fallback to single car position for backward compatibility
            self.camera.update_car_follow_camera(car_position)
            
        self.window.fill(BACKGROUND_COLOR)
        
        # Render track if available
        if self.track:
            self._render_track()
            
        # Render all cars if multi-car data provided
        if cars_data:
            self._render_multiple_cars(cars_data, followed_car_index)
        elif car_position is not None and car_angle is not None:
            # Fallback to single car rendering for backward compatibility
            self._render_car(car_position, car_angle)
        
        # Render track info if enabled
        if self.show_track_info and self.track:
            self._render_track_info()
        
        # Render debug visualizations if enabled
        if self.show_debug and self.track:
            self._render_debug_info(debug_data)
        
        # Render action bars if action data is available
        if current_action is not None:
            # Get car name for display
            car_name = None
            if cars_data and followed_car_index < len(cars_data):
                car_name = cars_data[followed_car_index].get('name', f"Car {followed_car_index}")
            self._render_action_bars(current_action, car_name)
        
        # Render lap times if timing info is available
        if lap_timing_info is not None:
            self._render_lap_times(lap_timing_info)
        
        # Render reward if info is available
        if reward_info is not None and reward_info.get('show', False):
            self._render_reward(reward_info)
        
        # Render race tables if data is available and not in car follow mode
        if (race_positions_data is not None or best_lap_times_data is not None) and self.camera.get_camera_mode() != CAMERA_MODE_CAR_FOLLOW:
            self._render_race_tables(race_positions_data, best_lap_times_data)
        
        # Render countdown clock if timing info is available
        if countdown_info is not None:
            self._render_countdown_clock(countdown_info)
        
        # Calculate FPS - use pygame clock when FPS limited, manual calculation when uncapped
        if self.enable_fps_limit:
            current_fps = self.clock.get_fps()
        else:
            current_fps = self._calculate_manual_fps()
        
        if current_fps < FPS_THRESHOLD:
            fps_color = FPS_COLOR_LOW
        else:
            fps_color = FPS_COLOR_NORMAL
            
        fps_text = self.font.render(f"FPS: {current_fps:.1f}", True, fps_color)
        
        text_rect = fps_text.get_rect()
        text_rect.right = self.window_size[0] - 10  # 10 pixels from right edge
        text_rect.top = FPS_TEXT_TOP_MARGIN
        
        self.window.blit(fps_text, text_rect)
        
        pygame.display.flip()
        
        # Only limit FPS if enabled (for frame rate independence)
        if self.enable_fps_limit:
            self.clock.tick(self.render_fps)
        else:
            # Use high FPS cap instead of unlimited to prevent physics timing issues
            self.clock.tick(UNLIMITED_FPS_CAP)
    
    def _calculate_manual_fps(self) -> float:
        """Calculate FPS manually when not using pygame's FPS limiting."""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_delta = current_time - self.last_frame_time
            self.frame_times.append(frame_delta)
            
            # Keep only recent frame times (last 30 frames for smooth averaging)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
        
        # Calculate average FPS from recent frame times
        if len(self.frame_times) > 5:  # Need at least a few frames for accurate measurement
            average_frame_time = sum(self.frame_times) / len(self.frame_times)
            if average_frame_time > 0:
                return 1.0 / average_frame_time
        
        return 0.0  # Return 0 if we don't have enough data yet
        
    def close(self):
        """Clean up renderer resources safely"""
        try:
            if self.window is not None:
                # Clear references first
                self.window = None
                self.clock = None
                self.font = None
                
                # Reset FPS calculation
                self.frame_times = []
                self.last_frame_time = None
                
                # Close display only if it's initialized
                if pygame.display.get_init():
                    pygame.display.quit()
                
                # Mark as not initialized
                self._initialized_pygame = False
                
                # Note: pygame.quit() is now handled by car_env.py close method
        except Exception as e:
            print(f"Warning: Error during renderer cleanup: {e}")
            # Clear references anyway to prevent further issues
            self.window = None
            self.clock = None
            self.font = None
            self.frame_times = []
            self.last_frame_time = None
            self._initialized_pygame = False
    
    def _handle_events(self):
        """Handle pygame events, particularly fullscreen toggle"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Don't handle quit here - let the environment handle it
                pygame.event.post(event)  # Re-post the event
            elif event.type == pygame.KEYDOWN:
                if event.unicode.lower() == FULLSCREEN_TOGGLE_KEY:
                    self._toggle_fullscreen()
                elif event.unicode.lower() == TRACK_INFO_TOGGLE_KEY:
                    self.toggle_track_info()
                elif event.unicode.lower() == DEBUG_TOGGLE_KEY:
                    self.toggle_debug()
                elif event.unicode.lower() == CAMERA_MODE_TOGGLE_KEY:
                    self._toggle_camera_mode()
                else:
                    # Re-post other key events
                    pygame.event.post(event)
            else:
                # Re-post other events
                pygame.event.post(event)
    
    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode with robust handling"""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            # Switch to fullscreen - this usually works fine
            logger.info("Switching to fullscreen...")
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            fullscreen_size = self.window.get_size()
            logger.info(f"Fullscreen size: {fullscreen_size}")
            
            # Update camera with fullscreen size
            self.camera.set_window_size(fullscreen_size)
            self.window_size = fullscreen_size
            
        else:
            # Switch back to windowed - this is where problems occur
            logger.info(f"Switching to windowed mode {self.original_window_size}...")
            
            # Phase 1: Force display reset for problematic platforms
            if self.is_wsl or self.is_linux:
                logger.debug("Linux/WSL detected - using display reset method")
                self._force_display_reset()
            
            # Phase 2: Set SDL hints
            self._set_sdl_hints()
            
            # Phase 3: Robust window creation with multiple fallbacks
            try:
                self.window = self._create_windowed_window_robust(self.original_window_size)
                pygame.display.set_caption(WINDOW_CAPTION)
                
                # Phase 5: Validate and handle success/failure
                actual_size = self.window.get_size()
                if actual_size == self.original_window_size:
                    logger.info("Windowed mode switch successful!")
                    self.window_size = self.original_window_size
                    self.camera.set_window_size(self.original_window_size)
                else:
                    logger.warning(f"Partial success - got {actual_size} instead of {self.original_window_size}")
                    # Use whatever size we got
                    self.window_size = actual_size
                    self.camera.set_window_size(actual_size)
                
            except Exception as e:
                logger.error(f"Windowed mode switch failed: {e}")
                # Fallback: try to at least get a working window
                try:
                    self.window = pygame.display.set_mode(self.original_window_size)
                    self.window_size = self.window.get_size()
                    self.camera.set_window_size(self.window_size)
                    logger.info(f"Fallback successful: {self.window_size}")
                except pygame.error as fallback_error:
                    logger.error(f"Complete failure - fallback window creation failed: {fallback_error}")
                    # Revert to fullscreen to maintain functionality
                    self.is_fullscreen = True
                    self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    self.window_size = self.window.get_size()
                    self.camera.set_window_size(self.window_size)
    
    def set_track(self, track: Track):
        """Set the track to render"""
        self.track = track
        self.camera.set_track(track)
    
    def toggle_track_info(self):
        """Toggle the display of track information"""
        self.show_track_info = not self.show_track_info
    
    def toggle_debug(self):
        """Toggle debug visualization mode"""
        self.show_debug = not self.show_debug
        if self.show_debug:
            # Enable all debug features when turning on
            self.debug_centerline = True
        else:
            # Disable all debug features when turning off
            self.debug_centerline = False
    
    def _toggle_camera_mode(self):
        """Toggle camera mode between track view and car follow"""
        self.camera.toggle_camera_mode()
        logger.info(f"Camera mode switched to: {self.camera.get_camera_mode()}")
    
    def get_rendering_info(self) -> dict:
        """Get information about current rendering system"""
        info = {
            'system': 'polygon',
            'polygon_available': True
        }
        
        if self.polygon_renderer and self.track:
            stats = self.polygon_renderer.get_rendering_stats(self.track)
            info.update(stats)
        
        return info
    
    
    def _render_track(self):
        """Render the track segments using polygon rendering"""
        if not self.track:
            return
        
        # Use polygon-based rendering system
        if self.polygon_renderer:
            success = self.polygon_renderer.render_track_polygon(self.window, self.track)
            if not success:
                logger.warning("Polygon rendering failed")
    
    
    def _render_track_info(self):
        """Render track information in the middle-right portion of the screen"""
        if not self.track:
            return
        
        # Create track info font (smaller than main font)
        try:
            info_font = pygame.font.Font(None, TRACK_INFO_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if track info font fails
            if not self.font:
                return
            info_font = self.font
        
        # Get track metrics
        total_length = self.track.get_total_track_length()
        start_to_finish_length = self.track.get_start_to_finish_length()
        
        # Create info text lines
        info_lines = [
            f"Total Track Length: {total_length:.1f}m",
            f"Start to Finish: {start_to_finish_length:.1f}m"
        ]
        
        # Calculate text dimensions
        text_surfaces = []
        max_width = 0
        line_height = info_font.get_height()
        
        for line in info_lines:
            surface = info_font.render(line, True, TRACK_INFO_TEXT_COLOR)
            text_surfaces.append(surface)
            max_width = max(max_width, surface.get_width())
        
        # Position in middle-right of screen
        info_width = max_width + 2 * TRACK_INFO_PADDING
        info_height = len(info_lines) * line_height + 2 * TRACK_INFO_PADDING
        
        # Position: right side, vertically centered
        info_x = self.window_size[0] - info_width - TRACK_INFO_MARGIN
        info_y = (self.window_size[1] - info_height) // 2
        
        # Draw semi-transparent background
        bg_surface = pygame.Surface((info_width, info_height))
        bg_surface.set_alpha(TRACK_INFO_BG_ALPHA)
        bg_surface.fill(TRACK_INFO_BG_COLOR)
        self.window.blit(bg_surface, (info_x, info_y))
        
        # Draw text lines
        for i, surface in enumerate(text_surfaces):
            text_x = info_x + TRACK_INFO_PADDING
            text_y = info_y + TRACK_INFO_PADDING + i * line_height
            self.window.blit(surface, (text_x, text_y))
    
    def _render_debug_info(self, debug_data=None):
        """Render debug visualizations"""
        if not self.track or not self.polygon_renderer:
            return
        
        # Render centerline if enabled
        if self.debug_centerline:
            self.polygon_renderer.render_centerline(self.window)
            
        # Render enhanced debug info if data is available
        if debug_data and self.debug_info_renderer:
            self.debug_info_renderer.render_debug_info(self.window, debug_data)
            
    def _render_car(self, car_position: tuple, car_angle: float):
        """Render the car at the specified position and angle
        
        Args:
            car_position: (x, y) position of car in world coordinates
            car_angle: Car orientation angle in radians
        """
        import math
        
        if not self.window or not self.camera:
            return
            
        # Convert world coordinates to screen coordinates
        screen_x, screen_y = self.camera.world_to_screen(car_position)
        
        # Calculate car corners in local car coordinate system
        half_length = CAR_VISUAL_LENGTH / 2.0
        half_width = CAR_VISUAL_WIDTH / 2.0
        
        # Car corners relative to center (before rotation)
        corners = [
            (-half_length, -half_width),  # rear left
            (half_length, -half_width),   # front left  
            (half_length, half_width),    # front right
            (-half_length, half_width)    # rear right
        ]
        
        # Rotate corners based on car angle
        cos_angle = math.cos(car_angle)
        sin_angle = math.sin(car_angle)
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Rotate the corner
            rotated_x = corner_x * cos_angle - corner_y * sin_angle
            rotated_y = corner_x * sin_angle + corner_y * cos_angle
            
            # Convert to screen coordinates
            world_x = car_position[0] + rotated_x
            world_y = car_position[1] + rotated_y
            screen_corner_x, screen_corner_y = self.camera.world_to_screen((world_x, world_y))
            
            rotated_corners.append((screen_corner_x, screen_corner_y))
        
        # Draw the car as a filled polygon
        try:
            pygame.draw.polygon(self.window, CAR_COLOR, rotated_corners)
            pygame.draw.polygon(self.window, CAR_OUTLINE_COLOR, rotated_corners, 2)  # 2px outline
        except (ValueError, TypeError):
            # Fallback: draw as a simple circle if polygon fails
            pygame.draw.circle(self.window, CAR_COLOR, (int(screen_x), int(screen_y)), 10)
    
    def _render_car_with_color(self, car_position: tuple, car_angle: float, car_color: tuple):
        """Render a car with a specific color at the specified position and angle
        
        Args:
            car_position: (x, y) position of car in world coordinates
            car_angle: Car orientation angle in radians
            car_color: (R, G, B) color tuple for the car
        """
        import math
        
        if not self.window or not self.camera:
            return
            
        # Convert world coordinates to screen coordinates
        screen_x, screen_y = self.camera.world_to_screen(car_position)
        
        # Calculate car corners in local car coordinate system
        half_length = CAR_VISUAL_LENGTH / 2.0
        half_width = CAR_VISUAL_WIDTH / 2.0
        
        # Car corners relative to center (before rotation)
        corners = [
            (-half_length, -half_width),  # rear left
            (half_length, -half_width),   # front left  
            (half_length, half_width),    # front right
            (-half_length, half_width)    # rear right
        ]
        
        # Rotate corners based on car angle
        cos_angle = math.cos(car_angle)
        sin_angle = math.sin(car_angle)
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Rotate the corner
            rotated_x = corner_x * cos_angle - corner_y * sin_angle
            rotated_y = corner_x * sin_angle + corner_y * cos_angle
            
            # Convert to screen coordinates
            world_x = car_position[0] + rotated_x
            world_y = car_position[1] + rotated_y
            screen_corner_x, screen_corner_y = self.camera.world_to_screen((world_x, world_y))
            
            rotated_corners.append((screen_corner_x, screen_corner_y))
        
        # Draw the car as a filled polygon with specified color
        try:
            pygame.draw.polygon(self.window, car_color, rotated_corners)
            pygame.draw.polygon(self.window, CAR_OUTLINE_COLOR, rotated_corners, 2)  # 2px outline
        except (ValueError, TypeError):
            # Fallback: draw as a simple circle if polygon fails
            pygame.draw.circle(self.window, car_color, (int(screen_x), int(screen_y)), 10)
    
    def _render_multiple_cars(self, cars_data: list, followed_car_index: int):
        """Render multiple cars with different colors
        
        Args:
            cars_data: List of car data dictionaries with 'position', 'angle', and 'color' keys
            followed_car_index: Index of the currently followed car (for special highlighting)
        """
        if not cars_data:
            return
        
        # First pass: Render all car bodies
        for i, car_data in enumerate(cars_data):
            position = car_data.get('position')
            angle = car_data.get('angle')
            color = car_data.get('color', (255, 255, 255))  # Default to white
            
            if position is not None and angle is not None:
                # Render the car with its assigned color
                self._render_car_with_color(position, angle, color)
        
        # Second pass: Render all nameplates (non-followed cars first)
        for i, car_data in enumerate(cars_data):
            if i != followed_car_index:  # Skip followed car for now
                position = car_data.get('position')
                color = car_data.get('color', (255, 255, 255))
                name = car_data.get('name', f"Car {i}")
                
                if position is not None:
                    # Render car name above the car (non-followed)
                    self._render_car_name(position, name, color, is_followed=False)
        
        # Third pass: Render followed car's UI elements on top
        if 0 <= followed_car_index < len(cars_data):
            followed_car = cars_data[followed_car_index]
            position = followed_car.get('position')
            color = followed_car.get('color', (255, 255, 255))
            name = followed_car.get('name', f"Car {followed_car_index}")
            
            if position is not None:
                # Add circle indicator for the followed car
                try:
                    screen_x, screen_y = self.camera.world_to_screen(position)
                    # Draw a small circle indicator for the followed car
                    pygame.draw.circle(self.window, (255, 255, 255), (int(screen_x), int(screen_y)), 8, 2)
                except:
                    pass  # Ignore errors in highlighting
                
                # Render followed car's name above the car (always on top)
                self._render_car_name(position, name, color, is_followed=True)
    
    def _render_car_name(self, position: tuple, name: str, color: tuple, is_followed: bool = False):
        """Render car name above the car
        
        Args:
            position: Car position in world coordinates (x, y)
            name: Name to display
            color: Car color for text background
            is_followed: Whether this is the currently followed car
        """
        try:
            # Convert world position to screen coordinates
            screen_x, screen_y = self.camera.world_to_screen(position)
            
            # Create font if not already created
            if not hasattr(self, '_name_font'):
                self._name_font = pygame.font.Font(None, 20)  # Small font for names
            
            # Choose text color based on car color brightness
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            
            # Add highlighting for followed car
            if is_followed:
                text_color = (255, 255, 0)  # Yellow for followed car
                name = f"* {name} *"  # Add arrows for emphasis
            
            # Render text
            text_surface = self._name_font.render(name, True, text_color)
            text_rect = text_surface.get_rect()
            
            # Position text above the car (offset by car height + margin)
            text_x = int(screen_x - text_rect.width // 2)
            text_y = int(screen_y - 25)  # 25 pixels above car center
            
            # Draw semi-transparent background for readability
            bg_rect = pygame.Rect(text_x - 2, text_y - 2, text_rect.width + 4, text_rect.height + 4)
            bg_color = (*color, 128)  # Semi-transparent car color
            
            # Create a surface for the background with alpha
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill(bg_color)
            self.window.blit(bg_surface, (bg_rect.x, bg_rect.y))
            
            # Draw the text
            self.window.blit(text_surface, (text_x, text_y))
            
        except Exception as e:
            # Silently ignore rendering errors (e.g., when car is off-screen)
            pass
    
    def _render_action_bars(self, current_action, car_name=None):
        """Render action input bars at the top center of screen
        
        Args:
            current_action: numpy array [throttle, brake, steering] with values in range [0,1], [0,1], [-1,1]
            car_name: Optional name of the car whose actions are being displayed
        """
        if not self.window:
            return
            
        # Extract action values
        throttle, brake, steering = current_action[0], current_action[1], current_action[2]
        
        # Create action bar font (smaller than main font)
        try:
            action_font = pygame.font.Font(None, ACTION_BAR_LABEL_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if action font fails
            if not self.font:
                return
            action_font = self.font
        
        # Calculate total height needed for all bars
        total_height = 3 * ACTION_BAR_HEIGHT + 2 * ACTION_BAR_SPACING
        
        # Calculate starting position (centered horizontally, positioned below FPS)
        start_x = (self.window_size[0] - ACTION_BAR_WIDTH) // 2
        start_y = ACTION_BAR_TOP_MARGIN
        
        # Render car name header if provided
        if car_name:
            try:
                # Render car name above the action bars
                header_text = f"Controls: {car_name}"
                header_surface = action_font.render(header_text, True, (255, 255, 255))
                header_rect = header_surface.get_rect()
                header_x = (self.window_size[0] - header_rect.width) // 2
                header_y = start_y - 25  # Position above the bars
                
                # Draw semi-transparent background for readability
                bg_rect = pygame.Rect(header_x - 5, header_y - 2, header_rect.width + 10, header_rect.height + 4)
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, 128))  # Semi-transparent black
                self.window.blit(bg_surface, (bg_rect.x, bg_rect.y))
                
                # Draw the header text
                self.window.blit(header_surface, (header_x, header_y))
            except:
                pass  # Ignore rendering errors
        
        # Bar definitions: (label, value, fill_color, bg_color, is_bidirectional)
        bars = [
            ("Throttle", throttle, THROTTLE_BAR_COLOR, THROTTLE_BAR_BG_COLOR, False),
            ("Brake", brake, BRAKE_BAR_COLOR, BRAKE_BAR_BG_COLOR, False),
            ("Steering", steering, STEERING_BAR_COLOR, STEERING_BAR_BG_COLOR, True)
        ]
        
        # Render each bar
        for i, (label, value, fill_color, bg_color, is_bidirectional) in enumerate(bars):
            bar_y = start_y + i * (ACTION_BAR_HEIGHT + ACTION_BAR_SPACING)
            
            # Draw background
            bg_rect = pygame.Rect(start_x, bar_y, ACTION_BAR_WIDTH, ACTION_BAR_HEIGHT)
            pygame.draw.rect(self.window, bg_color, bg_rect)
            pygame.draw.rect(self.window, ACTION_BAR_BORDER_COLOR, bg_rect, 1)
            
            # Calculate fill width based on value and type
            inner_width = ACTION_BAR_WIDTH - 2 * ACTION_BAR_PADDING
            inner_x = start_x + ACTION_BAR_PADDING
            inner_y = bar_y + ACTION_BAR_PADDING
            inner_height = ACTION_BAR_HEIGHT - 2 * ACTION_BAR_PADDING
            
            if is_bidirectional:
                # Steering bar: center-outward fill (-1 to 1)
                center_x = inner_x + inner_width // 2
                if steering >= 0:
                    # Right side fill
                    fill_width = int(abs(steering) * (inner_width // 2))
                    fill_rect = pygame.Rect(center_x, inner_y, fill_width, inner_height)
                else:
                    # Left side fill  
                    fill_width = int(abs(steering) * (inner_width // 2))
                    fill_rect = pygame.Rect(center_x - fill_width, inner_y, fill_width, inner_height)
                    
                # Draw center line
                pygame.draw.line(self.window, ACTION_BAR_BORDER_COLOR, 
                               (center_x, inner_y), (center_x, inner_y + inner_height), 1)
            else:
                # Unidirectional bars: left-to-right fill (0 to 1)
                fill_width = int(max(0, min(1, value)) * inner_width)
                fill_rect = pygame.Rect(inner_x, inner_y, fill_width, inner_height)
            
            # Draw the fill
            if fill_rect.width > 0:
                pygame.draw.rect(self.window, fill_color, fill_rect)
            
            # Draw label and value text  
            label_text = action_font.render(f"{label}: {value:.2f}", True, ACTION_BAR_TEXT_COLOR)
            label_rect = label_text.get_rect()
            
            # Position text to the right of the bar
            text_x = start_x + ACTION_BAR_WIDTH + 10
            text_y = bar_y + (ACTION_BAR_HEIGHT - label_rect.height) // 2
            
            self.window.blit(label_text, (text_x, text_y))
    
    def _render_lap_times(self, lap_timing_info):
        """Render lap timing information at the bottom center of the screen
        
        Args:
            lap_timing_info: Dictionary containing timing information from LapTimer
        """
        if not self.window or not lap_timing_info:
            return
        
        # Create lap timer font
        try:
            lap_font = pygame.font.Font(None, LAP_TIMER_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if lap font fails
            if not self.font:
                return
            lap_font = self.font
        
        # Extract timing information
        current_time = lap_timing_info.get('formatted_current', '--:--.---')
        last_time = lap_timing_info.get('formatted_last', '--:--.---') 
        best_time = lap_timing_info.get('formatted_best', '--:--.---')
        is_timing = lap_timing_info.get('is_timing', False)
        car_name = lap_timing_info.get('car_name', 'Car')
        
        # Only show current time if we're actually timing
        display_current = current_time if is_timing else '--:--.---'
        
        # Prepare text lines with labels and times (include car name in header)
        lines = [
            (f"{car_name}", "", (255, 255, 255)),  # Car name as header
            ("Current:", display_current, LAP_TIMER_CURRENT_COLOR),
            ("Last:", last_time, LAP_TIMER_LAST_COLOR),
            ("Best:", best_time, LAP_TIMER_BEST_COLOR)
        ]
        
        # Calculate total height needed
        line_height = lap_font.get_height()
        total_height = len(lines) * line_height + (len(lines) - 1) * (LAP_TIMER_LINE_SPACING - line_height)
        
        # Calculate starting position (centered horizontally, positioned from bottom)
        start_y = self.window_size[1] - LAP_TIMER_BOTTOM_MARGIN - total_height
        
        # Create background surface with alpha
        bg_width = max(lap_font.size(f"{label} {time_str}")[0] for label, time_str, _ in lines) + 20
        bg_height = total_height + 20
        bg_x = (self.window_size[0] - bg_width) // 2
        bg_y = start_y - 10
        
        # Create semi-transparent background
        bg_surface = pygame.Surface((bg_width, bg_height))
        bg_surface.set_alpha(LAP_TIMER_BG_ALPHA)
        bg_surface.fill(LAP_TIMER_BG_COLOR)
        self.window.blit(bg_surface, (bg_x, bg_y))
        
        # Render each line
        for i, (label, time_str, color) in enumerate(lines):
            line_y = start_y + i * LAP_TIMER_LINE_SPACING
            
            # Render label and time as one string for consistent alignment
            full_text = f"{label:<8} {time_str}"
            text_surface = lap_font.render(full_text, True, color)
            text_rect = text_surface.get_rect()
            text_rect.centerx = self.window_size[0] // 2
            text_rect.y = line_y
            
            self.window.blit(text_surface, text_rect)
    
    def _render_race_tables(self, race_positions_data, best_lap_times_data):
        """Render race position and best lap times tables in the center of the screen
        
        Args:
            race_positions_data: List of tuples (car_index, car_name, total_progress, completed_laps) sorted by position
            best_lap_times_data: List of tuples (car_index, car_name, best_time) sorted by time
        """
        if not self.window:
            return
        
        # Create race tables font
        try:
            tables_font = pygame.font.Font(None, RACE_TABLES_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if tables font fails
            if not self.font:
                return
            tables_font = self.font
        
        # Calculate center position for both tables
        screen_center_x = self.window_size[0] // 2
        screen_center_y = self.window_size[1] // 2
        
        # Calculate table positions (left and right of center)
        left_table_x = screen_center_x - RACE_TABLES_WIDTH - RACE_TABLES_SPACING // 2
        right_table_x = screen_center_x + RACE_TABLES_SPACING // 2
        table_y = screen_center_y - RACE_TABLES_HEIGHT // 2
        
        # Render positions table (left)
        if race_positions_data:
            self._render_positions_table(tables_font, left_table_x, table_y, race_positions_data)
        
        # Render best lap times table (right)  
        if best_lap_times_data:
            self._render_lap_times_table(tables_font, right_table_x, table_y, best_lap_times_data)
    
    def _render_positions_table(self, font, x, y, race_positions_data):
        """Render the race positions table (left table)"""
        # Draw background
        bg_surface = pygame.Surface((RACE_TABLES_WIDTH, RACE_TABLES_HEIGHT))
        bg_surface.set_alpha(RACE_TABLES_BG_ALPHA)
        bg_surface.fill(RACE_TABLES_BG_COLOR)
        self.window.blit(bg_surface, (x, y))
        
        # Draw header
        header_text = "POSITIONS"
        header_surface = font.render(header_text, True, RACE_TABLES_HEADER_COLOR)
        header_rect = header_surface.get_rect()
        header_x = x + (RACE_TABLES_WIDTH - header_rect.width) // 2
        header_y = y + RACE_TABLES_PADDING
        self.window.blit(header_surface, (header_x, header_y))
        
        # Draw position entries
        current_y = header_y + RACE_TABLES_LINE_HEIGHT + RACE_TABLES_PADDING
        
        for position, position_data in enumerate(race_positions_data, 1):
            if current_y + RACE_TABLES_LINE_HEIGHT > y + RACE_TABLES_HEIGHT - RACE_TABLES_PADDING:
                break  # Don't draw outside table bounds
            
            # Handle both old format (3 items) and new format (4 items) for backward compatibility
            if len(position_data) >= 4:
                car_index, car_name, total_progress, completed_laps = position_data[:4]
                # Format: "1st CarName (L2)" where L2 means 2 laps completed
                position_suffix = self._get_position_suffix(position)
                if completed_laps > 0:
                    position_text = f"{position}{position_suffix} {car_name} (L{completed_laps})"
                else:
                    position_text = f"{position}{position_suffix} {car_name}"
            else:
                # Legacy format: (car_index, car_name, progress)
                car_index, car_name, progress = position_data
                position_suffix = self._get_position_suffix(position)
                position_text = f"{position}{position_suffix} {car_name}"
            
            # Truncate if too long
            if len(position_text) > 18:  # Adjust based on table width
                truncated_name = car_name[:8] + "..."
                if len(position_data) >= 4 and completed_laps > 0:
                    position_text = f"{position}{position_suffix} {truncated_name} (L{completed_laps})"
                else:
                    position_text = f"{position}{position_suffix} {truncated_name}"
            
            text_surface = font.render(position_text, True, RACE_TABLES_POSITION_COLOR)
            text_x = x + RACE_TABLES_PADDING
            self.window.blit(text_surface, (text_x, current_y))
            
            current_y += RACE_TABLES_LINE_HEIGHT
    
    def _render_lap_times_table(self, font, x, y, best_lap_times_data):
        """Render the best lap times table (right table)"""
        # Draw background
        bg_surface = pygame.Surface((RACE_TABLES_WIDTH, RACE_TABLES_HEIGHT))
        bg_surface.set_alpha(RACE_TABLES_BG_ALPHA)
        bg_surface.fill(RACE_TABLES_BG_COLOR)
        self.window.blit(bg_surface, (x, y))
        
        # Draw header
        header_text = "BEST LAP TIMES"
        header_surface = font.render(header_text, True, RACE_TABLES_HEADER_COLOR)
        header_rect = header_surface.get_rect()
        header_x = x + (RACE_TABLES_WIDTH - header_rect.width) // 2
        header_y = y + RACE_TABLES_PADDING
        self.window.blit(header_surface, (header_x, header_y))
        
        # Draw lap time entries
        current_y = header_y + RACE_TABLES_LINE_HEIGHT + RACE_TABLES_PADDING
        
        for car_index, car_name, best_time in best_lap_times_data:
            if current_y + RACE_TABLES_LINE_HEIGHT > y + RACE_TABLES_HEIGHT - RACE_TABLES_PADDING:
                break  # Don't draw outside table bounds
            
            # Format lap time using LapTimer format
            from .lap_timer import LapTimer
            formatted_time = LapTimer.format_time(best_time)
            
            # Format: "CarName MM:SS.mmm"
            # Truncate car name if necessary to fit time
            max_name_length = 8  # Adjust based on table width
            display_name = car_name[:max_name_length] + "..." if len(car_name) > max_name_length else car_name
            time_text = f"{display_name:<{max_name_length}} {formatted_time}"
            
            text_surface = font.render(time_text, True, RACE_TABLES_LAP_TIME_COLOR)
            text_x = x + RACE_TABLES_PADDING
            self.window.blit(text_surface, (text_x, current_y))
            
            current_y += RACE_TABLES_LINE_HEIGHT
    
    def _get_position_suffix(self, position):
        """Get ordinal suffix for position (1st, 2nd, 3rd, 4th, etc.)"""
        if 10 <= position % 100 <= 20:  # Special case for 11th, 12th, 13th
            return "th"
        else:
            suffix_map = {1: "st", 2: "nd", 3: "rd"}
            return suffix_map.get(position % 10, "th")
    
    def _render_reward(self, reward_info):
        """Render reward information in the bottom right corner of the screen
        
        Args:
            reward_info: Dictionary containing reward information
        """
        if not self.window or not reward_info:
            return
        
        # Create reward font (same size as lap timer)
        try:
            reward_font = pygame.font.Font(None, LAP_TIMER_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if reward font fails
            if not self.font:
                return
            reward_font = self.font
        
        # Extract reward values
        current_reward = reward_info.get('current_reward', 0.0)
        cumulative_reward = reward_info.get('cumulative_reward', 0.0)
        
        # Format reward texts
        lines = [
            f"Reward: {current_reward:+.4f}",
            f"Total: {cumulative_reward:+.2f}"
        ]
        
        # Calculate dimensions
        line_height = reward_font.get_height()
        max_width = max(reward_font.size(line)[0] for line in lines)
        total_height = len(lines) * line_height + (len(lines) - 1) * 5  # 5 pixels spacing
        
        # Position in bottom right corner
        start_x = self.window_size[0] - max_width - 30  # 30 pixels from right edge
        start_y = self.window_size[1] - LAP_TIMER_BOTTOM_MARGIN - total_height
        
        # Create semi-transparent background
        bg_padding = 10
        bg_rect = pygame.Rect(start_x - bg_padding, start_y - bg_padding, 
                              max_width + bg_padding * 2, total_height + bg_padding * 2)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(LAP_TIMER_BG_ALPHA)
        bg_surface.fill(LAP_TIMER_BG_COLOR)
        self.window.blit(bg_surface, bg_rect)
        
        # Draw each line of text
        for i, line in enumerate(lines):
            text_color = LAP_TIMER_CURRENT_COLOR if i == 0 else LAP_TIMER_BEST_COLOR
            text_surface = reward_font.render(line, True, text_color)
            text_y = start_y + i * (line_height + 5)
            self.window.blit(text_surface, (start_x, text_y))
    
    def _render_countdown_clock(self, countdown_info):
        """Render countdown clock in the top left corner of the screen
        
        Args:
            countdown_info: Dictionary containing timing information
        """
        if not self.window or not countdown_info:
            return
        
        # Extract timing values
        current_time = countdown_info.get('current_time', 0.0)
        time_limit = countdown_info.get('time_limit', 600.0)
        reset_on_lap = countdown_info.get('reset_on_lap', False)
        
        # Calculate remaining time
        remaining_time = max(0.0, time_limit - current_time)
        
        # Format time as MM:SS
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        
        # Determine text color based on remaining time
        if remaining_time > COUNTDOWN_WARNING_THRESHOLD:
            text_color = COUNTDOWN_TIME_SAFE_COLOR
        elif remaining_time > COUNTDOWN_DANGER_THRESHOLD:
            text_color = COUNTDOWN_TIME_WARNING_COLOR
        else:
            text_color = COUNTDOWN_TIME_DANGER_COLOR
        
        # Create countdown font
        try:
            countdown_font = pygame.font.Font(None, COUNTDOWN_FONT_SIZE)
        except pygame.error:
            # Fallback to main font if countdown font fails
            if not self.font:
                return
            countdown_font = self.font
        
        # Render text and get dimensions
        text_surface = countdown_font.render(time_text, True, text_color)
        text_width, text_height = text_surface.get_size()
        
        # Calculate position (top left corner with margins)
        text_x = COUNTDOWN_LEFT_MARGIN
        text_y = COUNTDOWN_TOP_MARGIN
        
        # Create semi-transparent background
        bg_width = text_width + COUNTDOWN_BG_PADDING * 2
        bg_height = text_height + COUNTDOWN_BG_PADDING * 2
        bg_rect = pygame.Rect(text_x - COUNTDOWN_BG_PADDING, text_y - COUNTDOWN_BG_PADDING, 
                              bg_width, bg_height)
        bg_surface = pygame.Surface((bg_width, bg_height))
        bg_surface.set_alpha(COUNTDOWN_BG_ALPHA)
        bg_surface.fill(COUNTDOWN_BG_COLOR)
        self.window.blit(bg_surface, bg_rect)
        
        # Draw countdown text
        self.window.blit(text_surface, (text_x, text_y))