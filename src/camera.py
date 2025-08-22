from typing import Tuple, Optional
from .track_generator import Track
from .constants import (
    CAMERA_MARGIN_FACTOR, 
    MIN_ZOOM_FACTOR, 
    MAX_ZOOM_FACTOR,
    DEFAULT_ZOOM,
    DEFAULT_TRACK_WIDTH_FALLBACK,
    DEFAULT_TRACK_HEIGHT_FALLBACK,
    PIXELS_PER_METER,
    CAR_FOLLOW_ZOOM_FACTOR,
    CAMERA_MODE_TRACK_VIEW,
    CAMERA_MODE_CAR_FOLLOW
)


class Camera:
    def __init__(self, window_size: Tuple[int, int]):
        self.window_size = window_size
        self.track = None
        self.zoom = DEFAULT_ZOOM
        self.offset = (0.0, 0.0)  # Camera offset in world coordinates
        self.pixels_per_meter = 1.0  # Dynamic scale factor
        
        # Camera mode state
        self.camera_mode = CAMERA_MODE_TRACK_VIEW  # Default to track view
        self.car_follow_zoom = CAR_FOLLOW_ZOOM_FACTOR  # Configurable zoom for car follow mode
        
    def set_window_size(self, window_size: Tuple[int, int]):
        """Update window size and recalculate camera parameters"""
        self.window_size = window_size
        if self.track:
            self.calculate_auto_fit()
    
    def set_track(self, track: Optional[Track]):
        """Set the track and calculate optimal camera parameters"""
        self.track = track
        if track:
            self.calculate_auto_fit()
        else:
            # Reset to default view
            self.zoom = DEFAULT_ZOOM
            self.offset = (0.0, 0.0)
            self.pixels_per_meter = PIXELS_PER_METER
    
    def calculate_auto_fit(self):
        """Calculate camera parameters to fit the entire track on screen"""
        if not self.track or not self.track.segments:
            return
        
        # Get track bounds in world coordinates
        bounds = self.track.get_track_bounds()
        min_pos, max_pos = bounds
        
        # Calculate track dimensions
        track_width = max_pos[0] - min_pos[0]
        track_height = max_pos[1] - min_pos[1]
        
        # Add margins (as percentage of track size)
        margin_x = track_width * CAMERA_MARGIN_FACTOR
        margin_y = track_height * CAMERA_MARGIN_FACTOR
        
        total_width = track_width + 2 * margin_x
        total_height = track_height + 2 * margin_y
        
        # Handle zero-dimension cases
        if total_width <= 0:
            total_width = DEFAULT_TRACK_WIDTH_FALLBACK
        if total_height <= 0:
            total_height = DEFAULT_TRACK_HEIGHT_FALLBACK
        
        # Calculate pixels per meter to fit track in window
        scale_x = self.window_size[0] / total_width
        scale_y = self.window_size[1] / total_height
        
        # Use the smaller scale to ensure track fits in both dimensions
        self.pixels_per_meter = min(scale_x, scale_y)
        
        # Clamp to reasonable limits
        self.pixels_per_meter = max(MIN_ZOOM_FACTOR, 
                                   min(MAX_ZOOM_FACTOR, self.pixels_per_meter))
        
        # Calculate camera center to center the track
        track_center_x = (min_pos[0] + max_pos[0]) / 2
        track_center_y = (min_pos[1] + max_pos[1]) / 2
        
        # Calculate offset to center the track in the window
        # Note: Y-axis offset accounts for inverted coordinate system in world_to_screen
        self.offset = (
            self.window_size[0] / 2 - track_center_x * self.pixels_per_meter,
            self.window_size[1] / 2 + track_center_y * self.pixels_per_meter
        )
    
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(world_pos[0] * self.pixels_per_meter + self.offset[0])
        screen_y = int(-world_pos[1] * self.pixels_per_meter + self.offset[1])
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_pos[0] - self.offset[0]) / self.pixels_per_meter
        world_y = -(screen_pos[1] - self.offset[1]) / self.pixels_per_meter
        return (world_x, world_y)
    
    def get_scale_factor(self) -> float:
        """Get the current pixels per meter scale factor"""
        return self.pixels_per_meter
    
    def toggle_camera_mode(self):
        """Toggle between track view and car follow modes"""
        if self.camera_mode == CAMERA_MODE_TRACK_VIEW:
            self.camera_mode = CAMERA_MODE_CAR_FOLLOW
        else:
            self.camera_mode = CAMERA_MODE_TRACK_VIEW
            # Return to track view by recalculating auto-fit
            if self.track:
                self.calculate_auto_fit()
    
    def set_car_follow_mode(self, car_position: Tuple[float, float]):
        """Set camera to follow the car at the specified position
        
        Args:
            car_position: (x, y) position of car in world coordinates
        """
        self.camera_mode = CAMERA_MODE_CAR_FOLLOW
        self.update_car_follow_camera(car_position)
    
    def set_track_view_mode(self):
        """Set camera to show the entire track (auto-fit mode)"""
        self.camera_mode = CAMERA_MODE_TRACK_VIEW
        if self.track:
            self.calculate_auto_fit()
    
    def update_car_follow_camera(self, car_position: Tuple[float, float]):
        """Update camera position and zoom to follow the car
        
        Args:
            car_position: (x, y) position of car in world coordinates
        """
        if self.camera_mode != CAMERA_MODE_CAR_FOLLOW:
            return
            
        # Set zoom level for car following
        self.pixels_per_meter = self.car_follow_zoom
        
        # Clamp to reasonable limits
        self.pixels_per_meter = max(MIN_ZOOM_FACTOR, 
                                   min(MAX_ZOOM_FACTOR, self.pixels_per_meter))
        
        # Center camera on car position
        # Calculate offset to center the car in the window
        # Note: Y-axis offset accounts for inverted coordinate system in world_to_screen
        self.offset = (
            self.window_size[0] / 2 - car_position[0] * self.pixels_per_meter,
            self.window_size[1] / 2 + car_position[1] * self.pixels_per_meter
        )
    
    def get_camera_mode(self) -> str:
        """Get the current camera mode"""
        return self.camera_mode