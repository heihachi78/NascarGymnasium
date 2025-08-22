"""
Debug information renderer for enhanced debug visualization.

This module provides the DebugInfoRenderer class which renders detailed
debug information including tyre data, vectors, and car physics data.
"""

import pygame
import math
from typing import Dict, Tuple, Optional
from .constants import (
    DEBUG_INFO_PANEL_X,
    DEBUG_INFO_PANEL_Y,
    DEBUG_INFO_PANEL_WIDTH,
    DEBUG_INFO_PANEL_HEIGHT,
    DEBUG_INFO_PANEL_BG_COLOR,
    DEBUG_INFO_PANEL_BG_ALPHA,
    DEBUG_INFO_PANEL_TEXT_COLOR,
    DEBUG_INFO_PANEL_PADDING,
    DEBUG_INFO_TEXT_FONT_SIZE,
    DEBUG_INFO_LINE_HEIGHT,
    DEBUG_VELOCITY_VECTOR_COLOR,
    DEBUG_ACCELERATION_VECTOR_COLOR,
    DEBUG_STEERING_INPUT_VECTOR_COLOR,
    DEBUG_STEERING_ACTUAL_VECTOR_COLOR,
    DEBUG_VECTOR_VELOCITY_SCALE_FACTOR,
    DEBUG_VECTOR_ACCELERATION_SCALE_FACTOR,
    DEBUG_VECTOR_STEERING_SCALE_FACTOR,
    DEBUG_VECTOR_MAX_LENGTH,
    DEBUG_VECTOR_ARROW_WIDTH,
    DEBUG_VECTOR_ARROW_HEAD_SIZE,
    DEBUG_VECTOR_MIN_LENGTH,
    DEBUG_SENSOR_VECTOR_COLOR,
    DEBUG_SENSOR_VECTOR_SCALE,
    DEBUG_SENSOR_TEXT_COLOR,
    TYRE_IDEAL_TEMPERATURE_MIN,
    TYRE_IDEAL_TEMPERATURE_MAX
)


class DebugInfoRenderer:
    """Renders detailed debug information for car simulation"""
    
    def __init__(self, camera):
        """
        Initialize debug info renderer.
        
        Args:
            camera: Camera object for world-to-screen coordinate conversion
        """
        self.camera = camera
        self.font = None
    
    def _ensure_font(self):
        """Ensure font is initialized with monospace font"""
        if self.font is None:
            # Try to load a monospace font, fall back to system default if not available
            try:
                # Try common monospace fonts
                monospace_fonts = [
                    'Consolas',      # Windows
                    'Menlo',         # macOS
                    'DejaVu Sans Mono',  # Linux
                    'Liberation Mono',   # Linux
                    'Courier New',   # Cross-platform
                    'monospace'      # Generic fallback
                ]
                
                font_loaded = False
                for font_name in monospace_fonts:
                    try:
                        self.font = pygame.font.SysFont(font_name, DEBUG_INFO_TEXT_FONT_SIZE)
                        if self.font.get_bold():  # Test if font loaded successfully
                            font_loaded = True
                            break
                        self.font = pygame.font.SysFont(font_name, DEBUG_INFO_TEXT_FONT_SIZE)
                        font_loaded = True
                        break
                    except:
                        continue
                
                # If no system font worked, use pygame's default monospace
                if not font_loaded:
                    self.font = pygame.font.Font(None, DEBUG_INFO_TEXT_FONT_SIZE)
                    
            except Exception:
                # Ultimate fallback
                self.font = pygame.font.Font(None, DEBUG_INFO_TEXT_FONT_SIZE)
    
    def render_debug_info(self, window: pygame.Surface, debug_data: dict) -> bool:
        """
        Render complete debug information.
        
        Args:
            window: pygame surface to draw on
            debug_data: Debug data dictionary from CarEnv.get_debug_data()
            
        Returns:
            True if rendering succeeded, False otherwise
        """
        if not debug_data:
            return False
            
        try:
            self._ensure_font()
            
            # Render info panel with tyre data and physics info
            self._render_info_panel(window, debug_data)
            
            # Render velocity and acceleration vectors
            self._render_vectors(window, debug_data)
            
            # Render sensor vectors
            self._render_sensor_vectors(window, debug_data)
            
            return True
            
        except Exception as e:
            return False
    
    def _render_info_panel(self, window: pygame.Surface, debug_data: dict) -> None:
        """Render the debug info panel with text information"""
        # Create semi-transparent background
        bg_surface = pygame.Surface((DEBUG_INFO_PANEL_WIDTH, DEBUG_INFO_PANEL_HEIGHT))
        bg_surface.set_alpha(DEBUG_INFO_PANEL_BG_ALPHA)
        bg_surface.fill(DEBUG_INFO_PANEL_BG_COLOR)
        window.blit(bg_surface, (DEBUG_INFO_PANEL_X, DEBUG_INFO_PANEL_Y))
        
        # Prepare text lines
        lines = self._prepare_text_lines(debug_data)
        
        # Render text lines
        y_offset = DEBUG_INFO_PANEL_Y + DEBUG_INFO_PANEL_PADDING
        for line in lines:
            text_surface = self.font.render(line, True, DEBUG_INFO_PANEL_TEXT_COLOR)
            window.blit(text_surface, (DEBUG_INFO_PANEL_X + DEBUG_INFO_PANEL_PADDING, y_offset))
            y_offset += DEBUG_INFO_LINE_HEIGHT
    
    def _prepare_text_lines(self, debug_data: dict) -> list:
        """Prepare text lines for debug info panel"""
        lines = []
        
        # Header section
        lines.append("VEHICLE TELEMETRY")
        lines.append("")
        
        # Physics data section
        drag_force = debug_data.get('drag_force', 0.0)
        velocity_vector = debug_data.get('velocity_vector', (0, 0))
        acceleration_vector = debug_data.get('acceleration_vector', (0, 0))
        
        # Calculate speed and acceleration magnitudes
        speed_ms = (velocity_vector[0]**2 + velocity_vector[1]**2)**0.5
        speed_kmh = speed_ms * 3.6
        accel_mag = (acceleration_vector[0]**2 + acceleration_vector[1]**2)**0.5
        
        lines.append("PHYSICS DATA")
        lines.append(f"Speed      : {speed_kmh:8.1f} km/h")
        lines.append(f"Drag Force : {drag_force:8.1f} N")
        lines.append(f"Accel Mag  : {accel_mag:8.2f} m/s2")
        lines.append("")
        
        # Tyre data section
        tyre_data = debug_data.get('tyre_data', {})
        loads = tyre_data.get('loads', {})
        temps = tyre_data.get('temperatures', {})
        pressures = tyre_data.get('pressures', {})
        wear = tyre_data.get('wear', {})
        
        lines.append("TYRE DATA")
        lines.append("Position   Load    Temp    Press   Wear")
        lines.append("           (N)     (C)     (PSI)   (%)")
        lines.append("----------------------------------------") 
        
        tyre_order = ['front_left', 'front_right', 'rear_left', 'rear_right']
        tyre_labels = ['Front L', 'Front R', 'Rear L ', 'Rear R ']
        
        for pos, label in zip(tyre_order, tyre_labels):
            load = loads.get(pos, 0.0)
            temp = temps.get(pos, 0.0)
            pressure = pressures.get(pos, 0.0)
            wear_val = wear.get(pos, 0.0)
            
            # Add status indicators for critical values
            temp_indicator = self._get_temperature_indicator(temp)
            pressure_indicator = self._get_pressure_indicator(pressure)
            wear_indicator = self._get_wear_indicator(wear_val)
            
            # Perfect monospace alignment
            lines.append(f"{label:<8} {load:6.0f}  {temp:5.1f} {temp_indicator:<4} {pressure:5.1f} {pressure_indicator:<4} {wear_val:4.1f} {wear_indicator}")
        
        # Sensor data section
        sensor_data = debug_data.get('sensor_data', {})
        sensor_distances = sensor_data.get('distances', [])
        
        if len(sensor_distances) >= 8:
            lines.append("")
            lines.append("SENSOR DATA (8-DIR)")
            lines.append("Direction  Distance")
            lines.append("           (meters) ")
            lines.append("------------------")
            
            direction_labels = ['Front   ', 'Fr-Right', 'Right   ', 'Bk-Right', 
                              'Back    ', 'Bk-Left ', 'Left    ', 'Fr-Left ']
            
            for label, distance in zip(direction_labels, sensor_distances):
                lines.append(f"{label:<8} {distance:8.1f}")
        
        return lines
    
    def _get_temperature_indicator(self, temp: float) -> str:
        """Get temperature status indicator"""
        # Use the actual constants for ideal temperature range
        if TYRE_IDEAL_TEMPERATURE_MIN <= temp <= TYRE_IDEAL_TEMPERATURE_MAX:  # Ideal range (85-105°C)
            return " OPT"
        elif (TYRE_IDEAL_TEMPERATURE_MIN - 10) <= temp <= (TYRE_IDEAL_TEMPERATURE_MAX + 10):  # Warning range (75-115°C)
            return " WARN"
        else:  # Critical range (outside warning range)
            return " CRIT"
    
    def _get_pressure_indicator(self, pressure: float) -> str:
        """Get pressure status indicator"""
        if 30.0 <= pressure <= 34.0:  # Optimal range
            return " OPT"
        elif 25.0 <= pressure <= 40.0:  # Warning range
            return " WARN"
        else:  # Critical range
            return " CRIT"
    
    def _get_wear_indicator(self, wear: float) -> str:
        """Get wear status indicator"""
        if wear <= 20.0:  # Good condition
            return " GOOD"
        elif wear <= 50.0:  # Moderate wear
            return " MODERATE"
        else:  # High wear
            return " HIGH WEAR"
    
    def _render_vectors(self, window: pygame.Surface, debug_data: dict) -> None:
        """Render velocity, acceleration, and steering vectors"""
        car_position = debug_data.get('car_position', (0, 0))
        car_angle = debug_data.get('car_angle', 0.0)
        
        # Convert car position to screen coordinates
        screen_pos = self.camera.world_to_screen(car_position)
        
        # Render velocity vector
        velocity_vector = debug_data.get('velocity_vector', (0, 0))
        self._render_vector(window, screen_pos, velocity_vector, DEBUG_VELOCITY_VECTOR_COLOR, 
                          scale_factor=DEBUG_VECTOR_VELOCITY_SCALE_FACTOR)
        
        # Render acceleration vector
        acceleration_vector = debug_data.get('acceleration_vector', (0, 0))
        self._render_vector(window, screen_pos, acceleration_vector, DEBUG_ACCELERATION_VECTOR_COLOR, 
                          scale_factor=DEBUG_VECTOR_ACCELERATION_SCALE_FACTOR)
        
        # Render steering vectors
        steering_vectors = debug_data.get('steering_vectors', {})
        input_vector = steering_vectors.get('input', (0, 0))
        actual_vector = steering_vectors.get('actual', (0, 0))
        
        self._render_vector(window, screen_pos, input_vector, DEBUG_STEERING_INPUT_VECTOR_COLOR, scale_factor=DEBUG_VECTOR_STEERING_SCALE_FACTOR)
        self._render_vector(window, screen_pos, actual_vector, DEBUG_STEERING_ACTUAL_VECTOR_COLOR, scale_factor=DEBUG_VECTOR_STEERING_SCALE_FACTOR)
    
    def _render_vector(self, window: pygame.Surface, start_pos: Tuple[int, int], 
                      vector: Tuple[float, float], color: Tuple[int, int, int],
                      scale_factor: float = 1.0) -> None:
        """Render a single vector as an arrow"""
        if not vector or (vector[0] == 0 and vector[1] == 0):
            return
        
        # Calculate vector magnitude and direction
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
        if magnitude < 0.01:  # Skip very small vectors
            return
        
        # Scale vector for display
        display_length = magnitude * scale_factor
        if display_length < DEBUG_VECTOR_MIN_LENGTH:
            return
        
        # Clamp to maximum length to prevent screen overflow
        if display_length > DEBUG_VECTOR_MAX_LENGTH:
            display_length = DEBUG_VECTOR_MAX_LENGTH
        
        # Calculate end position
        angle = math.atan2(-vector[1], vector[0])  # Negative Y for screen coordinates
        end_x = start_pos[0] + display_length * math.cos(angle)
        end_y = start_pos[1] + display_length * math.sin(angle)
        end_pos = (int(end_x), int(end_y))
        
        # Draw vector line
        pygame.draw.line(window, color, start_pos, end_pos, DEBUG_VECTOR_ARROW_WIDTH)
        
        # Draw arrow head
        self._draw_arrow_head(window, end_pos, angle, color)
    
    def _draw_arrow_head(self, window: pygame.Surface, tip_pos: Tuple[int, int], 
                        angle: float, color: Tuple[int, int, int]) -> None:
        """Draw arrow head at the tip of a vector"""
        head_size = DEBUG_VECTOR_ARROW_HEAD_SIZE
        
        # Calculate arrow head points
        head_angle1 = angle + math.pi - math.pi/6  # 30 degrees back
        head_angle2 = angle + math.pi + math.pi/6  # 30 degrees back
        
        head_point1 = (
            int(tip_pos[0] + head_size * math.cos(head_angle1)),
            int(tip_pos[1] + head_size * math.sin(head_angle1))
        )
        head_point2 = (
            int(tip_pos[0] + head_size * math.cos(head_angle2)),
            int(tip_pos[1] + head_size * math.sin(head_angle2))
        )
        
        # Draw arrow head lines
        pygame.draw.line(window, color, tip_pos, head_point1, DEBUG_VECTOR_ARROW_WIDTH)
        pygame.draw.line(window, color, tip_pos, head_point2, DEBUG_VECTOR_ARROW_WIDTH)
    
    def _render_sensor_vectors(self, window: pygame.Surface, debug_data: dict) -> None:
        """Render distance sensor vectors"""
        car_position = debug_data.get('car_position', (0, 0))
        sensor_data = debug_data.get('sensor_data', {})
        sensor_distances = sensor_data.get('distances', [])
        sensor_angles = sensor_data.get('angles', [])
        
        if len(sensor_distances) < 8 or len(sensor_angles) < 8:
            return
            
        # Convert car position to screen coordinates
        screen_pos = self.camera.world_to_screen(car_position)
        
        # Render each sensor ray
        for i in range(8):
            distance = sensor_distances[i]
            angle = sensor_angles[i]
            
            if distance <= 0:
                continue
                
            # Calculate sensor end point in world coordinates
            end_world_x = car_position[0] + distance * math.cos(angle)
            end_world_y = car_position[1] + distance * math.sin(angle)
            end_screen_pos = self.camera.world_to_screen((end_world_x, end_world_y))
            
            # Draw sensor line
            pygame.draw.line(window, DEBUG_SENSOR_VECTOR_COLOR, screen_pos, end_screen_pos, 2)
            
            # Draw small circle at sensor hit point
            pygame.draw.circle(window, DEBUG_SENSOR_VECTOR_COLOR, end_screen_pos, 3)