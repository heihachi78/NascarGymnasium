"""
Polygon-based track renderer.

This module provides smooth track rendering using filled polygons instead of thick lines,
eliminating visual artifacts and providing production-quality rendering.
"""

import pygame
import math
import logging
from typing import List, Tuple, Optional
from .track_generator import Track
from .centerline_generator import CenterlineGenerator
from .track_boundary import TrackBoundary
from .constants import (
    TRACK_COLOR, GRID_COLOR, STARTLINE_COLOR, FINISHLINE_COLOR,
    MINIMUM_LINE_WIDTH,
    POLYGON_CENTERLINE_DEFAULT_SPACING,
    POLYGON_POSITION_MATCHING_TOLERANCE,
    POLYGON_MIN_CENTERLINE_SPACING,
    DEBUG_CENTERLINE_COLOR,
    DEBUG_CENTERLINE_WIDTH
)

# Setup module logger
logger = logging.getLogger(__name__)


class TrackPolygonRenderer:
    """High-quality polygon-based track renderer"""
    
    def __init__(self, camera):
        self.camera = camera
        self.centerline_generator = CenterlineGenerator(adaptive_sampling=True)
        self.track_boundary = TrackBoundary()
        self.use_antialiasing = True  # Enable anti-aliased rendering
        self.centerline_spacing = POLYGON_CENTERLINE_DEFAULT_SPACING
        
        # Polygon caching
        self._cached_track = None
        self._cached_centerline = None
        self._cached_boundaries = None
        self._cached_polygon = None
        self._cached_special_segments = None
        self._cached_window_size = None
        self._cached_spacing = None
        
        # Camera state caching for invalidation
        self._cached_camera_scale = None
        self._cached_camera_offset = None
        self._cached_camera_mode = None
    
    def render_track_polygon(self, window: pygame.Surface, track: Track) -> bool:
        """
        Render track as smooth filled polygons with caching.
        
        Args:
            window: Pygame surface to render on
            track: Track object to render
            
        Returns:
            True if rendering succeeded, False otherwise
        """
        if not track or not track.segments:
            return False
        
        try:
            current_window_size = window.get_size()
            
            # Get current camera state
            current_camera_scale = self.camera.get_scale_factor()
            current_camera_offset = self.camera.offset
            current_camera_mode = self.camera.get_camera_mode()
            
            # Check if we need to regenerate cached data
            need_regenerate = (
                self._cached_track != track or
                self._cached_window_size != current_window_size or
                self._cached_polygon is None or
                self._cached_spacing != self.centerline_spacing or
                self._cached_camera_scale != current_camera_scale or
                self._cached_camera_offset != current_camera_offset or
                self._cached_camera_mode != current_camera_mode
            )
            
            if need_regenerate:
                # Step 1: Generate centerline for entire track
                centerline = self.centerline_generator.generate_centerline(track, target_spacing=self.centerline_spacing)
                
                if len(centerline) < 2:
                    return False
                
                # Step 2: Generate track boundaries
                left_boundary, right_boundary = self.track_boundary.generate_boundaries(
                    centerline, track.width
                )
                
                # Step 3: Create main track polygon
                track_polygon = self.track_boundary.create_track_polygon(left_boundary, right_boundary)
                
                if not self.track_boundary.validate_polygon(track_polygon):
                    return False
                
                # Step 4: Convert to screen coordinates
                screen_polygon = self._convert_polygon_to_screen(track_polygon)
                
                if len(screen_polygon) < 3:
                    return False
                
                # Step 5: Cache special segments
                special_segments = self._prepare_special_segments(track, centerline)
                
                # Cache everything
                self._cached_track = track
                self._cached_centerline = centerline
                self._cached_boundaries = (left_boundary, right_boundary)
                self._cached_polygon = screen_polygon
                self._cached_special_segments = special_segments
                self._cached_window_size = current_window_size
                self._cached_spacing = self.centerline_spacing
                
                # Cache camera state
                self._cached_camera_scale = current_camera_scale
                self._cached_camera_offset = current_camera_offset
                self._cached_camera_mode = current_camera_mode
            
            # Render from cache
            self._render_polygon(window, self._cached_polygon, TRACK_COLOR)
            
            # Render cached special segments
            if self._cached_special_segments:
                for segment_data in self._cached_special_segments:
                    self._render_polygon(window, segment_data['polygon'], segment_data['color'])
            
            return True
            
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Data error in polygon track rendering: {e}")
            # Clear cache on error
            self._clear_cache()
            return False
        except Exception as e:
            logger.error(f"Unexpected error in polygon track rendering: {e}")
            # Clear cache on error
            self._clear_cache()
            return False
    
    def _convert_polygon_to_screen(self, world_polygon: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convert world coordinate polygon to screen coordinates"""
        screen_polygon = []
        
        for world_point in world_polygon:
            screen_point = self.camera.world_to_screen(world_point)
            # Ensure integer coordinates for pygame
            screen_polygon.append((int(screen_point[0]), int(screen_point[1])))
        
        return screen_polygon
    
    def _render_polygon(self, window: pygame.Surface, polygon: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
        """Render a filled polygon with optional anti-aliasing"""
        if len(polygon) < 3:
            return
        
        try:
            # Use anti-aliased rendering if available and enabled
            if self.use_antialiasing and hasattr(pygame, 'gfxdraw'):
                try:
                    # Render filled polygon with anti-aliasing
                    pygame.gfxdraw.filled_polygon(window, polygon, color)
                    # Add anti-aliased outline for smoother edges
                    pygame.gfxdraw.aapolygon(window, polygon, color)
                except (ValueError, OverflowError, AttributeError) as e:
                    # Fall back to regular polygon if gfxdraw fails
                    logger.debug(f"gfxdraw anti-aliasing failed, using regular polygon: {e}")
                    pygame.draw.polygon(window, color, polygon)
            else:
                # Regular polygon rendering without anti-aliasing
                pygame.draw.polygon(window, color, polygon)
                    
        except (ValueError, OverflowError):
            # Handle coordinate overflow cases
            pass
    
    def _prepare_special_segments(self, track: Track, centerline: List[Tuple[float, float]]) -> List[dict]:
        """Prepare special segment overlays for caching"""
        special_segments = []
        
        # Build mapping of world positions to centerline indices
        position_to_index = self._build_position_index_map(track, centerline)
        
        for segment in track.segments:
            color = None
            
            if segment.segment_type == "GRID":
                color = GRID_COLOR
            elif segment.segment_type == "STARTLINE":
                color = STARTLINE_COLOR
            elif segment.segment_type == "FINISHLINE":
                color = FINISHLINE_COLOR
            else:
                continue  # Skip STRAIGHT and CURVE segments
            
            # Find centerline points corresponding to this segment
            start_idx = position_to_index.get(segment.start_position)
            end_idx = position_to_index.get(segment.end_position)
            
            if start_idx is not None and end_idx is not None and start_idx != end_idx:
                segment_polygon = self._prepare_segment_overlay(centerline, start_idx, end_idx, track.width)
                if segment_polygon and len(segment_polygon) >= 3:
                    special_segments.append({
                        'polygon': segment_polygon,
                        'color': color
                    })
        
        return special_segments
    
    def _prepare_segment_overlay(self, centerline: List[Tuple[float, float]], 
                                start_idx: int, end_idx: int, track_width: float) -> Optional[List[Tuple[int, int]]]:
        """Prepare a segment overlay polygon for caching"""
        if start_idx >= end_idx or end_idx >= len(centerline):
            return None
        
        # Extract centerline portion for this segment
        segment_centerline = centerline[start_idx:end_idx + 1]
        
        if len(segment_centerline) < 2:
            return None
        
        # Generate boundaries for this segment portion
        left_boundary, right_boundary = self.track_boundary.generate_boundaries(
            segment_centerline, track_width
        )
        
        # Create polygon for this segment
        segment_polygon = self.track_boundary.create_track_polygon(left_boundary, right_boundary)
        
        if not segment_polygon:
            return None
        
        # Convert to screen coordinates
        return self._convert_polygon_to_screen(segment_polygon)
    
    def _clear_cache(self):
        """Clear all cached data"""
        self._cached_track = None
        self._cached_centerline = None
        self._cached_boundaries = None
        self._cached_polygon = None
        self._cached_special_segments = None
        self._cached_window_size = None
        self._cached_spacing = None
        
        # Clear camera state cache
        self._cached_camera_scale = None
        self._cached_camera_offset = None
        self._cached_camera_mode = None
    
    def _render_special_segments(self, window: pygame.Surface, track: Track, centerline: List[Tuple[float, float]]) -> None:
        """Render colored overlays for special track segments"""
        
        # Build mapping of world positions to centerline indices
        position_to_index = self._build_position_index_map(track, centerline)
        
        for segment in track.segments:
            color = None
            
            if segment.segment_type == "GRID":
                color = GRID_COLOR
            elif segment.segment_type == "STARTLINE":
                color = STARTLINE_COLOR
            elif segment.segment_type == "FINISHLINE":
                color = FINISHLINE_COLOR
            else:
                continue  # Skip STRAIGHT and CURVE segments
            
            # Find centerline points corresponding to this segment
            start_idx = position_to_index.get(segment.start_position)
            end_idx = position_to_index.get(segment.end_position)
            
            if start_idx is not None and end_idx is not None and start_idx != end_idx:
                self._render_segment_overlay(window, centerline, start_idx, end_idx, track.width, color)
    
    def _build_position_index_map(self, track: Track, centerline: List[Tuple[float, float]]) -> dict:
        """Build mapping from segment positions to centerline indices"""
        position_map = {}
        tolerance = POLYGON_POSITION_MATCHING_TOLERANCE
        
        # Map each segment start/end to nearest centerline point
        for segment in track.segments:
            # Find closest centerline points to segment start/end
            start_idx = self._find_closest_centerline_point(centerline, segment.start_position, tolerance)
            end_idx = self._find_closest_centerline_point(centerline, segment.end_position, tolerance)
            
            if start_idx is not None:
                position_map[segment.start_position] = start_idx
            if end_idx is not None:
                position_map[segment.end_position] = end_idx
        
        return position_map
    
    def _find_closest_centerline_point(self, centerline: List[Tuple[float, float]], target: Tuple[float, float], tolerance: float) -> Optional[int]:
        """Find index of centerline point closest to target position"""
        best_idx = None
        best_distance = tolerance
        
        for i, point in enumerate(centerline):
            distance = math.sqrt((point[0] - target[0])**2 + (point[1] - target[1])**2)
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        return best_idx
    
    def _render_segment_overlay(self, window: pygame.Surface, centerline: List[Tuple[float, float]], 
                               start_idx: int, end_idx: int, track_width: float, color: Tuple[int, int, int]) -> None:
        """Render colored overlay for a specific segment portion"""
        
        if start_idx >= end_idx or end_idx >= len(centerline):
            return
        
        # Extract centerline portion for this segment
        segment_centerline = centerline[start_idx:end_idx + 1]
        
        if len(segment_centerline) < 2:
            return
        
        # Generate boundaries for this segment portion
        left_boundary, right_boundary = self.track_boundary.generate_boundaries(
            segment_centerline, track_width
        )
        
        # Create polygon for this segment
        segment_polygon = self.track_boundary.create_track_polygon(left_boundary, right_boundary)
        
        if not segment_polygon:
            return
        
        # Convert to screen coordinates and render
        screen_polygon = self._convert_polygon_to_screen(segment_polygon)
        
        if len(screen_polygon) >= 3:
            self._render_polygon(window, screen_polygon, color)
    
    def set_antialiasing(self, enabled: bool) -> None:
        """Enable or disable anti-aliased rendering"""
        self.use_antialiasing = enabled
    
    def set_centerline_spacing(self, spacing: float) -> None:
        """Set target spacing between centerline points"""
        self.centerline_spacing = max(POLYGON_MIN_CENTERLINE_SPACING, spacing)
        # Clear cache when spacing changes
        self._clear_cache()
    
    def get_rendering_stats(self, track: Track) -> dict:
        """Get statistics about the rendering process"""
        if not track:
            return {}
        
        centerline = self.centerline_generator.generate_centerline(track)
        left_boundary, right_boundary = self.track_boundary.generate_boundaries(centerline, track.width)
        
        return {
            'centerline_points': len(centerline),
            'boundary_points': len(left_boundary) + len(right_boundary),
            'polygon_points': len(left_boundary) + len(right_boundary) if left_boundary and right_boundary else 0,
            'segments': len(track.segments)
        }
    
    def render_centerline(self, window: pygame.Surface) -> bool:
        """
        Render the track centerline for debug visualization.
        
        Args:
            window: Pygame surface to render on
            
        Returns:
            True if rendering succeeded, False otherwise
        """
        if not self._cached_centerline or len(self._cached_centerline) < 2:
            return False
        
        try:
            # Convert centerline to screen coordinates
            screen_points = []
            for world_point in self._cached_centerline:
                screen_point = self.camera.world_to_screen(world_point)
                # Ensure integer coordinates for pygame
                screen_points.append((int(screen_point[0]), int(screen_point[1])))
            
            if len(screen_points) < 2:
                return False
            
            # Draw the centerline as connected lines
            pygame.draw.lines(window, DEBUG_CENTERLINE_COLOR, False, screen_points, DEBUG_CENTERLINE_WIDTH)
            
            return True
            
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error rendering centerline: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error rendering centerline: {e}")
            return False