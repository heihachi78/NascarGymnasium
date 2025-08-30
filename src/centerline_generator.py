"""
Centerline generation system for smooth track rendering.

This module generates smooth continuous centerlines from track segments,
handling both straight and curved sections with proper interpolation.
"""

import math
from typing import List, Tuple
from .track_generator import Track, TrackSegment
from .constants import (
    CENTERLINE_DEFAULT_SPACING,
    CENTERLINE_MIN_CURVE_POINTS,
    CENTERLINE_TIGHT_CURVE_THRESHOLD,
    CENTERLINE_ADAPTIVE_MIN_FACTOR,
    CENTERLINE_SMOOTHING_FACTOR
)


class CenterlineGenerator:
    """Generates smooth centerlines for track rendering.

    This class takes a Track object and generates a smooth, continuous centerline
    by interpolating points along the track segments.

    Args:
        adaptive_sampling (bool): If True, use adaptive sampling for curves.
    """
    
    def __init__(self, adaptive_sampling: bool = True):
        """Initializes the CenterlineGenerator.

        Args:
            adaptive_sampling (bool): If True, use adaptive sampling for curves.
        """
        self.adaptive_sampling = adaptive_sampling
    
    def generate_centerline(self, track: Track, target_spacing: float = CENTERLINE_DEFAULT_SPACING) -> List[Tuple[float, float]]:
        """
        Generates a smooth centerline from the track segments.
        
        Args:
            track (Track): The track object containing the segments.
            target_spacing (float): The target distance between centerline points.
            
        Returns:
            A list of (x, y) points forming the centerline.
        """
        if not track or not track.segments:
            return []
        
        centerline_points = []
        
        for segment in track.segments:
            segment_points = self._generate_segment_centerline(segment, target_spacing)
            
            if not centerline_points:
                # First segment - include all points
                centerline_points.extend(segment_points)
            else:
                # Subsequent segments - skip first point to avoid duplication
                centerline_points.extend(segment_points[1:])
        
        # Apply smoothing to reduce sharp corners
        smoothed_points = self._smooth_centerline(centerline_points)
        
        return smoothed_points
    
    def _generate_segment_centerline(self, segment: TrackSegment, target_spacing: float) -> List[Tuple[float, float]]:
        """Generates centerline points for a single segment.

        Args:
            segment (TrackSegment): The track segment.
            target_spacing (float): The target distance between points.

        Returns:
            A list of (x, y) points for the segment's centerline.
        """
        
        if segment.segment_type == "CURVE" and segment.curve_radius > 0:
            return self._generate_curve_centerline(segment, target_spacing)
        else:
            return self._generate_straight_centerline(segment, target_spacing)
    
    def _generate_straight_centerline(self, segment: TrackSegment, target_spacing: float) -> List[Tuple[float, float]]:
        """Generates the centerline for a straight segment.

        Args:
            segment (TrackSegment): The straight track segment.
            target_spacing (float): The target distance between points.

        Returns:
            A list of (x, y) points for the straight centerline.
        """
        start = segment.start_position
        end = segment.end_position
        
        # Calculate segment length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length <= 0:
            return [start]
        
        # Calculate number of points based on target spacing
        num_points = max(2, int(length / target_spacing) + 1)
        
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = start[0] + t * dx
            y = start[1] + t * dy
            points.append((x, y))
        
        return points
    
    def _generate_curve_centerline(self, segment: TrackSegment, target_spacing: float) -> List[Tuple[float, float]]:
        """Generates the centerline for a curved segment with adaptive sampling.

        Args:
            segment (TrackSegment): The curved track segment.
            target_spacing (float): The target distance between points.

        Returns:
            A list of (x, y) points for the curved centerline.
        """
        
        # Calculate curve parameters
        start_heading_rad = math.radians(segment.start_heading)
        curve_angle_rad = math.radians(segment.curve_angle)
        
        if segment.curve_direction == "LEFT":
            turn_multiplier = 1.0
        else:  # RIGHT
            turn_multiplier = -1.0
        
        # Calculate curve center
        perpendicular_heading = start_heading_rad + turn_multiplier * math.pi / 2
        center_x = segment.start_position[0] + segment.curve_radius * math.cos(perpendicular_heading)
        center_y = segment.start_position[1] + segment.curve_radius * math.sin(perpendicular_heading)
        
        # Calculate arc length and number of points
        arc_length = abs(segment.curve_radius * curve_angle_rad)
        
        if self.adaptive_sampling:
            # More points for tighter curves (smaller radius)
            # Minimum points defined by constant, more for longer arcs or smaller radii
            base_points = max(CENTERLINE_MIN_CURVE_POINTS, int(arc_length / target_spacing))
            radius_factor = max(CENTERLINE_ADAPTIVE_MIN_FACTOR, CENTERLINE_TIGHT_CURVE_THRESHOLD / segment.curve_radius)
            num_points = int(base_points * radius_factor)
        else:
            num_points = max(CENTERLINE_MIN_CURVE_POINTS, int(arc_length / target_spacing))
        
        # Generate curve points
        points = []
        angle_from_center_to_start = start_heading_rad - turn_multiplier * math.pi / 2
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            angle = angle_from_center_to_start + turn_multiplier * curve_angle_rad * t
            
            x = center_x + segment.curve_radius * math.cos(angle)
            y = center_y + segment.curve_radius * math.sin(angle)
            points.append((x, y))
        
        # Ensure exact endpoints
        points[0] = segment.start_position
        points[-1] = segment.end_position
        
        return points
    
    def _smooth_centerline(self, points: List[Tuple[float, float]], smoothing_factor: float = CENTERLINE_SMOOTHING_FACTOR) -> List[Tuple[float, float]]:
        """
        Applies smoothing to the centerline to reduce sharp corners.
        
        Args:
            points (List[Tuple[float, float]]): The input centerline points.
            smoothing_factor (float): The amount of smoothing to apply.
            
        Returns:
            A list of smoothed centerline points.
        """
        if len(points) < 3 or smoothing_factor <= 0:
            return points
        
        smoothed = [points[0]]  # Keep first point unchanged
        
        for i in range(1, len(points) - 1):
            prev_point = points[i - 1]
            curr_point = points[i]
            next_point = points[i + 1]
            
            # Calculate average of neighbors
            avg_x = (prev_point[0] + next_point[0]) / 2
            avg_y = (prev_point[1] + next_point[1]) / 2
            
            # Interpolate between current point and average
            smooth_x = curr_point[0] * (1 - smoothing_factor) + avg_x * smoothing_factor
            smooth_y = curr_point[1] * (1 - smoothing_factor) + avg_y * smoothing_factor
            
            smoothed.append((smooth_x, smooth_y))
        
        smoothed.append(points[-1])  # Keep last point unchanged
        
        return smoothed
    
    def get_centerline_tangent(self, points: List[Tuple[float, float]], index: int) -> Tuple[float, float]:
        """
        Calculates the tangent vector at a specific point on the centerline.
        
        Args:
            points (List[Tuple[float, float]]): The centerline points.
            index (int): The index of the point to calculate the tangent for.
            
        Returns:
            A normalized tangent vector (dx, dy).
        """
        if len(points) < 2:
            return (1.0, 0.0)  # Default to horizontal
        
        if index == 0:
            # Use forward difference for first point
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif index == len(points) - 1:
            # Use backward difference for last point
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            # Use central difference for middle points
            dx = points[index + 1][0] - points[index - 1][0]
            dy = points[index + 1][1] - points[index - 1][1]
        
        # Normalize the tangent vector
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            return (dx / length, dy / length)
        else:
            return (1.0, 0.0)