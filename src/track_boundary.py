"""
Track boundary calculation system.

This module calculates left and right track boundaries from centerline points,
handling proper offsetting and corner cases for clean polygon rendering.
"""

import math
from typing import List, Tuple, Optional
from .centerline_generator import CenterlineGenerator
from .constants import (
    BOUNDARY_SMOOTHING_MAX_ANGLE,
    BOUNDARY_SMOOTHING_MAX_FACTOR,
    BOUNDARY_POINTS_EQUAL_TOLERANCE,
    BOUNDARY_MIN_POLYGON_AREA
)


class TrackBoundary:
    """Calculates track boundaries from centerline.

    This class generates the left and right boundaries of a track based on a
    given centerline and track width. It also includes methods for smoothing
    corners and validating the generated polygons.
    """
    
    def __init__(self):
        """Initializes the TrackBoundary."""
        self.centerline_generator = CenterlineGenerator()
    
    def generate_boundaries(self, centerline: List[Tuple[float, float]], track_width: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Generates left and right track boundaries from a centerline.
        
        Args:
            centerline (List[Tuple[float, float]]): A list of centerline points.
            track_width (float): The width of the track.
            
        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: A tuple containing
            two lists of points: (left_boundary, right_boundary).
        """
        if len(centerline) < 2:
            return [], []
        
        half_width = track_width / 2.0
        left_boundary = []
        right_boundary = []
        
        for i, point in enumerate(centerline):
            # Get tangent vector at this point
            tangent = self.centerline_generator.get_centerline_tangent(centerline, i)
            
            # Calculate normal vector (perpendicular to tangent)
            normal = self._get_normal_vector(tangent)
            
            # Calculate boundary points by offsetting along normal
            left_point = (
                point[0] + normal[0] * half_width,
                point[1] + normal[1] * half_width
            )
            right_point = (
                point[0] - normal[0] * half_width,
                point[1] - normal[1] * half_width
            )
            
            left_boundary.append(left_point)
            right_boundary.append(right_point)
        
        # Apply corner smoothing to prevent intersections
        left_boundary = self._smooth_boundary_corners(left_boundary)
        right_boundary = self._smooth_boundary_corners(right_boundary)
        
        return left_boundary, right_boundary
    
    def _get_normal_vector(self, tangent: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculates the normal (perpendicular) vector from a tangent vector.
        
        Args:
            tangent (Tuple[float, float]): The normalized tangent vector (dx, dy).
            
        Returns:
            Tuple[float, float]: The normal vector pointing to the left of the tangent.
        """
        # Rotate tangent 90 degrees counterclockwise to get left normal
        return (-tangent[1], tangent[0])
    
    def _smooth_boundary_corners(self, boundary: List[Tuple[float, float]], max_angle: float = BOUNDARY_SMOOTHING_MAX_ANGLE) -> List[Tuple[float, float]]:
        """
        Smooths sharp corners in a boundary to prevent self-intersections.
        
        Args:
            boundary (List[Tuple[float, float]]): The input boundary points.
            max_angle (float): The maximum angle (in degrees) before smoothing kicks in.
            
        Returns:
            List[Tuple[float, float]]: The smoothed boundary points.
        """
        if len(boundary) < 3:
            return boundary
        
        smoothed = [boundary[0]]  # Keep first point
        max_angle_rad = math.radians(max_angle)
        
        for i in range(1, len(boundary) - 1):
            prev_point = boundary[i - 1]
            curr_point = boundary[i]
            next_point = boundary[i + 1]
            
            # Calculate angle at current point
            angle = self._calculate_angle(prev_point, curr_point, next_point)
            
            if angle < max_angle_rad:
                # Sharp corner - apply smoothing
                smoothing_factor = 1.0 - (angle / max_angle_rad)
                smoothing_amount = BOUNDARY_SMOOTHING_MAX_FACTOR * smoothing_factor
                
                # Average with neighbors
                avg_x = (prev_point[0] + next_point[0]) / 2
                avg_y = (prev_point[1] + next_point[1]) / 2
                
                smooth_x = curr_point[0] * (1 - smoothing_amount) + avg_x * smoothing_amount
                smooth_y = curr_point[1] * (1 - smoothing_amount) + avg_y * smoothing_amount
                
                smoothed.append((smooth_x, smooth_y))
            else:
                # Normal corner - keep as is
                smoothed.append(curr_point)
        
        smoothed.append(boundary[-1])  # Keep last point
        
        return smoothed
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculates the angle at point p2 formed by the line segments p1-p2 and p2-p3.
        
        Args:
            p1 (Tuple[float, float]): The first point.
            p2 (Tuple[float, float]): The second (middle) point.
            p3 (Tuple[float, float]): The third point.
            
        Returns:
            float: The angle in radians (0 to π).
        """
        # Vectors from p2 to p1 and p2 to p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return math.pi  # Straight line
        
        # Calculate angle using dot product
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        
        return math.acos(cos_angle)
    
    def create_track_polygon(self, left_boundary: List[Tuple[float, float]], right_boundary: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Creates a single polygon from the left and right boundaries.
        
        Args:
            left_boundary (List[Tuple[float, float]]): Points forming the left edge of the track.
            right_boundary (List[Tuple[float, float]]): Points forming the right edge of the track.
            
        Returns:
            List[Tuple[float, float]]: Polygon points suitable for pygame.draw.polygon().
        """
        if not left_boundary or not right_boundary:
            return []
        
        # Create polygon by going along left boundary, then back along right boundary
        polygon = []
        
        # Add left boundary points
        polygon.extend(left_boundary)
        
        # Add right boundary points in reverse order
        polygon.extend(reversed(right_boundary))
        
        return polygon
    
    def validate_polygon(self, polygon: List[Tuple[float, float]]) -> bool:
        """
        Validates that a polygon is suitable for rendering.
        
        Args:
            polygon (List[Tuple[float, float]]): Polygon points to validate.
            
        Returns:
            bool: True if the polygon is valid for rendering, False otherwise.
        """
        if len(polygon) < 3:
            return False
        
        # Check for duplicate consecutive points
        for i in range(len(polygon)):
            next_i = (i + 1) % len(polygon)
            if self._points_equal(polygon[i], polygon[next_i]):
                return False
        
        # Check for reasonable polygon size
        if self._calculate_polygon_area(polygon) < BOUNDARY_MIN_POLYGON_AREA:
            return False
        
        return True
    
    def _points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = BOUNDARY_POINTS_EQUAL_TOLERANCE) -> bool:
        """Checks if two points are equal within a given tolerance.

        Args:
            p1 (Tuple[float, float]): The first point.
            p2 (Tuple[float, float]): The second point.
            tolerance (float): The tolerance for equality.

        Returns:
            bool: True if the points are equal within the tolerance, False otherwise.
        """
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
    
    def _calculate_polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculates the area of a polygon using the shoelace formula.

        Args:
            polygon (List[Tuple[float, float]]): The polygon points.

        Returns:
            float: The area of the polygon.
        """
        if len(polygon) < 3:
            return 0.0
        
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0