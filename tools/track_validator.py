"""
Enhanced Track Validation System

Provides comprehensive validation of race tracks including loop closure,
self-intersection detection, and geometric consistency checks.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.track_generator import Track, TrackSegment
from src.centerline_generator import CenterlineGenerator
from src.track_boundary import TrackBoundary


@dataclass
class ValidationResult:
    """Result of track validation with detailed feedback."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class TrackValidator:
    """Enhanced track validation system with comprehensive checks."""
    
    def __init__(self):
        """Initialize the track validator."""
        self.centerline_generator = CenterlineGenerator()
        self.boundary_generator = TrackBoundary()
        
        # Validation tolerances
        self.position_tolerance = 2.0  # meters
        self.heading_tolerance = 3.0   # degrees
        self.min_radius = 20.0         # minimum safe curve radius
        self.max_turn_rate = 5.0       # degrees per meter
        self.min_segment_length = 1.0  # meters
        
    def validate_track(self, track: Track) -> ValidationResult:
        """
        Perform comprehensive validation of a track.
        
        Args:
            track (Track): The track to validate
            
        Returns:
            ValidationResult: Detailed validation results
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Basic structural validation
        struct_result = self._validate_structure(track)
        errors.extend(struct_result['errors'])
        warnings.extend(struct_result['warnings'])
        
        # Loop closure validation
        loop_result = self._validate_loop_closure(track)
        errors.extend(loop_result['errors'])
        warnings.extend(loop_result['warnings'])
        
        # Geometric consistency
        geom_result = self._validate_geometry(track)
        errors.extend(geom_result['errors'])
        warnings.extend(geom_result['warnings'])
        
        # Safety and realism checks
        safety_result = self._validate_safety(track)
        errors.extend(safety_result['errors'])
        warnings.extend(safety_result['warnings'])
        suggestions.extend(safety_result['suggestions'])
        
        # Self-intersection detection
        if len(errors) == 0:  # Only check if basic validation passes
            intersection_result = self._check_self_intersections(track)
            errors.extend(intersection_result['errors'])
            warnings.extend(intersection_result['warnings'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_structure(self, track: Track) -> Dict[str, List[str]]:
        """Validate basic track structure."""
        errors = []
        warnings = []
        
        if not track.segments:
            errors.append("Track has no segments")
            return {'errors': errors, 'warnings': warnings}
        
        # Check for required segments
        has_grid = any(seg.segment_type == "GRID" for seg in track.segments)
        has_startline = any(seg.segment_type == "STARTLINE" for seg in track.segments)
        
        if not has_grid:
            warnings.append("Track has no GRID segment")
        
        if not has_startline:
            errors.append("Track must have a STARTLINE segment")
        
        # Check segment order
        if has_grid and track.segments[0].segment_type != "GRID":
            warnings.append("GRID segment should typically be first")
        
        # Check for duplicate special segments
        grid_count = sum(1 for seg in track.segments if seg.segment_type == "GRID")
        startline_count = sum(1 for seg in track.segments if seg.segment_type == "STARTLINE")
        finishline_count = sum(1 for seg in track.segments if seg.segment_type == "FINISHLINE")
        
        if grid_count > 1:
            warnings.append(f"Track has {grid_count} GRID segments (typically only 1)")
        if startline_count > 1:
            errors.append(f"Track has {startline_count} STARTLINE segments (must be 1)")
        if finishline_count > 1:
            errors.append(f"Track has {finishline_count} FINISHLINE segments (max 1)")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_loop_closure(self, track: Track) -> Dict[str, List[str]]:
        """Validate that the track forms a proper closed loop."""
        errors = []
        warnings = []
        
        if not track.segments:
            return {'errors': errors, 'warnings': warnings}
        
        start_pos = track.segments[0].start_position
        end_pos = track.segments[-1].end_position
        
        # Check position closure
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        if distance > self.position_tolerance:
            errors.append(f"Track doesn't form a closed loop. Position gap: {distance:.2f}m "
                         f"(tolerance: {self.position_tolerance}m)")
        elif distance > self.position_tolerance / 2:
            warnings.append(f"Small gap in track loop closure: {distance:.2f}m")
        
        # Check heading continuity
        start_heading = track.segments[0].start_heading
        end_heading = track.segments[-1].end_heading
        
        heading_diff = abs(end_heading - start_heading)
        
        # Handle wraparound (e.g., 359° vs 1°)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        
        if heading_diff > self.heading_tolerance:
            errors.append(f"Heading discontinuity at loop closure: {heading_diff:.2f}° "
                         f"(tolerance: {self.heading_tolerance}°)")
        elif heading_diff > self.heading_tolerance / 2:
            warnings.append(f"Small heading discontinuity: {heading_diff:.2f}°")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_geometry(self, track: Track) -> Dict[str, List[str]]:
        """Validate geometric properties of track segments."""
        errors = []
        warnings = []
        
        for i, seg in enumerate(track.segments):
            # Check segment length
            if seg.length <= 0:
                errors.append(f"Segment {i} ({seg.segment_type}) has invalid length: {seg.length}")
            elif seg.length < self.min_segment_length:
                warnings.append(f"Segment {i} ({seg.segment_type}) is very short: {seg.length:.2f}m")
            
            # Check width
            if seg.width <= 0:
                errors.append(f"Segment {i} ({seg.segment_type}) has invalid width: {seg.width}")
            elif seg.width < 8.0:
                warnings.append(f"Segment {i} ({seg.segment_type}) is very narrow: {seg.width:.1f}m")
            elif seg.width > 50.0:
                warnings.append(f"Segment {i} ({seg.segment_type}) is very wide: {seg.width:.1f}m")
            
            # Check curve-specific properties
            if seg.segment_type == "CURVE":
                if seg.curve_radius <= 0:
                    errors.append(f"Curve segment {i} has invalid radius: {seg.curve_radius}")
                if seg.curve_angle <= 0 or seg.curve_angle > 360:
                    errors.append(f"Curve segment {i} has invalid angle: {seg.curve_angle}°")
                if not seg.curve_direction in ["LEFT", "RIGHT"]:
                    errors.append(f"Curve segment {i} has invalid direction: '{seg.curve_direction}'")
            
            # Check position consistency between segments
            if i > 0:
                prev_seg = track.segments[i-1]
                pos_diff = math.sqrt((seg.start_position[0] - prev_seg.end_position[0])**2 + 
                                   (seg.start_position[1] - prev_seg.end_position[1])**2)
                
                if pos_diff > 0.1:  # 10cm tolerance
                    errors.append(f"Position discontinuity between segments {i-1} and {i}: {pos_diff:.3f}m")
                
                # Check heading consistency
                heading_diff = abs(seg.start_heading - prev_seg.end_heading)
                if heading_diff > 180:
                    heading_diff = 360 - heading_diff
                
                if heading_diff > 1.0:  # 1 degree tolerance
                    warnings.append(f"Heading jump between segments {i-1} and {i}: {heading_diff:.2f}°")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_safety(self, track: Track) -> Dict[str, List[str]]:
        """Validate track safety and realism."""
        errors = []
        warnings = []
        suggestions = []
        
        for i, seg in enumerate(track.segments):
            if seg.segment_type == "CURVE":
                # Check minimum safe radius
                if seg.curve_radius < self.min_radius:
                    warnings.append(f"Curve segment {i} has tight radius: {seg.curve_radius:.1f}m "
                                  f"(minimum recommended: {self.min_radius}m)")
                
                # Check turn rate (degrees per meter)
                turn_rate = seg.curve_angle / seg.length if seg.length > 0 else 0
                if turn_rate > self.max_turn_rate:
                    warnings.append(f"Curve segment {i} has high turn rate: {turn_rate:.2f}°/m "
                                  f"(maximum recommended: {self.max_turn_rate}°/m)")
                
                # Suggest banking for high-speed turns
                if seg.curve_radius > 200 and seg.curve_angle > 45:
                    suggestions.append(f"Consider banking for high-speed turn at segment {i}")
        
        # Check overall track characteristics
        total_curves = sum(1 for seg in track.segments if seg.segment_type == "CURVE")
        total_length = track.get_total_track_length()
        
        if total_curves == 0:
            warnings.append("Track has no curves - may be too simple")
        elif total_curves / len(track.segments) > 0.8:
            warnings.append("Track is very technical with many curves")
        
        if total_length < 500:
            suggestions.append("Track is quite short - consider extending for more interesting racing")
        elif total_length > 10000:
            suggestions.append("Track is very long - ensure it remains engaging throughout")
        
        return {'errors': errors, 'warnings': warnings, 'suggestions': suggestions}
    
    def _check_self_intersections(self, track: Track) -> Dict[str, List[str]]:
        """Check for track boundary self-intersections."""
        errors = []
        warnings = []
        
        try:
            # Generate centerline and boundaries
            centerline = self.centerline_generator.generate_centerline(track)
            if len(centerline) < 4:
                return {'errors': errors, 'warnings': warnings}
            
            left_boundary, right_boundary = self.boundary_generator.generate_boundaries(
                centerline, track.width
            )
            
            # Check for self-intersections in boundaries
            left_intersections = self._find_polygon_self_intersections(left_boundary)
            right_intersections = self._find_polygon_self_intersections(right_boundary)
            
            if left_intersections:
                errors.append(f"Left track boundary has {len(left_intersections)} self-intersections")
            
            if right_intersections:
                errors.append(f"Right track boundary has {len(right_intersections)} self-intersections")
            
            # Check boundary validity
            track_polygon = self.boundary_generator.create_track_polygon(left_boundary, right_boundary)
            if not self.boundary_generator.validate_polygon(track_polygon):
                warnings.append("Generated track polygon failed validation")
        
        except Exception as e:
            warnings.append(f"Could not check for self-intersections: {str(e)}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _find_polygon_self_intersections(self, polygon: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Find self-intersections in a polygon."""
        intersections = []
        
        if len(polygon) < 4:
            return intersections
        
        # Check each edge against every other edge (excluding adjacent edges)
        for i in range(len(polygon)):
            for j in range(i + 2, len(polygon)):
                # Skip the wrap-around case for adjacent edges
                if i == 0 and j == len(polygon) - 1:
                    continue
                
                edge1_start = polygon[i]
                edge1_end = polygon[(i + 1) % len(polygon)]
                edge2_start = polygon[j]
                edge2_end = polygon[(j + 1) % len(polygon)]
                
                if self._line_segments_intersect(edge1_start, edge1_end, edge2_start, edge2_end):
                    intersections.append((i, j))
        
        return intersections
    
    def _line_segments_intersect(self, p1: Tuple[float, float], q1: Tuple[float, float], 
                                p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        """Check if two line segments intersect."""
        def orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
            """Find orientation of ordered triplet (p, q, r)."""
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10:  # Colinear
                return 0
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
            """Check if point q lies on segment pr."""
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases - colinear points
        if (o1 == 0 and on_segment(p1, p2, q1) or
            o2 == 0 and on_segment(p1, q2, q1) or
            o3 == 0 and on_segment(p2, p1, q2) or
            o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False