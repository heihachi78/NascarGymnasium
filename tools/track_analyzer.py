"""
Track Analysis Module

Provides comprehensive analysis of race track files including geometry,
statistics, validation, and performance metrics.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.track_generator import Track, TrackLoader, TrackSegment
from src.centerline_generator import CenterlineGenerator
from src.track_boundary import TrackBoundary
from src.constants import DEFAULT_TRACK_WIDTH


@dataclass
class TrackStatistics:
    """Container for comprehensive track statistics."""
    
    # Basic metrics
    total_length: float
    segment_count: int
    curve_count: int
    straight_count: int
    
    # Geometry metrics
    average_width: float
    min_width: float
    max_width: float
    
    # Curve analysis
    total_curve_angle: float
    average_curve_radius: float
    min_curve_radius: float
    max_curve_radius: float
    sharpest_turn_angle: float
    
    # Track bounds
    track_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    track_area: float
    
    # Validation status
    is_valid_loop: bool
    validation_errors: List[str]
    
    # Performance estimates
    estimated_lap_time: float
    technical_difficulty: float


class TrackAnalyzer:
    """Comprehensive track analysis system."""
    
    def __init__(self):
        """Initialize the track analyzer."""
        self.loader = TrackLoader()
        self.centerline_generator = CenterlineGenerator()
        self.boundary_generator = TrackBoundary()
    
    def analyze_track_file(self, file_path: str) -> TrackStatistics:
        """
        Analyze a track file and return comprehensive statistics.
        
        Args:
            file_path (str): Path to the track file
            
        Returns:
            TrackStatistics: Comprehensive track analysis results
        """
        # Load the track
        track = self.loader.load_track(file_path)
        
        return self.analyze_track(track)
    
    def analyze_track(self, track: Track) -> TrackStatistics:
        """
        Analyze a Track object and return comprehensive statistics.
        
        Args:
            track (Track): The track object to analyze
            
        Returns:
            TrackStatistics: Comprehensive track analysis results
        """
        # Basic metrics
        basic_stats = self._calculate_basic_metrics(track)
        
        # Geometry analysis
        geometry_stats = self._analyze_geometry(track)
        
        # Curve analysis
        curve_stats = self._analyze_curves(track)
        
        # Track bounds and area
        bounds = track.get_track_bounds()
        area = self._calculate_track_area(track)
        
        # Validation
        validation_result = self._validate_track_loop(track)
        
        # Performance estimates
        performance_stats = self._estimate_performance_metrics(track)
        
        return TrackStatistics(
            # Basic metrics
            total_length=basic_stats['total_length'],
            segment_count=basic_stats['segment_count'],
            curve_count=basic_stats['curve_count'],
            straight_count=basic_stats['straight_count'],
            
            # Geometry metrics
            average_width=geometry_stats['average_width'],
            min_width=geometry_stats['min_width'],
            max_width=geometry_stats['max_width'],
            
            # Curve analysis
            total_curve_angle=curve_stats['total_angle'],
            average_curve_radius=curve_stats['average_radius'],
            min_curve_radius=curve_stats['min_radius'],
            max_curve_radius=curve_stats['max_radius'],
            sharpest_turn_angle=curve_stats['sharpest_angle'],
            
            # Track bounds
            track_bounds=bounds,
            track_area=area,
            
            # Validation status
            is_valid_loop=validation_result['is_valid'],
            validation_errors=validation_result['errors'],
            
            # Performance estimates
            estimated_lap_time=performance_stats['lap_time'],
            technical_difficulty=performance_stats['difficulty']
        )
    
    def _calculate_basic_metrics(self, track: Track) -> Dict[str, Any]:
        """Calculate basic track metrics."""
        segment_count = len(track.segments)
        curve_count = sum(1 for seg in track.segments if seg.segment_type == "CURVE")
        straight_count = segment_count - curve_count
        
        return {
            'total_length': track.get_total_track_length(),
            'segment_count': segment_count,
            'curve_count': curve_count,
            'straight_count': straight_count
        }
    
    def _analyze_geometry(self, track: Track) -> Dict[str, float]:
        """Analyze track geometry metrics."""
        if not track.segments:
            return {
                'average_width': 0.0,
                'min_width': 0.0,
                'max_width': 0.0
            }
        
        widths = [seg.width for seg in track.segments]
        
        return {
            'average_width': sum(widths) / len(widths),
            'min_width': min(widths),
            'max_width': max(widths)
        }
    
    def _analyze_curves(self, track: Track) -> Dict[str, float]:
        """Analyze curve characteristics."""
        curves = [seg for seg in track.segments if seg.segment_type == "CURVE"]
        
        if not curves:
            return {
                'total_angle': 0.0,
                'average_radius': 0.0,
                'min_radius': 0.0,
                'max_radius': 0.0,
                'sharpest_angle': 0.0
            }
        
        total_angle = sum(seg.curve_angle for seg in curves)
        radii = [seg.curve_radius for seg in curves]
        angles = [seg.curve_angle for seg in curves]
        
        return {
            'total_angle': total_angle,
            'average_radius': sum(radii) / len(radii),
            'min_radius': min(radii),
            'max_radius': max(radii),
            'sharpest_angle': max(angles)
        }
    
    def _calculate_track_area(self, track: Track) -> float:
        """Calculate approximate track area using bounding box."""
        bounds = track.get_track_bounds()
        width = bounds[1][0] - bounds[0][0]
        height = bounds[1][1] - bounds[0][1]
        return width * height
    
    def _validate_track_loop(self, track: Track) -> Dict[str, Any]:
        """Validate that the track forms a proper closed loop."""
        errors = []
        
        if not track.segments:
            errors.append("Track has no segments")
            return {'is_valid': False, 'errors': errors}
        
        # Check basic validation first
        try:
            self.loader.validate_track(track)
        except ValueError as e:
            errors.append(f"Basic validation failed: {str(e)}")
        
        # Check loop closure
        start_pos = track.segments[0].start_position
        end_pos = track.segments[-1].end_position
        
        position_tolerance = 5.0  # 5 meter tolerance
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        if distance > position_tolerance:
            errors.append(f"Track doesn't form a closed loop. Gap: {distance:.2f}m")
        
        # Check heading continuity
        start_heading = track.segments[0].start_heading
        end_heading = track.segments[-1].end_heading
        
        heading_tolerance = 5.0  # 5 degree tolerance
        heading_diff = abs(end_heading - start_heading)
        
        # Handle wraparound (e.g., 359° vs 1°)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        
        if heading_diff > heading_tolerance:
            errors.append(f"Heading discontinuity at loop closure: {heading_diff:.2f}°")
        
        # Check for reasonable track segments
        for i, seg in enumerate(track.segments):
            if seg.length <= 0:
                errors.append(f"Segment {i} has invalid length: {seg.length}")
            
            if seg.segment_type == "CURVE":
                if seg.curve_radius <= 0:
                    errors.append(f"Curve segment {i} has invalid radius: {seg.curve_radius}")
                if seg.curve_angle <= 0 or seg.curve_angle > 360:
                    errors.append(f"Curve segment {i} has invalid angle: {seg.curve_angle}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _estimate_performance_metrics(self, track: Track) -> Dict[str, float]:
        """Estimate performance metrics like lap time and difficulty."""
        if not track.segments:
            return {'lap_time': 0.0, 'difficulty': 0.0}
        
        # Simple lap time estimation based on segment types
        estimated_time = 0.0
        difficulty_score = 0.0
        
        for seg in track.segments:
            if seg.segment_type == "CURVE":
                # Slower through curves, difficulty increases with sharper turns
                curve_speed = max(20.0, 60.0 - (seg.curve_angle / 10.0))  # m/s
                segment_time = seg.length / curve_speed
                
                # Difficulty based on radius and angle
                turn_difficulty = (seg.curve_angle / 90.0) * (100.0 / max(seg.curve_radius, 50.0))
                difficulty_score += turn_difficulty
                
            else:
                # Faster on straights
                straight_speed = 50.0  # m/s
                segment_time = seg.length / straight_speed
                
                # Long straights add some difficulty (high speed)
                if seg.length > 500:
                    difficulty_score += seg.length / 1000.0
            
            estimated_time += segment_time
        
        # Normalize difficulty (0-10 scale)
        difficulty_score = min(10.0, max(0.0, difficulty_score / len(track.segments)))
        
        return {
            'lap_time': estimated_time,
            'difficulty': difficulty_score
        }
    
    def generate_report(self, stats: TrackStatistics, track_name: str = "Unknown") -> str:
        """
        Generate a comprehensive text report of track analysis.
        
        Args:
            stats (TrackStatistics): The track statistics
            track_name (str): Name of the track
            
        Returns:
            str: Formatted text report
        """
        report = []
        report.append("=" * 60)
        report.append(f"TRACK ANALYSIS REPORT: {track_name}")
        report.append("=" * 60)
        report.append("")
        
        # Basic Information
        report.append("BASIC INFORMATION:")
        report.append(f"  Total Length: {stats.total_length:.1f} m")
        report.append(f"  Total Segments: {stats.segment_count}")
        report.append(f"  Curve Segments: {stats.curve_count}")
        report.append(f"  Straight Segments: {stats.straight_count}")
        report.append("")
        
        # Track Geometry
        report.append("TRACK GEOMETRY:")
        report.append(f"  Average Width: {stats.average_width:.1f} m")
        report.append(f"  Width Range: {stats.min_width:.1f} - {stats.max_width:.1f} m")
        report.append(f"  Track Area: {stats.track_area:.0f} m²")
        
        bounds = stats.track_bounds
        track_width = bounds[1][0] - bounds[0][0]
        track_height = bounds[1][1] - bounds[0][1]
        report.append(f"  Bounding Box: {track_width:.1f} × {track_height:.1f} m")
        report.append("")
        
        # Curve Analysis
        if stats.curve_count > 0:
            report.append("CURVE ANALYSIS:")
            report.append(f"  Total Turn Angle: {stats.total_curve_angle:.1f}°")
            report.append(f"  Average Radius: {stats.average_curve_radius:.1f} m")
            report.append(f"  Radius Range: {stats.min_curve_radius:.1f} - {stats.max_curve_radius:.1f} m")
            report.append(f"  Sharpest Turn: {stats.sharpest_turn_angle:.1f}°")
            report.append("")
        
        # Performance Estimates
        report.append("PERFORMANCE ESTIMATES:")
        report.append(f"  Estimated Lap Time: {stats.estimated_lap_time:.1f} seconds")
        report.append(f"  Technical Difficulty: {stats.technical_difficulty:.1f}/10")
        report.append("")
        
        # Validation Status
        report.append("VALIDATION STATUS:")
        if stats.is_valid_loop:
            report.append("  ✓ Track is a valid closed loop")
        else:
            report.append("  ✗ Track validation failed")
            for error in stats.validation_errors:
                report.append(f"    - {error}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)