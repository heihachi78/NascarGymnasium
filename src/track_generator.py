from dataclasses import dataclass
from typing import List, Tuple
import os
import math
from .constants import (
    DEFAULT_TRACK_WIDTH, 
    DEFAULT_GRID_LENGTH, 
    STARTLINE_LENGTH, 
    FINISHLINE_LENGTH
)


@dataclass
class TrackSegment:
    segment_type: str  # "GRID", "STARTLINE", "STRAIGHT", "FINISHLINE", "CURVE"
    length: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    width: float
    curve_angle: float = 0.0  # Degrees
    curve_radius: float = 0.0  # Meters
    curve_direction: str = ""  # "LEFT", "RIGHT", or ""
    start_heading: float = 0.0  # Start direction in degrees
    end_heading: float = 0.0  # End direction in degrees


class Track:
    def __init__(self, width: float = DEFAULT_TRACK_WIDTH):
        self.width = width
        self.segments: List[TrackSegment] = []
        self.total_length = 0.0
        self.has_startline = False
        self.has_finishline = False
        self.current_position = (0.0, 0.0)
        self.current_heading = 0.0  # Track direction in degrees (0 = positive x)
    
    def add_segment(self, segment_type: str, length: float, curve_angle: float = 0.0, curve_radius: float = 0.0, curve_direction: str = ""):
        start_pos = self.current_position
        start_heading = self.current_heading
        
        if curve_angle == 0.0:
            # Straight segment - extend along current heading
            end_heading = start_heading
            end_pos = self._calculate_straight_end_position(start_pos, start_heading, length)
        else:
            # Curved segment
            segment_type = "CURVE"
            end_pos, end_heading = self._calculate_curve_geometry(start_pos, start_heading, curve_angle, curve_radius, curve_direction)
            # Calculate arc length for curves
            arc_length = abs(curve_radius * math.radians(curve_angle))
            length = arc_length
        
        segment = TrackSegment(
            segment_type=segment_type,
            length=length,
            start_position=start_pos,
            end_position=end_pos,
            width=self.width,
            curve_angle=curve_angle,
            curve_radius=curve_radius,
            curve_direction=curve_direction,
            start_heading=start_heading,
            end_heading=end_heading
        )
        
        self.segments.append(segment)
        self.total_length += length
        self.current_position = end_pos
        self.current_heading = end_heading
        
        # Track special segments
        if segment_type == "STARTLINE":
            self.has_startline = True
        elif segment_type == "FINISHLINE":
            self.has_finishline = True
    
    def _calculate_straight_end_position(self, start_pos: Tuple[float, float], heading: float, length: float) -> Tuple[float, float]:
        """Calculate end position for a straight segment"""
        heading_rad = math.radians(heading)
        end_x = start_pos[0] + length * math.cos(heading_rad)
        end_y = start_pos[1] + length * math.sin(heading_rad)
        return (end_x, end_y)
    
    def _calculate_curve_geometry(self, start_pos: Tuple[float, float], start_heading: float, 
                                 curve_angle: float, curve_radius: float, curve_direction: str) -> Tuple[Tuple[float, float], float]:
        """Calculate end position and heading for a curved segment"""
        # Convert to radians
        start_heading_rad = math.radians(start_heading)
        curve_angle_rad = math.radians(curve_angle)
        
        # Determine turn direction
        if curve_direction == "LEFT":
            turn_multiplier = 1.0
        else:  # RIGHT
            turn_multiplier = -1.0
        
        # Calculate center of the curve
        # For a left turn, center is to the left of current direction
        perpendicular_heading = start_heading_rad + turn_multiplier * math.pi / 2
        center_x = start_pos[0] + curve_radius * math.cos(perpendicular_heading)
        center_y = start_pos[1] + curve_radius * math.sin(perpendicular_heading)
        
        # Calculate end position
        end_heading = start_heading + turn_multiplier * curve_angle
        end_heading_rad = math.radians(end_heading)
        
        # End position is on the circle, rotated by the curve angle
        angle_from_center_to_start = start_heading_rad - turn_multiplier * math.pi / 2
        angle_from_center_to_end = angle_from_center_to_start + turn_multiplier * curve_angle_rad
        
        end_x = center_x + curve_radius * math.cos(angle_from_center_to_end)
        end_y = center_y + curve_radius * math.sin(angle_from_center_to_end)
        
        return ((end_x, end_y), end_heading)
    
    def get_track_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if not self.segments:
            return ((0, 0), (0, 0))
        
        all_points = []
        
        for seg in self.segments:
            # Add start and end positions
            all_points.append(seg.start_position)
            all_points.append(seg.end_position)
            
            # For curved segments, also consider the curve extremes
            if seg.segment_type == "CURVE" and seg.curve_radius > 0:
                curve_bounds = self._get_curve_bounds(seg)
                all_points.extend(curve_bounds)
        
        # Calculate bounds including track width
        min_x = min(point[0] for point in all_points) - self.width/2
        max_x = max(point[0] for point in all_points) + self.width/2
        min_y = min(point[1] for point in all_points) - self.width/2
        max_y = max(point[1] for point in all_points) + self.width/2
        
        return ((min_x, min_y), (max_x, max_y))
    
    def _get_curve_bounds(self, segment: TrackSegment) -> List[Tuple[float, float]]:
        """Get additional points for curve bounds calculation"""
        if segment.curve_radius <= 0:
            return []
        
        # Calculate curve center
        start_heading_rad = math.radians(segment.start_heading)
        if segment.curve_direction == "LEFT":
            turn_multiplier = 1.0
        else:  # RIGHT
            turn_multiplier = -1.0
        
        perpendicular_heading = start_heading_rad + turn_multiplier * math.pi / 2
        center_x = segment.start_position[0] + segment.curve_radius * math.cos(perpendicular_heading)
        center_y = segment.start_position[1] + segment.curve_radius * math.sin(perpendicular_heading)
        
        # Generate points along the curve arc
        curve_points = []
        angle_from_center_to_start = start_heading_rad - turn_multiplier * math.pi / 2
        curve_angle_rad = math.radians(segment.curve_angle)
        
        # Sample points along the curve
        num_samples = max(4, int(abs(segment.curve_angle) / 15))  # Sample every ~15 degrees
        for i in range(num_samples + 1):
            t = i / num_samples
            angle = angle_from_center_to_start + turn_multiplier * curve_angle_rad * t
            x = center_x + segment.curve_radius * math.cos(angle)
            y = center_y + segment.curve_radius * math.sin(angle)
            curve_points.append((x, y))
        
        return curve_points
    
    def get_total_track_length(self) -> float:
        """Get the total length of the track including all segments"""
        return self.total_length
    
    def get_start_to_finish_length(self) -> float:
        """Get the length from start of STARTLINE to end of FINISHLINE"""
        if not self.has_startline:
            return 0.0
        
        start_index = None
        finish_index = None
        grid_index = None
        
        # Find GRID, STARTLINE and FINISHLINE indices
        for i, segment in enumerate(self.segments):
            if segment.segment_type == "GRID" and grid_index is None:
                grid_index = i
            elif segment.segment_type == "STARTLINE" and start_index is None:
                start_index = i
            elif segment.segment_type == "FINISHLINE":
                finish_index = i
        
        if start_index is None:
            return 0.0
        
        # Determine calculation start point based on track type
        if finish_index is None:
            # Circular track: include GRID if it exists, race through entire circuit
            calculation_start = grid_index if grid_index is not None else start_index
            finish_index = len(self.segments) - 1
        else:
            # Linear track: exclude GRID, race from STARTLINE to FINISHLINE
            calculation_start = start_index
        
        # Calculate total length 
        total_length = 0.0
        for i in range(calculation_start, finish_index + 1):
            total_length += self.segments[i].length
        
        return total_length


class TrackLoader:
    def __init__(self):
        pass
    
    def load_track(self, file_path: str) -> Track:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Track file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        track = Track()
        
        for line in lines:
            line = line.strip().upper()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Remove inline comments
            if '#' in line:
                comment_index = line.find('#')
                line = line[:comment_index].strip()
                parts = line.split()
                if not parts:
                    continue
            
            command = parts[0]
            
            if command == "WIDTH":
                if len(parts) != 2:
                    raise ValueError(f"WIDTH command requires exactly one argument: {line}")
                try:
                    track.width = float(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid width value: {parts[1]}")
                    
            elif command == "GRID":
                track.add_segment("GRID", DEFAULT_GRID_LENGTH)
                
            elif command == "STARTLINE":
                track.add_segment("STARTLINE", STARTLINE_LENGTH)
                
            elif command == "STRAIGHT":
                if len(parts) != 2:
                    raise ValueError(f"STRAIGHT command requires exactly one argument: {line}")
                try:
                    length = float(parts[1])
                    track.add_segment("STRAIGHT", length)
                except ValueError:
                    raise ValueError(f"Invalid straight length value: {parts[1]}")
                    
            elif command == "FINISHLINE":
                track.add_segment("FINISHLINE", FINISHLINE_LENGTH)
                
            elif command in ["LEFT", "RIGHT"]:
                if len(parts) != 3:
                    raise ValueError(f"{command} command requires exactly two arguments (angle, radius): {line}")
                try:
                    angle = float(parts[1])
                    radius = float(parts[2])
                    
                    # Validate parameters
                    if angle <= 0 or angle > 360:
                        raise ValueError(f"Curve angle must be between 0 and 360 degrees: {angle}")
                    if radius <= 0:
                        raise ValueError(f"Curve radius must be positive: {radius}")
                    
                    track.add_segment("CURVE", 0, curve_angle=angle, curve_radius=radius, curve_direction=command)
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid numeric values for {command} command: {parts[1]}, {parts[2]}")
                    else:
                        raise
                
            else:
                raise ValueError(f"Unknown command: {command}")
        
        return track
    
    def validate_track(self, track: Track) -> bool:
        if not track.segments:
            raise ValueError("Track must have at least one segment")
        
        # Check if track has a grid (should be first)
        if track.segments[0].segment_type != "GRID":
            raise ValueError("Track must start with GRID")
        
        # Check if track has startline
        if not track.has_startline:
            raise ValueError("Track must have a STARTLINE")
        
        # If no finishline, startline serves as finishline (circular track)
        return True