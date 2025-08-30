"""
Track Builder GUI

Interactive track visualization and analysis tool with GUI interface.
Provides real-time track display, analysis information, and validation feedback.
"""

import pygame
import math
import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.track_generator import Track, TrackLoader
from src.centerline_generator import CenterlineGenerator
from src.track_boundary import TrackBoundary
from tools.track_analyzer import TrackAnalyzer, TrackStatistics
from tools.track_validator import TrackValidator, ValidationResult


class ViewMode(Enum):
    """Available view modes for track display."""
    OVERHEAD = "overhead"
    CENTERLINE = "centerline"
    BOUNDARIES = "boundaries"
    SEGMENTS = "segments"


class TrackBuilderGUI:
    """Interactive track builder and analyzer GUI."""
    
    def __init__(self):
        """Initialize the track builder GUI."""
        pygame.init()
        
        # Window settings
        self.width = 1200
        self.height = 800
        self.info_panel_width = 300
        self.track_area_width = self.width - self.info_panel_width
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Track Builder & Analyzer")
        
        # Initialize components
        self.loader = TrackLoader()
        self.analyzer = TrackAnalyzer()
        self.validator = TrackValidator()
        self.centerline_generator = CenterlineGenerator()
        self.boundary_generator = TrackBoundary()
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Colors
        self.colors = {
            'background': (40, 40, 40),
            'panel_bg': (60, 60, 60),
            'panel_border': (100, 100, 100),
            'text': (255, 255, 255),
            'text_highlight': (255, 255, 0),
            'text_error': (255, 100, 100),
            'text_warning': (255, 200, 100),
            'text_success': (100, 255, 100),
            'track': (128, 128, 128),
            'centerline': (255, 255, 0),
            'boundary_left': (100, 200, 255),
            'boundary_right': (255, 100, 100),
            'grid': (192, 192, 192),
            'startline': (0, 255, 0),
            'finishline': (255, 255, 0),
            'curve': (150, 100, 255),
            'straight': (100, 255, 150)
        }
        
        # State
        self.current_track: Optional[Track] = None
        self.current_stats: Optional[TrackStatistics] = None
        self.current_validation: Optional[ValidationResult] = None
        self.track_name = "No track loaded"
        self.current_track_index = 0
        
        # View settings
        self.view_mode = ViewMode.OVERHEAD
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_info = True
        
        # Available track files
        self.available_tracks = self._find_available_tracks()
        self.status_message = ""
        self.status_timer = 0
        
        # Generated data
        self.centerline = []
        self.boundaries = ([], [])
        self.track_polygon = []
        
        self.clock = pygame.time.Clock()
        self.running = True
    
    def _find_available_tracks(self) -> List[str]:
        """Find all available track files in the tracks directory."""
        tracks_dir = "tracks"
        available = []
        
        if os.path.exists(tracks_dir):
            for file in os.listdir(tracks_dir):
                if file.endswith('.track'):
                    available.append(os.path.join(tracks_dir, file))
        
        return sorted(available)
    
    def load_track_file(self, file_path: str) -> bool:
        """
        Load a track file and analyze it.
        
        Args:
            file_path (str): Path to the track file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load track
            self.current_track = self.loader.load_track(file_path)
            self.track_name = os.path.basename(file_path)
            
            # Analyze track
            self.current_stats = self.analyzer.analyze_track(self.current_track)
            
            # Validate track
            self.current_validation = self.validator.validate_track(self.current_track)
            
            # Generate visualization data
            self._generate_visualization_data()
            
            # Auto-fit view
            self._auto_fit_view()
            
            # Show status message
            self.status_message = f"Loaded: {self.track_name}"
            self.status_timer = pygame.time.get_ticks()
            
            return True
            
        except Exception as e:
            self.status_message = f"Error loading {os.path.basename(file_path)}: {str(e)}"
            self.status_timer = pygame.time.get_ticks()
            print(f"Error loading track: {e}")
            return False
    
    def load_next_track(self):
        """Load the next available track from the tracks directory."""
        if not self.available_tracks:
            self.status_message = "No track files found in tracks/ directory"
            self.status_timer = pygame.time.get_ticks()
            return
        
        # Cycle to next track
        self.current_track_index = (self.current_track_index + 1) % len(self.available_tracks)
        track_path = self.available_tracks[self.current_track_index]
        
        self.load_track_file(track_path)
    
    def _generate_visualization_data(self):
        """Generate data for track visualization."""
        if not self.current_track:
            return
        
        try:
            # Generate centerline
            self.centerline = self.centerline_generator.generate_centerline(self.current_track)
            
            # Generate boundaries
            if self.centerline:
                self.boundaries = self.boundary_generator.generate_boundaries(
                    self.centerline, self.current_track.width
                )
                
                # Create track polygon
                self.track_polygon = self.boundary_generator.create_track_polygon(
                    self.boundaries[0], self.boundaries[1]
                )
        except Exception as e:
            print(f"Error generating visualization data: {e}")
    
    def _auto_fit_view(self):
        """Auto-fit the view to show the entire track."""
        if not self.current_track:
            return
        
        bounds = self.current_track.get_track_bounds()
        track_width = bounds[1][0] - bounds[0][0]
        track_height = bounds[1][1] - bounds[0][1]
        
        # Calculate zoom to fit track in view area with margin
        margin = 50
        zoom_x = (self.track_area_width - 2 * margin) / track_width if track_width > 0 else 1.0
        zoom_y = (self.height - 2 * margin) / track_height if track_height > 0 else 1.0
        
        self.zoom = min(zoom_x, zoom_y) * 0.9  # 90% to ensure margins
        
        # Center the track
        track_center_x = (bounds[0][0] + bounds[1][0]) / 2
        track_center_y = (bounds[0][1] + bounds[1][1]) / 2
        
        self.pan_x = self.track_area_width / 2 - track_center_x * self.zoom
        self.pan_y = self.height / 2 - track_center_y * self.zoom
    
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(world_pos[0] * self.zoom + self.pan_x)
        screen_y = int(world_pos[1] * self.zoom + self.pan_y)
        return (screen_x, screen_y)
    
    def draw_track(self):
        """Draw the track based on current view mode."""
        if not self.current_track:
            return
        
        # Draw track polygon (background)
        if self.track_polygon and self.view_mode in [ViewMode.OVERHEAD, ViewMode.BOUNDARIES]:
            screen_polygon = [self.world_to_screen(pos) for pos in self.track_polygon]
            if len(screen_polygon) > 2:
                try:
                    pygame.draw.polygon(self.screen, self.colors['track'], screen_polygon)
                except:
                    pass  # Skip if polygon is invalid
        
        if self.view_mode == ViewMode.CENTERLINE:
            # Draw centerline
            if len(self.centerline) > 1:
                screen_points = [self.world_to_screen(pos) for pos in self.centerline]
                pygame.draw.lines(self.screen, self.colors['centerline'], False, screen_points, 2)
        
        elif self.view_mode == ViewMode.BOUNDARIES:
            # Draw boundaries
            if self.boundaries[0] and len(self.boundaries[0]) > 1:
                screen_left = [self.world_to_screen(pos) for pos in self.boundaries[0]]
                pygame.draw.lines(self.screen, self.colors['boundary_left'], False, screen_left, 2)
            
            if self.boundaries[1] and len(self.boundaries[1]) > 1:
                screen_right = [self.world_to_screen(pos) for pos in self.boundaries[1]]
                pygame.draw.lines(self.screen, self.colors['boundary_right'], False, screen_right, 2)
        
        elif self.view_mode == ViewMode.SEGMENTS:
            # Draw individual segments with different colors
            for i, segment in enumerate(self.current_track.segments):
                start_screen = self.world_to_screen(segment.start_position)
                end_screen = self.world_to_screen(segment.end_position)
                
                # Choose color based on segment type
                if segment.segment_type == "GRID":
                    color = self.colors['grid']
                elif segment.segment_type == "STARTLINE":
                    color = self.colors['startline']
                elif segment.segment_type == "FINISHLINE":
                    color = self.colors['finishline']
                elif segment.segment_type == "CURVE":
                    color = self.colors['curve']
                else:
                    color = self.colors['straight']
                
                pygame.draw.line(self.screen, color, start_screen, end_screen, 3)
                
                # Draw segment number
                mid_x = (start_screen[0] + end_screen[0]) // 2
                mid_y = (start_screen[1] + end_screen[1]) // 2
                text = self.font_small.render(str(i), True, self.colors['text'])
                self.screen.blit(text, (mid_x - 5, mid_y - 5))
    
    def draw_info_panel(self):
        """Draw the information panel."""
        panel_rect = pygame.Rect(self.track_area_width, 0, self.info_panel_width, self.height)
        pygame.draw.rect(self.screen, self.colors['panel_bg'], panel_rect)
        pygame.draw.line(self.screen, self.colors['panel_border'], 
                        (self.track_area_width, 0), (self.track_area_width, self.height), 2)
        
        y_pos = 10
        x_pos = self.track_area_width + 10
        line_height = 20
        
        # Track name
        text = self.font_large.render(self.track_name, True, self.colors['text_highlight'])
        self.screen.blit(text, (x_pos, y_pos))
        y_pos += 30
        
        # View controls
        text = self.font_medium.render("VIEW CONTROLS:", True, self.colors['text'])
        self.screen.blit(text, (x_pos, y_pos))
        y_pos += line_height
        
        text = self.font_small.render(f"Mode: {self.view_mode.value}", True, self.colors['text'])
        self.screen.blit(text, (x_pos, y_pos))
        y_pos += line_height
        
        text = self.font_small.render(f"Zoom: {self.zoom:.2f}x", True, self.colors['text'])
        self.screen.blit(text, (x_pos, y_pos))
        y_pos += line_height + 10
        
        # Track statistics
        if self.current_stats:
            text = self.font_medium.render("TRACK STATISTICS:", True, self.colors['text'])
            self.screen.blit(text, (x_pos, y_pos))
            y_pos += line_height
            
            stats_lines = [
                f"Length: {self.current_stats.total_length:.1f}m",
                f"Segments: {self.current_stats.segment_count}",
                f"Curves: {self.current_stats.curve_count}",
                f"Width: {self.current_stats.average_width:.1f}m",
                f"Area: {self.current_stats.track_area:.0f}m²",
                f"Lap Time: {self.current_stats.estimated_lap_time:.1f}s",
                f"Difficulty: {self.current_stats.technical_difficulty:.1f}/10"
            ]
            
            for line in stats_lines:
                text = self.font_small.render(line, True, self.colors['text'])
                self.screen.blit(text, (x_pos, y_pos))
                y_pos += line_height
        
        y_pos += 10
        
        # Validation results
        if self.current_validation:
            text = self.font_medium.render("VALIDATION:", True, self.colors['text'])
            self.screen.blit(text, (x_pos, y_pos))
            y_pos += line_height
            
            # Status
            if self.current_validation.is_valid:
                text = self.font_small.render("✓ Valid", True, self.colors['text_success'])
            else:
                text = self.font_small.render("✗ Invalid", True, self.colors['text_error'])
            self.screen.blit(text, (x_pos, y_pos))
            y_pos += line_height
            
            # Errors
            if self.current_validation.errors:
                text = self.font_small.render("Errors:", True, self.colors['text_error'])
                self.screen.blit(text, (x_pos, y_pos))
                y_pos += line_height
                
                for error in self.current_validation.errors[:3]:  # Show max 3 errors
                    wrapped_lines = self._wrap_text(error, self.info_panel_width - 30)
                    for line in wrapped_lines:
                        text = self.font_small.render(f"• {line}", True, self.colors['text_error'])
                        self.screen.blit(text, (x_pos + 10, y_pos))
                        y_pos += line_height
            
            # Warnings
            if self.current_validation.warnings:
                text = self.font_small.render("Warnings:", True, self.colors['text_warning'])
                self.screen.blit(text, (x_pos, y_pos))
                y_pos += line_height
                
                for warning in self.current_validation.warnings[:2]:  # Show max 2 warnings
                    wrapped_lines = self._wrap_text(warning, self.info_panel_width - 30)
                    for line in wrapped_lines:
                        text = self.font_small.render(f"• {line}", True, self.colors['text_warning'])
                        self.screen.blit(text, (x_pos + 10, y_pos))
                        y_pos += line_height
        
        # Available tracks
        y_pos += 10
        if self.available_tracks:
            text = self.font_medium.render("AVAILABLE TRACKS:", True, self.colors['text'])
            self.screen.blit(text, (x_pos, y_pos))
            y_pos += line_height
            
            for i, track_path in enumerate(self.available_tracks):
                track_name = os.path.basename(track_path)
                color = self.colors['text_highlight'] if i == self.current_track_index else self.colors['text']
                indicator = "► " if i == self.current_track_index else "  "
                text = self.font_small.render(f"{indicator}{track_name}", True, color)
                self.screen.blit(text, (x_pos, y_pos))
                y_pos += line_height
        
        # Controls help
        y_pos = self.height - 120
        text = self.font_medium.render("CONTROLS:", True, self.colors['text'])
        self.screen.blit(text, (x_pos, y_pos))
        y_pos += line_height
        
        controls = [
            "O - Next track",
            "M - Change view mode", 
            "R - Reset view",
            "Mouse wheel - Zoom",
            "Click & drag - Pan"
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, self.colors['text'])
            self.screen.blit(text, (x_pos, y_pos))
            y_pos += line_height
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within specified width."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " " if current_line else word
            if len(test_line) * 8 < max_width:  # Rough character width estimation
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def draw_status_message(self):
        """Draw status message at the bottom of the screen."""
        if self.status_message and self.status_timer > 0:
            current_time = pygame.time.get_ticks()
            if current_time - self.status_timer < 3000:  # Show for 3 seconds
                # Draw semi-transparent background
                status_bg = pygame.Surface((self.track_area_width, 30))
                status_bg.set_alpha(128)
                status_bg.fill(self.colors['panel_bg'])
                self.screen.blit(status_bg, (0, self.height - 30))
                
                # Draw status text
                text = self.font_medium.render(self.status_message, True, self.colors['text_highlight'])
                text_rect = text.get_rect()
                text_rect.center = (self.track_area_width // 2, self.height - 15)
                self.screen.blit(text, text_rect)
            else:
                self.status_message = ""
                self.status_timer = 0
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    # Cycle through available tracks
                    self.load_next_track()
                
                elif event.key == pygame.K_m:
                    # Change view mode
                    modes = list(ViewMode)
                    current_index = modes.index(self.view_mode)
                    self.view_mode = modes[(current_index + 1) % len(modes)]
                
                elif event.key == pygame.K_r:
                    # Reset view
                    self._auto_fit_view()
                
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.zoom *= zoom_factor
                self.zoom = max(0.1, min(self.zoom, 10.0))
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.drag_start = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if hasattr(self, 'dragging') and self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.pan_x += dx
                    self.pan_y += dy
                    self.drag_start = event.pos
    
    def run(self):
        """Run the main application loop."""
        # Load the first available track if any exist
        if self.available_tracks:
            self.current_track_index = 0
            self.load_track_file(self.available_tracks[0])
        else:
            self.status_message = "No track files found in tracks/ directory"
            self.status_timer = pygame.time.get_ticks()
        
        while self.running:
            self.handle_events()
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw track
            self.draw_track()
            
            # Draw info panel
            if self.show_info:
                self.draw_info_panel()
            
            # Draw status message
            self.draw_status_message()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main entry point."""
    app = TrackBuilderGUI()
    app.run()


if __name__ == "__main__":
    main()