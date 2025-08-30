"""
Optimized Observation Visualizer for Car Racing Environment

This module provides high-performance real-time visualization of car observations
using pygame directly instead of matplotlib for better performance.
"""

import numpy as np
import pygame
from collections import deque
from typing import List, Optional, Tuple, Dict
import math

# Import constants for observation structure
from .constants import SENSOR_NUM_DIRECTIONS

class ObservationVisualizerOptimized:
    """High-performance real-time visualization of car observations using pure pygame.

    This class is responsible for visualizing the car's observation space in real-time.
    It uses pygame for rendering to ensure high performance.

    Args:
        history_length (int): The number of data points to keep in the rolling history.
        graph_width (int): The width of each graph in pixels.
        graph_height (int): The height of each graph in pixels.
    """
    
    def __init__(self, history_length: int = 300, graph_width: int = 500, graph_height: int = 350):
        """
        Initializes the optimized observation visualizer.
        
        Args:
            history_length (int): The number of data points to keep in the rolling history.
            graph_width (int): The width of each graph in pixels.
            graph_height (int): The height of each graph in pixels.
        """
        self.history_length = history_length
        self.graph_width = graph_width
        self.graph_height = graph_height
        
        # Data storage - per-car observation histories
        self.car_observation_histories = {}  # Dict[car_id, List[observation]]
        self.car_time_histories = {}  # Dict[car_id, List[time]]
        self.max_display_points = history_length  # Still limit what we display for performance
        self.current_time = 0.0
        self.current_car_id = 0  # Track which car's data to display
        
        # Graph categories and their observation indices
        self.graph_categories = {
            "Position & Motion": {
                "indices": [0, 1, 2, 3, 4, 5, 6],  # pos_x, pos_y, vel_x, vel_y, speed, orientation, angular_vel
                "labels": ["Position X", "Position Y", "Velocity X", "Velocity Y", "Speed", "Orientation", "Angular Velocity"],
                "colors": [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 200, 100), 
                          (255, 100, 255), (100, 255, 255), (255, 255, 100)]
            },
            "Tire Loads": {
                "indices": [7, 8, 9, 10],  # Front-left, front-right, rear-left, rear-right
                "labels": ["Front Left", "Front Right", "Rear Left", "Rear Right"],
                "colors": [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 200, 100)]
            },
            "Tire Temperatures": {
                "indices": [11, 12, 13, 14],  # Same order as loads
                "labels": ["Front Left", "Front Right", "Rear Left", "Rear Right"],
                "colors": [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 200, 100)]
            },
            "Tire Wear": {
                "indices": [15, 16, 17, 18],  # Same order as loads
                "labels": ["Front Left", "Front Right", "Rear Left", "Rear Right"],
                "colors": [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 200, 100)]
            },
            "Collision Data": {
                "indices": [19, 20, 21],  # collision_impulse, collision_angle, cumulative_impact
                "labels": ["Collision Impulse", "Collision Angle", "Cumulative Impact"],
                "colors": [(255, 100, 100), (255, 200, 100), (200, 100, 100)]
            },
            "Distance Sensors": {
                "indices": list(range(22, 22 + SENSOR_NUM_DIRECTIONS)),  # 16 sensor distances
                "labels": [f"Sensor {i}" for i in range(SENSOR_NUM_DIRECTIONS)],
                "colors": [(100 + i * 10, 200 - i * 5, 150 + i * 7) for i in range(SENSOR_NUM_DIRECTIONS)]  # Varied colors
            }
        }
        
        # Graph surfaces (will be created/resized as needed)
        self.graph_surfaces = {}
        self.current_screen_size = None  # Track screen size changes
        
        # Frame counter for update throttling
        self.frame_counter = 0
        self.update_frequency = 2  # Update graphs every N frames (was 3, now 2 for more frequent updates)
        
        # Font for labels (larger for better readability)
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 22)
        
    def add_observation(self, observation: np.ndarray, time: float, car_id: int = 0) -> None:
        """
        Adds a new observation to the history for a specific car.
        
        Args:
            observation (np.ndarray): The normalized observation array from the car environment.
            time (float): The current simulation time.
            car_id (int): The ID of the car this observation belongs to.
        """
        # Initialize history for this car if it doesn't exist
        if car_id not in self.car_observation_histories:
            self.car_observation_histories[car_id] = []
            self.car_time_histories[car_id] = []
        
        self.car_observation_histories[car_id].append(observation.copy())
        self.car_time_histories[car_id].append(time)
        self.current_time = time
    
    def set_displayed_car(self, car_id: int) -> None:
        """
        Sets which car's observation history to display.
        
        Args:
            car_id (int): The ID of the car whose data should be displayed.
        """
        self.current_car_id = car_id
        
    def _draw_graph_fast(self, surface: pygame.Surface, category_name: str, category_data: dict) -> None:
        """
        Draws a graph directly using pygame primitives for maximum performance.
        
        Args:
            surface (pygame.Surface): The pygame Surface to draw on.
            category_name (str): The name of the observation category.
            category_data (dict): A dictionary containing indices, labels, and colors.
        """
        # Clear surface with dark background
        surface.fill((20, 20, 20))
        
        # Get data for the current car
        observation_history = self.car_observation_histories.get(self.current_car_id, [])
        time_history = self.car_time_histories.get(self.current_car_id, [])
        
        if len(observation_history) < 2:
            # Not enough data - show loading text with car info
            car_text = f"Car {self.current_car_id}: Collecting data..."
            text = self.font.render(car_text, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.graph_width//2, self.graph_height//2))
            surface.blit(text, text_rect)
            return
        
        # Draw graph border
        pygame.draw.rect(surface, (100, 100, 100), (0, 0, self.graph_width, self.graph_height), 1)
        
        # Draw title
        title_text = self.title_font.render(category_name, True, (255, 255, 255))
        surface.blit(title_text, (5, 5))
        
        # Calculate graph area (leave margins for labels)
        margin_left = 40
        margin_right = 20  
        margin_top = 25
        margin_bottom = 20
        graph_area_width = self.graph_width - margin_left - margin_right
        graph_area_height = self.graph_height - margin_top - margin_bottom
        
        if graph_area_width <= 0 or graph_area_height <= 0:
            return
        
        # Extract time data
        times = np.array(list(time_history))
        
        # Fix time range calculation - ensure we always show meaningful range
        if len(times) > 1:
            time_range = times[-1] - times[0]
            # If time range is very small, use a minimum range to show data properly
            if time_range < 1.0:
                time_range = max(1.0, len(times) * 0.016)  # Assume 60 FPS, minimum 1 second range
        else:
            time_range = 1.0
        
        # Get data for this category
        indices = category_data["indices"]
        colors = category_data["colors"]
        
        # For distance sensors, show only a subset to avoid clutter
        if category_name == "Distance Sensors":
            # Show every 4th sensor for better visibility
            indices = indices[::4]  
            colors = colors[::4]
        
        # Intelligently sample data points if we have too many for good performance
        total_points = len(observation_history)
        if total_points > self.max_display_points:
            # Sample evenly across the full timeline to show the complete episode
            step = total_points // self.max_display_points
            sample_indices = list(range(0, total_points, step))
            # Always include the last point to show current state
            if sample_indices[-1] != total_points - 1:
                sample_indices.append(total_points - 1)
        else:
            # Show all data if it's not too much
            sample_indices = list(range(total_points))

        # Draw each data series
        for i, (obs_index, color) in enumerate(zip(indices, colors)):
            if not observation_history or obs_index >= len(observation_history[0]):
                continue
                
            # Extract data for this observation component using sampled indices
            data_points = []
            time_points = []
            
            for j in sample_indices:
                obs = observation_history[j]
                if obs_index < len(obs):
                    data_points.append(obs[obs_index])
                    time_points.append(times[j])
            
            if len(data_points) < 2:
                continue
                
            data_points = np.array(data_points)
            time_points = np.array(time_points)
            
            # Scale data to graph area
            # Time scaling (X-axis)
            try:
                if time_range > 0:
                    x_coords = margin_left + (time_points - times[0]) / time_range * graph_area_width
                else:
                    x_coords = np.full_like(time_points, margin_left + graph_area_width // 2)
                
                # Data scaling (Y-axis) - assume normalized data [-1, 1]
                # Clamp data to valid range to avoid issues
                data_points_clamped = np.clip(data_points, -1.0, 1.0)
                y_coords = margin_top + (1.0 - data_points_clamped) / 2.0 * graph_area_height
            except (ValueError, TypeError, ZeroDivisionError):
                continue  # Skip this data series if there are issues
            
            # Draw the line using pygame
            try:
                if len(x_coords) > 1:
                    points = list(zip(x_coords.astype(int), y_coords.astype(int)))
                    # Filter out invalid points with safer comparisons
                    valid_points = []
                    for x, y in points:
                        if (not np.isnan(x) and not np.isnan(y) and 
                            0 <= x < self.graph_width and 0 <= y < self.graph_height):
                            valid_points.append((x, y))
                    points = valid_points
                
                    if len(points) > 1:
                        pygame.draw.lines(surface, color, False, points, 2)
            except (ValueError, TypeError, AttributeError):
                # Skip this data series if there are any issues
                continue
        
        # Draw Y-axis labels with category-specific ranges
        if category_name == "Distance Sensors":
            y_labels = ["Max", "Mid", "Min"]  # Distance sensors show max detection range
        elif category_name == "Tire Temperatures":
            y_labels = ["Hot", "Warm", "Cold"]
        elif category_name == "Collision Data":
            y_labels = ["High", "Med", "Low"]
        else:
            y_labels = ["1.0", "0.0", "-1.0"]  # Default normalized range
            
        for i, label in enumerate(y_labels):
            y = margin_top + i * graph_area_height // 2
            label_surface = self.font.render(label, True, (200, 200, 200))
            surface.blit(label_surface, (2, y - 8))
        
        # Draw center line (Y=0)
        center_y = margin_top + graph_area_height // 2
        pygame.draw.line(surface, (80, 80, 80), 
                        (margin_left, center_y), 
                        (margin_left + graph_area_width, center_y), 1)
        
        # Draw legend if we have space and multiple data series
        labels = category_data["labels"]
        if len(indices) > 1 and self.graph_height > 200:  # Only show legend if graph is tall enough
            self._draw_legend(surface, labels[:len(indices)], colors[:len(indices)])
    
    def _draw_legend(self, surface: pygame.Surface, labels: List[str], colors: List[Tuple[int, int, int]]) -> None:
        """
        Draws a legend for the graph showing color-coded labels.
        
        Args:
            surface (pygame.Surface): The pygame Surface to draw on.
            labels (List[str]): A list of data series labels.
            colors (List[Tuple[int, int, int]]): A list of colors corresponding to each label.
        """
        if not labels or not colors:
            return
        
        # Legend positioning - bottom right corner of graph
        legend_width = 150
        legend_item_height = 16
        legend_height = len(labels) * legend_item_height + 10
        legend_x = self.graph_width - legend_width - 5
        legend_y = self.graph_height - legend_height - 5
        
        # Don't draw legend if it would overlap with graph area too much
        if legend_y < 50:
            return
        
        # Draw legend background (semi-transparent dark background)
        legend_bg = pygame.Surface((legend_width, legend_height))
        legend_bg.set_alpha(180)
        legend_bg.fill((0, 0, 0))
        surface.blit(legend_bg, (legend_x, legend_y))
        
        # Draw legend border
        pygame.draw.rect(surface, (100, 100, 100), 
                        (legend_x, legend_y, legend_width, legend_height), 1)
        
        # Draw each legend item
        for i, (label, color) in enumerate(zip(labels, colors)):
            item_y = legend_y + 5 + i * legend_item_height
            
            # Draw color indicator (small line)
            line_start_x = legend_x + 5
            line_end_x = legend_x + 20
            line_y = item_y + legend_item_height // 2
            pygame.draw.line(surface, color, (line_start_x, line_y), (line_end_x, line_y), 3)
            
            # Draw label text (truncate if too long)
            label_text = label[:18] + "..." if len(label) > 18 else label
            text_surface = self.font.render(label_text, True, (255, 255, 255))
            surface.blit(text_surface, (legend_x + 25, item_y))
            
    def update_graphs(self, screen_width: int = None, screen_height: int = None) -> None:
        """Updates all graph surfaces with the current data (throttled for performance)."""
        self.frame_counter += 1
        
        # Only update every N frames to maintain performance
        if self.frame_counter % self.update_frequency != 0:
            return
        
        # Check if screen size changed and update graph dimensions
        current_size = (screen_width, screen_height) if screen_width and screen_height else None
        if current_size and current_size != self.current_screen_size:
            self._update_graph_sizes(screen_width, screen_height)
            self.current_screen_size = current_size
            
        # Ensure surfaces exist for all categories
        for category_name in self.graph_categories:
            if category_name not in self.graph_surfaces:
                self.graph_surfaces[category_name] = pygame.Surface((self.graph_width, self.graph_height))
            
        for category_name, category_data in self.graph_categories.items():
            surface = self.graph_surfaces[category_name]
            self._draw_graph_fast(surface, category_name, category_data)
            
    def _update_graph_sizes(self, screen_width: int, screen_height: int) -> None:
        """Updates the graph dimensions based on the screen size and recreates the surfaces."""
        # Recalculate optimal graph size for current screen
        # This also updates self.graph_width and self.graph_height
        self.get_layout_positions(screen_width, screen_height)
        
        # Recreate surfaces with new dimensions
        for category_name in self.graph_categories:
            self.graph_surfaces[category_name] = pygame.Surface((self.graph_width, self.graph_height))
            
    def get_graph_surfaces(self) -> dict:
        """
        Gets all the graph surfaces for rendering.
        
        Returns:
            A dictionary mapping category names to pygame surfaces.
        """
        return self.graph_surfaces.copy()
        
    def clear_history(self, car_id: int = None) -> None:
        """
        Clears the observation history for a specific car or all cars.
        
        Args:
            car_id (int, optional): The ID of the car to clear the history for. If None, clears all cars.
        """
        if car_id is None:
            # Clear all cars' histories
            self.car_observation_histories.clear()
            self.car_time_histories.clear()
        else:
            # Clear specific car's history
            if car_id in self.car_observation_histories:
                self.car_observation_histories[car_id].clear()
            if car_id in self.car_time_histories:
                self.car_time_histories[car_id].clear()
        
        self.current_time = 0.0
        self.frame_counter = 0
        
    def get_layout_positions(self, screen_width: int, screen_height: int) -> dict:
        """
        Calculates the optimal positions for arranging the graphs on the screen.
        
        Args:
            screen_width (int): The available screen width.
            screen_height (int): The available screen height.
            
        Returns:
            A dictionary mapping category names to (x, y) positions.
        """
        # Arrange graphs in a 2x3 grid
        cols = 2
        rows = 3
        
        # Calculate spacing and margins (reduced for more space)
        margin_x = 10
        margin_y = 30  # Top margin for title
        spacing_x = 15
        spacing_y = 15
        
        available_width = screen_width - (2 * margin_x) - ((cols - 1) * spacing_x)
        available_height = screen_height - (2 * margin_y) - ((rows - 1) * spacing_y)
        
        graph_width = available_width // cols
        graph_height = available_height // rows
        
        # Update graph dimensions if needed
        self.graph_width = graph_width
        self.graph_height = graph_height
        
        positions = {}
        categories = list(self.graph_categories.keys())
        
        for i, category in enumerate(categories):
            row = i // cols
            col = i % cols
            
            x = margin_x + col * (graph_width + spacing_x)
            y = margin_y + row * (graph_height + spacing_y)
            
            positions[category] = (x, y)
            
        return positions