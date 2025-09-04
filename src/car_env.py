"""
Main car racing environment.

This module provides the complete car racing simulation environment with
realistic physics, track integration, and comprehensive observation space.
"""

import time
import math
import numpy as np
import pygame
import os
import random
import glob
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .base_env import BaseEnv
from .car import Car
from .car_physics import CarPhysics
from .collision import CollisionReporter
from .track_generator import TrackLoader
from .renderer import Renderer
from .distance_sensor import DistanceSensor
from .lap_timer import LapTimer
from .observation_visualizer_optimized import ObservationVisualizerOptimized as ObservationVisualizer
from .constants import (
    DEFAULT_WINDOW_SIZE,
    RENDER_MODE_HUMAN,
    TYRE_START_TEMPERATURE,
    CAR_MASS,
    GRAVITY_MS2,
    SENSOR_MAX_DISTANCE,
    SENSOR_NUM_DIRECTIONS,
    STUCK_SPEED_THRESHOLD,
    BACKWARD_MOVEMENT_THRESHOLD,
    PENALTY_BACKWARD_PER_METER,
    BACKWARD_DISABLE_THRESHOLD,
    STUCK_TIME_THRESHOLD,
    STUCK_DISTANCE_THRESHOLD,
    STUCK_EXTENDED_TIME_THRESHOLD,
    # Reward constants
    REWARD_DISTANCE_MULTIPLIER,
    REWARD_LAP_COMPLETION,
    REWARD_FAST_LAP_TIME,
    REWARD_FAST_LAP_BONUS,
    PENALTY_PER_STEP,
    PENALTY_WALL_COLLISION_PER_STEP,
    PENALTY_DISABLED,
    # Collision constants
    COLLISION_FORCE_THRESHOLD,
    INSTANT_DISABLE_IMPACT_THRESHOLD,
    CUMULATIVE_DISABLE_IMPACT_THRESHOLD,
    # Termination constants
    TERMINATION_MIN_REWARD,
    TERMINATION_MAX_TIME,
    TRUNCATION_MAX_TIME,
    # Normalization constants
    NORM_MAX_POSITION,
    NORM_MAX_VELOCITY,
    NORM_MAX_ANGULAR_VEL,
    NORM_MAX_TYRE_LOAD,
    NORM_MAX_TYRE_TEMP,
    NORM_MAX_TYRE_WEAR,
    INSTANT_DISABLE_IMPACT_THRESHOLD,
    # Multi-car constants
    MAX_CARS,
    MULTI_CAR_COLORS,
    CAR_SELECT_KEYS,
    # Observation visualization constants
    OBSERVATION_HISTORY_LENGTH
)




class CarEnv(BaseEnv):
    """Complete car racing environment with realistic physics"""
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 track_file: Optional[str] = None,
                 start_position: Optional[Tuple[float, float]] = None,
                 start_angle: float = 0.0,
                 reset_on_lap: bool = False,
                 discrete_action_space: bool = False,
                 num_cars: int = 1,
                 car_names: Optional[list] = None,
                ):
        """
        Initialize car racing environment.
        
        Args:
            render_mode: Rendering mode ("human" or None)
            track_file: Path to track definition file (if None, a random track will be selected automatically)
            start_position: Car starting position (auto-detected if None)
            start_angle: Car starting angle in radians
            reset_on_lap: If True, reset environment automatically when a lap is completed
            discrete_action_space: If True, use discrete action space (5 actions) instead of continuous
            num_cars: Number of cars to create (1-10)
            car_names: List of names for each car (optional, defaults to "Car 0", "Car 1", etc.)
        """
        super().__init__(discrete_action_space=discrete_action_space, num_cars=num_cars)
        
        # Validate num_cars parameter
        if num_cars < 1 or num_cars > MAX_CARS:
            raise ValueError(f"Number of cars must be between 1 and {MAX_CARS}")
        
        self.render_mode = render_mode
        self.track_file = track_file
        # Store original track file to prevent random tracks when explicit file provided
        if track_file:
            self._original_track_file = track_file
        self.track = None
        self.start_position = start_position or (0.0, 0.0)
        self.start_angle = start_angle
        self.reset_on_lap = reset_on_lap
        
        # Multi-car attributes
        self.num_cars = num_cars
        self.followed_car_index = 0  # Index of car being followed by camera and RL agent
        self.car_colors = MULTI_CAR_COLORS[:num_cars]  # Assign colors to each car
        self.car_lap_timers = []  # One lap timer per car
        
        # Initialize car names
        if car_names is None:
            self.car_names = [f"Car {i}" for i in range(num_cars)]
        else:
            if len(car_names) != num_cars:
                raise ValueError(f"Number of car names ({len(car_names)}) must match number of cars ({num_cars})")
            self.car_names = list(car_names)  # Make a copy
        
        # Random track mode tracking
        self._is_random_track_mode = track_file is None
        
        # Load track: explicit file or random selection
        if track_file:
            self._load_track(track_file)
        else:
            # Automatically use random tracks when no track file specified
            self._load_random_track()
            
        # Create physics system
        self.car_physics_worlds = []
        self.cars = []   # List of all cars in the simulation
        
        # Collision system
        self.collision_reporter = CollisionReporter()
        
        # Distance sensor system
        self.distance_sensor = DistanceSensor()
        
        # Lap timing system (create one timer per car)
        self._initialize_lap_timers()
        
        # Rendering system
        self.renderer = None
        
        if render_mode == RENDER_MODE_HUMAN:
            self.renderer = Renderer(
                window_size=DEFAULT_WINDOW_SIZE,
                render_fps=self.metadata["render_fps"],
                track=self.track
            )
            
        # Environment state
        self.simulation_time = 0.0
        
        # Store latest action for physics updates
        self._current_physics_action = None
        
        # Performance tracking
        self.episode_stats = {}
        
        
        # Current action for rendering
        self.current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [throttle, brake, steering]
        self.all_actions = None  # Will store all actions for multi-car scenarios
        
        # Track disabled cars (for multi-car collision handling)
        self.disabled_cars = set()  # Set of car indices that are disabled due to collisions
        self._just_disabled_cars = set() # Set of car indices that were just disabled in the current step
        
        
        # Track cumulative collision impacts for new disabling features
        self.cumulative_collision_impacts = {}  # Dict of car_index -> total accumulated collision impulse
        
        # Lap reset control - prevent immediate reset on first lap
        self._lap_reset_pending = False
        
        # Track previous lap count for lap completion rewards
        self._previous_lap_count = 0
        
        # Track car position for distance-based rewards
        self._previous_car_position = None
        self._total_distance_traveled = 0.0
        
        # Track last penalized collision to avoid double penalties
        self._last_penalized_collision_time = -float('inf')
        
        # Reward display control
        self._show_reward = False
        self._current_reward = 0.0
        self._cumulative_rewards = [0.0]  # Use array for consistency with multi-car
        
        # Observation display control
        self._show_observations = False
        self.observation_visualizer = ObservationVisualizer(history_length=OBSERVATION_HISTORY_LENGTH)
        
        # Termination reason tracking
        self.termination_reason = None
        
        # Backward movement tracking
        self._track_progress_history = [0.0] * self.num_cars  # Previous track progress for each car
        self._backward_distance = [0.0] * self.num_cars  # Accumulated backward distance for each car
        self._previous_backward_distance = [0.0] * self.num_cars  # Previous backward distance for delta calculation
        self._backward_penalty_active = [False] * self.num_cars  # Whether penalty is active for each car
        self._first_step_after_reset = [True] * self.num_cars  # Skip backward penalty on first step
        
        # Environment initialized (logging removed)
    
    def _initialize_lap_timers(self) -> None:
        """Initialize lap timers for all cars"""
        self.car_lap_timers = []
        for i in range(self.num_cars):
            car_name = self.car_names[i] if i < len(self.car_names) else f"Car {i}"
            lap_timer = LapTimer(self.track, car_id=car_name)
            self.car_lap_timers.append(lap_timer)
        
        
    def _load_track(self, track_file: str) -> None:
        """Load track from file"""
        track_loader = TrackLoader()
        self.track = track_loader.load_track(track_file)
        
        # Set start position to track start if not specified
        if self.start_position == (0.0, 0.0) and self.track.segments:
            # Find GRID or STARTLINE segment for starting position
            for segment in self.track.segments:
                if segment.segment_type in ["GRID", "STARTLINE"]:
                    self.start_position = segment.start_position
                    break
    
    def _discover_available_tracks(self) -> List[str]:
        """Discover all available .track files in the tracks directory"""
        if not hasattr(self, '_available_tracks'):
            # Find tracks directory relative to the current module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            tracks_dir = os.path.join(project_root, 'tracks')
            
            # Discover all .track files
            track_pattern = os.path.join(tracks_dir, '*.track')
            track_files = glob.glob(track_pattern)
            
            # Store relative paths from project root
            self._available_tracks = []
            for track_file in track_files:
                relative_path = os.path.relpath(track_file, project_root)
                self._available_tracks.append(relative_path)
            
        
        return self._available_tracks
    
    def _select_random_track(self) -> Optional[str]:
        """Select a random track file from available tracks"""
        available_tracks = self._discover_available_tracks()
        
        if not available_tracks:
            return None
        
        # Ensure we get a different track if possible
        previous_track = getattr(self, 'track_file', None)
        
        # If we have more than one track available, try to avoid repeating
        if len(available_tracks) > 1 and previous_track in available_tracks:
            other_tracks = [t for t in available_tracks if t != previous_track]
            if other_tracks:
                available_tracks = other_tracks
        
        # Add process-specific entropy to avoid synchronized selection in multiprocess environments
        process_entropy = os.getpid() + int(time.time() * 1000) % 1000
        random.seed(process_entropy)
        
        selected_track = random.choice(available_tracks)
        return selected_track
    
    def _load_random_track(self) -> None:
        """Load a randomly selected track"""
        selected_track = self._select_random_track()
        if selected_track:
            self._load_track(selected_track)
            # Store the selected track file for reference, but mark it as random
            self.track_file = selected_track
            self._is_random_track_mode = True  # Flag to indicate we should keep loading random tracks
            
            # Update start position if using default and track has GRID or STARTLINE
            if self.start_position == (0.0, 0.0) and self.track and self.track.segments:
                for segment in self.track.segments:
                    if segment.segment_type in ["GRID", "STARTLINE"]:
                        self.start_position = segment.start_position
                        break
        else:
            self.track = None
    
    def switch_to_random(self):
        """Switch to random track selection mode for training curriculum"""
        print(f"🔄 Switching to random track mode...")
        # Clear fixed track settings to enable random selection
        self.track_file = None
        if hasattr(self, '_original_track_file'):
            delattr(self, '_original_track_file')
        
        # Enable random track mode flag
        self._is_random_track_mode = True
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self.seed(seed_value=seed)
        
        # Load a new random track if in random track mode
        if self._is_random_track_mode:
            previous_track_file = self.track_file
            self._load_random_track()
            
            # Print track name when track is loaded or changed
            if self.track_file:
                track_name = os.path.splitext(os.path.basename(self.track_file))[0]
                track_length = self.track.get_total_track_length() if self.track else 0
                print(f"🏁 Track: {track_name} (length: {track_length:.1f}m)")
            
            # Update renderer with new track if it exists and track changed
            if (self.renderer and 
                self.track and 
                self.track_file != previous_track_file):
                self.renderer.set_track(self.track)
                
                # Update lap timers with new track data
                for lap_timer in self.car_lap_timers:
                    lap_timer.track = self.track
                    lap_timer._find_startline_segment()  # Re-find startline in new track
                    # Recalculate minimum lap distance for new track
                    if self.track:
                        track_length = self.track.get_total_track_length()
                        if track_length > 0:
                            lap_timer.minimum_lap_distance = track_length * lap_timer.minimum_lap_distance_percent
        elif hasattr(self, '_original_track_file') and self.track_file:
            # For explicit tracks, print track name on first reset only
            if not hasattr(self, '_track_name_printed'):
                track_name = os.path.splitext(os.path.basename(self.track_file))[0]
                print(f"🏁 Track: {track_name}")
                self._track_name_printed = True
                
                # Ensure lap timers have correct track data on first reset
                if self.track:
                    for lap_timer in self.car_lap_timers:
                        lap_timer.track = self.track
                        lap_timer._find_startline_segment()  # Re-find startline
                        # Recalculate minimum lap distance
                        track_length = self.track.get_total_track_length()
                        if track_length > 0:
                            lap_timer.minimum_lap_distance = track_length * lap_timer.minimum_lap_distance_percent
        
        # Create or reset cars - if track changed, recreate physics worlds
        if self._is_random_track_mode:
            previous_file = locals().get('previous_track_file', None)
            track_changed = previous_file != self.track_file
        else:
            track_changed = False
        
        if not self.cars or track_changed:
            # Clean up existing physics worlds if they exist
            if self.cars and track_changed:
                print(f"🔄 Track changed ({previous_track_file} → {self.track_file}), recreating physics worlds...")
                for physics_world in self.car_physics_worlds:
                    physics_world.cleanup()
                self.car_physics_worlds.clear()
                self.cars.clear()
            
            # Create multiple cars with new physics worlds
            for i in range(self.num_cars):
                car = Car(world=None, start_position=self.start_position, start_angle=self.start_angle, car_id=f"car_{i}")
                self.cars.append(car)
                physics_world = CarPhysics(car, self.track)
                self.car_physics_worlds.append(physics_world)
        else:
            # Reset existing cars (same track)
            for i in range(self.num_cars):
                self.car_physics_worlds[i].reset_car(self.start_position, self.start_angle)
            
        # Reset systems
        self.collision_reporter.reset()
        
        # Always ensure lap timers have correct track data (regardless of track change detection)
        if self.track:
            track_length = self.track.get_total_track_length()
            for i, lap_timer in enumerate(self.car_lap_timers):
                lap_timer.track = self.track
                lap_timer._find_startline_segment()  # Re-find startline in current track
                # Recalculate minimum lap distance for current track
                if track_length > 0:
                    lap_timer.minimum_lap_distance = track_length * lap_timer.minimum_lap_distance_percent
                    # Print corrected lap timer info (replaces misleading initialization logs)
                    #print(f"LapTimer: Track length {track_length:.1f}m, minimum lap distance: {lap_timer.minimum_lap_distance:.1f}m ({lap_timer.minimum_lap_distance_percent*100:.0f}%)")
        
        # Reset all lap timers (after ensuring correct track data)
        for lap_timer in self.car_lap_timers:
            lap_timer.reset()
        
        self.simulation_time = 0.0
        
        # Reset physics action tracking
        self._current_physics_action = None
        
        # Reset input tracking
        self.input_history = []
        self.last_meaningful_input_time = 0.0
        
        # Reset current action
        self.current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.all_actions = None
        self._last_rewards = None
        
        # Reset disabled cars
        self.disabled_cars.clear()
        
        # Reset cumulative impact force tracking for all cars
        
        # Reset cumulative collision impacts for new disabling features
        self.cumulative_collision_impacts = {i: 0.0 for i in range(self.num_cars)}
        
        # Reset lap reset control
        self._lap_reset_pending = False
        
        # Reset lap count tracking for rewards
        self._previous_lap_count = [0] * self.num_cars
        
        # Reset distance tracking for rewards (use followed car)
        self._previous_car_position = {}
        if self.cars:
            for i in range(self.num_cars):
                car_state = self.car_physics_worlds[i].get_car_state()
                self._previous_car_position[i] = (car_state[0], car_state[1]) if car_state else None
        
        # Reset collision penalty tracking
        self._last_penalized_collision_time = -float('inf')
        self._total_distance_traveled = 0.0
        
        # Reset reward tracking
        self._current_reward = 0.0
        
        # Reset cumulative rewards array for both single and multi-car modes
        self._cumulative_rewards = [0.0] * self.num_cars
        
        # Reset survival time tracking for all cars
        self._survival_time = [0.0] * self.num_cars
        
        # Reset backward movement tracking - initialize with actual car positions
        self._backward_distance = [0.0] * self.num_cars  
        self._previous_backward_distance = [0.0] * self.num_cars
        self._backward_penalty_active = [False] * self.num_cars
        self._first_step_after_reset = [True] * self.num_cars
        
        # Initialize track progress history with actual car positions to prevent false backward detection
        self._track_progress_history = []
        for car_index in range(self.num_cars):
            if car_index < len(self.cars):
                car_state = self.car_physics_worlds[car_index].get_car_state()
                if car_state:
                    current_position = (car_state[0], car_state[1])
                    initial_progress = self._calculate_track_progress(current_position)
                    self._track_progress_history.append(initial_progress)
                else:
                    self._track_progress_history.append(0.0)
            else:
                self._track_progress_history.append(0.0)
        
        # Reset stuck detection state for all cars (both single and multi-car modes)
        for car_index in range(self.num_cars):
            # Reset stuck duration tracking
            stuck_duration_attr = f'_stuck_duration_{car_index}'
            if hasattr(self, stuck_duration_attr):
                setattr(self, stuck_duration_attr, 0.0)
            
            # Reset stuck printed flag
            stuck_printed_attr = f'_stuck_printed_{car_index}'
            if hasattr(self, stuck_printed_attr):
                setattr(self, stuck_printed_attr, False)
            
            # Reset stuck start position
            stuck_start_pos_attr = f'_stuck_start_position_{car_index}'
            if hasattr(self, stuck_start_pos_attr):
                setattr(self, stuck_start_pos_attr, None)
            
            # Reset collision duration tracking
            collision_duration_attr = f'_collision_duration_{car_index}'
            if hasattr(self, collision_duration_attr):
                setattr(self, collision_duration_attr, 0.0)
        
        # Reset just disabled cars tracking
        self._just_disabled_cars.clear()
        
        # Reset termination reason
        self.termination_reason = None
        
        # Reset episode statistics
        self.episode_stats = {
            "max_speed": 0.0,
            "distance_traveled": 0.0,
            "collisions": 0,
            "time_on_track": 0.0,
            "performance_valid": False
        }
        
        # Reset observation visualizer
        self.observation_visualizer.clear_history()
        
        # Get initial observation using unified multi-car methods
        observations = self._get_multi_obs()
        infos = self._get_multi_info()
        
        # For single-car environments, return scalars to maintain Gymnasium compatibility
        if self.num_cars == 1:
            return observations[0], infos
        else:
            return observations, infos
        
    def update_physics(self, actions) -> None:
        """
        Update physics simulation with identical accumulator logic for both modes.
        
        Both modes use the exact same accumulator system:
        - Render mode: Accumulates real elapsed time from perf_counter()
        - Non-render mode: Accumulates fake fixed 1/60 second per frame
        
        This ensures identical physics logic with different time sources.
        
        Args:
            actions: Multi-car actions array with shape (num_cars, 3)
        """
        if not self.cars:
            return
            
        # Store the latest action(s)
        self._current_physics_action = actions
        
        # Actions should already be in multi-car format: (num_cars, 3)
        physics_actions = np.array(actions, dtype=np.float32)

        # Fixed physics timestep (60Hz) - consistent across all modes
        physics_dt = 1.0 / 60.0
        
        # Update collision reporter time once for this step
        self.collision_reporter.update_time(self.simulation_time)
        
        # Always run exactly one physics step per env.step() call
        # This ensures 1:1 action-to-physics ratio regardless of rendering mode
        for i in range(self.num_cars):
            # Update physics world time before stepping
            self.car_physics_worlds[i].simulation_time = self.simulation_time
            self._run_single_physics_step(i, physics_actions[i], physics_dt)
            
        # Increment simulation time once per step, after all physics worlds have been stepped
        self.simulation_time += physics_dt

    def _run_single_physics_step(self, car_idx, physics_action, dt):
        """Run a single physics step with the given timestep"""
                
        # Perform physics step with given timestep
        self.car_physics_worlds[car_idx].step(physics_action, dt)
        
        # Accumulate collision impacts and stuck durations for each car during physics steps only
        if car_idx not in self.disabled_cars:
            # Check collision impulse for this car using continuous collision data
            collision_impulse = self.car_physics_worlds[car_idx].get_continuous_collision_impulse()
            
            # Note: Collision impulse reset moved to after observations are gathered
            # This ensures observations can see current collision impulses
            
            
            # Check for instant disable on severe single impact
            if collision_impulse > INSTANT_DISABLE_IMPACT_THRESHOLD:
                if car_idx not in self.disabled_cars:
                    self.disabled_cars.add(car_idx)
                    
                    self._just_disabled_cars.add(car_idx)
                    car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                    print(f"🚫 {car_name} disabled due to CATASTROPHIC IMPACT ({collision_impulse:.0f} N⋅s > {INSTANT_DISABLE_IMPACT_THRESHOLD:.0f} N⋅s)")
            
            # Track cumulative collision impacts for gradual disabling
            if collision_impulse > COLLISION_FORCE_THRESHOLD:
                if car_idx not in self.cumulative_collision_impacts:
                    self.cumulative_collision_impacts[car_idx] = 0.0
                self.cumulative_collision_impacts[car_idx] += collision_impulse
                
            # Check for cumulative disable threshold
            if self.cumulative_collision_impacts[car_idx] > CUMULATIVE_DISABLE_IMPACT_THRESHOLD:
                if car_idx not in self.disabled_cars:
                    self.disabled_cars.add(car_idx)
                    
                    self._just_disabled_cars.add(car_idx)
                    car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                    print(f"🚫 {car_name} disabled due to ACCUMULATED DAMAGE ({self.cumulative_collision_impacts[car_idx]:.0f} N⋅s > {CUMULATIVE_DISABLE_IMPACT_THRESHOLD:.0f} N⋅s)")
            
            # Update stuck duration tracking during physics steps only
            if car_idx < len(self.cars) and self.cars[car_idx]:
                car = self.cars[car_idx]
                speed = car.get_velocity_magnitude()
                
                # Initialize stuck duration tracking if needed
                stuck_duration_attr = f'_stuck_duration_{car_idx}'
                if not hasattr(self, stuck_duration_attr):
                    setattr(self, stuck_duration_attr, 0.0)
                
                # Check if car is moving slowly and accumulate stuck duration
                if speed < STUCK_SPEED_THRESHOLD:
                    prev_stuck_duration = getattr(self, stuck_duration_attr)
                    current_stuck_duration = prev_stuck_duration + dt  # Use physics dt, not frame dt
                    setattr(self, stuck_duration_attr, current_stuck_duration)
                else:
                    # Reset stuck duration if car is moving fast enough
                    setattr(self, stuck_duration_attr, 0.0)
                    # Also reset the stuck start position and printed flag
                    stuck_start_pos_attr = f'_stuck_start_position_{car_idx}'
                    stuck_printed_attr = f'_stuck_printed_{car_idx}'
                    if hasattr(self, stuck_start_pos_attr):
                        setattr(self, stuck_start_pos_attr, None)
                    if hasattr(self, stuck_printed_attr):
                        setattr(self, stuck_printed_attr, False)
        
        # Update episode stats tracking during physics steps only (for followed car)
        if hasattr(self, 'episode_stats') and self.followed_car_index < len(self.cars):
            followed_car = self.cars[self.followed_car_index]
            if followed_car:
                speed = followed_car.get_velocity_magnitude()
                
                # Track distance traveled
                self.episode_stats["distance_traveled"] += speed * dt
                
                # Track time on track
                if self.car_physics_worlds[self.followed_car_index].is_car_on_track():
                    self.episode_stats["time_on_track"] += dt
                    
                # Check for collisions using physics dt
                if self.collision_reporter.has_recent_collision(dt):
                    self.episode_stats["collisions"] += 1
        
        # Update survival time tracking during physics steps only
        if hasattr(self, '_survival_time'):
            if car_idx not in self.disabled_cars and car_idx < len(self._survival_time):
                self._survival_time[car_idx] += dt
        
        # Update lap timers for all cars
        if car_idx < len(self.car_lap_timers):
            car_state = self.car_physics_worlds[car_idx].get_car_state()
            if car_state:
                car_position = (car_state[0], car_state[1])  # x, y position
                
                # Update this car's lap timer with simulation time
                lap_timer = self.car_lap_timers[car_idx]
                lap_completed = lap_timer.update(car_position, self.simulation_time)
                
                if lap_completed:
                    
                    # Mark reset as pending if reset_on_lap is enabled and all active cars have completed a lap
                    if self.reset_on_lap and self._all_active_cars_completed_lap():
                        self._lap_reset_pending = True
        
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Control inputs as array of actions for all cars.
                   Shape: (num_cars, 2) for continuous [throttle_brake, steering] or (num_cars,) for discrete
            
        Returns:
            observations: Array of observations for all cars (num_cars, obs_size)
            rewards: Array of rewards for all cars (num_cars,)
            terminated: Boolean indicating if episode is terminated
            truncated: Boolean indicating if episode is truncated
            infos: List of info dictionaries for all cars
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        if not self.cars:
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        # Convert scalar actions to multi-car format for internal processing
        if self.num_cars == 1:
            if self.discrete_action_space:
                # Discrete action is a scalar, wrap in array
                action = np.array([action])
            elif hasattr(action, 'ndim') and action.ndim == 1:
                action = action.reshape(1, -1)  # Convert (2,) to (1,2)
        
        # Convert discrete to continuous if needed, or convert 2-element to 3-element
        if self.discrete_action_space:
            # Convert discrete to 2-element continuous, then to 3-element internal
            continuous_actions_2d = [self._discrete_to_continuous(a) for a in action]
            continuous_actions = [self._convert_to_internal_action(a) for a in continuous_actions_2d]
            actions_array = np.array(continuous_actions, dtype=np.float32)
        else:
            # Convert 2-element actions to 3-element internal format
            continuous_actions = [self._convert_to_internal_action(a) for a in action]
            actions_array = np.array(continuous_actions, dtype=np.float32)
        
        # Use unified multi-car step logic
        return self._step_multi_car(actions_array)
    
    # _step_single_car method removed - now handled by unified step() method
    
    def _step_multi_car(self, actions):
        """Handle multi-car step - unified for both single and multi-car modes"""
        # Actions should already be in continuous format as an array
        # Store all actions for rendering
        self.all_actions = np.array(actions, dtype=np.float32)
        
        # Store current action for the followed car for rendering
        if hasattr(self, 'followed_car_index') and 0 <= self.followed_car_index < len(actions):
            self.current_action = np.array(actions[self.followed_car_index], dtype=np.float32)
        else:
            self.current_action = np.array(actions[0], dtype=np.float32) if len(actions) > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        
        # Filter actions for disabled cars (set to zero action)
        filtered_actions = np.copy(actions)
        for car_idx in self.disabled_cars:
            if car_idx < len(filtered_actions):
                filtered_actions[car_idx] = [0.0, 0.0, 0.0]  # No throttle, no brake, no steering
        
        # Update physics with filtered actions
        self.update_physics(filtered_actions)
        
        # Check for new collisions and disable cars if necessary
        self._check_and_disable_cars()
        
        # Update physics system with disabled cars info (to suppress collision messages)
        for i in range(self.num_cars):
            self.car_physics_worlds[i].set_disabled_cars(self.disabled_cars)
        
        # Update episode statistics (for followed car)
        self._update_episode_stats()
        
        # Get observations, rewards, and info for all cars
        observations = self._get_multi_obs()
        rewards = self._calculate_multi_rewards()
        terminated, truncated = self._check_multi_termination()
        infos = self._get_multi_info()
        
        # Update observation history for visualization (for all cars)
        if self.num_cars > 1:
            # Multi-car: add observations for all cars
            for car_id, obs in enumerate(observations):
                self.observation_visualizer.add_observation(obs, self.simulation_time, car_id)
        else:
            # Single car: add observation for car 0
            self.observation_visualizer.add_observation(observations, self.simulation_time, 0)
        
        # Set which car's data to display
        self.observation_visualizer.set_displayed_car(self.followed_car_index)
        
        # Reset collision impulses after observations are gathered
        # This prevents double-counting while allowing observations to see current collisions
        for car_idx in range(self.num_cars):
            if car_idx < len(self.cars) and self.cars[car_idx]:
                car_id = self.cars[car_idx].car_id
                physics_world = self.car_physics_worlds[car_idx]
                physics_world.collision_listener.car_collision_impulses[car_id] = 0.0
        
        # Store last rewards for display purposes
        self._last_rewards = rewards
        
        # Update cumulative rewards
        if not hasattr(self, '_cumulative_rewards'):
            self._cumulative_rewards = [0.0] * self.num_cars
        for i, reward in enumerate(rewards):
            self._cumulative_rewards[i] += reward
        
        
        # Check if lap reset is pending
        if self._lap_reset_pending:
            self._lap_reset_pending = False
            terminated = True
        
        # Clear the set of just disabled cars for the next timestep
        self._just_disabled_cars.clear()

        # For single-car environments, return scalars to maintain Gymnasium compatibility
        if self.num_cars == 1:
            return observations[0], rewards[0], terminated, truncated, infos
        else:
            return observations, rewards, terminated, truncated, infos
    
    def _check_and_disable_cars(self):
        """Check for sustained severe collisions and disable cars"""
        # Apply stuck detection for all environments (single-car and multi-car)
        
        # Track cars disabled in this timestep for final penalty application
            
        for car_idx in range(self.num_cars):
            if car_idx in self.disabled_cars:
                continue  # Car is already disabled
            
            # Check for stuck conditions (stuck duration is now accumulated in physics steps)
            if car_idx < len(self.cars) and self.cars[car_idx]:
                car = self.cars[car_idx]
                speed = car.get_velocity_magnitude()
                
                # Initialize stuck duration and position tracking if needed
                stuck_duration_attr = f'_stuck_duration_{car_idx}'
                stuck_printed_attr = f'_stuck_printed_{car_idx}'  # Track if we've printed stuck message
                stuck_start_pos_attr = f'_stuck_start_position_{car_idx}'  # Track position when stuck started
                if not hasattr(self, stuck_duration_attr):
                    setattr(self, stuck_duration_attr, 0.0)
                if not hasattr(self, stuck_printed_attr):
                    setattr(self, stuck_printed_attr, False)
                if not hasattr(self, stuck_start_pos_attr):
                    setattr(self, stuck_start_pos_attr, None)
                
                # Get current stuck duration (updated in physics steps)
                current_stuck_duration = getattr(self, stuck_duration_attr)
                
                # Track position when stuck detection starts (only for slow cars)
                if speed < STUCK_SPEED_THRESHOLD and current_stuck_duration > 0:
                    # Get current car position
                    car_state = self.car_physics_worlds[car_idx].get_car_state()
                    current_position = (car_state[0], car_state[1]) if car_state else (0, 0)
                    
                    # Save starting position when stuck detection begins
                    stuck_start_pos = getattr(self, stuck_start_pos_attr)
                    if stuck_start_pos is None:
                        setattr(self, stuck_start_pos_attr, current_position)
                        stuck_start_pos = current_position
                    
                    # Calculate distance moved since stuck started
                    distance_moved = 0.0
                    if stuck_start_pos:
                        dx = current_position[0] - stuck_start_pos[0]
                        dy = current_position[1] - stuck_start_pos[1]
                        distance_moved = (dx**2 + dy**2)**0.5
                    
                    # Print when stuck detection starts (only once)
                    if not getattr(self, stuck_printed_attr):
                        car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                        #print(f"⚠️  {car_name} stuck detection started (speed: {speed:.2f} m/s)")
                        setattr(self, stuck_printed_attr, True)
                    
                    # Check if car has been stuck long enough
                    if current_stuck_duration > STUCK_TIME_THRESHOLD:
                        # Determine if car should be disabled based on multiple criteria
                        should_disable = False
                        disable_reason = ""
                        
                        # Criterion 1: Movement-based detection (most reliable)
                        # If car moved less than threshold distance in the stuck period
                        if distance_moved < STUCK_DISTANCE_THRESHOLD:
                            should_disable = True
                            disable_reason = f"moved only {distance_moved:.1f}m in {current_stuck_duration:.1f}s"
                        
                        # Criterion 2: Extended stuck time override
                        # After extended time threshold, disable regardless of movement
                        elif current_stuck_duration > STUCK_EXTENDED_TIME_THRESHOLD:
                            should_disable = True
                            disable_reason = f"stuck for {current_stuck_duration:.1f}s (override)"
                        
                        
                        if should_disable:
                            # Car is stuck - disable it
                            self.disabled_cars.add(car_idx)
                            self._just_disabled_cars.add(car_idx)  # Track for final penalty
                            car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                            print(f"🚫 {car_name} disabled due to being STUCK ({disable_reason})")
                else:
                    # Reset stuck tracking when car is moving normally
                    setattr(self, stuck_duration_attr, 0.0)
                    setattr(self, stuck_printed_attr, False)
                    setattr(self, stuck_start_pos_attr, None)
            
    
    def _get_multi_obs(self):
        """Get observations for all cars"""
        observations = []
        for car_index in range(self.num_cars):
            if car_index < len(self.cars) and self.cars[car_index]:
                # Get car state
                car_state = self.car_physics_worlds[car_index].get_car_state()
                if car_state:
                    pos_x, pos_y, vel_x, vel_y, orientation, angular_vel = car_state
                    
                    # Normalize position, velocity, etc.
                    norm_pos_x = np.clip(pos_x / NORM_MAX_POSITION, -1.0, 1.0)
                    norm_pos_y = np.clip(pos_y / NORM_MAX_POSITION, -1.0, 1.0)
                    norm_vel_x = np.clip(vel_x / NORM_MAX_VELOCITY, -1.0, 1.0)
                    norm_vel_y = np.clip(vel_y / NORM_MAX_VELOCITY, -1.0, 1.0)
                    speed_magnitude_ms = (vel_x**2 + vel_y**2)**0.5
                    norm_speed_magnitude = np.clip(speed_magnitude_ms / NORM_MAX_VELOCITY, 0.0, 1.0)
                    norm_orientation = np.clip(orientation / np.pi, -1.0, 1.0)
                    norm_angular_vel = np.clip(angular_vel / NORM_MAX_ANGULAR_VEL, -1.0, 1.0)
                    
                    # Get tyre data
                    tyre_data = self.car_physics_worlds[car_index].get_tyre_data()
                    tyre_loads, tyre_temps, tyre_wear = tyre_data
                    norm_tyre_loads = [np.clip(load / NORM_MAX_TYRE_LOAD, 0.0, 1.0) for load in tyre_loads]
                    norm_tyre_temps = [np.clip(temp / NORM_MAX_TYRE_TEMP, 0.0, 1.0) for temp in tyre_temps]
                    norm_tyre_wear = [np.clip(wear / NORM_MAX_TYRE_WEAR, 0.0, 1.0) for wear in tyre_wear]
                    
                    # Get collision data (simplified for multi-car)
                    collision_impulse, collision_angle = self.car_physics_worlds[car_index].get_collision_data()
                    norm_collision_impulse = np.clip(collision_impulse / INSTANT_DISABLE_IMPACT_THRESHOLD, 0.0, 1.0)
                    norm_collision_angle = np.clip(collision_angle / np.pi, -1.0, 1.0)
                    
                    # Calculate cumulative impact percentage
                    cumulative_impact = self.cumulative_collision_impacts.get(car_index, 0.0)
                    cumulative_impact_percentage = np.clip(cumulative_impact / CUMULATIVE_DISABLE_IMPACT_THRESHOLD, 0.0, 1.0)
                    
                    # Get sensor distances
                    world = self.car_physics_worlds[car_index].world
                    sensor_distances = self.distance_sensor.get_sensor_distances(
                        world, (pos_x, pos_y), orientation
                    )
                    normalized_sensor_distances = [np.clip(d / SENSOR_MAX_DISTANCE, 0.0, 1.0) for d in sensor_distances]
                    
                    # Construct observation
                    observation = np.array([
                        norm_pos_x, norm_pos_y,
                        norm_vel_x, norm_vel_y,
                        norm_speed_magnitude,
                        norm_orientation, norm_angular_vel,
                        norm_tyre_loads[0], norm_tyre_loads[1], norm_tyre_loads[2], norm_tyre_loads[3],
                        norm_tyre_temps[0], norm_tyre_temps[1], norm_tyre_temps[2], norm_tyre_temps[3],
                        norm_tyre_wear[0], norm_tyre_wear[1], norm_tyre_wear[2], norm_tyre_wear[3],
                        norm_collision_impulse, norm_collision_angle,
                        cumulative_impact_percentage,
                        *normalized_sensor_distances
                    ], dtype=np.float32)
                    
                    observations.append(observation)
                else:
                    # Car state not available - add default observation
                    observations.append(self._get_default_observation())
            else:
                # Car doesn't exist - add default observation
                observations.append(self._get_default_observation())
        
        return np.array(observations, dtype=np.float32)
    
    def _get_default_observation(self):
        """Get default normalized observation for non-existent or invalid cars"""
        static_load = CAR_MASS * GRAVITY_MS2 / 4.0
        normalized_static_load = static_load / NORM_MAX_TYRE_LOAD
        normalized_start_temp = TYRE_START_TEMPERATURE / NORM_MAX_TYRE_TEMP
        default_normalized_sensor_distances = [1.0] * SENSOR_NUM_DIRECTIONS
        
        return np.array([
            0.0, 0.0,  # position
            0.0, 0.0,  # velocity
            0.0,       # speed magnitude
            0.0, 0.0,  # orientation, angular velocity
            normalized_static_load, normalized_static_load, 
            normalized_static_load, normalized_static_load,  # tyre loads
            normalized_start_temp, normalized_start_temp,
            normalized_start_temp, normalized_start_temp,  # tyre temperatures
            0.0, 0.0, 0.0, 0.0,  # tyre wear
            0.0, 0.0,  # collision data
            0.0,       # cumulative impact percentage
            *default_normalized_sensor_distances  # sensor distances
        ], dtype=np.float32)
    
    def _calculate_multi_rewards(self):
        """Calculate rewards for all cars"""
        rewards = []
        
        for car_index in range(self.num_cars):
            if car_index < len(self.cars) and self.cars[car_index]:
                # Calculate reward even for disabled cars (to apply final collision penalty)
                # but only for the timestep they were disabled
                car_just_disabled = hasattr(self, '_just_disabled_cars') and car_index in getattr(self, '_just_disabled_cars', set())

                if car_index in self.disabled_cars and not car_just_disabled:
                    rewards.append(0.0)
                    continue
                
                if car_just_disabled:
                    reward = -PENALTY_DISABLED
                else:
                    reward = 0.0
                
                # Apply per-step penalty to encourage speed
                if car_index not in self.disabled_cars:
                    reward -= PENALTY_PER_STEP
                
                # Wall collision penalty based on accumulated impulse ratio
                if car_index not in self.disabled_cars:
                    # Get collision impulse for this car
                    collision_impulse = self.car_physics_worlds[car_index].get_continuous_collision_impulse()
                    # Apply penalty for current collision
                    if abs(collision_impulse) > 0:
                        reward -= PENALTY_WALL_COLLISION_PER_STEP
                
                # Distance reward (track per car if needed)
                car_state = self.car_physics_worlds[car_index].get_car_state()
                if car_state:
                    current_position = (car_state[0], car_state[1])
                    
                    # Calculate distance from last position
                    prev_pos = self._previous_car_position.get(car_index)
                    if prev_pos:
                        dx = current_position[0] - prev_pos[0]
                        dy = current_position[1] - prev_pos[1]
                        distance = (dx**2 + dy**2)**0.5
                        reward += distance * REWARD_DISTANCE_MULTIPLIER

                    # Update previous position for this car
                    self._previous_car_position[car_index] = current_position
                    
                    # Track backward movement for penalty (skip on first step after reset)
                    if not self._first_step_after_reset[car_index]:
                        current_track_progress = self._calculate_track_progress(current_position)
                        previous_track_progress = self._track_progress_history[car_index]
                        
                        # Calculate progress delta (accounting for lap wrap-around)
                        track_length = self.track.total_length if self.track else 1000.0  # Fallback length
                        progress_delta = current_track_progress - previous_track_progress
                        
                        # Handle lap boundary wrap-around
                        if progress_delta > track_length / 2:
                            progress_delta -= track_length  # Large positive = backward through start/finish
                        elif progress_delta < -track_length / 2:
                            progress_delta += track_length  # Large negative = forward through start/finish
                        
                        if progress_delta < 0:  # Moving backward
                            self._backward_distance[car_index] += abs(progress_delta)
                            
                            # Check if car should be disabled for excessive backward driving
                            if self._backward_distance[car_index] > BACKWARD_DISABLE_THRESHOLD:
                                if car_index not in self.disabled_cars:
                                    self.disabled_cars.add(car_index)
                                    if hasattr(self, '_just_disabled_cars'):
                                        self._just_disabled_cars.add(car_index)  # Track for final penalty
                                    car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
                                    print(f"🚫 {car_name} disabled for driving {self._backward_distance[car_index]:.0f}m backwards (threshold: {BACKWARD_DISABLE_THRESHOLD:.0f}m)")
                            
                            if self._backward_distance[car_index] > BACKWARD_MOVEMENT_THRESHOLD:
                                if not self._backward_penalty_active[car_index]:
                                    self._backward_penalty_active[car_index] = True
                                    car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
                                
                                # Apply penalty only for NEW backward movement beyond threshold
                                current_excess = max(0, self._backward_distance[car_index] - BACKWARD_MOVEMENT_THRESHOLD)
                                previous_excess = max(0, self._previous_backward_distance[car_index] - BACKWARD_MOVEMENT_THRESHOLD)
                                new_backward_distance = current_excess - previous_excess
                                
                                if new_backward_distance > 0:
                                    backward_penalty = new_backward_distance * PENALTY_BACKWARD_PER_METER
                                    reward -= backward_penalty
                        else:  # Moving forward - reset backward accumulation
                            if self._backward_penalty_active[car_index]:
                                car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
                            self._backward_distance[car_index] = 0.0
                            self._previous_backward_distance[car_index] = 0.0
                            self._backward_penalty_active[car_index] = False
                        
                        # Update track progress history
                        self._track_progress_history[car_index] = current_track_progress
                    else:
                        # First step after reset - just update progress history without penalty
                        current_track_progress = self._calculate_track_progress(current_position)
                        self._track_progress_history[car_index] = current_track_progress
                        self._first_step_after_reset[car_index] = False
                
                # Collision penalties have been removed - cars are now disabled instead of penalized
                
                # Lap completion bonus
                if car_index < len(self.car_lap_timers):
                    lap_timer = self.car_lap_timers[car_index]
                    lap_info = lap_timer.get_timing_info()
                    current_lap_count = lap_info.get("lap_count", 0)
                    
                    # Track previous lap count per car
                    prev_lap_count = self._previous_lap_count[car_index]
                    
                    if current_lap_count > prev_lap_count:
                        laps_completed = current_lap_count - prev_lap_count
                        reward += REWARD_LAP_COMPLETION * laps_completed
                        
                        # Fast lap bonus
                        last_lap_time = lap_info.get("last_lap_time", None)
                        if last_lap_time and last_lap_time < REWARD_FAST_LAP_TIME:
                            reward += REWARD_FAST_LAP_BONUS * (REWARD_FAST_LAP_TIME - last_lap_time)**2
                        
                        self._previous_lap_count[car_index] = current_lap_count
                
                # Update previous backward distance for next step
                if car_index not in self.disabled_cars:
                    self._previous_backward_distance[car_index] = self._backward_distance[car_index]
                
                rewards.append(reward)
            else:
                # Car doesn't exist - no reward
                rewards.append(0.0)
        
        return np.array(rewards, dtype=np.float32)
    
    def _check_multi_termination(self):
        """Check termination conditions for multi-car environment"""
        # In multi-car mode, we use a different strategy:
        # - Don't terminate on individual car collisions (cars get disabled instead)
        # - Only terminate on time limits or if ALL cars are disabled
        # - Terminate if ALL active (non-disabled) cars have low rewards
        
        # Check if all cars are disabled
        if len(self.disabled_cars) >= self.num_cars:
            self.termination_reason = "all_cars_disabled"
            return True, False
        
        # Check if ALL active (non-disabled) cars have low rewards
        if  hasattr(self, '_cumulative_rewards'):
            active_cars_below_threshold = []
            active_car_count = 0
            
            for car_index in range(self.num_cars):
                if car_index not in self.disabled_cars:
                    active_car_count += 1
                    if car_index < len(self._cumulative_rewards):
                        if self._cumulative_rewards[car_index] < TERMINATION_MIN_REWARD:
                            active_cars_below_threshold.append(car_index)
            
            # Terminate if all active cars are below threshold
            if active_car_count > 0 and len(active_cars_below_threshold) == active_car_count:
                self.termination_reason = f"all_active_cars_low_reward (threshold: {TERMINATION_MIN_REWARD})"
                return True, False
            elif len(active_cars_below_threshold) > 0:
                # In demo mode, just log which cars have low rewards but don't terminate
                if not self.reset_on_lap:
                    car_names = [self.car_names[i] if i < len(self.car_names) else f"Car {i}" 
                                for i in active_cars_below_threshold]
        
        # Time-based termination
        if self.reset_on_lap and self.simulation_time > TERMINATION_MAX_TIME:
            self.termination_reason = "time_limit"
            return True, False
            
        if self.simulation_time > TRUNCATION_MAX_TIME:
            self.termination_reason = "truncated"
            return False, True
        
        return False, False
    
    def _get_multi_info(self):
        """Get info dictionary for all cars following Gymnasium standard"""
        car_infos = []
        
        for car_index in range(self.num_cars):
            car_info = {
                "car_index": car_index,
                "disabled": car_index in self.disabled_cars,
            }
            
            if car_index < len(self.cars) and self.cars[car_index]:
                car = self.cars[car_index]
                car_state = self.car_physics_worlds[car_index].get_car_state()
                
                if car_state:
                    car_info.update({
                        "car_position": (car_state[0], car_state[1]),
                        "car_speed_kmh": car.get_velocity_kmh(),
                        "car_speed_ms": car.get_velocity_magnitude(),
                        "on_track": self.car_physics_worlds[car_index].is_car_on_track(),
                    })
                
                # Add performance info
                performance = car.validate_performance()
                car_info["performance"] = performance
                
                # Add lap timing info for this car
                if car_index < len(self.car_lap_timers):
                    lap_timer = self.car_lap_timers[car_index]
                    lap_timing = lap_timer.get_timing_info()
                    car_info["lap_timing"] = lap_timing
                
                # Add cumulative reward for this car
                if hasattr(self, '_cumulative_rewards') and car_index < len(self._cumulative_rewards):
                    car_info["cumulative_reward"] = self._cumulative_rewards[car_index]
                
                # Add cumulative collision force for this car
                if hasattr(self, 'cumulative_collision_impacts') and car_index in self.cumulative_collision_impacts:
                    car_info["cumulative_impact_force"] = self.cumulative_collision_impacts[car_index]
                
            else:
                # Car doesn't exist
                car_info.update({
                    "car_position": (0.0, 0.0),
                    "car_speed_kmh": 0.0,
                    "car_speed_ms": 0.0,
                    "on_track": True,
                    "performance": {"performance_valid": False},
                    "lap_timing": {},
                    "cumulative_reward": 0.0,
                    "cumulative_impact_force": 0.0,
                })
            
            car_infos.append(car_info)
        
        # Create single info dictionary following Gymnasium standard
        physics_stats = [p.get_performance_stats() for p in self.car_physics_worlds]
        
        info = {
            "simulation_time": self.simulation_time,
            "num_cars": self.num_cars,
            "followed_car_index": self.followed_car_index,
            "termination_reason": self.termination_reason,
            "cars": car_infos,  # All per-car info nested here
            "physics": physics_stats,
        }
        
        return info
        
    
        
                
    def _update_episode_stats(self) -> None:
        """Update episode statistics (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            return
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            return
            
        speed = followed_car.get_velocity_magnitude()
        self.episode_stats["max_speed"] = max(self.episode_stats["max_speed"], speed)
        
        # Episode stats are now updated in physics steps to be frame-rate independent
            
    def check_quit_requested(self) -> bool:
        """Check if user has requested to quit (e.g., by clicking window close button)"""
        if self.render_mode != RENDER_MODE_HUMAN:
            return False
            
        # Check if pygame is initialized before trying to get events
        if not pygame.get_init():
            return False
            
        # Check for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Toggle reward display
                    self._show_reward = not self._show_reward
                elif event.key == pygame.K_o:
                    # Toggle observation display
                    self._show_observations = not self._show_observations
                elif event.key in CAR_SELECT_KEYS:
                    # Handle car switching (keys 0-9)
                    key_index = CAR_SELECT_KEYS.index(event.key)
                    if key_index < self.num_cars:
                        old_car_index = self.followed_car_index
                        self.followed_car_index = key_index
                        
                        # Switch to display the new car's observation history
                        self.observation_visualizer.set_displayed_car(key_index)
                        
                        # Update legacy car reference for backward compatibility
                        if self.cars and key_index < len(self.cars):
                            self.car = self.cars[key_index]
                    else:
                        pass  # Car does not exist
                else:
                    # Re-post other key events
                    pygame.event.post(event)
            else:
                # Re-post non-quit events so they can be handled by renderer
                pygame.event.post(event)
                
        return False
        
    def is_lap_reset_pending(self) -> bool:
        """Check if a lap reset is pending (useful for external code)"""
        return self._lap_reset_pending
        
    def render(self) -> None:
        """Render the environment"""
        if self.render_mode == RENDER_MODE_HUMAN and self.renderer:
            # Prepare multi-car data for rendering
            cars_data = []
            if self.cars:
                for i, car in enumerate(self.cars):
                    if car:
                        car_state = self.car_physics_worlds[i].get_car_state()
                        if car_state:
                            car_data = {
                                'position': (car_state[0], car_state[1]),
                                'angle': car_state[4],
                                'color': self.car_colors[i] if i < len(self.car_colors) else (255, 255, 255),
                                'name': self.car_names[i] if i < len(self.car_names) else f"Car {i}"
                            }
                            cars_data.append(car_data)
            
            # Get debug data for followed car only
            debug_data = None
            if self.cars and self.followed_car_index < len(self.cars):
                debug_data = self.get_debug_data()
                
            # Get lap timing info for followed car
            followed_lap_timer = self.car_lap_timers[self.followed_car_index] if self.followed_car_index < len(self.car_lap_timers) else self.lap_timer
            lap_timing_info = followed_lap_timer.get_timing_info()
            # Add car name to timing info
            lap_timing_info['car_name'] = self.car_names[self.followed_car_index] if self.followed_car_index < len(self.car_names) else f"Car {self.followed_car_index}"
            
            # Prepare reward info if display is enabled
            reward_info = None
            if self._show_reward:
                current_reward = 0.0
                cumulative_reward = 0.0
                
                # Get current reward from last step if available
                if hasattr(self, '_last_rewards') and self._last_rewards is not None and self.followed_car_index < len(self._last_rewards):
                    current_reward = self._last_rewards[self.followed_car_index]
                
                # Get cumulative reward for followed car
                if hasattr(self, '_cumulative_rewards') and self._cumulative_rewards is not None and self.followed_car_index < len(self._cumulative_rewards):
                    cumulative_reward = self._cumulative_rewards[self.followed_car_index]
                
                reward_info = {
                    'current_reward': current_reward,
                    'cumulative_reward': cumulative_reward,
                    'show': True
                }
            
            # Also provide followed car data
            followed_car_position = None
            followed_car_angle = None
            if cars_data and self.followed_car_index < len(cars_data):
                followed_car_data = cars_data[self.followed_car_index]
                followed_car_position = followed_car_data['position']
                followed_car_angle = followed_car_data['angle']
            
            # Get current action for followed car
            followed_car_action = None
            if hasattr(self, 'all_actions') and self.all_actions is not None and self.followed_car_index < len(self.all_actions):
                followed_car_action = self.all_actions[self.followed_car_index]
            elif hasattr(self, 'current_action'):
                followed_car_action = self.current_action
            
            # Calculate race data for tables display
            race_positions_data = self._calculate_race_positions()
            best_lap_times_data = self._get_best_lap_times_data()
            
            # Prepare countdown timing info
            time_limit = TERMINATION_MAX_TIME if self.reset_on_lap else TRUNCATION_MAX_TIME
            countdown_info = {
                'current_time': self.simulation_time,
                'time_limit': time_limit,
                'reset_on_lap': self.reset_on_lap
            }
            
            # Prepare observation visualization info if enabled
            observation_info = None
            if self._show_observations:
                # Update graphs with latest data, passing current screen size for dynamic scaling
                if self.renderer and self.renderer.window:
                    # Get actual current window size (handles fullscreen properly)
                    screen_width, screen_height = self.renderer.window.get_size()
                else:
                    # Fallback size
                    screen_width, screen_height = (1280, 720)
                self.observation_visualizer.update_graphs(screen_width, screen_height)
                
                observation_info = {
                    'show': True,
                    'visualizer': self.observation_visualizer
                }
            
            self.renderer.render_frame(
                car_position=followed_car_position, 
                car_angle=followed_car_angle, 
                debug_data=debug_data, 
                current_action=followed_car_action, 
                lap_timing_info=lap_timing_info,
                reward_info=reward_info,
                cars_data=cars_data,
                followed_car_index=self.followed_car_index,
                race_positions_data=race_positions_data,
                best_lap_times_data=best_lap_times_data,
                countdown_info=countdown_info,
                observation_info=observation_info,
                track_file_name=self.track_file
            )
            
    def get_track(self):
        """Get the loaded track (for compatibility)"""
        return self.track
        
    def is_position_on_track(self, position: Tuple[float, float]) -> bool:
        """Check if position is on track (for compatibility)"""
        # This method is problematic with multiple worlds. For now, just check the first world.
        if self.car_physics_worlds:
            return self.car_physics_worlds[0]._is_position_on_track(position)
        return True
        
    def seed(self, seed_value: int = None) -> list:
        """Set random seed for reproducible track selection.
        
        Args:
            seed_value: Random seed value
            
        Returns:
            List containing the seed value used
        """
        if seed_value is None:
            seed_value = random.randint(0, 2**32 - 1)
        
        # Seed Python's random module for track selection
        random.seed(seed_value)
        
        # Seed numpy if available
        try:
            np.random.seed(seed_value)
        except ImportError:
            pass
        
        print(f"🌱 Environment seeded with: {seed_value}")
        return [seed_value]
        
    def close(self) -> None:
        """Clean up environment resources safely to prevent segfaults"""
        cleanup_start = time.time()
        CLEANUP_TIMEOUT = 3.0  # 3 second timeout for total cleanup
        
        # Step 1: Close renderer first (safest)
        try:
            if hasattr(self, 'renderer') and self.renderer:
                self.renderer.close()
                self.renderer = None
        except Exception as e:
            pass  # Error closing renderer
        
        # Step 2: Clean up physics world (most critical for segfault prevention)
        try:
            for world in self.car_physics_worlds:
                world.cleanup()
            self.car_physics_worlds = []
        except Exception as e:
            # Force clear reference to prevent further access
            self.car_physics_worlds = []
        
        # Step 3: Clean up other resources with timeout protection
        try:
            if time.time() - cleanup_start < CLEANUP_TIMEOUT:
                # Clear other references safely
                if hasattr(self, 'track'):
                    self.track = None
                if hasattr(self, 'car_lap_timers'):
                    self.car_lap_timers = None
                if hasattr(self, 'disabled_cars'):
                    self.disabled_cars = None
                if hasattr(self, 'cars'):
                    self.cars = None
                if hasattr(self, 'distance_sensors'):
                    self.distance_sensors = None
                if hasattr(self, 'distance_sensor'):
                    self.distance_sensor = None
        except Exception as e:
            pass  # Error clearing references
        
        # Step 4: Don't call pygame.quit() here during interrupt cleanup
        # pygame.quit() during signal handling can cause segfaults
        # Instead, let Python's cleanup handle it or use atexit
        
        # Environment closed successfully
    
    def _calculate_race_positions(self) -> list:
        """
        Calculate current race positions based on completed laps and track progress.
        
        Returns:
            List of tuples (car_index, car_name, total_progress, completed_laps) sorted by race position (1st, 2nd, etc.)
        """
        if not self.cars or not self.track or not self.track.segments:
            return []
        
        car_positions = []
        track_length = self.track.get_total_track_length()
        
        for car_index, car in enumerate(self.cars):
            if not car or car_index in self.disabled_cars:
                continue
                
            car_state = self.car_physics_worlds[car_index].get_car_state()
            if not car_state:
                continue
                
            car_pos = (car_state[0], car_state[1])  # x, y position
            car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
            
            # Get completed laps for this car
            completed_laps = 0
            lap_timer = None
            if car_index < len(self.car_lap_timers):
                lap_timer = self.car_lap_timers[car_index]
                completed_laps = lap_timer.get_lap_count()
            
            # Calculate current track progress (distance along track from start)
            current_progress = self._calculate_track_progress(car_pos)
            
            # Virtual lap increment for position calculation: if car is near start line 
            # but has traveled significant distance, they likely just crossed finish
            virtual_laps = completed_laps
            if (lap_timer and 
                lap_timer.is_timing() and 
                lap_timer.has_crossed_startline and
                current_progress < (track_length * 0.15) and  # Within 15% of start
                lap_timer.total_distance_traveled > (track_length * 0.8)):  # Traveled 80%+ of track
                # Car likely just crossed finish line but validation pending
                virtual_laps = completed_laps + 1
            
            # Total progress = (virtual_laps * track_length) + current_progress
            # Use virtual_laps for sorting to handle finish line crossings
            total_progress = virtual_laps * track_length + current_progress
            
            # Store virtual lap count and current progress for proper sorting
            car_positions.append((car_index, car_name, total_progress, virtual_laps, current_progress))
        
        # Sort by completed laps first (descending), then by current progress (descending)
        # This ensures cars with more laps always rank higher, and among cars on the same lap,
        # those further along rank higher
        car_positions.sort(key=lambda x: (x[3], x[4]), reverse=True)
        
        return car_positions
    
    def _calculate_track_progress(self, car_position: Tuple[float, float]) -> float:
        """
        Calculate how far along the track a car is (distance from start).
        
        Args:
            car_position: Car position (x, y) in world coordinates
            
        Returns:
            Distance traveled along track from start in meters
        """
        if not self.track or not self.track.segments:
            return 0.0
        
        car_x, car_y = car_position
        min_distance = float('inf')
        closest_segment_index = 0
        closest_point_on_segment = (0.0, 0.0)
        
        # Find closest point on any track segment
        for i, segment in enumerate(self.track.segments):
            # Calculate closest point on this segment
            start_x, start_y = segment.start_position
            end_x, end_y = segment.end_position
            
            # Vector from start to end of segment
            dx = end_x - start_x
            dy = end_y - start_y
            segment_length_sq = dx * dx + dy * dy
            
            if segment_length_sq < 1e-6:  # Very short segment
                closest_x, closest_y = start_x, start_y
                t = 0.0
            else:
                # Project car position onto line segment
                t = max(0, min(1, ((car_x - start_x) * dx + (car_y - start_y) * dy) / segment_length_sq))
                closest_x = start_x + t * dx
                closest_y = start_y + t * dy
            
            # Calculate distance from car to this closest point
            dist_sq = (car_x - closest_x) ** 2 + (car_y - closest_y) ** 2
            
            if dist_sq < min_distance:
                min_distance = dist_sq
                closest_segment_index = i
                closest_point_on_segment = (closest_x, closest_y)
        
        # Calculate total distance from start to closest point
        total_progress = 0.0
        
        # Add lengths of all complete segments before the closest one
        for i in range(closest_segment_index):
            segment = self.track.segments[i]
            segment_length = math.sqrt(
                (segment.end_position[0] - segment.start_position[0]) ** 2 +
                (segment.end_position[1] - segment.start_position[1]) ** 2
            )
            total_progress += segment_length
        
        # Add partial distance within the closest segment
        if closest_segment_index < len(self.track.segments):
            closest_segment = self.track.segments[closest_segment_index]
            partial_distance = math.sqrt(
                (closest_point_on_segment[0] - closest_segment.start_position[0]) ** 2 +
                (closest_point_on_segment[1] - closest_segment.start_position[1]) ** 2
            )
            total_progress += partial_distance
        
        return total_progress
    
    def _get_best_lap_times_data(self) -> list:
        """
        Get best lap times for all cars sorted by fastest time.
        
        Returns:
            List of tuples (car_index, car_name, best_lap_time) sorted by fastest time
        """
        if not self.car_lap_timers:
            return []
        
        lap_times_data = []
        
        for car_index, lap_timer in enumerate(self.car_lap_timers):
            if car_index in self.disabled_cars:
                continue
                
            car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
            best_time = lap_timer.get_best_lap_time()
            
            if best_time is not None:
                lap_times_data.append((car_index, car_name, best_time))
        
        # Sort by best lap time (fastest first)
        lap_times_data.sort(key=lambda x: x[2])
        
        return lap_times_data
    
    def _all_active_cars_completed_lap(self) -> bool:
        """
        Check if all active (non-disabled) cars have completed at least one lap.
        
        Returns:
            bool: True if all active cars have completed at least one lap, False otherwise
        """
        if not self.car_lap_timers:
            return False
        
        # Find all active (non-disabled) cars
        active_car_indices = []
        for car_index in range(self.num_cars):
            if car_index not in self.disabled_cars:
                active_car_indices.append(car_index)
        
        # If no active cars, return False (shouldn't happen in normal gameplay)
        if not active_car_indices:
            return False
        
        # Check if all active cars have completed at least one lap
        for car_index in active_car_indices:
            if car_index < len(self.car_lap_timers):
                lap_timer = self.car_lap_timers[car_index]
                if lap_timer.get_lap_count() < 1:
                    return False  # This active car hasn't completed a lap yet
            else:
                return False  # No lap timer for this car
        
        return True  # All active cars have completed at least one lap
        
    def get_debug_info(self) -> str:
        """Get debug information string"""
        if not self.car:
            return "No car initialized"
            
        # Use car's debug string which includes engine info
        car_info = str(self.car)
        collision_info = self.collision_reporter.get_debug_info()
        
        return f"{car_info}, {collision_info}"
        
    def get_debug_data(self) -> dict:
        """Get structured debug data for enhanced debug visualization (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            return {}
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            return {}
            
        # Car physics data
        car_position = followed_car.body.position
        car_angle = followed_car.body.angle
        drag_force = followed_car.get_drag_force()
        velocity_vector = followed_car.get_velocity_vector()
        acceleration_vector = followed_car.get_last_acceleration_vector()  # Use cached value to avoid side effects
        input_steering_vector, actual_steering_vector = followed_car.get_steering_vectors()
        
        # Tyre data
        tyre_loads = followed_car.tyre_manager.get_tyre_loads()
        tyre_temperatures = followed_car.tyre_manager.get_tyre_temperatures()
        tyre_wear = followed_car.tyre_manager.get_tyre_wear()
        tyre_pressures = followed_car.tyre_manager.get_tyre_pressures()
        
        # Sensor data
        world = self.car_physics_worlds[self.followed_car_index].world
        sensor_distances = self.distance_sensor.get_sensor_distances(
            world, (car_position.x, car_position.y), car_angle
        )
        sensor_angles = self.distance_sensor.get_sensor_angles(car_angle)
        
        # Cumulative impact data
        cumulative_impact = self.cumulative_collision_impacts.get(self.followed_car_index, 0.0)
        cumulative_impact_percentage = np.clip(cumulative_impact / CUMULATIVE_DISABLE_IMPACT_THRESHOLD, 0.0, 1.0)
        
        return {
            'car_position': (car_position.x, car_position.y),
            'car_angle': car_angle,
            'drag_force': drag_force,
            'velocity_vector': velocity_vector,
            'acceleration_vector': acceleration_vector,
            'steering_vectors': {
                'input': input_steering_vector,
                'actual': actual_steering_vector
            },
            'tyre_data': {
                'loads': tyre_loads,
                'temperatures': tyre_temperatures,
                'wear': tyre_wear,
                'pressures': tyre_pressures
            },
            'sensor_data': {
                'distances': sensor_distances,
                'angles': sensor_angles
            },
            'collision_damage': {
                'cumulative_impact': cumulative_impact,
                'cumulative_percentage': cumulative_impact_percentage
            }
        }
        
    def __str__(self) -> str:
        """String representation of environment"""
        track_name = self.track_file if self.track_file else "No track"
        return (f"CarEnv: {track_name}, "
                f"Time: {self.simulation_time:.1f}s, "
                f"Car: {'Yes' if self.car else 'No'}")