"""
CarPhysics system.

This module provides complete physics simulation including car dynamics
and track collision detection in a unified Box2D world.
"""

import Box2D
import math
from typing import Optional, Tuple, List, Dict, Any
from .car import Car
from .track_generator import Track
from .constants import (
    BOX2D_TIME_STEP,
    BOX2D_VELOCITY_ITERATIONS,
    BOX2D_POSITION_ITERATIONS,
    COLLISION_FORCE_SCALE_FACTOR,
    COLLISION_MIN_FORCE,
    COLLISION_MAX_FORCE,
    MIN_REALISTIC_FPS,
    MAX_REALISTIC_FPS,
    TWO_PI,
    TRACK_WALL_THICKNESS,
    BOX2D_WALL_DENSITY,
    BOX2D_WALL_FRICTION,
    BOX2D_WALL_RESTITUTION,
    # Collision filtering constants
    COLLISION_CATEGORY_TRACK_WALLS,
    COLLISION_MASK_TRACK_WALLS,
    # Physics detail constants
    PHYSICS_CURVE_DEGREES_PER_SEGMENT,
    PHYSICS_CURVE_MIN_SEGMENTS,
    PHYSICS_CURVE_MAX_SEGMENTS
)


class CollisionData:
    """Data structure for collision events"""
    
    def __init__(self, position: Tuple[float, float], impulse: float, normal: Tuple[float, float], car_id: str = "unknown"):
        self.position = position  # World collision point
        self.impulse = impulse  # Collision impulse magnitude (mass * delta_velocity)
        self.normal = normal  # Collision normal vector
        self.car_id = car_id  # Identifier of the car involved in collision
        self.timestamp = 0.0  # Will be set by collision system
        self.reported_to_environment = False  # Flag to track if collision has been reported


class CarPhysics:
    """Complete physics simulation with car dynamics and track collision detection"""
    
    def __init__(self, track: Optional[Track] = None):
        """
        Initialize car physics system.
        
        Args:
            track: Track for collision detection (optional)
        """
        # Create Box2D world
        self.world = Box2D.b2World(gravity=(0, 0))  # Top-down view, no gravity
        
        # Track integration - walls created directly in our world
        self.track = track
        
        # Track disabled cars to suppress collision messages
        self.disabled_cars = set()
        self.wall_bodies: List[Box2D.b2Body] = []
        if track:
            # Create track walls directly in our physics world
            self._create_track_walls()
        
        # Car instances (support for multiple cars)
        self.car = None  # Legacy single car reference (for backward compatibility)
        self.cars = []  # List of all cars in the simulation
        
        # Collision tracking
        self.collision_listener = CarCollisionListener(self)
        self.world.contactListener = self.collision_listener
        self.recent_collisions: List[CollisionData] = []
        self.simulation_time = 0.0
        
        # Performance tracking
        self.physics_steps = 0
        self.average_fps = 60.0
        self.last_fps_update_time = 0.0
        
    def create_car(self, start_position: Tuple[float, float] = (0.0, 0.0), 
                   start_angle: float = 0.0) -> Car:
        """
        Create a car in the physics world (legacy single-car method).
        
        Args:
            start_position: Initial position (x, y) in meters
            start_angle: Initial orientation in radians
            
        Returns:
            Car instance
        """
        if self.car is not None:
            # Remove existing car
            self.world.DestroyBody(self.car.body)
            
        self.car = Car(self.world, start_position, start_angle, "car_0")
        self.cars = [self.car]  # Update cars list for compatibility
        return self.car
    
    def create_cars(self, num_cars: int, start_position: Tuple[float, float] = (0.0, 0.0), 
                    start_angle: float = 0.0) -> List[Car]:
        """
        Create multiple cars in the physics world at the same starting position.
        
        Args:
            num_cars: Number of cars to create (1-10)
            start_position: Initial position (x, y) in meters for all cars
            start_angle: Initial orientation in radians for all cars
            
        Returns:
            List of Car instances
        """
        from .constants import MAX_CARS
        
        if num_cars < 1 or num_cars > MAX_CARS:
            raise ValueError(f"Number of cars must be between 1 and {MAX_CARS}")
        
        # Clear existing cars
        self.clear_cars()
        
        # Create new cars
        self.cars = []
        for i in range(num_cars):
            # Add small random offset to prevent cars from being exactly on top of each other
            offset_x = (i % 3 - 1) * 0.1  # -0.1, 0, 0.1 pattern
            offset_y = (i // 3) * 0.1     # Rows of 3 cars
            car_position = (start_position[0] + offset_x, start_position[1] + offset_y)
            
            car = Car(self.world, car_position, start_angle, f"car_{i}")
            self.cars.append(car)
        
        # Set first car as legacy reference
        self.car = self.cars[0] if self.cars else None
        
        return self.cars
    
    def set_disabled_cars(self, disabled_cars: set) -> None:
        """
        Update the set of disabled cars to suppress collision messages.
        
        Args:
            disabled_cars: Set of car indices that are disabled
        """
        self.disabled_cars = disabled_cars.copy()
    
    def clear_cars(self) -> None:
        """
        Remove all cars from the physics world.
        """
        # Check if world is locked first
        if hasattr(self.world, 'IsLocked') and self.world.IsLocked():
            return
            
        for car in self.cars:
            if car and hasattr(car, 'body') and car.body:
                try:
                    # Check if body is still valid before destroying
                    if car.body in self.world.bodies:
                        self.world.DestroyBody(car.body)
                    car.body = None  # Clear reference
                except:
                    pass  # Body may already be destroyed
        
        self.cars = []
        self.car = None
        
    def _create_track_walls(self) -> None:
        """Create track collision walls directly in our physics world"""
        if not self.track:
            return
            
        for segment in self.track.segments:
            self._create_segment_walls(segment)
            
    def _create_segment_walls(self, segment) -> None:
        """Create walls for a single track segment"""
        if segment.segment_type == "CURVE":
            self._create_curved_walls(segment)
        else:
            self._create_straight_walls(segment)
                
    def _create_straight_walls(self, segment) -> None:
        """Create walls for straight track segments"""
        start_x, start_y = segment.start_position
        end_x, end_y = segment.end_position
        half_width = segment.width / 2
        
        # Calculate perpendicular direction for walls
        segment_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if segment_length > 0:
            # Unit vector along the segment
            dx = (end_x - start_x) / segment_length
            dy = (end_y - start_y) / segment_length
            
            # Perpendicular vector (left side)
            perp_dx = -dy
            perp_dy = dx
        else:
            # Fallback for zero-length segments
            perp_dx, perp_dy = 0, 1
        
        # Calculate wall positions for left and right sides
        left_start = (start_x + perp_dx * half_width, start_y + perp_dy * half_width)
        left_end = (end_x + perp_dx * half_width, end_y + perp_dy * half_width)
        right_start = (start_x - perp_dx * half_width, start_y - perp_dy * half_width)
        right_end = (end_x - perp_dx * half_width, end_y - perp_dy * half_width)
        
        # Create left wall
        left_wall = self._create_wall_body_from_line(
            left_start[0], left_start[1],
            left_end[0], left_end[1],
            TRACK_WALL_THICKNESS
        )
        if left_wall:
            self.wall_bodies.append(left_wall)
        
        # Create right wall
        right_wall = self._create_wall_body_from_line(
            right_start[0], right_start[1],
            right_end[0], right_end[1],
            TRACK_WALL_THICKNESS
        )
        if right_wall:
            self.wall_bodies.append(right_wall)
            
    def _create_curved_walls(self, segment) -> None:
        """Create walls for curved track segments"""
        if segment.curve_radius <= 0 or segment.curve_angle <= 0:
            return
        
        # Generate points along inner and outer curves
        inner_points, outer_points = self._generate_curve_wall_points(segment)
        
        if len(inner_points) < 2 or len(outer_points) < 2:
            return
        
        # Create wall segments along inner curve
        for i in range(len(inner_points) - 1):
            wall = self._create_wall_body_from_line(
                inner_points[i][0], inner_points[i][1],
                inner_points[i+1][0], inner_points[i+1][1],
                TRACK_WALL_THICKNESS
            )
            if wall:
                self.wall_bodies.append(wall)
        
        # Create wall segments along outer curve
        for i in range(len(outer_points) - 1):
            wall = self._create_wall_body_from_line(
                outer_points[i][0], outer_points[i][1],
                outer_points[i+1][0], outer_points[i+1][1],
                TRACK_WALL_THICKNESS
            )
            if wall:
                self.wall_bodies.append(wall)
            
    def _generate_curve_wall_points(self, segment):
        """Generate points for inner and outer curve walls"""
        
        half_width = segment.width / 2
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
        
        # Calculate radii for inner and outer walls
        if segment.curve_direction == "LEFT":
            inner_radius = segment.curve_radius - half_width
            outer_radius = segment.curve_radius + half_width
        else:  # RIGHT
            inner_radius = segment.curve_radius + half_width
            outer_radius = segment.curve_radius - half_width
        
        # Generate points along the curves
        inner_points = []
        outer_points = []
        angle_from_center_to_start = start_heading_rad - turn_multiplier * math.pi / 2
        
        # Use configurable segments for physics accuracy (smaller than rendering)
        num_segments = max(
            PHYSICS_CURVE_MIN_SEGMENTS, 
            min(PHYSICS_CURVE_MAX_SEGMENTS, int(abs(segment.curve_angle) / PHYSICS_CURVE_DEGREES_PER_SEGMENT))
        )
        
        for i in range(num_segments + 1):
            t = i / num_segments
            angle = angle_from_center_to_start + turn_multiplier * curve_angle_rad * t
            
            # Inner wall point
            if inner_radius > 0:  # Avoid negative radius
                inner_x = center_x + inner_radius * math.cos(angle)
                inner_y = center_y + inner_radius * math.sin(angle)
                inner_points.append((inner_x, inner_y))
            
            # Outer wall point
            outer_x = center_x + outer_radius * math.cos(angle)
            outer_y = center_y + outer_radius * math.sin(angle)
            outer_points.append((outer_x, outer_y))
        
        return inner_points, outer_points
            
    def _create_wall_body_from_line(self, x1: float, y1: float, x2: float, y2: float, thickness: float) -> Box2D.b2Body:
        """Create a Box2D body for a wall segment"""
        # Calculate wall center and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        if length < 0.1:  # Skip very short segments
            return None
        
        # Calculate rotation angle
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # Create body definition
        body_def = Box2D.b2BodyDef()
        body_def.type = Box2D.b2_staticBody
        body_def.position = (center_x, center_y)
        body_def.angle = angle
        
        # Create body
        body = self.world.CreateBody(body_def)
        
        # Create box shape for the wall
        box_shape = Box2D.b2PolygonShape()
        box_shape.SetAsBox(length / 2, thickness / 2)
        
        # Create fixture with collision filtering
        fixture_def = Box2D.b2FixtureDef()
        fixture_def.shape = box_shape
        fixture_def.density = BOX2D_WALL_DENSITY
        fixture_def.friction = BOX2D_WALL_FRICTION
        fixture_def.restitution = BOX2D_WALL_RESTITUTION
        
        # Set collision filtering: track walls collide with cars
        fixture_def.filter.categoryBits = COLLISION_CATEGORY_TRACK_WALLS
        fixture_def.filter.maskBits = COLLISION_MASK_TRACK_WALLS
        
        fixture = body.CreateFixture(fixture_def)
        # Set userData on fixture for raycast detection
        fixture.userData = {'type': 'track_wall'}
        
        # Store metadata on body as well for compatibility
        body.userData = {
            'type': 'track_wall',
            'start': (x1, y1),
            'end': (x2, y2)
        }
        
        return body
        
    def step(self, actions, dt: float = BOX2D_TIME_STEP) -> None:
        """
        Perform one physics simulation step with multi-car actions.
        
        Args:
            actions: Either single action (throttle, brake, steering) for backward compatibility
                    or array of actions [(throttle, brake, steering), ...] for multi-car
            dt: Time step in seconds
        """
        if not self.cars:
            raise ValueError("No cars created. Call create_cars() or create_car() first.")
            
        # Note: simulation_time is updated in car_env.py to avoid double increment
        
        # Handle both single action and multi-action formats
        if isinstance(actions, (list, tuple)) and len(actions) == 3 and isinstance(actions[0], (int, float)):
            # Single action format for backward compatibility
            throttle, brake, steering = actions
            if len(self.cars) > 0 and self.cars[0]:
                self.cars[0].set_inputs(throttle, brake, steering)
        else:
            # Multi-action format: apply each action to corresponding car
            if hasattr(actions, 'shape') and len(actions.shape) == 2:
                # NumPy array format: (num_cars, 3)
                for i, car in enumerate(self.cars):
                    if car and i < len(actions):
                        throttle, brake, steering = actions[i]
                        car.set_inputs(float(throttle), float(brake), float(steering))
            else:
                # List/tuple format: [(throttle, brake, steering), ...]
                for i, car in enumerate(self.cars):
                    if car and i < len(actions):
                        throttle, brake, steering = actions[i]
                        car.set_inputs(float(throttle), float(brake), float(steering))
        
        # Update physics for all cars
        for car in self.cars:
            if car:
                car.update_physics(dt)
        
        # Step the physics world
        self.world.Step(dt, BOX2D_VELOCITY_ITERATIONS, BOX2D_POSITION_ITERATIONS)
        
        # Process collisions
        self._process_collisions()
        
        # Update performance tracking
        self.physics_steps += 1
        # Update FPS calculation every second worth of steps
        steps_per_second = int(1.0 / dt) if dt > 0 else 60
        if self.physics_steps % steps_per_second == 0:
            elapsed_time = self.simulation_time - self.last_fps_update_time
            if elapsed_time > 0:
                self.average_fps = steps_per_second / elapsed_time
            else:
                self.average_fps = 60.0
            self.last_fps_update_time = self.simulation_time
    
    def _process_collisions(self) -> None:
        """Process collision events from the collision listener"""
        # Get new collisions from listener
        new_collisions = self.collision_listener.get_collisions()
        
        for collision in new_collisions:
            collision.timestamp = self.simulation_time
            self.recent_collisions.append(collision)
            
        # Keep only recent collisions (last 2 seconds)
        cutoff_time = self.simulation_time - 2.0
        self.recent_collisions = [c for c in self.recent_collisions if c.timestamp > cutoff_time]
        
    def get_collision_data(self, car_index: int = 0) -> Tuple[float, float]:
        """
        Get current collision data for environment observation.
        Now returns CURRENT collision state, not just recent history.
        
        Args:
            car_index: Index of the car (0-based)
        
        Returns:
            Tuple of (collision_impulse, collision_angle_relative_to_car)
        """
        if not self.cars or car_index < 0 or car_index >= len(self.cars):
            return (0.0, 0.0)
            
        car = self.cars[car_index]
        if not car:
            return (0.0, 0.0)
            
        # Get car ID for this index
        car_id = f"car_{car_index}"
        
        # Get current collision impulse from listener
        current_impulse = self.collision_listener.get_car_collision_impulse(car_id)
        
        if current_impulse < 50.0:  # Minimum threshold for significant collision
            return (0.0, 0.0)
            
        # For angle, use the most recent collision data if available
        collision_angle = 0.0
        for key, collision_data in self.collision_listener.active_collisions.items():
            if key[0] == car_id:
                # Calculate collision angle relative to car
                car_angle = car.body.angle
                collision_normal = collision_data.normal
                
                normal_angle = math.atan2(collision_normal[1], collision_normal[0])
                collision_angle = normal_angle - car_angle
                
                # Normalize angle to [-π, π]
                while collision_angle > math.pi:
                    collision_angle -= TWO_PI
                while collision_angle < -math.pi:
                    collision_angle += TWO_PI
                break
            
        return (current_impulse, collision_angle)
    
    def get_continuous_collision_impulse(self, car_index: int = 0) -> float:
        """
        Get the current collision impulse for a car (for continuous penalty calculation).
        
        Args:
            car_index: Index of the car (0-based)
        
        Returns:
            Current collision impulse (0 if no collision)
        """
        if not self.cars or car_index < 0 or car_index >= len(self.cars):
            return 0.0
            
        # Get car ID for this index
        car_id = f"car_{car_index}"
        
        # Get current collision impulse from listener
        return self.collision_listener.get_car_collision_impulse(car_id)
        
    def is_car_on_track(self, car_index: int = 0) -> bool:
        """
        Check if specified car is currently on track.
        
        Args:
            car_index: Index of the car (0-based)
        
        Returns:
            True if car is on track, False otherwise
        """
        if not self.cars or car_index < 0 or car_index >= len(self.cars) or not self.track:
            return True  # No track means always "on track"
            
        car = self.cars[car_index]
        if not car:
            return True
            
        car_position = car.body.position
        return self._is_position_on_track((car_position.x, car_position.y))
        
    def _is_position_on_track(self, position: Tuple[float, float]) -> bool:
        """Check if a position is within the track boundaries (not colliding with walls)"""
        return not self._check_wall_collision(position)
        
    def _check_wall_collision(self, position: Tuple[float, float], radius: float = 0.5) -> bool:
        """Check if a circular object at position collides with track walls"""
        # Use AABB query to check for overlaps without stepping the simulation
        
        # Create AABB (Axis-Aligned Bounding Box) for the query
        aabb = Box2D.b2AABB()
        aabb.lowerBound = (position[0] - radius, position[1] - radius)
        aabb.upperBound = (position[0] + radius, position[1] + radius)
        
        # Query callback to check fixtures
        class QueryCallback(Box2D.b2QueryCallback):
            def __init__(self, center, radius):
                super().__init__()
                self.center = center
                self.radius = radius
                self.hit = False
                
            def ReportFixture(self, fixture):
                # Check if this is a track wall
                if fixture.userData and fixture.userData.get('type') == 'track_wall':
                    # Get the shape and check for actual collision
                    shape = fixture.shape
                    transform = fixture.body.transform
                    
                    if isinstance(shape, Box2D.b2PolygonShape):
                        # For polygon shapes, use Box2D's TestPoint
                        if shape.TestPoint(transform, self.center):
                            self.hit = True
                            return False
                        
                        # Also check if circle edge intersects polygon
                        for i in range(shape.vertexCount):
                            v = Box2D.b2Mul(transform, shape.vertices[i])
                            dx = self.center[0] - v[0]
                            dy = self.center[1] - v[1]
                            dist = (dx * dx + dy * dy) ** 0.5
                            if dist < self.radius:
                                self.hit = True
                                return False
                                
                return True  # Continue searching
        
        # Create and run the query
        callback = QueryCallback(position, radius)
        self.world.QueryAABB(callback, aabb)
        
        return callback.hit
        
    def get_car_state(self, car_index: int = 0) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Get current car state for specified car.
        
        Args:
            car_index: Index of the car (0-based)
            
        Returns:
            Car state tuple or None if car doesn't exist
        """
        if not self.cars or car_index < 0 or car_index >= len(self.cars):
            return None
            
        car = self.cars[car_index]
        if not car:
            return None
            
        return car.get_state()
        
    def get_tyre_data(self, car_index: int = 0) -> Optional[Tuple]:
        """
        Get tyre data from specified car.
        
        Args:
            car_index: Index of the car (0-based)
            
        Returns:
            Tyre data tuple or None if car doesn't exist
        """
        if not self.cars or car_index < 0 or car_index >= len(self.cars):
            return None
            
        car = self.cars[car_index]
        if not car:
            return None
            
        return car.get_tyre_data()
        
    def reset_car(self, position: Tuple[float, float] = (0.0, 0.0), angle: float = 0.0) -> None:
        """Reset car to specified position and angle (legacy single-car method)"""
        if self.car:
            self.car.body.position = position
            self.car.body.angle = angle
            self.car.reset()
            
        # Clear collision history
        self.recent_collisions.clear()
        self.simulation_time = 0.0
    
    def reset_cars(self, position: Tuple[float, float] = (0.0, 0.0), angle: float = 0.0) -> None:
        """
        Reset all cars to the same starting position and angle.
        
        Args:
            position: Reset position (x, y) in meters for all cars
            angle: Reset orientation in radians for all cars
        """
        for i, car in enumerate(self.cars):
            if car:
                # Add small offset like in create_cars
                offset_x = (i % 3 - 1) * 0.1
                offset_y = (i // 3) * 0.1
                car_position = (position[0] + offset_x, position[1] + offset_y)
                car.body.position = car_position
                car.body.angle = angle
                car.reset()
        
        # Clear collision history
        self.recent_collisions.clear()
        self.simulation_time = 0.0
        
        # Reset performance tracking
        self.physics_steps = 0
        self.average_fps = 60.0
        self.last_fps_update_time = 0.0
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get physics performance statistics"""
        # Validate FPS value to ensure it's realistic
        validated_fps = max(MIN_REALISTIC_FPS, min(self.average_fps, MAX_REALISTIC_FPS))
        
        stats = {
            "physics_steps": self.physics_steps,
            "simulation_time": self.simulation_time,
            "average_fps": validated_fps,
            "recent_collisions": len(self.recent_collisions),
            "bodies_in_world": len(self.world.bodies)
        }
        
        if self.car:
            stats.update(self.car.validate_performance())
            
        return stats
        
    def cleanup(self) -> None:
        """Clean up physics world safely to prevent segfaults"""
        # Check if world exists and is not locked
        if not self.world:
            return
            
        # Add timeout protection to prevent hanging during cleanup
        import time
        cleanup_start = time.time()
        CLEANUP_TIMEOUT = 2.0  # 2 second timeout
        
        try:
            # Check if world is locked (during simulation step)
            if hasattr(self.world, 'IsLocked') and self.world.IsLocked():
                print("Warning: Physics world is locked during cleanup, skipping body destruction")
                # Just clear references without destroying bodies
                self._clear_references_only()
                return
            
            # Clean up all cars first
            if time.time() - cleanup_start < CLEANUP_TIMEOUT:
                self.clear_cars()
            
            # Clean up wall bodies
            if hasattr(self, 'wall_bodies') and time.time() - cleanup_start < CLEANUP_TIMEOUT:
                for body in self.wall_bodies[:]:  # Use slice to create copy
                    try:
                        if body and hasattr(body, 'userData') and body in self.world.bodies:
                            self.world.DestroyBody(body)
                    except Exception:
                        pass  # Body may already be destroyed or invalid
                self.wall_bodies.clear()
            
            # Clear all remaining bodies from world with timeout protection
            if hasattr(self.world, 'bodies') and time.time() - cleanup_start < CLEANUP_TIMEOUT:
                try:
                    bodies_to_destroy = list(self.world.bodies)  # Create a copy
                    for body in bodies_to_destroy:
                        if time.time() - cleanup_start >= CLEANUP_TIMEOUT:
                            break
                        try:
                            # Double-check body is still valid and accessible
                            if body and hasattr(body, 'userData'):
                                self.world.DestroyBody(body)
                        except Exception:
                            pass  # Body may already be destroyed or invalid
                except Exception:
                    pass  # World may be in inconsistent state
                    
            # Clear collision data
            if hasattr(self, 'recent_collisions'):
                self.recent_collisions.clear()
            if hasattr(self, 'collision_listener'):
                self.collision_listener = None
            
            # Clear world reference last
            self.world = None
            
        except Exception as e:
            # Catch any cleanup errors to prevent segfault
            print(f"Warning: Error during physics cleanup: {e}")
            # Force clear all references to prevent further issues
            self._clear_references_only()
    
    def _clear_references_only(self):
        """Emergency cleanup that only clears references without destroying Box2D objects"""
        try:
            if hasattr(self, 'cars'):
                self.cars.clear()
            if hasattr(self, 'wall_bodies'):
                self.wall_bodies.clear()
            if hasattr(self, 'recent_collisions'):
                self.recent_collisions.clear()
            if hasattr(self, 'collision_listener'):
                self.collision_listener = None
            # Don't set world to None here as it may still be locked
            print("Physics cleanup: Cleared references only due to locked world")
        except Exception as e:
            print(f"Warning: Error during reference cleanup: {e}")
            pass


class CarCollisionListener(Box2D.b2ContactListener):
    """Collision listener for car-track collisions"""
    
    def __init__(self, car_physics=None):
        super().__init__()
        self.collisions: List[CollisionData] = []
        # Track active collisions: key is (car_id, wall_id), value is CollisionData
        self.active_collisions: Dict[Tuple[str, str], CollisionData] = {}
        # Track collision impulses per car for continuous penalties
        self.car_collision_impulses: Dict[str, float] = {}
        # Reference to parent CarPhysics for accessing disabled_cars
        self.car_physics = car_physics
        
    def BeginContact(self, contact):
        """Called when collision begins"""
        # Get collision details
        world_manifold = contact.worldManifold
        
        if world_manifold.points:
            collision_point = world_manifold.points[0]
            collision_normal = world_manifold.normal
            
            # Identify which body is the car and which is wall
            bodyA = contact.fixtureA.body
            bodyB = contact.fixtureB.body
            
            car_body = None
            car_id = None
            wall_body = None
            wall_id = None
            
            # Check which body is a car and which is wall
            if bodyA.userData and bodyA.userData.get("type") == "car":
                car_body = bodyA
                car_id = bodyA.userData.get("car_id", "unknown")
                if bodyB.userData and bodyB.userData.get("type") == "track_wall":
                    wall_body = bodyB
                    # Use a stable wall identifier based on position
                    pos = wall_body.position
                    wall_id = f"wall_{pos.x:.1f}_{pos.y:.1f}"
            elif bodyB.userData and bodyB.userData.get("type") == "car":
                car_body = bodyB  
                car_id = bodyB.userData.get("car_id", "unknown")
                if bodyA.userData and bodyA.userData.get("type") == "track_wall":
                    wall_body = bodyA
                    # Use a stable wall identifier based on position
                    pos = wall_body.position
                    wall_id = f"wall_{pos.x:.1f}_{pos.y:.1f}"
            
            if car_body is None or wall_body is None:
                # No car-wall collision, skip
                return
            
            # Calculate relative velocity at collision point
            velA = bodyA.GetLinearVelocityFromWorldPoint(collision_point)
            velB = bodyB.GetLinearVelocityFromWorldPoint(collision_point)
            relative_velocity = (velA.x - velB.x, velA.y - velB.y)
            
            # Estimate collision force based on mass and relative velocity
            massA = bodyA.mass if bodyA.mass > 0 else 1000.0
            massB = bodyB.mass if bodyB.mass > 0 else 1000.0
            reduced_mass = (massA * massB) / (massA + massB)
            
            # Calculate relative velocity component along collision normal
            # This gives the actual closing speed in the direction of impact
            normal_velocity = (relative_velocity[0] * collision_normal[0] + 
                             relative_velocity[1] * collision_normal[1])
            # Use absolute value of normal velocity for impulse magnitude
            collision_impulse = reduced_mass * abs(normal_velocity) * COLLISION_FORCE_SCALE_FACTOR
            
            # Apply realistic bounds to collision force
            collision_impulse = max(COLLISION_MIN_FORCE, min(collision_impulse, COLLISION_MAX_FORCE))
            
            # Store collision data
            collision = CollisionData(
                position=(collision_point[0], collision_point[1]),
                impulse=collision_impulse,
                normal=(collision_normal[0], collision_normal[1]),
                car_id=car_id
            )
            
            # Add to active collisions
            collision_key = (car_id, wall_id)
            self.active_collisions[collision_key] = collision
            
            # Initialize impulse tracking for this car if needed
            if car_id not in self.car_collision_impulses:
                self.car_collision_impulses[car_id] = 0.0
            
            # Add initial impulse
            self.car_collision_impulses[car_id] = collision_impulse
            
            # Debug print for collision force (only for active cars)
            # Extract car index from car_id format "car_X"
            if car_id and car_id.startswith("car_") and self.car_physics:
                try:
                    car_index = int(car_id.split("_")[1])
                    if car_index not in self.car_physics.disabled_cars:
                        #print(f"🔥 COLLISION: {car_id} force={collision_impulse:.1f} N⋅s")
                        pass
                except (ValueError, IndexError):
                    # Fallback for malformed car_id
                    #print(f"🔥 COLLISION: {car_id} force={collision_impulse:.1f} N⋅s")
                    pass
            elif not self.car_physics:
                # No reference to CarPhysics, show all collisions
                #print(f"🔥 COLLISION: {car_id} force={collision_impulse:.1f} N⋅s")
                pass
            
            # Also add to one-time collision list for logging
            self.collisions.append(collision)
            
    def EndContact(self, contact):
        """Called when collision ends"""
        # Identify which bodies were in contact
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        
        car_id = None
        wall_id = None
        
        # Find car and wall IDs
        if bodyA.userData and bodyA.userData.get("type") == "car":
            car_id = bodyA.userData.get("car_id", "unknown")
            if bodyB.userData and bodyB.userData.get("type") == "track_wall":
                # Use same stable wall identifier as BeginContact
                pos = bodyB.position
                wall_id = f"wall_{pos.x:.1f}_{pos.y:.1f}"
        elif bodyB.userData and bodyB.userData.get("type") == "car":
            car_id = bodyB.userData.get("car_id", "unknown")
            if bodyA.userData and bodyA.userData.get("type") == "track_wall":
                # Use same stable wall identifier as BeginContact
                pos = bodyA.position
                wall_id = f"wall_{pos.x:.1f}_{pos.y:.1f}"
        
        if car_id and wall_id:
            # Remove from active collisions
            collision_key = (car_id, wall_id)
            if collision_key in self.active_collisions:
                del self.active_collisions[collision_key]
            
            # Update impulse for this car - if no more active collisions, set to 0
            car_has_collisions = any(key[0] == car_id for key in self.active_collisions)
            if not car_has_collisions:
                self.car_collision_impulses[car_id] = 0.0
    
    def PostSolve(self, contact, impulse):
        """Called after collision resolution with impulse data"""
        # Get the normal impulse (force of collision)
        normal_impulses = impulse.normalImpulses
        if len(normal_impulses) > 0:
            total_impulse = sum(normal_impulses)
            
            # Identify car involved
            bodyA = contact.fixtureA.body
            bodyB = contact.fixtureB.body
            
            car_id = None
            if bodyA.userData and bodyA.userData.get("type") == "car":
                car_id = bodyA.userData.get("car_id", "unknown")
            elif bodyB.userData and bodyB.userData.get("type") == "car":
                car_id = bodyB.userData.get("car_id", "unknown")
            
            if car_id and car_id in self.car_collision_impulses:
                # Update with maximum impulse (to track strongest ongoing collision)
                self.car_collision_impulses[car_id] = max(self.car_collision_impulses[car_id], total_impulse)
    
    def get_collisions(self) -> List[CollisionData]:
        """Get and clear one-time collision list"""
        collisions = self.collisions.copy()
        self.collisions.clear()
        return collisions
    
    def get_car_collision_impulse(self, car_id: str) -> float:
        """Get current collision impulse for a specific car"""
        return self.car_collision_impulses.get(car_id, 0.0)
    
    def reset_impulses(self):
        """Reset impulse tracking for next physics step"""
        # Reset impulses for cars that have no active collisions
        for car_id in list(self.car_collision_impulses.keys()):
            car_has_active_collisions = any(key[0] == car_id for key in self.active_collisions)
            
            if not car_has_active_collisions:
                # No active collisions - completely reset impulse
                if self.car_collision_impulses[car_id] > 0:
                    print(f"🔄 IMPULSE RESET: {car_id} (no active collision)")
                self.car_collision_impulses[car_id] = 0.0
            else:
                # Has active collisions - apply light decay to smooth readings
                self.car_collision_impulses[car_id] *= 0.95