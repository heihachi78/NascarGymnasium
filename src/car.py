"""
Car physics implementation with Box2D integration.

This module implements realistic car physics including:
- Box2D rigid body dynamics
- Engine torque curves and power delivery
- Rear-wheel drive system
- Aerodynamic drag and rolling resistance
- Integration with TyreManager for grip and weight transfer
"""

import Box2D
import math
from collections import deque
from typing import Tuple, Optional
from .tyre_manager import TyreManager
from .constants import (
    CAR_MASS,
    CAR_LENGTH,
    CAR_WIDTH,
    CAR_WHEELBASE,
    CAR_MAX_POWER,
    CAR_MAX_TORQUE,
    CAR_MAX_SPEED_MS,
    CAR_TARGET_100KMH_MS,
    CAR_ACCELERATION_0_100_KMH,
    CAR_DRAG_COEFFICIENT,
    CAR_FRONTAL_AREA,
    ROLLING_RESISTANCE_FORCE,
    AIR_DENSITY,
    CAR_DENSITY,
    CAR_FRICTION,
    CAR_RESTITUTION,
    CAR_MOMENT_OF_INERTIA_FACTOR,
    MAX_STEERING_ANGLE,
    THROTTLE_MIN,
    THROTTLE_MAX,
    BRAKE_MIN,
    BRAKE_MAX,
    STEERING_MIN,
    STEERING_MAX,
    MAX_LONGITUDINAL_ACCELERATION,
    MAX_LATERAL_ACCELERATION,
    ACCELERATION_SANITY_CHECK_THRESHOLD,
    ACCELERATION_SANITY_DAMPENING,
    ACCELERATION_HISTORY_SIZE,
    # Engine and drivetrain constants
    ENGINE_IDLE_RPM,
    ENGINE_PEAK_TORQUE_RPM,
    ENGINE_MAX_RPM,
    # Collision filtering constants
    COLLISION_CATEGORY_CARS,
    COLLISION_MASK_CARS,
    ENGINE_MIN_RPM,
    ENGINE_MAX_RPM_LIMIT,
    ENGINE_REDLINE_RPM_RANGE,
    ENGINE_RPM_RESPONSE_RATE,
    ENGINE_RPM_RESPONSE_EPSILON,
    ENGINE_TORQUE_CURVE_LOW_FACTOR,
    ENGINE_TORQUE_CURVE_HIGH_FACTOR,
    ENGINE_TORQUE_CURVE_FALLOFF_FACTOR,
    FINAL_DRIVE_RATIO,
    WHEEL_RADIUS,
    WHEEL_CIRCUMFERENCE,
    WHEEL_CIRCUMFERENCE_TO_RPM,
    POWER_LIMIT_TRANSITION_START,
    POWER_LIMIT_TRANSITION_END,
    POWER_LIMIT_TRANSITION_RANGE,
    POWER_TRANSITION_CURVE_SHARPNESS,
    POWER_TRANSITION_MIN_BLEND,
    POWER_TRANSITION_MAX_BLEND,
    # Force and physics constants
    GRAVITY_MS2,
    FRICTION_CIRCLE_STEERING_REDUCTION_MAX,
    FRICTION_CIRCLE_STEERING_FACTOR,
    FRICTION_CIRCLE_BRAKE_REDUCTION_MAX,
    FRICTION_CIRCLE_BRAKE_FACTOR,
    VELOCITY_ALIGNMENT_FORCE_FACTOR,
    MAX_BRAKE_DECELERATION_G,
    BRAKE_FORCE_DISTRIBUTION_WHEELS,
    BRAKE_FRICTION_SPEED_THRESHOLD,
    BRAKE_FRICTION_MIN_SPEED_FACTOR,
    MINIMUM_SPEED_FOR_DRAG,
    MINIMUM_SPEED_FOR_BRAKE,
    MINIMUM_SPEED_FOR_STEERING,
    MINIMUM_THROTTLE_THRESHOLD,
    MINIMUM_BRAKE_THRESHOLD,
    MINIMUM_STEERING_THRESHOLD,
    STEERING_TORQUE_MULTIPLIER,
    STEERING_ANGULAR_DAMPING,
    LATERAL_FORCE_SPEED_THRESHOLD,
    MAX_LATERAL_FORCE,
    FRONT_TYRE_LATERAL_FACTOR,
    REAR_TYRE_LATERAL_FACTOR,
    MAX_FRICTION_FORCE_CAP,
    REAR_WHEEL_COUNT,
    # Performance validation constants
    VELOCITY_HISTORY_SIZE,
    PERFORMANCE_VALIDATION_MIN_SAMPLES,
    PERFORMANCE_SPEED_TOLERANCE,
    PERFORMANCE_TIME_TOLERANCE,
    WORLD_FORWARD_VECTOR,
    WORLD_RIGHT_VECTOR,
    COORDINATE_ZERO,
    BODY_CENTER_OFFSET,
    RESET_ANGULAR_VELOCITY,
    AERODYNAMIC_DRAG_FACTOR,
    DRAG_CONSTANT,
    CAR_MOMENT_OF_INERTIA,
    PERPENDICULAR_ANGLE_OFFSET,
    # Slip angle heating constants
    SLIP_ANGLE_THRESHOLD_DEGREES,
    SLIP_ANGLE_HEATING_BASE_MULTIPLIER,
    SLIP_ANGLE_HEATING_EXPONENTIAL_FACTOR,
    SLIP_ANGLE_MAX_HEATING_MULTIPLIER,
    SLIP_ANGLE_SPEED_THRESHOLD,
    LATERAL_FORCE_HEATING_FACTOR,
    LATERAL_FORCE_DISTRIBUTION_FRONT,
    LATERAL_FORCE_DISTRIBUTION_REAR,
    MAX_LATERAL_HEATING_PER_TYRE,
    CORNERING_LOAD_HEATING_FACTOR,
    CORNERING_SPEED_THRESHOLD,
    CORNERING_OUTER_TYRE_HEATING_FACTOR,
    CORNERING_INNER_TYRE_HEATING_FACTOR
)


class Car:
    """Realistic car physics simulation with Box2D"""
    
    def __init__(self, world: Box2D.b2World, start_position: Tuple[float, float] = (0.0, 0.0), start_angle: float = 0.0, car_id: str = "car"):
        """
        Initialize car with Box2D physics.
        
        Args:
            world: Box2D world instance
            start_position: Initial position (x, y) in meters
            start_angle: Initial orientation in radians
            car_id: Unique identifier for this car
        """
        self.world = world
        self.start_position = start_position
        self.start_angle = start_angle
        self.car_id = car_id
        
        # Create tyre manager for grip and weight transfer
        self.tyre_manager = TyreManager()
        
        # Engine and drivetrain state
        self.engine_rpm = ENGINE_IDLE_RPM
        self.throttle = 0.0
        self.brake = 0.0
        self.steering_angle = 0.0
        
        # Control inputs (clamped to valid ranges)
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steering_input = 0.0
        
        # Store current timestep for physics calculations
        self.current_dt = 1.0/60.0  # Default timestep, updated each physics step
        
        # Performance tracking
        self.velocity_history = deque(maxlen=VELOCITY_HISTORY_SIZE)  # For acceleration validation
        
        # Acceleration tracking for proper weight transfer calculation
        self.previous_velocity = COORDINATE_ZERO  # Previous velocity for acceleration calculation
        self.acceleration_history = deque(maxlen=ACCELERATION_HISTORY_SIZE)
        self.last_acceleration = (0.0, 0.0)  # Cache last computed acceleration (longitudinal, lateral) to avoid side effects
        
        # Lateral force and slip angle tracking for tyre heating
        self.lateral_force_magnitude = 0.0  # Total lateral force applied for slip correction
        self.slip_angle_degrees = 0.0  # Current slip angle between velocity and car orientation
        
        # Create Box2D car body
        self._create_car_body()
        
    def _create_car_body(self) -> None:
        """Create the main car body with Box2D"""
        # Use pre-computed moment of inertia (simplified rectangular body)
        moment_of_inertia = CAR_MOMENT_OF_INERTIA
        
        # Define car body
        body_def = Box2D.b2BodyDef()
        body_def.type = Box2D.b2_dynamicBody
        body_def.position = self.start_position
        body_def.angle = self.start_angle
        
        # Create the body
        self.body = self.world.CreateBody(body_def)
        
        # Set car identification in userData for collision tracking
        self.body.userData = {"type": "car", "car_id": self.car_id}
        
        # Create car shape (rectangle)
        shape = Box2D.b2PolygonShape()
        shape.SetAsBox(CAR_LENGTH / 2.0, CAR_WIDTH / 2.0)
        
        # Create fixture with collision filtering
        fixture_def = Box2D.b2FixtureDef()
        fixture_def.shape = shape
        fixture_def.density = CAR_DENSITY
        fixture_def.friction = CAR_FRICTION
        fixture_def.restitution = CAR_RESTITUTION
        
        # Set collision filtering: cars only collide with track walls, not with other cars
        fixture_def.filter.categoryBits = COLLISION_CATEGORY_CARS
        fixture_def.filter.maskBits = COLLISION_MASK_CARS
        
        self.body.CreateFixture(fixture_def)
        
        # Set manual mass properties for more precise control
        mass_data = Box2D.b2MassData()
        mass_data.mass = CAR_MASS
        mass_data.center = BODY_CENTER_OFFSET  # Center of mass at body center
        mass_data.I = moment_of_inertia
        self.body.massData = mass_data
        
    def set_inputs(self, throttle: float, brake: float, steering: float) -> None:
        """
        Set control inputs with proper clamping.
        
        Args:
            throttle: Throttle input [0.0, 1.0]
            brake: Brake input [0.0, 1.0] 
            steering: Steering input [-1.0, 1.0]
        """
        self.throttle_input = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))
        self.brake_input = max(BRAKE_MIN, min(BRAKE_MAX, brake))
        self.steering_input = max(STEERING_MIN, min(STEERING_MAX, steering))
        
    def _calculate_engine_torque(self, rpm: float, throttle: float) -> float:
        """
        Calculate engine torque based on RPM and throttle position.
        
        Implements a realistic torque curve for a high-performance engine.
        """
        # RPM ranges for torque curve
        idle_rpm = ENGINE_IDLE_RPM
        peak_torque_rpm = ENGINE_PEAK_TORQUE_RPM  # RPM at peak torque
        max_rpm = ENGINE_MAX_RPM
        
        # Normalize RPM for curve calculation
        if rpm < idle_rpm:
            rpm = idle_rpm
        elif rpm > max_rpm:
            rpm = max_rpm
            
        # Torque curve (simplified but realistic)
        if rpm <= peak_torque_rpm:
            # Rising torque curve from idle to peak
            torque_factor = ENGINE_TORQUE_CURVE_LOW_FACTOR + ENGINE_TORQUE_CURVE_HIGH_FACTOR * (rpm - idle_rpm) / (peak_torque_rpm - idle_rpm)
        else:
            # Falling torque curve from peak to redline - more aggressive falloff
            torque_factor = 1.0 - ENGINE_TORQUE_CURVE_FALLOFF_FACTOR * (rpm - peak_torque_rpm) / (max_rpm - peak_torque_rpm)
        
        # Apply throttle and return torque
        base_torque = CAR_MAX_TORQUE * torque_factor
        return base_torque * throttle
        
    def _calculate_wheel_rpm(self) -> float:
        """Calculate wheel RPM based on current velocity"""
        velocity = self.get_velocity_magnitude()
        wheel_circumference = WHEEL_CIRCUMFERENCE  # Pre-computed circumference
        
        if wheel_circumference > 0:
            return (velocity * WHEEL_CIRCUMFERENCE_TO_RPM) / wheel_circumference  # Convert to RPM
        return 0.0
    
    def _update_engine_rpm(self, dt: float) -> None:
        """Update engine RPM based on throttle and load"""
        # Simplified engine RPM calculation
        # In reality this would include gear ratios, clutch engagement, etc.
        target_rpm = ENGINE_IDLE_RPM + (ENGINE_REDLINE_RPM_RANGE * self.throttle_input)  # Idle to redline based on throttle
        
        # Simple RPM response (engine inertia)
        rpm_response_rate = ENGINE_RPM_RESPONSE_RATE  # RPM per second response rate
        rpm_difference = target_rpm - self.engine_rpm
        self.engine_rpm += rpm_difference * min(1.0, dt * rpm_response_rate / abs(rpm_difference + ENGINE_RPM_RESPONSE_EPSILON))
        
        # Clamp RPM
        self.engine_rpm = max(ENGINE_MIN_RPM, min(ENGINE_MAX_RPM_LIMIT, self.engine_rpm))
    
    def update_physics(self, dt: float) -> None:
        """
        Update car physics for one time step.
        
        Args:
            dt: Time step in seconds
        """
        # Store current timestep for use in other methods
        self.current_dt = dt
        
        # Update control values
        self.throttle = self.throttle_input
        self.brake = self.brake_input  
        self.steering_angle = self.steering_input * math.radians(MAX_STEERING_ANGLE)
        
        # Update engine RPM
        self._update_engine_rpm(dt)
        
        # Calculate forces
        self._apply_engine_force()
        self._apply_brake_force()
        self._apply_aerodynamic_drag()
        self._apply_rolling_resistance()
        
        # Update weight transfer and tyre physics
        acceleration = self._get_acceleration(dt)
        angular_acceleration = self.body.angularVelocity  # Simplified
        speed = self.get_velocity_magnitude()
        self.tyre_manager.update(dt, acceleration, angular_acceleration, speed, self.slip_angle_degrees)
        
        # Apply lateral tire forces for velocity alignment (independent of steering)
        speed = self.get_velocity_magnitude()
        if speed > LATERAL_FORCE_SPEED_THRESHOLD:
            self._apply_lateral_tire_forces(speed)
        
        # Always apply angular damping for stability
        self._apply_angular_damping()
        
        # Apply steering torque (separate from lateral grip forces)
        if abs(self.steering_angle) > MINIMUM_STEERING_THRESHOLD:
            self._apply_steering_torque()
            
        # Update friction forces if not already done by engine/brake forces
        if self.throttle <= MINIMUM_THROTTLE_THRESHOLD and self.brake <= MINIMUM_BRAKE_THRESHOLD:
            self._update_friction_forces(0.0)  # Rolling resistance only
        else:
            # If engine or brake forces were applied, we still need to update lateral heating
            # Get current friction forces from tyre manager and add lateral heating
            current_friction = self.tyre_manager.friction_forces.copy()
            self._add_lateral_force_heating(current_friction, speed)
            self.tyre_manager.set_friction_forces(current_friction)
            
        # Track velocity for performance validation
        velocity = self.get_velocity_magnitude()
        self.velocity_history.append((velocity, dt))
        # deque automatically maintains maxlen, no manual size management needed
            
    def _apply_engine_force(self) -> None:
        """Apply engine force to rear wheels with realistic power/torque physics"""
        current_speed = self.get_velocity_magnitude()
        engine_torque = self._calculate_engine_torque(self.engine_rpm, self.throttle)
        
        # Realistic final drive ratio for performance
        final_drive_ratio = FINAL_DRIVE_RATIO  # Fine-tuned balance of acceleration and top speed
        wheel_torque = engine_torque * final_drive_ratio
        
        # Convert torque to force
        wheel_radius = WHEEL_RADIUS  # meters
        torque_limited_force = wheel_torque / wheel_radius if wheel_radius > 0 else 0.0
        
        # Smooth power limiting at high speeds using exponential transition
        # At high speeds, available force = Power / Velocity
        if current_speed > POWER_LIMIT_TRANSITION_START:
            # Calculate power-limited force
            power_limited_force = (CAR_MAX_POWER * self.throttle) / current_speed
            
            if current_speed <= POWER_LIMIT_TRANSITION_END:
                # Smooth exponential transition in the blend zone
                # Normalize speed within transition range (0 to 1)
                normalized_speed = (current_speed - POWER_LIMIT_TRANSITION_START) / POWER_LIMIT_TRANSITION_RANGE
                
                # Exponential blend factor for smooth transition
                # exp(-sharpness * (1-x)) creates smooth S-curve from 0 to 1
                exponential_factor = 1.0 - math.exp(-POWER_TRANSITION_CURVE_SHARPNESS * normalized_speed)
                blend_factor = max(POWER_TRANSITION_MIN_BLEND, min(POWER_TRANSITION_MAX_BLEND, exponential_factor))
                
                # Smooth blend between torque-limited and power-limited
                engine_force = torque_limited_force * (1.0 - blend_factor) + power_limited_force * blend_factor
            else:
                # Fully power-limited above transition end, but still blend slightly
                engine_force = torque_limited_force * (1.0 - POWER_TRANSITION_MAX_BLEND) + power_limited_force * POWER_TRANSITION_MAX_BLEND
        else:
            # At low speeds, fully torque limited
            engine_force = torque_limited_force
            
        # Apply realistic maximum force limit (tyre grip limited)
        # Use actual tyre grip coefficient from tyre wear/temperature system
        actual_grip_coefficient = self.tyre_manager.get_total_grip_coefficient()
        
        # Implement friction circle - reduce available longitudinal grip when steering
        # The more we steer, the less grip is available for acceleration
        steering_factor = 1.0 - min(FRICTION_CIRCLE_STEERING_REDUCTION_MAX, abs(self.steering_angle) * FRICTION_CIRCLE_STEERING_FACTOR)
        
        # Use tire model directly for grip limit (no additional arbitrary factors)
        max_theoretical_force = CAR_MASS * GRAVITY_MS2 * actual_grip_coefficient * steering_factor
        engine_force = min(engine_force, max_theoretical_force)
        
        # Apply force in forward direction (rear wheel drive)
        forward_vector = self.body.GetWorldVector(WORLD_FORWARD_VECTOR)
        force_vector = (engine_force * forward_vector[0], engine_force * forward_vector[1])
        
        # Calculate friction forces for tyre heating
        rear_friction_force = min(MAX_FRICTION_FORCE_CAP, abs(engine_force) / REAR_WHEEL_COUNT)  # Split between rear tyres
        self._update_friction_forces(rear_friction_force)
        
        # Apply force at rear axle position
        rear_axle_pos = self.body.GetWorldPoint((-CAR_WHEELBASE / 2, 0.0))
        self.body.ApplyForce(force_vector, rear_axle_pos, True)
        
    def _apply_brake_force(self) -> None:
        """Apply braking force"""
        if self.brake > MINIMUM_BRAKE_THRESHOLD:
            # Maximum braking force (distributed across all wheels)
            # Implement friction circle - reduce available braking grip when steering
            steering_factor = 1.0 - min(FRICTION_CIRCLE_BRAKE_REDUCTION_MAX, abs(self.steering_angle) * FRICTION_CIRCLE_BRAKE_FACTOR)
            max_brake_force = CAR_MASS * MAX_BRAKE_DECELERATION_G * steering_factor
            brake_force = max_brake_force * self.brake
            
            # Apply opposing force to velocity direction
            velocity = self.body.linearVelocity
            speed = velocity.length
            if speed > MINIMUM_SPEED_FOR_BRAKE:
                brake_direction = (-velocity[0] / speed, -velocity[1] / speed)
                force_vector = (brake_force * brake_direction[0], brake_force * brake_direction[1])
                self.body.ApplyForceToCenter(force_vector, True)
                
                # Update friction forces for braking
                self._update_friction_forces(0.0)  # No driving force, but braking will be calculated
                
    def _apply_aerodynamic_drag(self) -> None:
        """Apply aerodynamic drag force"""
        velocity = self.body.linearVelocity
        speed = velocity.length
        
        if speed > MINIMUM_SPEED_FOR_DRAG:
            # Drag force = 0.5 * ρ * Cd * A * v² (using pre-computed drag constant)
            drag_magnitude = DRAG_CONSTANT * speed * speed
            
            # Apply drag opposite to velocity direction
            drag_direction = (-velocity[0] / speed, -velocity[1] / speed)
            drag_force = (drag_magnitude * drag_direction[0], drag_magnitude * drag_direction[1])
            
            self.body.ApplyForceToCenter(drag_force, True)
            
    def _apply_rolling_resistance(self) -> None:
        """Apply rolling resistance force"""
        velocity = self.body.linearVelocity
        speed = velocity.length
        
        if speed > MINIMUM_SPEED_FOR_DRAG:
            # Rolling resistance = Crr * N (normal force = weight)
            rolling_resistance = ROLLING_RESISTANCE_FORCE
            
            # Apply opposite to velocity direction
            resistance_direction = (-velocity[0] / speed, -velocity[1] / speed)
            resistance_force = (rolling_resistance * resistance_direction[0], 
                              rolling_resistance * resistance_direction[1])
            
            self.body.ApplyForceToCenter(resistance_force, True)
            
    def _apply_angular_damping(self) -> None:
        """Apply angular damping to stabilize rotation (always active)"""
        # Apply angular damping proportional to angular velocity
        # This provides stability at all times, not just when steering
        damping_torque = -self.body.angularVelocity * CAR_MASS * STEERING_ANGULAR_DAMPING
        self.body.ApplyTorque(damping_torque, True)
    
    def _apply_steering_torque(self) -> None:
        """Apply steering torque for vehicle rotation (separate from lateral grip forces)"""
        # Get current velocity
        velocity = self.body.linearVelocity
        speed = velocity.length
        
        if speed > MINIMUM_SPEED_FOR_STEERING:  # Only apply steering above minimum speed
            # Calculate desired angular velocity based on bicycle model
            # ω = v * tan(δ) / L, where δ is steering angle, L is wheelbase
            desired_angular_velocity = speed * math.tan(self.steering_angle) / CAR_WHEELBASE
            
            # Apply torque to achieve desired angular velocity
            # Note: damping is now applied separately in _apply_angular_damping
            angular_velocity_error = desired_angular_velocity - self.body.angularVelocity
            steering_torque = angular_velocity_error * CAR_MASS * STEERING_TORQUE_MULTIPLIER
            
            self.body.ApplyTorque(steering_torque, True)
    
    def _apply_lateral_tire_forces(self, speed: float) -> None:
        """Apply lateral forces to align car velocity with car orientation and track slip angle"""
        # Reset tracking variables
        self.lateral_force_magnitude = 0.0
        self.slip_angle_degrees = 0.0
        
        if speed < LATERAL_FORCE_SPEED_THRESHOLD:
            return
            
        # Get car orientation vectors
        car_forward = self.body.GetWorldVector(WORLD_FORWARD_VECTOR)
        velocity = self.body.linearVelocity
        
        if velocity.length < LATERAL_FORCE_SPEED_THRESHOLD:
            return
        
        # Calculate slip angle between velocity direction and car orientation
        current_speed = velocity.length
        velocity_normalized = (velocity[0] / current_speed, velocity[1] / current_speed)
        
        # Calculate angle between velocity vector and car forward vector
        # Using atan2 for more robust angle calculation
        cross_product = velocity_normalized[0] * car_forward[1] - velocity_normalized[1] * car_forward[0]
        dot_product = velocity_normalized[0] * car_forward[0] + velocity_normalized[1] * car_forward[1]
        slip_angle_radians = math.atan2(abs(cross_product), dot_product)
        self.slip_angle_degrees = math.degrees(abs(slip_angle_radians))
        
        # Calculate where the car "wants" to go based on its orientation and current speed
        desired_velocity = (car_forward[0] * current_speed, car_forward[1] * current_speed)
        
        # Calculate the velocity error (difference between current and desired velocity)
        velocity_error = (desired_velocity[0] - velocity[0], desired_velocity[1] - velocity[1])
        
        # The lateral force needed to correct this error
        # Use a strong proportional controller to align velocity with car orientation
        alignment_force_factor = CAR_MASS * VELOCITY_ALIGNMENT_FORCE_FACTOR
        
        # Calculate corrective force for velocity alignment
        corrective_force = (velocity_error[0] * alignment_force_factor, 
                           velocity_error[1] * alignment_force_factor)
        
        # Apply tire grip limits - use MAX_LATERAL_FORCE as the absolute limit
        actual_grip_coefficient = self.tyre_manager.get_total_grip_coefficient()
        physics_based_force = CAR_MASS * GRAVITY_MS2 * actual_grip_coefficient * 1.0
        max_force = min(MAX_LATERAL_FORCE * actual_grip_coefficient, physics_based_force)
        
        # Limit the total force magnitude
        force_magnitude = (corrective_force[0]**2 + corrective_force[1]**2)**0.5
        if force_magnitude > max_force:
            scale_factor = max_force / force_magnitude
            corrective_force = (corrective_force[0] * scale_factor, 
                              corrective_force[1] * scale_factor)
            force_magnitude = max_force
        
        # Store lateral force magnitude for tyre heating calculations
        self.lateral_force_magnitude = force_magnitude
        
        # Apply the force at the center of mass for direct velocity control
        self.body.ApplyForceToCenter(corrective_force, True)
    
    def _update_friction_forces(self, driving_force: float) -> None:
        """Update friction forces for tyre heating calculations"""
        # Calculate friction forces for each tyre
        # During acceleration, rear tyres do most work (RWD)
        # During braking, all tyres work
        # During cornering, outer tyres work more
        
        speed = self.get_velocity_magnitude()
        
        # Base friction from driving force (rear tyres for RWD during acceleration)
        # At steady state high speed, distribute some drag resistance to front tyres too
        if self.throttle > MINIMUM_THROTTLE_THRESHOLD and speed > 50.0:
            # At high speed, front tyres also experience aerodynamic drag friction
            front_aero_fraction = min(0.3, speed / 200.0)  # Up to 30% at very high speeds
            rear_friction = driving_force * (1.0 - front_aero_fraction)
            front_friction = driving_force * front_aero_fraction / 2.0  # Split between front tyres
        else:
            rear_friction = driving_force if self.throttle > MINIMUM_THROTTLE_THRESHOLD else 0.0
            front_friction = 0.0
        
        # Add braking friction (distributed across all tyres)
        # Scale brake friction by speed to prevent heating when stationary
        brake_friction = 0.0
        if self.brake > MINIMUM_BRAKE_THRESHOLD:
            max_brake_force = CAR_MASS * MAX_BRAKE_DECELERATION_G * self.brake
            base_brake_friction = max_brake_force / BRAKE_FORCE_DISTRIBUTION_WHEELS  # Distributed to all 4 tyres
            
            # Apply speed-dependent scaling for friction heating calculation
            # At very low speeds, brake friction for heating should be minimal
            if speed <= BRAKE_FRICTION_SPEED_THRESHOLD:
                speed_factor = BRAKE_FRICTION_MIN_SPEED_FACTOR + (1.0 - BRAKE_FRICTION_MIN_SPEED_FACTOR) * (speed / BRAKE_FRICTION_SPEED_THRESHOLD)
                brake_friction = base_brake_friction * speed_factor
            else:
                brake_friction = base_brake_friction
        
        # Add rolling resistance (always present when moving)
        rolling_friction = 0.0
        if speed > MINIMUM_SPEED_FOR_DRAG:
            rolling_resistance = ROLLING_RESISTANCE_FORCE
            rolling_friction = rolling_resistance / BRAKE_FORCE_DISTRIBUTION_WHEELS  # Distributed to all 4 tyres
        
        # Calculate individual tyre friction forces
        friction_forces = {
            'front_left': front_friction + brake_friction + rolling_friction,
            'front_right': front_friction + brake_friction + rolling_friction,
            'rear_left': rear_friction / REAR_WHEEL_COUNT + brake_friction + rolling_friction,
            'rear_right': rear_friction / REAR_WHEEL_COUNT + brake_friction + rolling_friction
        }
        
        # Add enhanced lateral force heating from slip correction and cornering
        self._add_lateral_force_heating(friction_forces, speed)
        
        # Update tyre manager with friction forces
        self.tyre_manager.set_friction_forces(friction_forces)
    
    def _add_lateral_force_heating(self, friction_forces: dict, speed: float) -> None:
        """Add enhanced lateral force heating from slip angle and cornering forces"""
        if speed < SLIP_ANGLE_SPEED_THRESHOLD:
            return
        
        # Calculate slip angle heating multiplier
        slip_heating_multiplier = 1.0
        if self.slip_angle_degrees > SLIP_ANGLE_THRESHOLD_DEGREES:
            # Exponential heating increase with slip angle
            slip_excess = self.slip_angle_degrees - SLIP_ANGLE_THRESHOLD_DEGREES
            slip_heating_multiplier = SLIP_ANGLE_HEATING_BASE_MULTIPLIER + (
                SLIP_ANGLE_HEATING_EXPONENTIAL_FACTOR * slip_excess ** 2
            )
            # Cap the heating multiplier
            slip_heating_multiplier = min(slip_heating_multiplier, SLIP_ANGLE_MAX_HEATING_MULTIPLIER)
        
        # Calculate base lateral heating from the lateral corrective forces
        base_lateral_heating = self.lateral_force_magnitude * LATERAL_FORCE_HEATING_FACTOR * slip_heating_multiplier
        
        # Distribute lateral heating between front and rear tyres
        front_lateral_heating = base_lateral_heating * LATERAL_FORCE_DISTRIBUTION_FRONT / 2.0  # Split between L/R
        rear_lateral_heating = base_lateral_heating * LATERAL_FORCE_DISTRIBUTION_REAR / 2.0   # Split between L/R
        
        # Add cornering load transfer effects
        if abs(self.steering_angle) > MINIMUM_STEERING_THRESHOLD and speed > CORNERING_SPEED_THRESHOLD:
            # Get current tyre loads for load-based heating distribution
            tyres = self.tyre_manager.tyres
            
            # Determine which tyres are on the outside of the turn
            # Negative steering = turning left = left tyres on inside, right tyres on outside
            # Positive steering = turning right = right tyres on inside, left tyres on outside
            if self.steering_angle < 0:  # Left turn (negative steering)
                # Left tyres (outside) get more heating when turning left
                outer_front_tyre = 'front_left'
                outer_rear_tyre = 'rear_left'
                inner_front_tyre = 'front_right'
                inner_rear_tyre = 'rear_right'
            else:  # Right turn (positive steering)
                # Right tyres (outside) get more heating when turning right
                outer_front_tyre = 'front_right'
                outer_rear_tyre = 'rear_right'
                inner_front_tyre = 'front_left'
                inner_rear_tyre = 'rear_left'
            
            # Calculate load-based heating bonus for outer tyres
            outer_front_load = tyres[outer_front_tyre].load
            outer_rear_load = tyres[outer_rear_tyre].load
            
            outer_front_bonus = outer_front_load * CORNERING_LOAD_HEATING_FACTOR
            outer_rear_bonus = outer_rear_load * CORNERING_LOAD_HEATING_FACTOR
            
            # Apply asymmetric heating distribution (outer tyres work much harder)
            friction_forces[outer_front_tyre] += front_lateral_heating * CORNERING_OUTER_TYRE_HEATING_FACTOR + outer_front_bonus
            friction_forces[outer_rear_tyre] += rear_lateral_heating * CORNERING_OUTER_TYRE_HEATING_FACTOR + outer_rear_bonus
            friction_forces[inner_front_tyre] += front_lateral_heating * CORNERING_INNER_TYRE_HEATING_FACTOR
            friction_forces[inner_rear_tyre] += rear_lateral_heating * CORNERING_INNER_TYRE_HEATING_FACTOR
        else:
            # No steering - distribute evenly
            friction_forces['front_left'] += front_lateral_heating
            friction_forces['front_right'] += front_lateral_heating
            friction_forces['rear_left'] += rear_lateral_heating
            friction_forces['rear_right'] += rear_lateral_heating
        
        # Note: Individual tyre heating capping can be added later if needed
        # For now, let the heating be applied naturally
            
    def _get_acceleration(self, dt: float) -> Tuple[float, float]:
        """Calculate current acceleration for weight transfer using proper physics
        
        Args:
            dt: Time step in seconds
        """
        # Get current velocity
        current_velocity = self.body.linearVelocity
        current_vel_tuple = (current_velocity.x, current_velocity.y)
        
        # Calculate acceleration as change in velocity
        if dt > 0:
            # Calculate raw acceleration
            accel_x = (current_vel_tuple[0] - self.previous_velocity[0]) / dt
            accel_y = (current_vel_tuple[1] - self.previous_velocity[1]) / dt
            
            # Transform to car-local coordinates for longitudinal/lateral separation
            car_forward = self.body.GetWorldVector(WORLD_FORWARD_VECTOR)  # Car forward direction
            car_right = self.body.GetWorldVector(WORLD_RIGHT_VECTOR)  # Car right direction
            
            # Project acceleration onto car axes
            longitudinal_accel = accel_x * car_forward[0] + accel_y * car_forward[1]
            lateral_accel = accel_x * car_right[0] + accel_y * car_right[1]
            
            # Apply realistic limits to prevent extreme values
            # Clamp to realistic car physics limits (1g = 9.81 m/s²)
            longitudinal_accel = max(-MAX_LONGITUDINAL_ACCELERATION, 
                                   min(MAX_LONGITUDINAL_ACCELERATION, longitudinal_accel))
            lateral_accel = max(-MAX_LATERAL_ACCELERATION, 
                              min(MAX_LATERAL_ACCELERATION, lateral_accel))
            
            # Additional sanity check - if acceleration seems unrealistic, dampen it
            if abs(longitudinal_accel) > ACCELERATION_SANITY_CHECK_THRESHOLD or abs(lateral_accel) > ACCELERATION_SANITY_CHECK_THRESHOLD:
                # Something is wrong, probably a timestep issue or force spike
                longitudinal_accel *= ACCELERATION_SANITY_DAMPENING
                lateral_accel *= ACCELERATION_SANITY_DAMPENING
            
            # Store in history for smoothing
            self.acceleration_history.append((longitudinal_accel, lateral_accel))
            # deque automatically maintains maxlen, no manual size management needed
            
            # Use smoothed acceleration (average of recent samples)
            if self.acceleration_history:
                avg_long = sum(a[0] for a in self.acceleration_history) / len(self.acceleration_history)
                avg_lat = sum(a[1] for a in self.acceleration_history) / len(self.acceleration_history)  # Preserve sign for weight transfer direction
                result = (avg_long, avg_lat)
            else:
                result = (longitudinal_accel, lateral_accel)
        else:
            result = COORDINATE_ZERO
        
        # Update previous velocity for next calculation
        self.previous_velocity = current_vel_tuple
        
        # Cache the result for debug display without side effects
        self.last_acceleration = result
        
        return result
        
    def get_state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get current car state.
        
        Returns:
            Tuple of (pos_x, pos_y, vel_x, vel_y, orientation, angular_velocity)
        """
        position = self.body.position
        velocity = self.body.linearVelocity
        
        return (
            position.x,
            position.y,
            velocity.x,
            velocity.y,
            self.body.angle,
            self.body.angularVelocity
        )
        
    def get_velocity_magnitude(self) -> float:
        """Get current speed in m/s"""
        return self.body.linearVelocity.length
        
    def get_velocity_kmh(self) -> float:
        """Get current speed in km/h"""
        return self.get_velocity_magnitude() * 3.6
        
    def get_tyre_data(self) -> Tuple:
        """Get tyre data for observation space"""
        return self.tyre_manager.get_observation_data()
        
    def get_drag_force(self) -> float:
        """Get current drag force in Newtons"""
        velocity = self.get_velocity_magnitude()
        return DRAG_CONSTANT * velocity * velocity
        
    def get_velocity_vector(self) -> Tuple[float, float]:
        """Get current velocity vector in m/s"""
        velocity = self.body.linearVelocity
        return (velocity.x, velocity.y)
        
    def get_acceleration_vector(self) -> Tuple[float, float]:
        """Get current acceleration vector in m/s² (world coordinates)"""
        # Get car-relative acceleration (longitudinal, lateral)
        longitudinal, lateral = self.last_acceleration
        
        # Get car orientation vectors in world space
        car_forward = self.body.GetWorldVector(WORLD_FORWARD_VECTOR)
        car_right = self.body.GetWorldVector(WORLD_RIGHT_VECTOR)
        
        # Transform car-relative acceleration to world coordinates
        world_accel_x = longitudinal * car_forward[0] + lateral * car_right[0]
        world_accel_y = longitudinal * car_forward[1] + lateral * car_right[1]
        
        return (world_accel_x, world_accel_y)
        
    def get_last_acceleration_vector(self) -> Tuple[float, float]:
        """Get last computed acceleration vector without side effects (for debug display)"""
        # Get cached car-relative acceleration (longitudinal, lateral)
        longitudinal, lateral = self.last_acceleration
        
        # Get car orientation vectors in world space
        car_forward = self.body.GetWorldVector(WORLD_FORWARD_VECTOR)
        car_right = self.body.GetWorldVector(WORLD_RIGHT_VECTOR)
        
        # Transform car-relative acceleration to world coordinates
        world_accel_x = longitudinal * car_forward[0] + lateral * car_right[0]
        world_accel_y = longitudinal * car_forward[1] + lateral * car_right[1]
        
        return (world_accel_x, world_accel_y)
        
    def get_steering_vectors(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get steering vectors for debug visualization.
        
        Returns:
            Tuple of (input_vector, actual_vector) representing steering input and actual angle
        """
        # Input steering vector (normalized to -1 to 1 range)
        input_magnitude = abs(self.steering_input)
        input_angle = self.body.angle + (PERPENDICULAR_ANGLE_OFFSET if self.steering_input > 0 else -PERPENDICULAR_ANGLE_OFFSET)
        input_vector = (
            input_magnitude * math.cos(input_angle),
            input_magnitude * math.sin(input_angle)
        )
        
        # Actual steering vector (based on actual steering angle)
        actual_magnitude = abs(self.steering_angle) / MAX_STEERING_ANGLE
        actual_angle = self.body.angle + self.steering_angle + math.pi/2
        actual_vector = (
            actual_magnitude * math.cos(actual_angle),
            actual_magnitude * math.sin(actual_angle)
        )
        
        return (input_vector, actual_vector)
        
    def reset(self) -> None:
        """Reset car to initial state"""
        self.body.position = self.start_position
        self.body.angle = self.start_angle
        self.body.linearVelocity = COORDINATE_ZERO
        self.body.angularVelocity = RESET_ANGULAR_VELOCITY
        
        # Reset control inputs
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steering_input = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steering_angle = 0.0
        
        # Reset engine state
        self.engine_rpm = ENGINE_IDLE_RPM
        
        # Reset tyre manager
        self.tyre_manager.reset()
        
        # Reset acceleration tracking
        self.previous_velocity = COORDINATE_ZERO
        self.acceleration_history = deque(maxlen=ACCELERATION_HISTORY_SIZE)
        self.last_acceleration = (0.0, 0.0)
        
        # Reset lateral force and slip angle tracking
        self.lateral_force_magnitude = 0.0
        self.slip_angle_degrees = 0.0
        
        # Clear performance history
        self.velocity_history.clear()
        
    def validate_performance(self) -> dict:
        """
        Validate car performance against specifications.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "max_speed_ms": CAR_MAX_SPEED_MS,
            "target_100kmh_ms": CAR_TARGET_100KMH_MS,
            "target_acceleration_time": CAR_ACCELERATION_0_100_KMH,
            "current_max_speed": 0.0,
            "estimated_0_100_time": 0.0,
            "performance_valid": False
        }
        
        if len(self.velocity_history) > PERFORMANCE_VALIDATION_MIN_SAMPLES:
            # Find maximum achieved speed
            max_speed = max(v[0] for v in self.velocity_history)
            results["current_max_speed"] = max_speed
            
            # Estimate 0-100 km/h time from velocity history
            target_speed = CAR_TARGET_100KMH_MS
            time_to_target = 0.0
            
            for i, (speed, dt) in enumerate(self.velocity_history):
                time_to_target += dt
                if speed >= target_speed:
                    results["estimated_0_100_time"] = time_to_target
                    break
                    
            # Validate performance
            speed_ok = max_speed >= CAR_MAX_SPEED_MS * PERFORMANCE_SPEED_TOLERANCE  # Within 5% of target
            accel_ok = (results["estimated_0_100_time"] <= CAR_ACCELERATION_0_100_KMH * PERFORMANCE_TIME_TOLERANCE 
                       if results["estimated_0_100_time"] > 0 else False)
            
            results["performance_valid"] = speed_ok and accel_ok
            
        return results
        
    def __str__(self) -> str:
        """String representation of car state"""
        pos_x, pos_y, vel_x, vel_y, angle, angular_vel = self.get_state()
        speed_kmh = self.get_velocity_kmh()
        
        # Calculate current engine torque for debug
        engine_torque = self._calculate_engine_torque(self.engine_rpm, self.throttle)
        gear_ratio = FINAL_DRIVE_RATIO  # Same as in _apply_engine_force
        wheel_torque = engine_torque * gear_ratio
        wheel_force = wheel_torque / WHEEL_RADIUS if wheel_torque > 0 else 0.0
        
        return (f"Car: Pos({pos_x:.1f}, {pos_y:.1f}), "
                f"Speed: {speed_kmh:.1f} km/h, "
                f"Throttle: {self.throttle:.2f}, "
                f"RPM: {self.engine_rpm:.0f}, "
                f"WheelForce: {wheel_force:.0f}N")