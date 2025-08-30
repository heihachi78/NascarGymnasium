"""
Tyre system implementation for realistic tyre physics simulation.

This module handles tyre temperature dynamics, grip calculation based on 
temperature, wear tracking, and load distribution effects.
"""

import math
from typing import Tuple
from .constants import (
    TYRE_START_TEMPERATURE,
    TYRE_IDEAL_TEMPERATURE_MIN,
    TYRE_IDEAL_TEMPERATURE_MAX,
    TYRE_MAX_GRIP_COEFFICIENT,
    TYRE_MIN_GRIP_COEFFICIENT,
    TYRE_GRIP_FALLOFF_RATE,
    TYRE_MAX_WEAR,
    TYRE_FRICTION_WEAR_COEFFICIENT,
    TYRE_GRIP_WEAR_FACTOR,
    TYRE_HEATING_RATE_FRICTION,
    TYRE_COOLING_RATE_AMBIENT,
    TYRE_HIGH_SPEED_COOLING_REDUCTION,
    TYRE_MIN_COOLING_SPEED_THRESHOLD,
    TYRE_AERODYNAMIC_HEATING_FACTOR,
    AMBIENT_TEMPERATURE,
    TYRE_THERMAL_MASS,
    STATIC_LOAD_PER_TYRE,
    TYRE_OPTIMAL_PRESSURE_PSI,
    TYRE_MIN_PRESSURE_PSI,
    TYRE_MAX_PRESSURE_PSI,
    TYRE_PRESSURE_TEMPERATURE_FACTOR,
    TYRE_PRESSURE_LOAD_FACTOR,
    MAX_TYRE_PRESSURE_INCREASE,
    CAR_MAX_SPEED_MS,
    # New speed-dependent wear constants
    TYRE_SPEED_WEAR_REFERENCE_SPEED,
    TYRE_SPEED_WEAR_MULTIPLIER,
    TYRE_SPEED_WEAR_MAX_MULTIPLIER,
    # New cornering wear constants
    TYRE_CORNERING_WEAR_BASE_G,
    TYRE_CORNERING_WEAR_MULTIPLIER,
    TYRE_CORNERING_WEAR_MAX_MULTIPLIER,
    # New slip wear constants
    TYRE_SLIP_WEAR_MAX_ANGLE,
    TYRE_SLIP_WEAR_MAX_MULTIPLIER,
    MS_TO_KMH,
    GRAVITY_MS2
)


class Tyre:
    """Individual tyre with temperature, grip, wear, and load simulation"""
    
    def __init__(self, position: str):
        """
        Initialize a tyre.
        
        Args:
            position: Tyre position ("front_left", "front_right", "rear_left", "rear_right")
        """
        self.position = position
        self.temperature = TYRE_START_TEMPERATURE
        self.wear = 0.0  # 0-100% wear
        self.load = STATIC_LOAD_PER_TYRE  # Equal initial load distribution
        self.friction_work = 0.0  # Accumulated friction work for heating
        self.pressure = TYRE_OPTIMAL_PRESSURE_PSI  # PSI pressure
        
    def update(self, dt: float, load: float, friction_force: float, speed: float = 0.0, lateral_accel: float = 0.0, slip_angle_degrees: float = 0.0) -> None:
        """
        Update tyre state for one physics time step.
        
        Args:
            dt: Time step in seconds
            load: Current load on the tyre in Newtons
            friction_force: Current friction force being applied
            speed: Current vehicle speed in m/s
            lateral_accel: Lateral acceleration in m/s² for cornering wear calculation
            slip_angle_degrees: Current slip angle in degrees for slip-based wear
        """
        self.load = load
        self._update_temperature(dt, friction_force, speed)
        self._update_wear(dt, friction_force, speed, lateral_accel, slip_angle_degrees)
        self._update_pressure(dt)
    
    def _update_temperature(self, dt: float, friction_force: float, speed: float) -> None:
        """Update tyre temperature based on friction work and ambient cooling"""
        # Heat generation from friction work with speed-dependent enhancement
        # Cap speed factor to prevent unrealistic heating at extreme speeds
        normalized_speed = min(speed / CAR_MAX_SPEED_MS, 2.0)  # Cap at 2x max speed
        speed_factor = 1.0 + normalized_speed ** 2  # Quadratic effect up to factor of 5
        friction_power = abs(friction_force) * TYRE_HEATING_RATE_FRICTION * speed_factor
        
        # Add aerodynamic heating at high speeds
        aerodynamic_heating = 0.0
        if speed > TYRE_MIN_COOLING_SPEED_THRESHOLD:
            aerodynamic_heating = (speed ** 2) * TYRE_AERODYNAMIC_HEATING_FACTOR
        
        total_heating = (friction_power + aerodynamic_heating) * dt / TYRE_THERMAL_MASS
        
        # Speed-dependent cooling reduction
        temperature_difference = self.temperature - AMBIENT_TEMPERATURE
        cooling_effectiveness = TYRE_COOLING_RATE_AMBIENT
        
        # Reduce cooling at high speeds (aerodynamic heating effect)
        if speed > TYRE_MIN_COOLING_SPEED_THRESHOLD:
            cooling_reduction = min(0.8, speed / 100.0)  # Up to 80% reduction at very high speeds
            cooling_effectiveness *= (1.0 - cooling_reduction * TYRE_HIGH_SPEED_COOLING_REDUCTION)
        
        cooling_rate = temperature_difference * cooling_effectiveness
        temperature_decrease = cooling_rate * dt
        
        # Update temperature
        self.temperature += total_heating - temperature_decrease
        
        # Ensure realistic temperature bounds (racing tires typically max ~120°C)
        self.temperature = max(AMBIENT_TEMPERATURE, min(120.0, self.temperature))
    
    def _update_wear(self, dt: float, friction_force: float, speed: float, lateral_accel: float, slip_angle_degrees: float) -> None:
        """Update tyre wear based on friction forces, temperature, load, speed, cornering intensity, and slip angle"""
        # Base wear rate is now proportional to actual friction forces (realistic physics-based wear)
        # No friction forces = no wear (e.g., stationary car)
        friction_wear_rate = abs(friction_force) * TYRE_FRICTION_WEAR_COEFFICIENT
        
        # Temperature-based wear multiplier
        if self.is_in_ideal_temperature_range():
            temperature_multiplier = 1.0
        else:
            # More aggressive wear increase for extreme temperatures
            if self.temperature > TYRE_IDEAL_TEMPERATURE_MAX:
                temp_excess = self.temperature - TYRE_IDEAL_TEMPERATURE_MAX
                temperature_multiplier = 1.0 + (temp_excess / 20.0)  # Exponential increase with heat
            else:
                temp_deficit = TYRE_IDEAL_TEMPERATURE_MIN - self.temperature
                temperature_multiplier = 1.0 + (temp_deficit / 30.0)  # Cold tyres wear more gradually
        
        # Load-based wear (proportional to work done)
        load_factor = self.load / STATIC_LOAD_PER_TYRE
        load_multiplier = max(0.5, load_factor)  # Reduced load = reduced wear
        
        # Work-based wear (heat generation indicates work done)
        work_factor = max(1.0, self.temperature / TYRE_START_TEMPERATURE)  # More work = more wear
        
        # Speed-dependent wear multiplier (proportional across all speeds)
        speed_kmh = speed * MS_TO_KMH
        # Formula: (actual_speed / 400 km/h) * multiplier + 1.0 (base multiplier)
        speed_wear_factor = 1.0 + (speed_kmh / TYRE_SPEED_WEAR_REFERENCE_SPEED) * TYRE_SPEED_WEAR_MULTIPLIER
        # Cap the multiplier to prevent extreme values
        speed_wear_factor = min(speed_wear_factor, TYRE_SPEED_WEAR_MAX_MULTIPLIER)
        
        # Cornering-intensity wear multiplier
        lateral_g = abs(lateral_accel) / GRAVITY_MS2  # Convert to G force (absolute value)
        if lateral_g > TYRE_CORNERING_WEAR_BASE_G:
            # Higher lateral G forces cause more wear
            cornering_excess = lateral_g - TYRE_CORNERING_WEAR_BASE_G
            cornering_wear_factor = 1.0 + (cornering_excess * TYRE_CORNERING_WEAR_MULTIPLIER)
            # Cap the multiplier
            cornering_wear_factor = min(cornering_wear_factor, TYRE_CORNERING_WEAR_MAX_MULTIPLIER)
        else:
            cornering_wear_factor = 1.0
        
        # Slip-angle wear multiplier (proportional scaling - no thresholds)
        slip_wear_factor = 1.0 + (abs(slip_angle_degrees) / TYRE_SLIP_WEAR_MAX_ANGLE) * (TYRE_SLIP_WEAR_MAX_MULTIPLIER - 1.0)
        # Cap the multiplier to prevent extreme values
        slip_wear_factor = min(slip_wear_factor, TYRE_SLIP_WEAR_MAX_MULTIPLIER)
        
        # Calculate total wear for this time step
        total_wear_multiplier = temperature_multiplier * load_multiplier * work_factor * speed_wear_factor * cornering_wear_factor * slip_wear_factor
        wear_rate = friction_wear_rate * total_wear_multiplier
        self.wear += wear_rate * dt
        
        # Clamp wear to maximum
        self.wear = min(TYRE_MAX_WEAR, self.wear)
    
    def _update_pressure(self, dt) -> None:
        """Update tyre pressure based on temperature and load"""
        # Base pressure starts at optimal
        base_pressure = TYRE_OPTIMAL_PRESSURE_PSI
        
        # Pressure increases with temperature
        temperature_delta = self.temperature - TYRE_START_TEMPERATURE
        temperature_pressure_change = temperature_delta * TYRE_PRESSURE_TEMPERATURE_FACTOR
        
        # Pressure changes with load (compression effect)
        load_delta = self.load - STATIC_LOAD_PER_TYRE
        load_pressure_change = load_delta * TYRE_PRESSURE_LOAD_FACTOR
        
        # Calculate new pressure with cap on increase
        total_pressure_increase = (temperature_pressure_change + load_pressure_change)
        # Cap the total increase
        total_pressure_increase = min(total_pressure_increase, MAX_TYRE_PRESSURE_INCREASE)
        
        self.pressure = base_pressure + total_pressure_increase
        
        # Clamp to safe limits
        self.pressure = max(TYRE_MIN_PRESSURE_PSI, min(TYRE_MAX_PRESSURE_PSI, self.pressure))
    
    def get_grip_coefficient(self) -> float:
        """Calculate current grip coefficient based on temperature and wear"""
        # Temperature-based grip
        temperature_grip = self._calculate_temperature_grip()
        
        # Wear-based grip reduction
        wear_factor = 1.0 - (self.wear / TYRE_MAX_WEAR) * (1.0 - TYRE_GRIP_WEAR_FACTOR)
        
        return temperature_grip * wear_factor
    
    def _calculate_temperature_grip(self) -> float:
        """Calculate grip coefficient based on temperature curve"""
        if self.is_in_ideal_temperature_range():
            # Peak grip at ideal temperature
            return TYRE_MAX_GRIP_COEFFICIENT
        
        # Calculate distance from ideal range
        if self.temperature < TYRE_IDEAL_TEMPERATURE_MIN:
            temp_deviation = TYRE_IDEAL_TEMPERATURE_MIN - self.temperature
        else:  # temperature > TYRE_IDEAL_TEMPERATURE_MAX
            temp_deviation = self.temperature - TYRE_IDEAL_TEMPERATURE_MAX
        
        # Linear falloff from ideal range
        grip_reduction = temp_deviation * TYRE_GRIP_FALLOFF_RATE
        grip_coefficient = TYRE_MAX_GRIP_COEFFICIENT - grip_reduction
        
        # Ensure minimum grip
        return max(TYRE_MIN_GRIP_COEFFICIENT, grip_coefficient)
    
    def is_in_ideal_temperature_range(self) -> bool:
        """Check if tyre is in ideal temperature range"""
        return TYRE_IDEAL_TEMPERATURE_MIN <= self.temperature <= TYRE_IDEAL_TEMPERATURE_MAX
    
    def get_max_friction_force(self) -> float:
        """Calculate maximum friction force this tyre can provide"""
        return self.get_grip_coefficient() * self.load
    
    def get_state(self) -> Tuple[float, float, float, float, float]:
        """
        Get current tyre state as tuple.
        
        Returns:
            Tuple of (temperature, wear, load, pressure, grip_coefficient)
        """
        return (
            self.temperature,
            self.wear,
            self.load,
            self.pressure,
            self.get_grip_coefficient()
        )
    
    def reset(self) -> None:
        """Reset tyre to initial state"""
        self.temperature = TYRE_START_TEMPERATURE
        self.wear = 0.0
        self.load = STATIC_LOAD_PER_TYRE
        self.friction_work = 0.0
        self.pressure = TYRE_OPTIMAL_PRESSURE_PSI
    
    def __str__(self) -> str:
        """String representation of tyre state"""
        temp_status = "IDEAL" if self.is_in_ideal_temperature_range() else "OFF-TEMP"
        return (f"Tyre({self.position}): {self.temperature:.1f}°C ({temp_status}), "
                f"Wear: {self.wear:.1f}%, Load: {self.load:.0f}N, "
                f"Grip: {self.get_grip_coefficient():.2f}")