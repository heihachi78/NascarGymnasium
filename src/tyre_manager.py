"""
TyreManager system for coordinating all four tyres and weight transfer physics.

This module manages the four individual tyres, calculates weight transfer effects
from acceleration/braking/turning, and coordinates tyre updates.
"""

import math
from typing import Dict, List, Tuple
from .tyre import Tyre
from .constants import (
    CAR_MASS,
    GRAVITY_MS2,
    EFFECTIVE_LONGITUDINAL_TRANSFER_FACTOR,
    EFFECTIVE_LATERAL_TRANSFER_FACTOR,
    TYRES_PER_AXLE,
    MIN_TYRE_LOAD_CONSTRAINT,
    CAR_WEIGHT_DISTRIBUTION_FRONT,
    CAR_WEIGHT_DISTRIBUTION_REAR,
    WHEELBASE_FRONT,
    WHEELBASE_REAR,
    TRACK_WIDTH,
    AERODYNAMIC_DOWNFORCE_COEFFICIENT,
    AERODYNAMIC_DOWNFORCE_REAR_BIAS,
    AERODYNAMIC_DOWNFORCE_SPEED_THRESHOLD,
    MAX_DOWNFORCE_MULTIPLIER,
    MIN_TYRE_LOAD,
    MAX_LONGITUDINAL_TRANSFER_RATIO
)


class TyreManager:
    """Manages all four tyres and calculates load distribution with weight transfer"""
    
    def __init__(self):
        """Initialize the tyre manager with four tyres"""
        # Validate weight distribution normalization
        total_weight_distribution = CAR_WEIGHT_DISTRIBUTION_FRONT + CAR_WEIGHT_DISTRIBUTION_REAR
        if abs(total_weight_distribution - 1.0) > 0.001:  # Allow small floating point errors
            print(f"Warning: Weight distribution does not sum to 1.0: {total_weight_distribution:.6f}")
            print(f"Front: {CAR_WEIGHT_DISTRIBUTION_FRONT}, Rear: {CAR_WEIGHT_DISTRIBUTION_REAR}")
            # Normalize the values
            normalized_front = CAR_WEIGHT_DISTRIBUTION_FRONT / total_weight_distribution
            normalized_rear = CAR_WEIGHT_DISTRIBUTION_REAR / total_weight_distribution
            print(f"Using normalized values - Front: {normalized_front:.6f}, Rear: {normalized_rear:.6f}")
            self._weight_front = normalized_front
            self._weight_rear = normalized_rear
        else:
            self._weight_front = CAR_WEIGHT_DISTRIBUTION_FRONT
            self._weight_rear = CAR_WEIGHT_DISTRIBUTION_REAR
        
        # Create four tyres
        self.tyres = {
            'front_left': Tyre('front_left'),
            'front_right': Tyre('front_right'),
            'rear_left': Tyre('rear_left'),
            'rear_right': Tyre('rear_right')
        }
        
        # Static load distribution (using validated values)
        self.static_weight = CAR_MASS * GRAVITY_MS2
        self.static_front_load = self.static_weight * self._weight_front
        self.static_rear_load = self.static_weight * self._weight_rear
        
        # Current dynamic loads (will be updated with weight transfer)
        self.current_loads = {
            'front_left': self.static_front_load / TYRES_PER_AXLE,
            'front_right': self.static_front_load / TYRES_PER_AXLE,
            'rear_left': self.static_rear_load / TYRES_PER_AXLE,
            'rear_right': self.static_rear_load / TYRES_PER_AXLE
        }
        
        # Friction forces for heating simulation
        self.friction_forces = {
            'front_left': 0.0,
            'front_right': 0.0,
            'rear_left': 0.0,
            'rear_right': 0.0
        }
    
    def update(self, dt: float, acceleration: Tuple[float, float], angular_acceleration: float, speed: float = 0.0, slip_angle_degrees: float = 0.0) -> None:
        """
        Update all tyres for one physics time step.
        
        Args:
            dt: Time step in seconds
            acceleration: Car acceleration (longitudinal, lateral) in m/s²
            angular_acceleration: Car angular acceleration in rad/s²
            speed: Current vehicle speed in m/s
            slip_angle_degrees: Current slip angle of the car in degrees
        """
        # Calculate weight transfer and update loads (including aerodynamic downforce)
        self._calculate_weight_transfer(acceleration, angular_acceleration, speed)
        
        # Extract lateral acceleration for tyre wear calculation
        longitudinal_accel, lateral_accel = acceleration
        
        # Update each tyre with its current load, friction force, speed, lateral acceleration, and slip angle
        for position, tyre in self.tyres.items():
            tyre.update(dt, self.current_loads[position], self.friction_forces[position], speed, lateral_accel, slip_angle_degrees)
    
    def _calculate_weight_transfer(self, acceleration: Tuple[float, float], angular_acceleration: float, speed: float) -> None:
        """Calculate load distribution with weight transfer effects and aerodynamic downforce"""
        longitudinal_accel, lateral_accel = acceleration
        
        # Debug removed for normal operation
        
        # Calculate aerodynamic downforce at high speeds with realistic cap
        aerodynamic_downforce = 0.0
        if speed > AERODYNAMIC_DOWNFORCE_SPEED_THRESHOLD:
            # Downforce increases quadratically with speed
            speed_factor = (speed / AERODYNAMIC_DOWNFORCE_SPEED_THRESHOLD) ** 2
            calculated_downforce = AERODYNAMIC_DOWNFORCE_COEFFICIENT * speed_factor * CAR_MASS * GRAVITY_MS2
            # Cap downforce at realistic maximum
            max_downforce = MAX_DOWNFORCE_MULTIPLIER * CAR_MASS * GRAVITY_MS2
            aerodynamic_downforce = min(calculated_downforce, max_downforce)
        
        # Distribute aerodynamic downforce (70% rear, 30% front for realistic sports car)
        aero_rear_load = aerodynamic_downforce * AERODYNAMIC_DOWNFORCE_REAR_BIAS
        aero_front_load = aerodynamic_downforce * (1.0 - AERODYNAMIC_DOWNFORCE_REAR_BIAS)
        
        # Base loads with aerodynamic enhancement
        base_front_load = self.static_front_load + aero_front_load
        base_rear_load = self.static_rear_load + aero_rear_load
        total_enhanced_weight = self.static_weight + aerodynamic_downforce
        
        # Longitudinal weight transfer (acceleration/braking) with realistic limits
        # Effective factor combines physics geometry and race car suspension rigidity
        raw_longitudinal_transfer = longitudinal_accel * total_enhanced_weight * EFFECTIVE_LONGITUDINAL_TRANSFER_FACTOR
        
        # Cap longitudinal transfer to prevent axles from going negative
        max_long_transfer_forward = base_front_load * MAX_LONGITUDINAL_TRANSFER_RATIO
        max_long_transfer_backward = base_rear_load * MAX_LONGITUDINAL_TRANSFER_RATIO
        
        if raw_longitudinal_transfer > 0:  # Acceleration (weight to rear)
            longitudinal_transfer = min(raw_longitudinal_transfer, max_long_transfer_forward)
        else:  # Braking (weight to front)
            longitudinal_transfer = max(raw_longitudinal_transfer, -max_long_transfer_backward)
        
        # Apply longitudinal transfer (front/rear distribution)
        front_total = base_front_load - longitudinal_transfer
        rear_total = base_rear_load + longitudinal_transfer
        
        # Lateral weight transfer (cornering) with realistic limits
        # Effective factor combines physics geometry and race car suspension rigidity
        raw_lateral_transfer = lateral_accel * total_enhanced_weight * EFFECTIVE_LATERAL_TRANSFER_FACTOR
        
        # Cap lateral transfer based on available load on each axle
        max_lateral_transfer_front = (front_total / TYRES_PER_AXLE) - MIN_TYRE_LOAD
        max_lateral_transfer_rear = (rear_total / TYRES_PER_AXLE) - MIN_TYRE_LOAD
        max_lateral_transfer = min(max_lateral_transfer_front, max_lateral_transfer_rear)
        
        # Apply the cap
        if max_lateral_transfer > 0:
            lateral_transfer = max(-max_lateral_transfer, min(max_lateral_transfer, raw_lateral_transfer))
        else:
            lateral_transfer = 0.0  # No room for lateral transfer
        
        # Apply lateral transfer (left/right distribution)
        # Positive lateral acceleration = left turn, centrifugal force pushes weight to RIGHT tyres (outside)
        front_left = front_total / TYRES_PER_AXLE - lateral_transfer / TYRES_PER_AXLE
        front_right = front_total / TYRES_PER_AXLE + lateral_transfer / TYRES_PER_AXLE
        rear_left = rear_total / TYRES_PER_AXLE - lateral_transfer / TYRES_PER_AXLE
        rear_right = rear_total / TYRES_PER_AXLE + lateral_transfer / TYRES_PER_AXLE
        
        # Apply minimum load constraint while preserving total weight
        min_load = MIN_TYRE_LOAD_CONSTRAINT  # Minimum load to prevent complete unloading
        
        # Calculate raw loads
        raw_loads = [front_left, front_right, rear_left, rear_right]
        
        # Apply minimum constraints and calculate deficit
        constrained_loads = []
        total_deficit = 0.0
        total_excess = 0.0
        
        for load in raw_loads:
            if load < min_load:
                constrained_loads.append(min_load)
                total_deficit += min_load - load
            else:
                constrained_loads.append(load)
                total_excess += load - min_load
        
        # Redistribute deficit proportionally among tyres with excess load
        if total_deficit > 0.0 and total_excess > 0.0:
            redistribution_factor = total_deficit / total_excess
            final_loads = []
            
            for i, load in enumerate(constrained_loads):
                if raw_loads[i] >= min_load:  # This tyre has excess load
                    excess = load - min_load
                    redistributed_load = load - excess * redistribution_factor
                    final_loads.append(max(min_load, redistributed_load))
                else:
                    final_loads.append(load)
        else:
            final_loads = constrained_loads
        
        self.current_loads = {
            'front_left': final_loads[0],
            'front_right': final_loads[1],
            'rear_left': final_loads[2],
            'rear_right': final_loads[3]
        }
        
        # Debug removed for normal operation
    
    def set_friction_forces(self, forces: Dict[str, float]) -> None:
        """Set friction forces for tyre heating calculation"""
        for position in self.friction_forces:
            self.friction_forces[position] = forces.get(position, 0.0)
    
    def get_total_grip_coefficient(self) -> float:
        """Get combined grip coefficient from all tyres"""
        total_grip = 0.0
        total_weight = 0.0
        
        for position, tyre in self.tyres.items():
            load = self.current_loads[position]
            grip = tyre.get_grip_coefficient()
            total_grip += grip * load
            total_weight += load
        
        return total_grip / total_weight if total_weight > 0 else 0.0
    
    def get_max_friction_forces(self) -> Dict[str, float]:
        """Get maximum friction force available from each tyre"""
        return {
            position: tyre.get_max_friction_force()
            for position, tyre in self.tyres.items()
        }
    
    def get_tyre_loads(self) -> Dict[str, float]:
        """Get current load on each tyre"""
        return self.current_loads.copy()
    
    def get_tyre_temperatures(self) -> Dict[str, float]:
        """Get current temperature of each tyre"""
        return {
            position: tyre.temperature
            for position, tyre in self.tyres.items()
        }
    
    def get_tyre_wear(self) -> Dict[str, float]:
        """Get current wear of each tyre"""
        return {
            position: tyre.wear
            for position, tyre in self.tyres.items()
        }
    
    def get_tyre_pressures(self) -> Dict[str, float]:
        """Get current pressure of each tyre"""
        return {
            position: tyre.pressure
            for position, tyre in self.tyres.items()
        }
    
    def get_tyre_grip_coefficients(self) -> Dict[str, float]:
        """Get current grip coefficient of each tyre"""
        return {
            position: tyre.get_grip_coefficient()
            for position, tyre in self.tyres.items()
        }
    
    def get_observation_data(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Get tyre data formatted for environment observation.
        
        Returns:
            Tuple of (loads, temperatures, wear_values) in order [FL, FR, RL, RR]
        """
        order = ['front_left', 'front_right', 'rear_left', 'rear_right']
        
        loads = [self.current_loads[pos] for pos in order]
        temperatures = [self.tyres[pos].temperature for pos in order]
        wear_values = [self.tyres[pos].wear for pos in order]
        
        return loads, temperatures, wear_values
    
    def reset(self) -> None:
        """Reset all tyres to initial state"""
        for tyre in self.tyres.values():
            tyre.reset()
        
        # Reset loads to static distribution
        self.current_loads = {
            'front_left': self.static_front_load / TYRES_PER_AXLE,
            'front_right': self.static_front_load / TYRES_PER_AXLE,
            'rear_left': self.static_rear_load / TYRES_PER_AXLE,
            'rear_right': self.static_rear_load / TYRES_PER_AXLE
        }
        
        # Reset friction forces
        self.friction_forces = {position: 0.0 for position in self.friction_forces}
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for all tyres"""
        loads = list(self.current_loads.values())
        temps = [tyre.temperature for tyre in self.tyres.values()]
        wear_vals = [tyre.wear for tyre in self.tyres.values()]
        grips = [tyre.get_grip_coefficient() for tyre in self.tyres.values()]
        
        return {
            'total_load': sum(loads),
            'avg_temperature': sum(temps) / len(temps),
            'max_temperature': max(temps),
            'min_temperature': min(temps),
            'avg_wear': sum(wear_vals) / len(wear_vals),
            'max_wear': max(wear_vals),
            'avg_grip': sum(grips) / len(grips),
            'min_grip': min(grips),
            'load_distribution_front': (loads[0] + loads[1]) / sum(loads),
            'load_distribution_rear': (loads[2] + loads[3]) / sum(loads)
        }
    
    def __str__(self) -> str:
        """String representation of tyre manager state"""
        stats = self.get_summary_stats()
        return (f"TyreManager: Total Load: {stats['total_load']:.0f}N, "
                f"Avg Temp: {stats['avg_temperature']:.1f}°C, "
                f"Avg Wear: {stats['avg_wear']:.1f}%, "
                f"Avg Grip: {stats['avg_grip']:.2f}")