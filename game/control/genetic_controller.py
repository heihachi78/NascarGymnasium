"""
Genetic Algorithm-optimized controller for car racing.

This module implements a rule-based controller whose parameters are evolved
using genetic algorithms. The controller maintains the same logic structure
as the base controller but with optimized parameters.
"""

import numpy as np
from .base_controller import BaseController


class GeneticController(BaseController):
    """
    Rule-based controller with genetic algorithm-optimized parameters.
    
    This controller uses the same control logic as the base controller
    but with evolved parameters for better performance.
    """
    
    def __init__(self, name=None, genome=None):
        """
        Initialize the genetic controller.
        
        Args:
            name: Optional name for this controller
            genome: List of evolved parameters, if None uses default values
        """
        super().__init__(name or "GeneticController")
        
        # Default genome (baseline parameters)
        self.default_genome = [
            0.95,   # speed_lower_multiplier
            1.05,   # speed_upper_multiplier  
            0.10,   # throttle_increment
            0.10,   # brake_increment
            1.00,   # steering_sensitivity
            0.50,   # throttle_reduction_when_steering
            0.90,   # steering_decay_rate
            0.25,   # max_steering_for_throttle_reduction
            1.00,   # left_sensor_weight
            1.00,   # right_sensor_weight
            0.80,   # collision_avoidance_threshold
            0.30,   # emergency_brake_threshold
        ]
        
        # Use provided genome or default
        self.genome = genome if genome is not None else self.default_genome.copy()
        
        # Extract parameters from genome
        self.speed_lower_multiplier = self.genome[0]
        self.speed_upper_multiplier = self.genome[1] 
        self.throttle_increment = self.genome[2]
        self.brake_increment = self.genome[3]
        self.steering_sensitivity = self.genome[4]
        self.throttle_reduction = self.genome[5]
        self.steering_decay = self.genome[6]
        self.max_steering_threshold = self.genome[7]
        self.left_sensor_weight = self.genome[8]
        self.right_sensor_weight = self.genome[9]
        self.collision_threshold = self.genome[10]
        self.emergency_brake_threshold = self.genome[11]
        
    def control(self, observation):
        """
        Generate control actions using evolved parameters.
        
        Args:
            observation: numpy array of shape (38,) containing car state
        
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
        """
        # Extract sensor data (16 sensors from index 22-37)
        sensors = observation[22:38]
        forward = sensors[0]           # Forward sensor (0°)
        current_speed = observation[4] # Speed from observation
        
        # Emergency braking for imminent collision
        if forward < self.emergency_brake_threshold:
            self.control_state['throttle_brake'] = -1.0  # Full brake
        else:
            # Speed control with evolved parameters
            if self.control_state['last_forward'] >= forward:
                self.control_state['speed_limit'] = forward * self.collision_threshold
            if self.control_state['last_forward'] < forward:
                self.control_state['speed_limit'] = 1.0
            
            # Throttle control with evolved increments
            if current_speed < self.control_state['speed_limit'] * self.speed_lower_multiplier:
                self.control_state['throttle_brake'] += self.throttle_increment
            if current_speed > self.control_state['speed_limit'] * self.speed_upper_multiplier:
                self.control_state['throttle_brake'] -= self.brake_increment
        
        # Enhanced steering control with evolved weights
        # Multi-sensor steering decision
        left_side_avg = np.mean(sensors[1:4]) * self.left_sensor_weight
        right_side_avg = np.mean(sensors[13:16]) * self.right_sensor_weight
        
        if right_side_avg > left_side_avg:
            steering_amount = (1 - (left_side_avg/right_side_avg)) * self.steering_sensitivity
            self.control_state['steering'] = min(steering_amount, 1.0)
        elif left_side_avg > right_side_avg:
            steering_amount = (1 - (right_side_avg/left_side_avg)) * self.steering_sensitivity
            self.control_state['steering'] = max(-steering_amount, -1.0)
        else:
            # Gradual return to center with evolved decay rate
            self.control_state['steering'] *= self.steering_decay
        
        # Reduce throttle when steering heavily (evolved threshold and reduction)
        if abs(self.control_state['steering']) > self.max_steering_threshold:
            self.control_state['throttle_brake'] *= self.throttle_reduction
        
        # Apply limits
        self.control_state['throttle_brake'] = max(min(self.control_state['throttle_brake'], 1), -1)
        self.control_state['steering'] = max(min(self.control_state['steering'], 1), -1)
        self.control_state['last_forward'] = forward
        
        return np.array([
            self.control_state['throttle_brake'],
            self.control_state['steering']
        ], dtype=np.float32)
    
    def get_genome(self):
        """Get the current genome parameters."""
        return self.genome.copy()
    
    def set_genome(self, genome):
        """Set new genome parameters and update internal parameters."""
        self.genome = genome.copy()
        
        # Update parameters from new genome
        self.speed_lower_multiplier = self.genome[0]
        self.speed_upper_multiplier = self.genome[1]
        self.throttle_increment = self.genome[2] 
        self.brake_increment = self.genome[3]
        self.steering_sensitivity = self.genome[4]
        self.throttle_reduction = self.genome[5]
        self.steering_decay = self.genome[6]
        self.max_steering_threshold = self.genome[7]
        self.left_sensor_weight = self.genome[8]
        self.right_sensor_weight = self.genome[9]
        self.collision_threshold = self.genome[10]
        self.emergency_brake_threshold = self.genome[11]
        
    def get_info(self):
        """Get information about this controller."""
        info = super().get_info()
        info.update({
            'genome_length': len(self.genome),
            'speed_params': [self.speed_lower_multiplier, self.speed_upper_multiplier],
            'control_params': [self.throttle_increment, self.brake_increment],
            'steering_params': [self.steering_sensitivity, self.steering_decay],
        })
        return info
    
    @staticmethod
    def get_genome_bounds():
        """
        Get the bounds for each genome parameter for GA optimization.
        
        Returns:
            List of (min, max) tuples for each parameter
        """
        return [
            (0.70, 0.99),   # speed_lower_multiplier
            (1.01, 1.30),   # speed_upper_multiplier
            (0.01, 0.30),   # throttle_increment
            (0.01, 0.30),   # brake_increment
            (0.10, 3.00),   # steering_sensitivity
            (0.10, 0.90),   # throttle_reduction_when_steering
            (0.70, 0.99),   # steering_decay_rate
            (0.10, 0.50),   # max_steering_for_throttle_reduction
            (0.50, 2.00),   # left_sensor_weight
            (0.50, 2.00),   # right_sensor_weight
            (0.50, 1.00),   # collision_avoidance_threshold
            (0.05, 0.50),   # emergency_brake_threshold
        ]
    
    @staticmethod
    def random_genome():
        """Generate a random genome within bounds."""
        bounds = GeneticController.get_genome_bounds()
        genome = []
        for min_val, max_val in bounds:
            genome.append(np.random.uniform(min_val, max_val))
        return genome
    
    @staticmethod
    def crossover(parent1_genome, parent2_genome, crossover_rate=0.5):
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1_genome: First parent genome
            parent2_genome: Second parent genome
            crossover_rate: Probability of crossover at each gene
            
        Returns:
            Two child genomes
        """
        child1 = []
        child2 = []
        
        for i in range(len(parent1_genome)):
            if np.random.random() < crossover_rate:
                # Swap genes
                child1.append(parent2_genome[i])
                child2.append(parent1_genome[i])
            else:
                # Keep original genes
                child1.append(parent1_genome[i])
                child2.append(parent2_genome[i])
                
        return child1, child2
    
    @staticmethod
    def mutate(genome, mutation_rate=0.1, mutation_strength=0.1):
        """
        Mutate a genome by adding random noise.
        
        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation at each gene
            mutation_strength: Maximum relative change for mutation
            
        Returns:
            Mutated genome
        """
        bounds = GeneticController.get_genome_bounds()
        mutated = []
        
        for i, gene in enumerate(genome):
            if np.random.random() < mutation_rate:
                # Apply mutation within bounds
                min_val, max_val = bounds[i]
                noise = np.random.uniform(-mutation_strength, mutation_strength) * gene
                mutated_gene = gene + noise
                # Ensure within bounds
                mutated_gene = max(min_val, min(max_val, mutated_gene))
                mutated.append(mutated_gene)
            else:
                mutated.append(gene)
                
        return mutated