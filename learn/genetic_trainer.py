"""
Genetic Algorithm trainer for evolving car racing controllers.

This module implements a genetic algorithm to evolve optimal parameters
for rule-based car controllers using fitness evaluation on racing tracks.
"""

import numpy as np
import os
import json
import time
import pickle
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from game.control.genetic_controller import GeneticController
from src.car_env import CarEnv


class GeneticTrainer:
    """
    Genetic Algorithm trainer for car racing controllers.
    
    Evolves controller parameters through fitness-based selection,
    crossover, and mutation operations.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.15,
                 mutation_strength: float = 0.2,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.2,
                 track_file: Optional[str] = None,
                 parallel_workers: int = 4):
        """
        Initialize the genetic algorithm trainer.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of gene mutation
            mutation_strength: Maximum relative change during mutation
            crossover_rate: Probability of crossover between parents
            elite_ratio: Fraction of population to preserve as elite
            track_file: Specific track to train on (None for random tracks)
            parallel_workers: Number of parallel processes for evaluation
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.track_file = track_file
        self.parallel_workers = parallel_workers
        
        # Evolution tracking
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.generation = 0
        
        # Create results directory
        self.results_dir = "learn/genetic_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def initialize_population(self):
        """Initialize random population of genomes."""
        self.population = []
        for _ in range(self.population_size):
            genome = GeneticController.random_genome()
            self.population.append(genome)
        print(f"🧬 Initialized population of {self.population_size} individuals")
        
    def evaluate_individual(self, genome: List[float], individual_id: int = 0) -> Dict:
        """
        Evaluate fitness of a single individual.
        
        Args:
            genome: Individual's genome to evaluate
            individual_id: ID for tracking/debugging
            
        Returns:
            Dictionary containing fitness metrics
        """
        # Create environment with specific or random track
        env = CarEnv(
            render_mode=None,
            track_file=self.track_file,
            reset_on_lap=False,
            discrete_action_space=False
        )
        
        # Create controller with this genome
        controller = GeneticController(
            name=f"Individual_{individual_id}",
            genome=genome
        )
        
        # Run evaluation episode
        observation, info = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 9999  # Prevent infinite episodes
        
        # Fitness metrics
        distance_traveled = 0.0
        max_speed = 0.0
        time_on_track = 0.0
        lap_completed = False
        lap_time = None
        
        try:
            while steps < max_steps:
                # Get action from controller
                action = controller.control(observation)
                
                # Take step in environment
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Accumulate metrics
                total_reward += reward
                steps += 1
                
                # Extract performance metrics from info
                if isinstance(info, dict):
                    if 'cars' in info and len(info['cars']) > 0:
                        car_info = info['cars'][0]  # Single car
                        
                        # Track distance and speed
                        if 'car_speed_ms' in car_info:
                            speed = car_info['car_speed_ms']
                            max_speed = max(max_speed, speed)
                            distance_traveled += speed * (1/60)  # Assuming 60Hz
                        
                        # Track time on track
                        if car_info.get('on_track', False):
                            time_on_track += 1/60
                        
                        # Check for lap completion
                        if 'lap_timing' in car_info:
                            lap_info = car_info['lap_timing']
                            if lap_info.get('lap_count', 0) > 0 and not lap_completed:
                                lap_completed = True
                                lap_time = lap_info.get('last_lap_time')
                
                # Check termination
                if terminated or truncated:
                    break
                    
        except Exception as e:
            print(f"⚠️ Error evaluating individual {individual_id}: {e}")
            # Return poor fitness for failed individuals
            env.close()
            return {
                'total_reward': -1000,
                'distance_traveled': 0,
                'max_speed': 0,
                'time_on_track': 0,
                'lap_completed': False,
                'lap_time': None,
                'steps': steps,
                'fitness': -1000
            }
        
        # Calculate composite fitness
        fitness = self._calculate_fitness(
            total_reward, distance_traveled, max_speed, 
            time_on_track, lap_completed, lap_time, steps
        )
        
        env.close()
        
        return {
            'total_reward': total_reward,
            'distance_traveled': distance_traveled,
            'max_speed': max_speed,
            'time_on_track': time_on_track,
            'lap_completed': lap_completed,
            'lap_time': lap_time,
            'steps': steps,
            'fitness': fitness
        }
    
    def _calculate_fitness(self, reward: float, distance: float, max_speed: float,
                          time_on_track: float, lap_completed: bool, lap_time: Optional[float],
                          steps: int) -> float:
        """
        Calculate composite fitness score from performance metrics.
        
        Multi-objective fitness combining:
        - Primary: Lap completion and time
        - Secondary: Distance, speed, track adherence
        - Penalties: Poor performance, crashes
        """
        fitness = 0.0
        
        # Primary objective: Lap completion
        if lap_completed and lap_time is not None:
            # Reward lap completion heavily, with bonus for faster times
            fitness += 1000  # Base lap completion bonus
            fitness += max(0, 200 - lap_time)  # Time bonus (assumes reasonable lap times < 200s)
        else:
            # Reward progress towards lap completion
            fitness += distance * 3  # Distance reward
            fitness += max_speed * 2  # Speed reward
        
        # Secondary objectives
        fitness += time_on_track  # Staying on track is important
        fitness += reward  # Environment reward (smaller weight)
        
        # Penalties
        if distance < 10:  # Very poor performance
            fitness -= 200
        if max_speed < 1:  # Car didn't move much
            fitness -= 100
            
        return fitness
    
    def _evaluate_batch(self, genomes: List[List[float]], start_idx: int = 0) -> List[Dict]:
        """
        Evaluate a batch of genomes simultaneously using multi-car environment.
        
        Args:
            genomes: List of genomes to evaluate
            start_idx: Starting index for individual IDs
            
        Returns:
            List of evaluation results for each genome
        """
        num_individuals = len(genomes)
        
        # Create multi-car environment
        env = CarEnv(
            render_mode=None,
            track_file=self.track_file,
            reset_on_lap=False,
            discrete_action_space=False,
            num_cars=num_individuals,
            car_names=[f"Individual_{start_idx + i}" for i in range(num_individuals)]
        )
        
        # Create controllers for each genome
        controllers = []
        for i, genome in enumerate(genomes):
            controller = GeneticController(
                name=f"Individual_{start_idx + i}",
                genome=genome
            )
            controllers.append(controller)
        
        # Run evaluation episode with all individuals simultaneously
        try:
            observations, info = env.reset()
            max_steps = 9999  # Prevent infinite episodes
            steps = 0
            
            # Initialize metrics for each individual
            individual_metrics = []
            for i in range(num_individuals):
                individual_metrics.append({
                    'total_reward': 0.0,
                    'distance_traveled': 0.0,
                    'max_speed': 0.0,
                    'time_on_track': 0.0,
                    'lap_completed': False,
                    'lap_time': None,
                    'steps': 0
                })
            
            while steps < max_steps:
                # Get actions from all controllers
                actions = []
                for i, controller in enumerate(controllers):
                    if i < len(observations):
                        action = controller.control(observations[i])
                        actions.append(action)
                    else:
                        # Fallback if observation missing
                        actions.append([0.0, 0.0])
                
                # Take step in environment
                observations, rewards, terminated, truncated, info = env.step(actions)
                steps += 1
                
                # Update metrics for each individual
                for i in range(num_individuals):
                    metrics = individual_metrics[i]
                    
                    # Accumulate reward
                    if i < len(rewards):
                        metrics['total_reward'] += rewards[i]
                    
                    # Extract car-specific metrics from info
                    if isinstance(info, dict) and 'cars' in info and i < len(info['cars']):
                        car_info = info['cars'][i]
                        
                        # Track speed and distance
                        if 'car_speed_ms' in car_info:
                            speed = car_info['car_speed_ms']
                            metrics['max_speed'] = max(metrics['max_speed'], speed)
                            metrics['distance_traveled'] += speed * (1/60)  # Assuming 60Hz
                        
                        # Track time on track
                        if car_info.get('on_track', False):
                            metrics['time_on_track'] += 1/60
                        
                        # Check for lap completion
                        if 'lap_timing' in car_info:
                            lap_info = car_info['lap_timing']
                            if lap_info.get('lap_count', 0) > 0 and not metrics['lap_completed']:
                                metrics['lap_completed'] = True
                                metrics['lap_time'] = lap_info.get('last_lap_time')
                    
                    metrics['steps'] = steps
                
                # Check termination
                if terminated or truncated:
                    break
            
            env.close()
            
            # Calculate fitness for each individual
            results = []
            for i, metrics in enumerate(individual_metrics):
                fitness = self._calculate_fitness(
                    metrics['total_reward'],
                    metrics['distance_traveled'],
                    metrics['max_speed'],
                    metrics['time_on_track'],
                    metrics['lap_completed'],
                    metrics['lap_time'],
                    metrics['steps']
                )
                
                result = {
                    **metrics,
                    'fitness': fitness
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error in batch evaluation: {e}")
            env.close()
            
            # Return poor fitness for all individuals on error
            return [{
                'total_reward': -1000,
                'distance_traveled': 0,
                'max_speed': 0,
                'time_on_track': 0,
                'lap_completed': False,
                'lap_time': None,
                'steps': max_steps,
                'fitness': -1000
            } for _ in genomes]
    
    def evaluate_population(self) -> List[Dict]:
        """Evaluate entire population using multi-car environments for efficiency."""
        print(f"🏁 Evaluating generation {self.generation} ({self.population_size} individuals)...")
        
        # Use multi-car evaluation for much faster training
        if self.population_size <= 10:
            # Single batch - all individuals in one environment
            print(f"  Evaluating all {self.population_size} individuals simultaneously...")
            results = self._evaluate_batch(self.population)
        else:
            # Multiple batches of up to 10 individuals each
            results = []
            batch_size = 10
            num_batches = (self.population_size + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.population_size)
                batch_genomes = self.population[start_idx:end_idx]
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Evaluating individuals {start_idx + 1}-{end_idx}...")
                batch_results = self._evaluate_batch(batch_genomes, start_idx)
                results.extend(batch_results)
        
        # Print summary
        fitness_scores = [r['fitness'] for r in results]
        print(f"  Fitness range: {min(fitness_scores):.1f} - {max(fitness_scores):.1f}")
        
        return results
    
    def selection(self, fitness_scores: List[float]) -> List[int]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            fitness_scores: List of fitness values for population
            
        Returns:
            List of indices of selected parents
        """
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size - self.elite_count):
            # Tournament selection
            tournament = np.random.choice(len(fitness_scores), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner_idx = tournament[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
            
        return selected
    
    def evolve_generation(self, results: List[Dict]):
        """Evolve population for one generation."""
        fitness_scores = [r['fitness'] for r in results]
        
        # Track best individual
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
            print(f"🏆 New best individual! Fitness: {self.best_fitness:.1f}")
        
        # Store fitness history (convert numpy types to Python types)
        self.fitness_history.append({
            'generation': int(self.generation),
            'best_fitness': float(max(fitness_scores)),
            'avg_fitness': float(np.mean(fitness_scores)),
            'std_fitness': float(np.std(fitness_scores))
        })
        
        # Create new population
        new_population = []
        
        # Keep elite individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring through crossover and mutation
        selected_parents = self.selection(fitness_scores)
        
        while len(new_population) < self.population_size:
            # Select two parents
            parent1_idx = np.random.choice(selected_parents)
            parent2_idx = np.random.choice(selected_parents)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = GeneticController.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = GeneticController.mutate(child1, self.mutation_rate, self.mutation_strength)
            child2 = GeneticController.mutate(child2, self.mutation_rate, self.mutation_strength)
            
            # Add children to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        
    def save_progress(self):
        """Save current evolution progress."""
        # Convert numpy types to Python native types for JSON serialization
        best_individual = None
        if self.best_individual:
            best_individual = [float(x) for x in self.best_individual]
        
        progress_data = {
            'generation': int(self.generation),
            'population_size': int(self.population_size),
            'best_fitness': float(self.best_fitness),
            'best_individual': best_individual,
            'fitness_history': self.fitness_history,
            'hyperparameters': {
                'mutation_rate': float(self.mutation_rate),
                'mutation_strength': float(self.mutation_strength),
                'crossover_rate': float(self.crossover_rate),
                'elite_ratio': float(self.elite_ratio)
            }
        }
        
        # Save progress data
        progress_file = os.path.join(self.results_dir, f"progress_gen_{self.generation}.json")
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Save best individual separately
        if self.best_individual:
            best_file = os.path.join(self.results_dir, f"best_individual_gen_{self.generation}.pkl")
            with open(best_file, 'wb') as f:
                pickle.dump(self.best_individual, f)
    
    def plot_fitness_evolution(self):
        """Plot fitness evolution over generations."""
        if not self.fitness_history:
            return
            
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        
        plot_file = os.path.join(self.results_dir, f"fitness_evolution_gen_{self.generation}.png")
        plt.savefig(plot_file)
        plt.close()
    
    def train(self):
        """Run complete genetic algorithm training."""
        print(f"🧬 Starting Genetic Algorithm Training")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Mutation: {self.mutation_rate:.2f}, Crossover: {self.crossover_rate:.2f}")
        print(f"Track: {self.track_file or 'Random tracks'}")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        start_time = time.time()
        
        for generation in range(self.generations):
            self.generation = generation
            
            print(f"\n📈 Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            results = self.evaluate_population()
            
            # Print generation statistics
            fitness_scores = [r['fitness'] for r in results]
            print(f"Best: {max(fitness_scores):.1f}, Avg: {np.mean(fitness_scores):.1f}, Std: {np.std(fitness_scores):.1f}")
            
            # Evolve to next generation
            self.evolve_generation(results)
            
            # Save progress periodically
            if generation % 10 == 0 or generation == self.generations - 1:
                self.save_progress()
                self.plot_fitness_evolution()
        
        total_time = time.time() - start_time
        print(f"\n🎯 Training completed in {total_time:.1f}s")
        print(f"Best fitness achieved: {self.best_fitness:.1f}")
        
        # Save final results
        self.save_final_results()
        
        return self.best_individual, self.best_fitness
    
    def save_final_results(self):
        """Save final training results and best controller."""
        if not self.best_individual:
            return
            
        # Save best controller (convert numpy types)
        best_genome = [float(x) for x in self.best_individual] if self.best_individual else None
        best_controller = GeneticController(
            name="BestEvolved",
            genome=best_genome
        )
        
        controller_file = os.path.join(self.results_dir, "best_evolved_controller.pkl")
        with open(controller_file, 'wb') as f:
            pickle.dump(best_controller, f)
        
        # Save readable genome parameters
        bounds = GeneticController.get_genome_bounds()
        param_names = [
            'speed_lower_multiplier', 'speed_upper_multiplier',
            'throttle_increment', 'brake_increment', 
            'steering_sensitivity', 'throttle_reduction',
            'steering_decay', 'max_steering_threshold',
            'left_sensor_weight', 'right_sensor_weight',
            'collision_threshold', 'emergency_brake_threshold'
        ]
        
        genome_info = {}
        for i, (name, value) in enumerate(zip(param_names, self.best_individual)):
            min_val, max_val = bounds[i]
            genome_info[name] = {
                'value': float(value),
                'bounds': [float(min_val), float(max_val)],
                'normalized': float((value - min_val) / (max_val - min_val))
            }
        
        genome_file = os.path.join(self.results_dir, "best_genome_analysis.json")
        with open(genome_file, 'w') as f:
            json.dump(genome_info, f, indent=2)
        
        print(f"💾 Best controller saved to {controller_file}")
        print(f"📊 Genome analysis saved to {genome_file}")


def main():
    """Main function to run genetic algorithm training."""
    # Training configuration
    trainer = GeneticTrainer(
        population_size=30,
        generations=50,
        mutation_rate=0.15,
        mutation_strength=0.2,
        crossover_rate=0.7,
        elite_ratio=0.2,
        track_file=None,  # Use random tracks
        parallel_workers=1  # Sequential for stability
    )
    
    # Run training
    best_genome, best_fitness = trainer.train()
    
    print(f"\n🏆 Evolution completed!")
    print(f"Best fitness: {best_fitness:.1f}")
    print(f"Best genome: {best_genome}")


if __name__ == "__main__":
    main()