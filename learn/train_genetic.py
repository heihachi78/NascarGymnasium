#!/usr/bin/env python3
"""
Standalone script to train genetic algorithm controllers.

This script can be run directly from the project root directory.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from game.control.genetic_controller import GeneticController
from learn.genetic_trainer import GeneticTrainer


def main():
    """Main function to run genetic algorithm training."""
    print("🧬 Genetic Algorithm Controller Training")
    print("=" * 50)
    
    # Training configuration - optimized for multi-car evaluation
    trainer = GeneticTrainer(
        population_size=10,      # Larger population since we can evaluate faster
        generations=50,          # More generations for better evolution  
        mutation_rate=0.15,
        mutation_strength=0.2,
        crossover_rate=0.7,
        elite_ratio=0.2,
        track_file=None,         # Use random tracks for diverse training
        parallel_workers=1       # Multi-car evaluation handles parallelism
    )
    
    print(f"Configuration:")
    print(f"- Population size: {trainer.population_size}")
    print(f"- Generations: {trainer.generations}")
    print(f"- Mutation rate: {trainer.mutation_rate}")
    print(f"- Crossover rate: {trainer.crossover_rate}")
    print(f"- Elite ratio: {trainer.elite_ratio}")
    print(f"- Track: {trainer.track_file or 'Random tracks'}")
    
    # Run training
    try:
        best_genome, best_fitness = trainer.train()
        
        print(f"\n🎯 Training completed successfully!")
        print(f"Best fitness achieved: {best_fitness:.3f}")
        print(f"Best genome parameters:")
        
        # Display best genome in readable format
        if best_genome:
            param_names = [
                'speed_lower_multiplier', 'speed_upper_multiplier',
                'throttle_increment', 'brake_increment',
                'steering_sensitivity', 'throttle_reduction',
                'steering_decay', 'max_steering_threshold',
                'left_sensor_weight', 'right_sensor_weight',
                'collision_threshold', 'emergency_brake_threshold'
            ]
            
            print("\nBest evolved parameters:")
            for name, value in zip(param_names, best_genome):
                print(f"  {name}: {value:.4f}")
        
        # Test the best evolved controller
        if best_genome:
            print(f"\n🏁 Testing best evolved controller...")
            best_controller = GeneticController(name="BestEvolved", genome=best_genome)
            test_controller_performance(best_controller)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        if hasattr(trainer, 'best_individual') and trainer.best_individual:
            print(f"Best fitness so far: {trainer.best_fitness:.3f}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_controller_performance(controller, num_episodes=3):
    """Test a controller's performance on the track."""
    from src.car_env import CarEnv
    
    print(f"Testing {controller.name} over {num_episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
        
        # Create environment
        env = CarEnv(
            render_mode=None,
            track_file=None,  # Random track
            reset_on_lap=False,
            discrete_action_space=False
        )
        
        try:
            # Run episode
            observation, _ = env.reset()
            total_reward = 0.0
            steps = 0
            max_steps = 9999
            
            while steps < max_steps:
                action = controller.control(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            print(f" Reward: {total_reward:.1f}, Steps: {steps}")
            
        except Exception as e:
            print(f" Error: {e}")
            total_rewards.append(-1000)
        finally:
            env.close()
    
    # Print summary
    if total_rewards:
        import numpy as np
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\n📊 Performance Summary:")
        print(f"  Average reward: {avg_reward:.1f} ± {std_reward:.1f}")
        print(f"  Best episode: {max(total_rewards):.1f}")
        print(f"  Worst episode: {min(total_rewards):.1f}")


if __name__ == "__main__":
    main()