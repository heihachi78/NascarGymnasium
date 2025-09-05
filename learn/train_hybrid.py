#!/usr/bin/env python3
"""
Standalone script to run hybrid training (Genetic Algorithm + Regression Models).

This script can be run directly from the project root directory.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from learn.hybrid_trainer import HybridTrainer
from game.control.regression_controller import SKLEARN_AVAILABLE


def main():
    """Main function for hybrid training pipeline."""
    print("🚀 Hybrid Training Pipeline (GA + Regression)")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available. Only genetic algorithm training will work.")
        print("For full hybrid training, install scikit-learn with: pip install scikit-learn")
    
    # Create hybrid trainer with demo-friendly configuration
    trainer = HybridTrainer(
        results_dir="hybrid_training_results",
        track_file=None,  # Use random tracks for diverse training
        ga_config={
            'population_size': 15,   # Smaller for faster demo
            'generations': 20,       # Reasonable for demo
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'elite_ratio': 0.2
        },
        regression_config={
            'model_types': ['random_forest', 'neural_network'],  # Most effective models
            'test_size': 0.2,
            'validation_size': 0.1
        }
    )
    
    print("Configuration:")
    print(f"- Results directory: {trainer.results_dir}")
    print(f"- Track: {trainer.track_file or 'Random tracks'}")
    print(f"- GA Population: {trainer.ga_config['population_size']}")
    print(f"- GA Generations: {trainer.ga_config['generations']}")
    print(f"- Regression Models: {trainer.regression_config['model_types']}")
    print(f"- Scikit-learn Available: {SKLEARN_AVAILABLE}")
    
    # Run the complete hybrid training pipeline
    try:
        print(f"\n🚀 Starting hybrid training pipeline...")
        results = trainer.run_complete_training(evaluate=True)
        
        print(f"\n🎯 HYBRID TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display results summary
        print("Results Summary:")
        print(f"- Best GA Fitness: {results.get('best_ga_fitness', 'N/A'):.1f}")
        print(f"- Evolved Controllers: {len(results.get('evolved_controllers', []))}")
        print(f"- Behavioral Samples Collected: {results.get('behavioral_samples', 0)}")
        
        if SKLEARN_AVAILABLE:
            hybrid_models = results.get('hybrid_models', {})
            successful_models = [k for k, v in hybrid_models.items() if v]
            print(f"- Successful Hybrid Models: {len(successful_models)} ({successful_models})")
        
        print(f"- Total Training Time: {results.get('training_time', 0):.1f}s")
        
        # Show evaluation results if available
        if 'evaluation' in results:
            evaluation = results['evaluation']
            print(f"\n📊 Performance Evaluation:")
            
            for approach, metrics in evaluation.items():
                if 'error' not in metrics:
                    avg_reward = metrics.get('avg_reward', 0)
                    success_rate = metrics.get('success_rate', 0)
                    print(f"  {approach:20}: Reward {avg_reward:6.1f}, Success {success_rate:.2f}")
        
        # Show where to find detailed results
        print(f"\n📂 Detailed results saved to: {trainer.results_dir}/")
        print(f"- Genetic Algorithm results: {trainer.ga_dir}/")
        if SKLEARN_AVAILABLE:
            print(f"- Regression models: {trainer.regression_dir}/")
            print(f"- Hybrid models: {trainer.hybrid_dir}/")
        print(f"- Complete results: {trainer.results_dir}/complete_results.json")
        print(f"- Evaluation summary: {trainer.results_dir}/comprehensive_evaluation.json")
        
        # Provide usage instructions
        print(f"\n🎮 Using the trained models:")
        print(f"The best models can be loaded and used in your game modes.")
        print(f"Check the generated model files in the results directories.")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_demo():
    """Run a very quick demo with minimal configuration."""
    print("⚡ Quick Demo Mode")
    print("=" * 30)
    
    trainer = HybridTrainer(
        results_dir="quick_demo_results",
        ga_config={
            'population_size': 8,    # Very small for quick demo
            'generations': 8,        # Very few generations
            'mutation_rate': 0.2,
            'crossover_rate': 0.7
        },
        regression_config={
            'model_types': ['random_forest'] if SKLEARN_AVAILABLE else [],
            'test_size': 0.3
        }
    )
    
    print("Quick demo configuration (faster but less thorough):")
    print(f"- Population: {trainer.ga_config['population_size']}")  
    print(f"- Generations: {trainer.ga_config['generations']}")
    print(f"- Models: {trainer.regression_config['model_types']}")
    
    try:
        results = trainer.run_complete_training(evaluate=False)  # Skip evaluation for speed
        
        print(f"\n⚡ Quick demo completed!")
        print(f"Best GA fitness: {results.get('best_ga_fitness', 'N/A')}")
        print(f"Time taken: {results.get('training_time', 0):.1f}s")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    # Check if user wants quick demo
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        main()