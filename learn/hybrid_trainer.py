"""
Hybrid trainer combining Genetic Algorithms and Regression Models.

This module implements a hybrid approach that uses genetic algorithms to evolve
optimal rule-based controllers, then trains regression models on the best 
evolved behaviors to create smooth, continuous controllers.
"""

import numpy as np
import os
import json
import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from game.control.genetic_controller import GeneticController
from .genetic_trainer import GeneticTrainer
from game.control.regression_controller import SKLEARN_AVAILABLE
from .regression_trainer import RegressionTrainer
from game.control.base_controller import BaseController


class HybridTrainer:
    """
    Hybrid trainer that combines genetic algorithms with regression learning.
    
    The process:
    1. Evolve optimal rule-based controllers using genetic algorithms
    2. Collect behavioral data from the best evolved controllers
    3. Train regression models on this optimized behavioral data
    4. Compare pure GA, pure regression, and hybrid approaches
    """
    
    def __init__(self,
                 results_dir: str = "hybrid_results",
                 track_file: Optional[str] = None,
                 ga_config: Optional[Dict] = None,
                 regression_config: Optional[Dict] = None):
        """
        Initialize the hybrid trainer.
        
        Args:
            results_dir: Directory to store all results
            track_file: Specific track for training (None for random tracks)
            ga_config: Configuration for genetic algorithm training
            regression_config: Configuration for regression training
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn not available. Regression training will be disabled.")
        
        self.results_dir = results_dir
        self.track_file = track_file
        
        # Create results directory structure
        os.makedirs(results_dir, exist_ok=True)
        self.ga_dir = os.path.join(results_dir, "genetic_algorithm")
        self.regression_dir = os.path.join(results_dir, "regression_models")
        self.hybrid_dir = os.path.join(results_dir, "hybrid_models")
        
        for dir_path in [self.ga_dir, self.regression_dir, self.hybrid_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configuration
        self.ga_config = ga_config or {
            'population_size': 30,
            'generations': 40,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'elite_ratio': 0.2
        }
        
        self.regression_config = regression_config or {
            'model_types': ['random_forest', 'neural_network'],
            'test_size': 0.2,
            'validation_size': 0.1
        }
        
        # Training components
        self.ga_trainer = None
        self.regression_trainer = None
        self.evolved_controllers = []
        self.hybrid_models = {}
        
    def run_genetic_evolution(self) -> List[GeneticController]:
        """
        Run genetic algorithm evolution to find optimal controllers.
        
        Returns:
            List of best evolved controllers
        """
        print("🧬 Phase 1: Genetic Algorithm Evolution")
        print("=" * 50)
        
        # Create GA trainer
        self.ga_trainer = GeneticTrainer(
            population_size=self.ga_config['population_size'],
            generations=self.ga_config['generations'],
            mutation_rate=self.ga_config['mutation_rate'],
            crossover_rate=self.ga_config['crossover_rate'],
            elite_ratio=self.ga_config['elite_ratio'],
            track_file=self.track_file,
            parallel_workers=1
        )
        
        # Change results directory to our GA subdirectory
        self.ga_trainer.results_dir = self.ga_dir
        
        # Run evolution
        best_genome, best_fitness = self.ga_trainer.train()
        
        # Create controllers from best individuals
        best_controllers = []
        
        if best_genome and self.ga_trainer.fitness_history:
            # Create controller from absolute best individual
            best_controller = GeneticController(
                name="BestEvolved",
                genome=best_genome
            )
            best_controllers.append(best_controller)
            
            # Create controllers from top performers in final generation
            if len(self.ga_trainer.population) >= 3:
                # Evaluate final population to get top performers
                final_results = self.ga_trainer.evaluate_population()
                fitness_scores = [r['fitness'] for r in final_results]
                top_indices = np.argsort(fitness_scores)[-3:]  # Top 3
                
                for i, idx in enumerate(top_indices):
                    if idx < len(self.ga_trainer.population):
                        genome = self.ga_trainer.population[idx]
                        controller = GeneticController(
                            name=f"Evolved_Top{i+1}",
                            genome=genome
                        )
                        if controller not in best_controllers:  # Avoid duplicates
                            best_controllers.append(controller)
        
        self.evolved_controllers = best_controllers
        
        print(f"✅ Evolution completed! Created {len(best_controllers)} elite controllers")
        print(f"Best fitness achieved: {best_fitness:.1f}")
        
        return best_controllers
    
    def collect_behavioral_data(self, controllers: List[BaseController] = None) -> int:
        """
        Collect behavioral data from evolved controllers.
        
        Args:
            controllers: Controllers to collect data from (uses evolved controllers if None)
            
        Returns:
            Total number of data samples collected
        """
        print("\\n📊 Phase 2: Behavioral Data Collection")
        print("=" * 50)
        
        if controllers is None:
            controllers = self.evolved_controllers
        
        if not controllers:
            raise ValueError("No controllers available for data collection")
        
        # Create regression trainer for data collection
        self.regression_trainer = RegressionTrainer(
            data_dir=os.path.join(self.regression_dir, "training_data"),
            models_dir=self.regression_dir,
            track_file=self.track_file
        )
        
        # Collect data from each evolved controller
        total_samples = 0
        episodes_per_controller = max(15, 50 // len(controllers))  # Scale episodes with controller count
        
        for controller in controllers:
            print(f"🎮 Collecting data from {controller.name}...")
            samples = self.regression_trainer.collect_data_from_controller(
                controller=controller,
                num_episodes=episodes_per_controller,
                max_steps_per_episode=4000,
                min_quality_threshold=-300.0  # More lenient for evolved controllers
            )
            total_samples += samples
        
        # Save the collected data
        self.regression_trainer.save_training_data("hybrid_behavioral_data.pkl")
        
        print(f"✅ Data collection completed! Total samples: {total_samples}")
        return total_samples
    
    def train_hybrid_models(self) -> Dict:
        """
        Train regression models on behavioral data from evolved controllers.
        
        Returns:
            Dictionary containing trained hybrid models
        """
        print("\\n🤖 Phase 3: Hybrid Model Training")
        print("=" * 50)
        
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available. Skipping regression training.")
            return {}
        
        if not self.regression_trainer:
            raise ValueError("Behavioral data collection must be completed first")
        
        if not self.regression_trainer.observations:
            raise ValueError("No behavioral data available for training")
        
        # Train models with hybrid-specific configuration
        model_types = self.regression_config['model_types']
        results = {}
        
        # Override model types in trainer
        original_model_types = self.regression_trainer.model_types
        self.regression_trainer.model_types = model_types
        
        try:
            # Train all specified models
            training_results = self.regression_trainer.train_all_models(
                test_size=self.regression_config['test_size'],
                validation_size=self.regression_config['validation_size']
            )
            
            # Store hybrid models with enhanced naming
            for model_type, result in training_results.items():
                if result['training_success']:
                    hybrid_name = f"Hybrid_{model_type}"
                    controller = result['controller']
                    controller.name = hybrid_name
                    
                    # Save to hybrid directory
                    hybrid_model_path = os.path.join(self.hybrid_dir, f"{hybrid_name}.pkl")
                    controller.save_model(hybrid_model_path)
                    
                    results[model_type] = {
                        'controller': controller,
                        'model_path': hybrid_model_path,
                        'test_metrics': result['test_metrics'],
                        'training_success': True
                    }
                else:
                    results[model_type] = result
            
            self.hybrid_models = results
            
            # Compare regression models
            print("\\n📈 Hybrid Model Performance:")
            self.regression_trainer.compare_models(plot_results=True)
            
        finally:
            # Restore original model types
            self.regression_trainer.model_types = original_model_types
        
        return results
    
    def comprehensive_evaluation(self, num_episodes: int = 5) -> Dict:
        """
        Comprehensive evaluation comparing all approaches.
        
        Args:
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Dictionary containing evaluation results for all approaches
        """
        print("\\n🏁 Phase 4: Comprehensive Evaluation")
        print("=" * 50)
        
        evaluation_results = {}
        
        # 1. Evaluate baseline controller
        print("1. Evaluating baseline rule-based controller...")
        baseline_controller = BaseController("Baseline")
        baseline_results = self._evaluate_controller(baseline_controller, num_episodes)
        evaluation_results['baseline'] = baseline_results
        
        # 2. Evaluate best evolved (GA) controller
        if self.evolved_controllers:
            print("2. Evaluating best evolved (GA) controller...")
            best_ga = self.evolved_controllers[0]  # First is typically the best
            ga_results = self._evaluate_controller(best_ga, num_episodes)
            evaluation_results['genetic_algorithm'] = ga_results
        
        # 3. Evaluate hybrid regression models
        if self.hybrid_models and SKLEARN_AVAILABLE:
            for model_type, model_info in self.hybrid_models.items():
                if model_info['training_success']:
                    print(f"3. Evaluating hybrid {model_type} model...")
                    controller = model_info['controller']
                    hybrid_results = self._evaluate_controller(controller, num_episodes)
                    evaluation_results[f'hybrid_{model_type}'] = hybrid_results
        
        # 4. Create comparison summary
        self._create_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_controller(self, controller: BaseController, num_episodes: int) -> Dict:
        """Evaluate a single controller on the track."""
        from src.car_env import CarEnv
        
        episode_results = []
        
        for _ in range(num_episodes):
            # Create environment
            env = CarEnv(
                render_mode=None,
                track_file=self.track_file,
                reset_on_lap=False,
                discrete_action_space=False
            )
            
            # Run episode
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            max_steps = 4000
            
            try:
                while steps < max_steps:
                    action = controller.control(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                # Extract performance metrics
                performance = {
                    'total_reward': total_reward,
                    'steps': steps,
                    'completed': terminated or truncated
                }
                
                # Add car-specific metrics if available
                if isinstance(info, dict) and 'cars' in info and info['cars']:
                    car_info = info['cars'][0]
                    performance.update({
                        'max_speed': car_info.get('car_speed_ms', 0),
                        'on_track': car_info.get('on_track', False),
                        'lap_completed': car_info.get('lap_timing', {}).get('lap_count', 0) > 0
                    })
                
                episode_results.append(performance)
                
            except Exception as e:
                episode_results.append({
                    'total_reward': -1000,
                    'error': str(e)
                })
            finally:
                env.close()
        
        # Calculate summary statistics
        valid_results = [r for r in episode_results if 'error' not in r]
        
        if valid_results:
            rewards = [r['total_reward'] for r in valid_results]
            steps = [r['steps'] for r in valid_results]
            
            summary = {
                'controller_name': controller.name,
                'num_episodes': len(valid_results),
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'avg_steps': np.mean(steps),
                'success_rate': len(valid_results) / num_episodes,
                'episodes': episode_results
            }
            
            # Add lap completion rate if available
            lap_completions = [r.get('lap_completed', False) for r in valid_results]
            if any(lap_completions):
                summary['lap_completion_rate'] = np.mean(lap_completions)
            
        else:
            summary = {
                'controller_name': controller.name,
                'num_episodes': 0,
                'avg_reward': -1000,
                'error': 'All episodes failed'
            }
        
        return summary
    
    def _create_evaluation_summary(self, evaluation_results: Dict):
        """Create and save comprehensive evaluation summary."""
        print("\\n📋 Creating evaluation summary...")
        
        # Console summary
        print("\\n" + "="*70)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*70)
        print(f"{'Approach':<20} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Steps':<10}")
        print("-"*70)
        
        summary_data = []
        
        for approach_name, results in evaluation_results.items():
            if 'error' not in results:
                avg_reward = results['avg_reward']
                success_rate = results.get('success_rate', 0.0)
                avg_steps = results.get('avg_steps', 0)
                
                summary_data.append({
                    'approach': approach_name,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'avg_steps': avg_steps,
                    'full_results': results
                })
                
                print(f"{approach_name:<20} {avg_reward:<12.1f} {success_rate:<12.2f} {avg_steps:<10.0f}")
            else:
                print(f"{approach_name:<20} {'ERROR':<12} {'-':<12} {'-':<10}")
        
        print("="*70)
        
        # Save detailed results
        summary_file = os.path.join(self.results_dir, "comprehensive_evaluation.json")
        with open(summary_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create comparison plots
        if len(summary_data) >= 2:
            self._plot_evaluation_comparison(summary_data)
        
        print(f"💾 Evaluation results saved to {summary_file}")
    
    def _plot_evaluation_comparison(self, summary_data: List[Dict]):
        """Create comparison plots for different approaches."""
        approaches = [d['approach'] for d in summary_data]
        rewards = [d['avg_reward'] for d in summary_data]
        success_rates = [d['success_rate'] for d in summary_data]
        
        # Create comparison plots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reward comparison
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(approaches)]
        bars1 = ax1.bar(approaches, rewards, color=colors)
        ax1.set_title('Average Reward by Approach')
        ax1.set_ylabel('Average Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, reward in zip(bars1, rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max(rewards) * 0.01),
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # Success rate comparison
        bars2 = ax2.bar(approaches, success_rates, color=colors)
        ax2.set_title('Success Rate by Approach')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars2, success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, "approach_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to {plot_path}")
        plt.close()
    
    def run_complete_training(self, evaluate: bool = True) -> Dict:
        """
        Run the complete hybrid training pipeline.
        
        Args:
            evaluate: Whether to run comprehensive evaluation
            
        Returns:
            Dictionary containing all training results
        """
        print("🚀 HYBRID TRAINING PIPELINE")
        print("=" * 50)
        print(f"Results directory: {self.results_dir}")
        print(f"Track: {self.track_file or 'Random tracks'}")
        print(f"GA Config: {self.ga_config}")
        print(f"Regression Config: {self.regression_config}")
        
        start_time = time.time()
        results = {}
        
        try:
            # Phase 1: Genetic Algorithm Evolution
            evolved_controllers = self.run_genetic_evolution()
            results['evolved_controllers'] = [c.name for c in evolved_controllers]
            results['best_ga_fitness'] = self.ga_trainer.best_fitness if self.ga_trainer else 0
            
            # Phase 2: Behavioral Data Collection
            total_samples = self.collect_behavioral_data(evolved_controllers)
            results['behavioral_samples'] = total_samples
            
            # Phase 3: Hybrid Model Training
            if SKLEARN_AVAILABLE and total_samples > 0:
                hybrid_models = self.train_hybrid_models()
                results['hybrid_models'] = {k: v['training_success'] for k, v in hybrid_models.items()}
            else:
                results['hybrid_models'] = {}
                print("⚠️  Skipping hybrid model training (no sklearn or insufficient data)")
            
            # Phase 4: Comprehensive Evaluation
            if evaluate:
                evaluation_results = self.comprehensive_evaluation()
                results['evaluation'] = evaluation_results
            
            # Save complete results
            results['training_time'] = time.time() - start_time
            results['timestamp'] = time.time()
            
            results_file = os.path.join(self.results_dir, "complete_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"Total time: {results['training_time']:.1f}s")
            print(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            print(f"\\n❌ Training failed: {e}")
            results['error'] = str(e)
            results['training_time'] = time.time() - start_time
            
            # Save partial results
            error_file = os.path.join(self.results_dir, "error_results.json")
            with open(error_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            raise


def main():
    """Main function for demonstration."""
    print("🚀 Hybrid Training Demo")
    
    # Create hybrid trainer with demo configuration
    trainer = HybridTrainer(
        results_dir="demo_hybrid_results",
        ga_config={
            'population_size': 20,  # Smaller for demo
            'generations': 15,      # Fewer generations for demo
            'mutation_rate': 0.15,
            'crossover_rate': 0.7
        },
        regression_config={
            'model_types': ['random_forest'],  # Just one model for demo
            'test_size': 0.2
        }
    )
    
    # Run complete training pipeline
    results = trainer.run_complete_training(evaluate=True)
    
    print("\\n📋 Demo Results Summary:")
    print(f"- GA Best Fitness: {results.get('best_ga_fitness', 'N/A')}")
    print(f"- Behavioral Samples: {results.get('behavioral_samples', 0)}")
    print(f"- Hybrid Models Trained: {len([k for k, v in results.get('hybrid_models', {}).items() if v])}")
    print(f"- Training Time: {results.get('training_time', 0):.1f}s")


if __name__ == "__main__":
    main()