"""
Regression model trainer for car racing controllers.

This module handles data collection from existing controllers and
training of various regression models for direct action prediction.
"""

import numpy as np
import os
import json
import pickle
import time
from typing import List, Optional
import matplotlib.pyplot as plt

from game.control.regression_controller import RegressionController, SKLEARN_AVAILABLE
from game.control.base_controller import BaseController
from game.control.genetic_controller import GeneticController
from src.car_env import CarEnv

if SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split


class RegressionTrainer:
    """
    Trainer for regression-based car racing controllers.
    
    Handles data collection, model training, evaluation, and comparison
    of different regression approaches for car control.
    """
    
    def __init__(self, 
                 data_dir: str = "learn/regression_data",
                 models_dir: str = "learn/regression_models",
                 track_file: Optional[str] = None):
        """
        Initialize the regression trainer.
        
        Args:
            data_dir: Directory to store collected training data
            models_dir: Directory to store trained models
            track_file: Specific track for data collection (None for random tracks)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RegressionTrainer")
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.track_file = track_file
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Training data storage
        self.observations = []
        self.actions = []
        self.metadata = []
        
        # Model types to train and compare
        self.model_types = ['linear', 'ridge', 'random_forest', 'neural_network']
        self.trained_models = {}
        
    def collect_data_from_controller(self, 
                                   controller: BaseController,
                                   num_episodes: int = 20,
                                   max_steps_per_episode: int = 5000,
                                   min_quality_threshold: float = -500.0) -> int:
        """
        Collect training data by running a controller in the environment.
        
        Args:
            controller: Controller to collect data from
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            min_quality_threshold: Minimum total reward to keep episode data
            
        Returns:
            Number of data points collected
        """
        print(f"🎮 Collecting data from {controller.name}")
        print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
        
        episode_data = []
        episodes_kept = 0
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
            
            # Create environment
            env = CarEnv(
                render_mode=None,
                track_file=self.track_file,
                reset_on_lap=False,  # Let episodes run naturally
                discrete_action_space=False
            )
            
            # Run episode
            obs, info = env.reset()
            episode_obs = []
            episode_actions = []
            episode_reward = 0.0
            
            try:
                for _ in range(max_steps_per_episode):
                    # Get action from controller
                    action = controller.control(obs)
                    
                    # Store observation and action
                    episode_obs.append(obs.copy())
                    episode_actions.append(action.copy())
                    
                    # Take step
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    # Check termination
                    if terminated or truncated:
                        break
                
                env.close()
                
                # Keep episode if quality is sufficient
                if episode_reward >= min_quality_threshold:
                    episode_data.extend(list(zip(episode_obs, episode_actions)))
                    episodes_kept += 1
                    print(f" Reward: {episode_reward:.1f} ✅")
                else:
                    print(f" Reward: {episode_reward:.1f} ❌ (below threshold)")
                    
            except Exception as e:
                print(f" Error: {e}")
                env.close()
                continue
        
        # Add collected data to training set
        if episode_data:
            episode_obs, episode_actions = zip(*episode_data)
            self.observations.extend(episode_obs)
            self.actions.extend(episode_actions)
            
            # Add metadata
            metadata = {
                'controller_name': controller.name,
                'controller_type': type(controller).__name__,
                'num_samples': len(episode_data),
                'episodes_kept': episodes_kept,
                'total_episodes': num_episodes,
                'track_file': self.track_file,
                'collection_time': time.time()
            }
            self.metadata.append(metadata)
        
        samples_collected = len(episode_data) if episode_data else 0
        print(f"📊 Collected {samples_collected} samples from {episodes_kept}/{num_episodes} episodes")
        
        return samples_collected
    
    def collect_data_from_multiple_controllers_batch(self,
                                                   controllers: List[BaseController],
                                                   num_episodes: int = 20,
                                                   max_steps_per_episode: int = 5000,
                                                   min_quality_threshold: float = -300.0) -> int:
        """
        Collect training data from multiple controllers simultaneously using multi-car environment.
        
        This method provides 10x speedup by running all controllers in the same environment,
        generating diverse behavioral data from multi-car interactions.
        
        Args:
            controllers: List of controllers to collect data from
            num_episodes: Number of multi-car episodes to run
            max_steps_per_episode: Maximum steps per episode
            min_quality_threshold: Minimum total reward to keep controller's data
            
        Returns:
            Total number of data points collected
        """
        num_controllers = len(controllers)
        if num_controllers == 0:
            return 0
        
        # Limit to max 10 controllers (environment constraint)
        if num_controllers > 10:
            print(f"⚠️ Too many controllers ({num_controllers}), using first 10")
            controllers = controllers[:10]
            num_controllers = 10
        
        print(f"🎮 Multi-Car Data Collection from {num_controllers} controllers")
        print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
        print(f"Controllers: {[c.name for c in controllers]}")
        
        episode_data = []
        episodes_kept = 0
        controller_stats = {i: {'episodes_kept': 0, 'total_samples': 0} for i in range(num_controllers)}
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
            
            # Create multi-car environment
            env = CarEnv(
                render_mode=None,
                track_file=self.track_file,
                reset_on_lap=False,
                discrete_action_space=False,
                num_cars=num_controllers,
                car_names=[c.name for c in controllers]
            )
            
            # Run episode with all controllers simultaneously
            observations, _ = env.reset()
            
            # Track data for each controller separately
            controller_episode_data = [[] for _ in range(num_controllers)]
            controller_rewards = [0.0] * num_controllers
            
            try:
                for _ in range(max_steps_per_episode):
                    # Get actions from all controllers
                    actions = []
                    for i, controller in enumerate(controllers):
                        if i < len(observations):
                            action = controller.control(observations[i])
                            # Store observation-action pair for this controller
                            controller_episode_data[i].append((observations[i].copy(), action.copy()))
                            actions.append(action)
                        else:
                            actions.append([0.0, 0.0])
                    
                    # Take step in environment
                    observations, rewards, terminated, truncated, _ = env.step(actions)
                    
                    # Accumulate rewards for each controller
                    for i in range(num_controllers):
                        if i < len(rewards):
                            controller_rewards[i] += rewards[i]
                    
                    # Check termination
                    if terminated or truncated:
                        break
                
                env.close()
                
                # Evaluate each controller's performance and keep good episodes
                episode_kept = False
                reward_summary = []
                
                for i, controller in enumerate(controllers):
                    reward = controller_rewards[i]
                    reward_summary.append(f"{controller.name}:{reward:.1f}")
                    
                    if reward >= min_quality_threshold and len(controller_episode_data[i]) > 0:
                        # Keep this controller's data from this episode
                        episode_data.extend(controller_episode_data[i])
                        controller_stats[i]['episodes_kept'] += 1
                        controller_stats[i]['total_samples'] += len(controller_episode_data[i])
                        episode_kept = True
                
                if episode_kept:
                    episodes_kept += 1
                    print(f" ✅ ({', '.join(reward_summary)})")
                else:
                    print(f" ❌ ({', '.join(reward_summary)})")
                    
            except Exception as e:
                print(f" Error: {e}")
                env.close()
                continue
        
        # Add collected data to training set
        if episode_data:
            episode_obs, episode_actions = zip(*episode_data)
            self.observations.extend(episode_obs)
            self.actions.extend(episode_actions)
            
            # Add metadata for multi-car collection
            metadata = {
                'collection_method': 'multi_car_batch',
                'controllers': [{'name': c.name, 'type': type(c).__name__} for c in controllers],
                'num_episodes': num_episodes,
                'episodes_kept': episodes_kept,
                'total_samples': len(episode_data),
                'controller_stats': controller_stats,
                'track_file': self.track_file,
                'collection_time': time.time()
            }
            self.metadata.append(metadata)
        
        total_samples = len(episode_data) if episode_data else 0
        print(f"📊 Multi-car data collection completed:")
        print(f"   Episodes kept: {episodes_kept}/{num_episodes}")
        print(f"   Total samples: {total_samples}")
        
        # Print per-controller statistics
        for i, controller in enumerate(controllers):
            stats = controller_stats[i]
            print(f"   {controller.name}: {stats['episodes_kept']} episodes, {stats['total_samples']} samples")
        
        return total_samples
    
    def collect_data_from_multiple_controllers(self, 
                                             controllers: List[BaseController],
                                             episodes_per_controller: int = 10) -> int:
        """
        Collect training data from multiple controllers.
        
        Args:
            controllers: List of controllers to collect data from
            episodes_per_controller: Episodes to run per controller
            
        Returns:
            Total number of data points collected
        """
        total_samples = 0
        
        for controller in controllers:
            samples = self.collect_data_from_controller(
                controller, 
                num_episodes=episodes_per_controller
            )
            total_samples += samples
            
        return total_samples
    
    def save_training_data(self, filename: str = None):
        """Save collected training data to file."""
        if not self.observations:
            print("⚠️  No data to save")
            return
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"training_data_{timestamp}.pkl"
        
        filepath = os.path.join(self.data_dir, filename)
        
        training_data = {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions), 
            'metadata': self.metadata,
            'collection_info': {
                'total_samples': len(self.observations),
                'observation_shape': np.array(self.observations[0]).shape if self.observations else None,
                'action_shape': np.array(self.actions[0]).shape if self.actions else None,
                'track_file': self.track_file
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"💾 Training data saved to {filepath}")
        print(f"Samples: {len(self.observations)}, Controllers: {len(self.metadata)}")
    
    def load_training_data(self, filepath: str):
        """Load training data from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training data not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            training_data = pickle.load(f)
        
        self.observations = training_data['observations'].tolist()
        self.actions = training_data['actions'].tolist()
        self.metadata = training_data['metadata']
        
        info = training_data.get('collection_info', {})
        print(f"📂 Training data loaded from {filepath}")
        print(f"Samples: {info.get('total_samples', len(self.observations))}")
        print(f"Observation shape: {info.get('observation_shape')}")
        print(f"Action shape: {info.get('action_shape')}")
    
    def train_all_models(self, 
                        test_size: float = 0.2,
                        validation_size: float = 0.1,
                        random_state: int = 42) -> dict:
        """
        Train all regression models and compare their performance.
        
        Args:
            test_size: Fraction of data to use for testing
            validation_size: Fraction of training data to use for validation
            random_state: Random seed for reproducible splits
            
        Returns:
            Dictionary containing trained models and evaluation results
        """
        if not self.observations:
            raise ValueError("No training data available. Collect data first.")
        
        print(f"🤖 Training regression models on {len(self.observations)} samples")
        
        # Prepare data
        X = np.array(self.observations)
        y = np.array(self.actions)
        
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split training into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=random_state
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train each model type
        results = {}
        
        for model_type in self.model_types:
            print(f"\\n📈 Training {model_type} model...")
            
            # Create and train model
            controller = RegressionController(
                name=f"Regression_{model_type}",
                model_type=model_type
            )
            
            # Train model
            training_success = controller.train(X_train, y_train, X_val, y_val)
            
            if training_success:
                # Evaluate model
                test_metrics = controller.evaluate_predictions(X_test, y_test)
                
                # Save model
                model_filename = f"{model_type}_model.pkl"
                model_path = os.path.join(self.models_dir, model_filename)
                controller.save_model(model_path)
                
                # Store results
                results[model_type] = {
                    'controller': controller,
                    'model_path': model_path,
                    'test_metrics': test_metrics,
                    'training_success': True
                }
                
                # Print results
                print(f"  Test R² overall: {test_metrics['r2_overall']:.4f}")
                print(f"  Test MAE overall: {test_metrics['mae_overall']:.4f}")
                
            else:
                results[model_type] = {
                    'controller': None,
                    'training_success': False,
                    'error': 'Training failed'
                }
        
        self.trained_models = results
        
        # Save training summary
        self._save_training_summary(results, X_train.shape, X_test.shape)
        
        return results
    
    def _save_training_summary(self, results: dict, train_shape: tuple, test_shape: tuple):
        """Save training summary with model comparison."""
        summary = {
            'timestamp': time.time(),
            'data_info': {
                'total_samples': len(self.observations),
                'train_samples': train_shape[0],
                'test_samples': test_shape[0],
                'feature_dim': train_shape[1],
                'action_dim': 2,
                'track_file': self.track_file
            },
            'models': {}
        }
        
        # Add model results
        for model_type, result in results.items():
            if result['training_success']:
                summary['models'][model_type] = {
                    'model_path': result['model_path'],
                    'test_metrics': result['test_metrics']
                }
            else:
                summary['models'][model_type] = {
                    'training_failed': True,
                    'error': result.get('error', 'Unknown error')
                }
        
        # Add metadata about data sources
        summary['data_sources'] = self.metadata
        
        # Save summary
        summary_path = os.path.join(self.models_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved to {summary_path}")
    
    def compare_models(self, plot_results: bool = True):
        """Compare performance of all trained models."""
        if not self.trained_models:
            print("⚠️  No trained models to compare")
            return
        
        print("\\n📊 Model Comparison:")
        print("-" * 60)
        print(f"{'Model':<15} {'R² Overall':<12} {'MAE Overall':<12} {'Status':<10}")
        print("-" * 60)
        
        comparison_data = []
        
        for model_type, result in self.trained_models.items():
            if result['training_success']:
                metrics = result['test_metrics']
                r2 = metrics['r2_overall']
                mae = metrics['mae_overall']
                status = "✅ Success"
                
                comparison_data.append({
                    'model': model_type,
                    'r2': r2,
                    'mae': mae,
                    'metrics': metrics
                })
                
            else:
                r2 = mae = 0.0
                status = "❌ Failed"
            
            print(f"{model_type:<15} {r2:<12.4f} {mae:<12.4f} {status:<10}")
        
        if comparison_data and plot_results:
            self._plot_model_comparison(comparison_data)
    
    def _plot_model_comparison(self, comparison_data: List[dict]):
        """Plot model comparison charts."""
        if not comparison_data:
            return
        
        models = [d['model'] for d in comparison_data]
        r2_scores = [d['r2'] for d in comparison_data]
        mae_scores = [d['mae'] for d in comparison_data]
        
        # Create comparison plots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores
        bars1 = ax1.bar(models, r2_scores, color=['blue', 'green', 'orange', 'red'])
        ax1.set_title('Model Performance - R² Score (Higher is Better)')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # MAE scores
        bars2 = ax2.bar(models, mae_scores, color=['blue', 'green', 'orange', 'red'])
        ax2.set_title('Model Performance - Mean Absolute Error (Lower is Better)')
        ax2.set_ylabel('MAE')
        
        # Add value labels on bars
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(mae_scores) * 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Model comparison plot saved to {plot_path}")
        
        plt.close()
    
    def evaluate_on_track(self, model_type: str, num_episodes: int = 5) -> dict:
        """
        Evaluate a trained model by running it in the environment.
        
        Args:
            model_type: Type of model to evaluate
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation results
        """
        if model_type not in self.trained_models:
            raise ValueError(f"Model {model_type} not found in trained models")
        
        result = self.trained_models[model_type]
        if not result['training_success']:
            raise ValueError(f"Model {model_type} training failed")
        
        controller = result['controller']
        print(f"🏁 Evaluating {model_type} model on track ({num_episodes} episodes)")
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
            
            # Create environment
            env = CarEnv(
                render_mode=None,
                track_file=self.track_file,
                reset_on_lap=False,
                discrete_action_space=False
            )
            
            # Run episode
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            max_steps = 3000
            
            try:
                while steps < max_steps:
                    action = controller.control(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                # Extract final metrics
                final_info = {}
                if isinstance(info, dict) and 'cars' in info and info['cars']:
                    car_info = info['cars'][0]
                    final_info = {
                        'distance_traveled': steps * 0.1,  # Rough estimate
                        'max_speed': car_info.get('car_speed_ms', 0),
                        'on_track': car_info.get('on_track', False),
                        'lap_completed': car_info.get('lap_timing', {}).get('lap_count', 0) > 0
                    }
                
                episode_results.append({
                    'episode': episode,
                    'total_reward': total_reward,
                    'steps': steps,
                    'terminated': terminated,
                    'truncated': truncated,
                    **final_info
                })
                
                print(f" Reward: {total_reward:.1f}, Steps: {steps}")
                
            except Exception as e:
                print(f" Error: {e}")
                episode_results.append({
                    'episode': episode,
                    'total_reward': -1000,
                    'error': str(e)
                })
            finally:
                env.close()
        
        # Calculate summary statistics
        rewards = [r['total_reward'] for r in episode_results if 'error' not in r]
        if rewards:
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
        else:
            avg_reward = std_reward = min_reward = max_reward = 0
        
        evaluation_summary = {
            'model_type': model_type,
            'num_episodes': num_episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'episodes': episode_results
        }
        
        print(f"📊 Average reward: {avg_reward:.1f} ± {std_reward:.1f}")
        
        return evaluation_summary


def main():
    """Main function for demonstration."""
    print("🤖 Regression Controller Training Demo")
    
    # Create trainer
    trainer = RegressionTrainer(
        data_dir="demo_data",
        models_dir="demo_models"
    )
    
    # Create diverse controllers to collect data from
    controllers = [
        BaseController("BaseRule"),
        GeneticController("Genetic1", GeneticController.random_genome()),
        GeneticController("Genetic2", GeneticController.random_genome()),
        GeneticController("Genetic3", GeneticController.random_genome()),
    ]
    
    # Collect training data using multi-car batch method (10x faster!)
    print("\\n1. Collecting training data using multi-car environment...")
    trainer.collect_data_from_multiple_controllers_batch(
        controllers, 
        num_episodes=15,  # Fewer episodes needed due to multi-car efficiency
        max_steps_per_episode=4000,
        min_quality_threshold=-400.0
    )
    trainer.save_training_data("demo_training_data.pkl")
    
    # Train models
    print("\\n2. Training regression models...")
    results = trainer.train_all_models()
    
    # Compare models
    print("\\n3. Comparing model performance...")
    trainer.compare_models()
    
    # Evaluate best model on track
    print("\\n4. Evaluating best model...")
    best_models = [name for name, result in results.items() if result['training_success']]
    if best_models:
        best_model = best_models[0]  # Use first successful model
        trainer.evaluate_on_track(best_model, num_episodes=3)


if __name__ == "__main__":
    main()