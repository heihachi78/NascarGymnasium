#!/usr/bin/env python3
"""
Standalone script to train regression model controllers.

This script can be run directly from the project root directory.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from game.control.regression_controller import SKLEARN_AVAILABLE
from learn.regression_trainer import RegressionTrainer
from game.control.base_controller import BaseController
from game.control.genetic_controller import GeneticController
from game.control.td3_control_class import TD3Controller
from game.control.ppo_control_class import PPOController
from game.control.sac_control_class import SACController
from game.control.a2c_control_class import A2CController
import pickle


def load_genetic_controller(pkl_path, name):
    """Load a trained genetic controller from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            controller = pickle.load(f)
        # Update name for competition display
        controller.name = name
        return controller, True
    except Exception as e:
        print(f"⚠️ Failed to load genetic controller {pkl_path}: {e}")
        # Return fallback controller
        return GeneticController(name=name), False


def create_competition_controllers():
    """
    Create the same high-quality controllers used in competition.py.
    Returns list of successfully loaded controllers.
    """
    # Model configurations from competition.py
    model_configs = [
        ("game/control/models/a2c_best_model_opt_1.zip", "A2C-O-1"),
        ("game/control/models/a2c_best_model3.zip", "A2C-B-3"),
        ("game/control/models/ppo_284.zip", "PPO-284"),
        ("game/control/models/ppo_best_model.zip", "PPO-B"),
        ("game/control/models/td3_best_model1.zip", "TD3-B-1"),
        ("game/control/models/td3_best_model2.zip", "TD3-B-2"),
        ("genetic_results/best_evolved_controller.pkl", "GA-Best"),
        (None, "BC"),  # BaseController
    ]
    
    controllers = []
    successful_loads = 0
    
    print("🏎️ Loading competition-quality controllers...")
    
    for i, (model_path, name) in enumerate(model_configs):
        print(f"Loading {name}...", end=' ')
        
        try:
            if model_path is None:
                # BaseController
                controller = BaseController(name=name)
                controllers.append(controller)
                print("✓ Rule-based control")
                successful_loads += 1
                
            elif "ppo" in model_path.lower():
                controller = PPOController(model_path, name)
                controllers.append(controller)
                info = controller.get_info()
                if info['model_loaded']:
                    print("✓ PPO model loaded")
                    successful_loads += 1
                else:
                    print("⚠ PPO fallback control")
                    
            elif "sac" in model_path.lower():
                controller = SACController(model_path, name)
                controllers.append(controller)
                info = controller.get_info()
                if info['model_loaded']:
                    print("✓ SAC model loaded")
                    successful_loads += 1
                else:
                    print("⚠ SAC fallback control")
                    
            elif "a2c" in model_path.lower():
                controller = A2CController(model_path, name)
                controllers.append(controller)
                info = controller.get_info()
                if info['model_loaded']:
                    print("✓ A2C model loaded")
                    successful_loads += 1
                else:
                    print("⚠ A2C fallback control")
                    
            elif "td3" in model_path.lower():
                controller = TD3Controller(model_path, name)
                controllers.append(controller)
                info = controller.get_info()
                if info['model_loaded']:
                    print("✓ TD3 model loaded")
                    successful_loads += 1
                else:
                    print("⚠ TD3 fallback control")
                    
            elif "genetic" in model_path.lower() or model_path.endswith(".pkl"):
                controller, loaded = load_genetic_controller(model_path, name)
                controllers.append(controller)
                if loaded:
                    print("✓ Genetic controller loaded")
                    successful_loads += 1
                else:
                    print("⚠ Genetic fallback control")
            else:
                # Fallback to BaseController
                controller = BaseController(name=name)
                controllers.append(controller)
                print("✓ Fallback rule-based control")
                successful_loads += 1
                
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            # Still add a fallback controller
            controllers.append(BaseController(name=f"Fallback_{name}"))
    
    print(f"📊 Successfully loaded {successful_loads}/{len(model_configs)} controllers")
    return controllers


def main():
    """Main function for regression model training."""
    print("🤖 Regression Model Controller Training")
    print("=" * 50)
    
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn is not available. Please install it with:")
        print("pip install scikit-learn")
        return
    
    # Create trainer
    trainer = RegressionTrainer(
        data_dir="regression_data",
        models_dir="regression_models",
        track_file=None  # Use random tracks
    )
    
    print("Configuration:")
    print(f"- Data directory: {trainer.data_dir}")
    print(f"- Models directory: {trainer.models_dir}")
    print(f"- Track: {trainer.track_file or 'Random tracks'}")
    print(f"- Model types: {trainer.model_types}")
    
    # Step 1: Collect training data from different controllers
    print(f"\n📊 Step 1: Collecting training data...")
    
    # Load competition-quality controllers instead of random genetic ones
    controllers_to_use = create_competition_controllers()
    
    print(f"\nMulti-car data collection from {len(controllers_to_use)} high-quality controllers...")
    print(f"Controllers: {[c.name for c in controllers_to_use]}")
    
    # Collect data using efficient multi-car batch method
    total_samples = trainer.collect_data_from_multiple_controllers_batch(
        controllers_to_use,
        num_episodes=50,  # Fewer episodes needed due to multi-car efficiency  
        max_steps_per_episode=9000,
        min_quality_threshold=-350.0
    )
    
    if total_samples == 0:
        print("❌ No training data collected. Exiting.")
        return
    
    # Save training data
    trainer.save_training_data("demo_training_data.pkl")
    
    # Step 2: Train regression models
    print(f"\n🤖 Step 2: Training regression models...")
    
    try:
        results = trainer.train_all_models(
            test_size=0.2,
            validation_size=0.1,
            random_state=42
        )
        
        print(f"\n✅ Model training completed!")
        
        # Step 3: Compare model performance
        print(f"\n📈 Step 3: Model performance comparison...")
        trainer.compare_models(plot_results=True)
        
        # Step 4: Test best model on track
        print(f"\n🏁 Step 4: Testing best model on track...")
        
        # Find best performing model
        successful_models = {name: result for name, result in results.items() 
                           if result['training_success']}
        
        if successful_models:
            # Use the first successful model for testing
            best_model_name = list(successful_models.keys())[0]
            best_result = successful_models[best_model_name]
            
            print(f"Testing best model: {best_model_name}")
            test_metrics = best_result['test_metrics']
            print(f"Model test R²: {test_metrics['r2_overall']:.4f}")
            print(f"Model test MAE: {test_metrics['mae_overall']:.4f}")
            
            # Test on actual track
            evaluation_results = trainer.evaluate_on_track(best_model_name, num_episodes=3)
            
            print(f"\n📊 Track Performance:")
            print(f"Average reward: {evaluation_results['avg_reward']:.1f} ± {evaluation_results['std_reward']:.1f}")
            
        else:
            print("❌ No successful models to test")
        
        print(f"\n🎯 Training pipeline completed successfully!")
        print(f"Check the '{trainer.models_dir}' directory for trained models")
        print(f"Check the '{trainer.data_dir}' directory for training data")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()