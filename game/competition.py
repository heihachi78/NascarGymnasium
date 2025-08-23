"""
Model Competition - Race different TD3 model checkpoints against each other.

This demo creates multiple TD3 controllers with different model checkpoints
and races them against each other to compare performance.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv
from game.control.td3_control_class import TD3Controller
from game.control.ppo_control_class import PPOController
from game.control.sac_control_class import SACController
from game.control.a2c_control_class import A2CController
from game.control.base_controller import BaseController


def calculate_finishing_order(num_cars, car_names, total_laps, best_lap_time, car_rewards, disabled_cars, finishing_times):
    """
    Calculate the finishing order based on race performance.
    
    Ranking criteria (in order of priority):
    1. Laps completed (more laps = better)
    2. Finishing time (earlier finish = better, for cars with same lap count)
    3. Best lap time (faster = better, fallback tiebreaker)
    4. Cumulative reward (higher = better, fallback)
    5. Disabled status (disabled cars ranked last)
    
    Returns:
        List of tuples: (car_idx, car_name, status, details)
    """
    race_results = []
    
    for car_idx in range(num_cars):
        car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
        laps = total_laps.get(car_idx, 0)
        best_time = best_lap_time.get(car_idx, None)
        finish_time = finishing_times.get(car_idx, None)
        reward = car_rewards.get(car_idx, 0.0)
        is_disabled = car_idx in disabled_cars
        
        # Create status and details
        if is_disabled:
            status = "DISABLED"
            details = f"Laps: {laps}, Reward: {reward:.1f}"
        elif laps > 0:
            if best_time:
                minutes = int(best_time // 60)
                seconds = best_time % 60
                time_str = f"{minutes}:{seconds:06.3f}"
                status = f"{laps} laps completed"
                details = f"Best lap: {time_str}, Reward: {reward:.1f}"
            else:
                status = f"{laps} laps completed"
                details = f"Reward: {reward:.1f}"
        else:
            status = "Did not complete a lap"
            details = f"Reward: {reward:.1f}"
        
        race_results.append((car_idx, car_name, laps, best_time, finish_time, reward, is_disabled, status, details))
    
    # Sort by finishing order criteria
    def sort_key(result):
        _, _, laps, best_time, finish_time, reward, is_disabled, _, _ = result
        return (
            -laps,                                    # More laps = better (descending)
            finish_time if finish_time else 999999,  # Earlier finish = better (ascending, None = worst)
            best_time if best_time else 999999,      # Faster time = better (ascending, None = worst)
            -reward,                                  # Higher reward = better (descending)
            is_disabled                               # Non-disabled = better (False < True)
        )
    
    race_results.sort(key=sort_key)
    
    # Return simplified results for display
    return [(result[0], result[1], result[7], result[8]) for result in race_results]


def main():
    print("=" * 60)
    print("🏎️  COMPETITION MODE")
    print("=" * 60)
    
    # Configure which models to compete
    # You can modify this list to include any models you want to test
    model_configs = [
        ("game/control/models/ppo_model.zip", "PPO"),
        ("game/control/models/td3_model.zip", "TD3"),
        ("game/control/models/td3_model_time_trial.zip", "TD3-TT"),
        ("game/control/models/sac_model.zip", "SAC"),
        ("game/control/models/a2c_model_competition.zip", "A2C-COMP"),
        ("game/control/models/a2c_model_time_trial.zip", "A2C-TT"),
        ("game/control/models/a2c_model.zip", "A2C"),
        (None, "Rule-Based"),  # Use None for rule-based control
    ]

    # Take only the first 10 models (environment supports max 10 cars)
    model_configs = model_configs[:10]
    num_cars = len(model_configs)
    
    # Extract car names from configs
    car_names = [config[1] for config in model_configs]
    
    print(f"📊 Competition Setup: {num_cars} cars")
    print("=" * 60)
    
    # Create controllers for each car
    controllers = []
    for i, (model_path, name) in enumerate(model_configs):
        print(f"Car {i}: {name}")
        if model_path is None:
            print(f"   → Using rule-based control")
            # Create a BaseController instance for rule-based control
            controllers.append(BaseController(name=name))
        elif "ppo" in model_path.lower():
            print(f"   → Loading PPO model: {model_path}")
            controller = PPOController(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Model loaded successfully")
            else:
                print(f"   ⚠ Using fallback control")
        elif "sac" in model_path.lower():
            print(f"   → Loading SAC model: {model_path}")
            controller = SACController(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Model loaded successfully")
            else:
                print(f"   ⚠ Using fallback control")
        elif "a2c" in model_path.lower():
            print(f"   → Loading A2C model: {model_path}")
            controller = A2CController(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Model loaded successfully")
            else:
                print(f"   ⚠ Using fallback control")
        else:
            # Default to TD3 for backward compatibility
            print(f"   → Loading TD3 model: {model_path}")
            controller = TD3Controller(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Model loaded successfully")
            else:
                print(f"   ⚠ Using fallback control")
    
    print("=" * 60)
    
    # Create environment
    env = CarEnv(
        track_file="tracks/nascar.track", 
        num_cars=num_cars, 
        reset_on_lap=False, 
        render_mode="human",
        enable_fps_limit=False,
        disable_cars_on_high_impact=True,
        discrete_action_space=False,
        car_names=car_names
    )
    
    print(f"🎮 CONTROLS:")
    print(f"   Keys 0-{min(num_cars-1, 9)}: Switch camera between cars")
    print(f"   R: Toggle reward display")
    print(f"   D: Toggle debug display")
    print(f"   I: Toggle track info display")
    print(f"   C: Change camera")
    print(f"   ESC: Exit")
    print(f"📺 Following {car_names[0]} - Press 0-{min(num_cars-1, 9)} to switch cars")
    print("=" * 60)
    
    # Lap time tracking (per car)
    all_lap_times = {}
    best_lap_time = {}
    total_laps = {}
    previous_lap_count = {}
    finishing_times = {}
    
    # Reward tracking (per car)
    car_rewards = {}
    lap_start_rewards = {}
    lap_timing_started = {}
    
    # Overall best lap tracking
    overall_best_lap_time = None
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = []
        best_lap_time[i] = None
        total_laps[i] = 0
        previous_lap_count[i] = 0
        finishing_times[i] = None
        car_rewards[i] = 0.0
        lap_start_rewards[i] = 0.0
        lap_timing_started[i] = False
    
    current_followed_car = 0
    
    try:
        # Reset environment
        obs, info = env.reset()
        print("🏁 RACE STARTED!")
        print("=" * 60)
        total_reward = 0.0
        
        for step in range(100000):
            if env.check_quit_requested():
                print(f"   User requested quit at step {step}")
                break
            
            # Calculate individual controls for each car
            car_actions = []
            
            for car_idx in range(num_cars):
                # Handle multi-car observations
                if isinstance(obs, np.ndarray) and len(obs.shape) == 2:
                    car_obs = obs[car_idx]
                else:
                    car_obs = obs
                
                # Get control action from controller
                controller = controllers[car_idx]
                action = controller.control(car_obs)
                
                car_actions.append(action)
            
            # Create actions array
            if num_cars == 1:
                action = np.array(car_actions[0], dtype=np.float32)
            else:
                action = np.array(car_actions, dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Handle multi-car rewards
            if isinstance(reward, np.ndarray):
                for car_idx in range(min(num_cars, len(reward))):
                    car_rewards[car_idx] += reward[car_idx]
                total_reward += reward[current_followed_car]
            else:
                car_rewards[current_followed_car] += reward
                total_reward += reward
            
            # Handle multi-car info and check lap completions
            if isinstance(info, list):
                followed_car_info = info[current_followed_car] if current_followed_car < len(info) else info[0]
                current_followed_car = followed_car_info.get('followed_car_index', current_followed_car)
                
                # Check lap completions for ALL cars
                for car_idx in range(min(num_cars, len(info))):
                    car_info = info[car_idx]
                    lap_timing = car_info.get('lap_timing', {})
                    current_lap_count = lap_timing.get('lap_count', 0)
                    is_timing = lap_timing.get('is_timing', False)
                    car_name = car_names[car_idx]
                    
                    # Check if lap timing just started
                    if is_timing and not lap_timing_started[car_idx]:
                        lap_timing_started[car_idx] = True
                        lap_start_rewards[car_idx] = car_rewards[car_idx]
                    
                    # Check if this car completed a lap
                    if current_lap_count > previous_lap_count[car_idx]:
                        last_lap_time = lap_timing.get('last_lap_time', None)
                        if last_lap_time:
                            all_lap_times[car_idx].append(last_lap_time)
                            total_laps[car_idx] += 1
                            
                            # Record finishing time
                            current_sim_time = car_info.get("simulation_time", 0)
                            finishing_times[car_idx] = current_sim_time
                            
                            # Calculate lap reward
                            lap_reward = car_rewards[car_idx] - lap_start_rewards[car_idx]
                            lap_start_rewards[car_idx] = car_rewards[car_idx]
                            
                            # Format lap time
                            minutes = int(last_lap_time // 60)
                            seconds = last_lap_time % 60
                            lap_time_str = f"{minutes}:{seconds:06.3f}"
                            
                            # Update best lap for this car
                            if best_lap_time[car_idx] is None or last_lap_time < best_lap_time[car_idx]:
                                best_lap_time[car_idx] = last_lap_time
                                print(f"🏁 {car_name} NEW BEST LAP! Time: {lap_time_str} | Reward: {lap_reward:.1f}")
                            else:
                                print(f"🏁 {car_name} Lap {total_laps[car_idx]}: {lap_time_str} | Reward: {lap_reward:.1f}")
                            
                            # Check overall best lap
                            if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                                overall_best_lap_time = last_lap_time
                                print(f"   🌟 NEW OVERALL BEST LAP by {car_name}!")
                        
                        previous_lap_count[car_idx] = current_lap_count
            
            env.render()
            
            if terminated or truncated:
                # Display termination info
                termination_type = "terminated" if terminated else "truncated"
                reason = info.get("termination_reason", "unknown") if not isinstance(info, list) else info[0].get("termination_reason", "unknown")
                sim_time = info.get("simulation_time", 0) if not isinstance(info, list) else info[0].get("simulation_time", 0)
                
                print(f"\n⚠️  Episode {termination_type} at step {step}")
                print(f"📊 Reason: {reason}")
                print(f"⏱️  Simulation time: {sim_time:.1f}s")
                
                # Display final results
                print("\n" + "=" * 60)
                print("🏁 COMPETITION FINISHED - FINAL RESULTS")
                print("=" * 60)
                
                finishing_order = calculate_finishing_order(
                    num_cars, car_names, total_laps, best_lap_time, 
                    car_rewards, env.disabled_cars, finishing_times
                )
                
                print("🏆 FINISHING ORDER:")
                for position, (car_idx, car_name, status, details) in enumerate(finishing_order, 1):
                    position_emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
                    print(f"   {position_emoji} {car_name} - {status}")
                    if details:
                        print(f"      {details}")
                
                # Display race statistics
                print(f"\n📊 RACE STATISTICS:")
                print(f"   🏁 Total laps completed: {sum(total_laps.values())}")
                print(f"   💰 Total rewards: {sum(car_rewards.values()):.2f}")
                
                # Display model performance comparison
                print(f"\n📈 MODEL PERFORMANCE:")
                for car_idx, car_name in enumerate(car_names):
                    controller = controllers[car_idx]
                    if isinstance(controller, BaseController) and not hasattr(controller, 'model_path'):
                        model_info = "Rule-based"
                    elif hasattr(controller, 'model_path'):
                        # Determine model type from path or class name
                        if "ppo" in controller.model_path.lower():
                            model_type = "PPO"
                        elif "sac" in controller.model_path.lower():
                            model_type = "SAC"
                        elif "td3" in controller.model_path.lower():
                            model_type = "TD3"
                        else:
                            # Fallback to class name
                            model_type = controller.__class__.__name__.replace('Controller', '')
                        model_info = f"{model_type} ({controller.model_path})"
                    else:
                        model_info = "Unknown"
                    
                    laps = total_laps[car_idx]
                    best = best_lap_time[car_idx]
                    reward = car_rewards[car_idx]
                    
                    print(f"   {car_name}:")
                    print(f"      Model: {model_info}")
                    best_str = f"{best:.3f}s" if best is not None else "N/A"
                    print(f"      Laps: {laps}, Best: {best_str}, Reward: {reward:.1f}")
                
                break
        
        if env.render_mode == "human":
            env.render()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Competition interrupted by user (Ctrl+C)")
        print("🔄 Performing cleanup...")
    except Exception as e:
        print(f"❌ Error during competition: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("=" * 60)
        try:
            env.close()
            print("🔒 Environment closed")
        except Exception as e:
            print(f"⚠️ Warning during environment cleanup: {e}")

        # Note: We intentionally don't call pygame.quit() here to avoid segfaults
        # The renderer and environment cleanup handle pygame display shutdown
        # pygame.quit() can cause segfaults when called after signal interrupts

if __name__ == "__main__":
    main()