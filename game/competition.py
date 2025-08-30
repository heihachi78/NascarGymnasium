"""
Model Competition - Race different TD3 model checkpoints against each other.

This demo creates multiple TD3 controllers with different model checkpoints
and races them against each other to compare performance.
"""

import sys
import os
import numpy as np
import signal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv
from game.control.td3_control_class import TD3Controller
from game.control.ppo_control_class import PPOController
from game.control.sac_control_class import SACController
from game.control.a2c_control_class import A2CController
from game.control.base_controller import BaseController


def calculate_finishing_order(num_cars, car_names, lap_counts, best_lap_time, car_rewards, disabled_cars, finishing_times):
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
        laps = lap_counts.get(car_idx, 0)
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


def signal_handler(signum, frame):
    """Immediately exit on interrupt to avoid segfault"""
    print("\n⚠️  Interrupt received...")
    os._exit(0)

def main():
    # Install signal handler for immediate exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🏎️  COMPETITION MODE")
    print("=" * 60)
    
    # Configure which models to compete
    # You can modify this list to include any models you want to test
    model_configs = [
        ("game/control/models/a2c_best_model2.zip", "A2C-B2"),
        ("game/control/models/a2c_best_model2.zip", "A2C-B2"),
        ("game/control/models/a2c_best_model2.zip", "A2C-B2"),
        ("game/control/models/a2c_best_model2.zip", "A2C-B2"),
        ("game/control/models/a2c_best_model2.zip", "A2C-B2"),
        ("game/control/models/a2c_best_model.zip", "A2C-B"),
        ("game/control/models/a2c_best_model.zip", "A2C-B"),
        ("game/control/models/a2c_best_model.zip", "A2C-B"),
        ("game/control/models/a2c_best_model.zip", "A2C-B"),
        ("game/control/models/a2c_best_model.zip", "A2C-B"),
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
        render_mode=None, #'human',
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
    previous_lap_count = {}
    finishing_times = {}
    
    # Reward tracking (per car)
    car_rewards = {}
    lap_start_rewards = {}
    lap_timing_started = {}
    
    # Note: Using direct physics collision tracking below instead of info dict
    
    # Raw physics collision tracking (per car)
    physics_collision_counts = {}  # Total collision events from physics
    physics_max_impulses = {}      # Maximum collision impulse seen
    physics_total_impulses = {}    # Sum of all collision impulses
    physics_collision_events = {}  # List to track all collision events for statistics
    
    # Overall best lap tracking
    overall_best_lap_time = None
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = []
        best_lap_time[i] = None
        previous_lap_count[i] = 0
        finishing_times[i] = None
        car_rewards[i] = 0.0
        lap_start_rewards[i] = 0.0
        lap_timing_started[i] = False
        # Initialize physics collision tracking
        physics_collision_counts[i] = 0
        physics_max_impulses[i] = 0.0
        physics_total_impulses[i] = 0.0
        physics_collision_events[i] = []
    
    current_followed_car = 0
    
    try:
        # Reset environment
        obs, info = env.reset(seed=42)
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
            
            # Create actions array - always use multi-car format
            if num_cars > 1:
                action = np.array(car_actions, dtype=np.float32)
            else:
                action = np.array(car_actions[0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect raw physics collision data each timestep
            for car_idx in range(num_cars):
                # Get current collision impulse directly from physics
                collision_impulse = env.car_physics_worlds[car_idx].get_continuous_collision_impulse()
                
                if collision_impulse > 0:
                    # Count this as a collision event
                    physics_collision_counts[car_idx] += 1
                    
                    # Update maximum impulse seen
                    physics_max_impulses[car_idx] = max(physics_max_impulses[car_idx], collision_impulse)
                    
                    # Add to total impulse sum
                    physics_total_impulses[car_idx] += collision_impulse
                    
                    # Store event for detailed statistics (keep last 1000 events to prevent memory issues)
                    physics_collision_events[car_idx].append(collision_impulse)
                    if len(physics_collision_events[car_idx]) > 1000:
                        physics_collision_events[car_idx].pop(0)
            
            # Handle multi-car rewards
            if isinstance(reward, np.ndarray):
                for car_idx in range(min(num_cars, len(reward))):
                    car_rewards[car_idx] += reward[car_idx]
                total_reward += reward[current_followed_car]
            else:
                car_rewards[current_followed_car] += reward
                total_reward += reward
            
            # Handle multi-car info and check lap completions
            if isinstance(info, dict) and "cars" in info:
                # Multi-car mode: info is a dict with cars list
                current_followed_car = info.get('followed_car_index', current_followed_car)
                
                # Check lap completions for ALL cars
                for car_idx in range(min(num_cars, len(info["cars"]))):
                    car_info = info["cars"][car_idx]
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
                            is_new_best = False
                            if best_lap_time[car_idx] is None:
                                # First lap for this car - always a personal best
                                best_lap_time[car_idx] = last_lap_time
                                is_new_best = True
                                print(f"🏁 {car_name} FIRST LAP COMPLETED! Time: {lap_time_str} | Reward: {lap_reward:.1f}")
                            elif last_lap_time < best_lap_time[car_idx]:
                                # Improved their personal best
                                best_lap_time[car_idx] = last_lap_time
                                is_new_best = True
                                print(f"🏁 {car_name} NEW PERSONAL BEST! Lap {current_lap_count}: {lap_time_str} | Reward: {lap_reward:.1f}")
                            else:
                                # Regular lap completion
                                print(f"🏁 {car_name} Lap {current_lap_count}: {lap_time_str} | Reward: {lap_reward:.1f}")
                            
                            # Check overall best lap
                            if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                                overall_best_lap_time = last_lap_time
                                if is_new_best:
                                    print(f"   🌟 NEW OVERALL BEST LAP by {car_name}!")
                                else:
                                    print(f"   🌟 {car_name} takes the OVERALL BEST LAP!")
                        
                        previous_lap_count[car_idx] = current_lap_count
                
                # Note: Collision forces tracked directly from physics above
            
            
            env.render()
            
            if terminated or truncated:
                # Display termination info
                termination_type = "terminated" if terminated else "truncated"
                reason = info.get("termination_reason", "unknown")
                sim_time = info.get("simulation_time", 0)
                
                print(f"\n⚠️  Episode {termination_type} at step {step}")
                print(f"📊 Reason: {reason}")
                print(f"⏱️  Simulation time: {sim_time:.1f}s")
                
                # Display final results
                print("\n" + "=" * 60)
                print("🏁 COMPETITION FINISHED - FINAL RESULTS")
                print("=" * 60)
                
                # Gather final lap counts from lap timers
                final_lap_counts = {}
                if isinstance(info, dict) and "cars" in info:
                    for car_idx in range(min(num_cars, len(info["cars"]))):
                        lap_timing = info["cars"][car_idx].get('lap_timing', {})
                        final_lap_counts[car_idx] = lap_timing.get('lap_count', 0)
                else:
                    # Single car mode or legacy format
                    lap_timing = info.get('lap_timing', {})
                    final_lap_counts[0] = lap_timing.get('lap_count', 0)
                
                finishing_order = calculate_finishing_order(
                    num_cars, car_names, final_lap_counts, best_lap_time, 
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
                print(f"   🏁 Total laps completed: {sum(final_lap_counts.values())}")
                print(f"   💰 Total rewards: {sum(car_rewards.values()):.2f}")
                
                # Display collision force summary using real physics data
                print(f"\n💥 COLLISION FORCE SUMMARY:")
                total_physics_collisions = sum(physics_collision_counts.values())
                total_physics_impulse = sum(physics_total_impulses.values())
                max_physics_impulse = max(physics_max_impulses.values()) if any(physics_max_impulses.values()) else 0
                
                print(f"   🏆 Total collision impulse (all cars): {total_physics_impulse:.2f} N⋅s")
                print(f"   📊 Total collision events: {total_physics_collisions}")
                
                if total_physics_collisions > 0:
                    print(f"   📋 Per-car collision breakdown:")
                    for car_idx in range(num_cars):
                        car_name = car_names[car_idx]
                        car_collisions = physics_collision_counts[car_idx]
                        car_max = physics_max_impulses[car_idx]
                        car_total = physics_total_impulses[car_idx]
                        if car_collisions > 0:
                            car_avg = car_total / car_collisions
                            percentage = (car_total / total_physics_impulse * 100) if total_physics_impulse > 0 else 0
                            print(f"      🚗 {car_name}: {car_collisions} events, max: {car_max:.1f} N⋅s, avg: {car_avg:.1f} N⋅s, total: {car_total:.1f} N⋅s ({percentage:.1f}%)")
                        else:
                            print(f"      🚗 {car_name}: 0 collision events")
                    
                    # Calculate collision statistics
                    avg_physics_impulse = total_physics_impulse / total_physics_collisions
                    print(f"   📈 Average collision impulse: {avg_physics_impulse:.1f} N⋅s")
                    print(f"   🔥 Maximum collision impulse: {max_physics_impulse:.1f} N⋅s")
                    
                    # Calculate collision rate based on simulation time
                    sim_time = info.get("simulation_time", 0)
                    if sim_time > 0:
                        collision_rate = (total_physics_collisions / sim_time) * 60  # per minute
                        print(f"   ⏱️ Collision rate: {collision_rate:.1f}/min")
                else:
                    print(f"   ✅ No collision events detected - clean race!")

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
                    
                    laps = final_lap_counts.get(car_idx, 0)
                    best = best_lap_time[car_idx]
                    reward = car_rewards[car_idx]
                    collision_force = physics_total_impulses[car_idx]  # Use actual physics collision data
                    
                    print(f"   {car_name}:")
                    print(f"      Model: {model_info}")
                    best_str = f"{best:.3f}s" if best is not None else "N/A"
                    print(f"      Laps: {laps}, Best: {best_str}, Reward: {reward:.1f}")
                    print(f"      Collision force: {collision_force:.2f} N⋅s")
                
                break
        
        if env.render_mode == "human":
            env.render()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Competition interrupted by user (Ctrl+C)")
        print("🔄 Skipping...")
        # Don't call env.close() during KeyboardInterrupt as it causes segfaults
        # Use os._exit to avoid any cleanup that might cause segfaults
        import os
        os._exit(0)
    except Exception as e:
        print(f"❌ Error during competition: {e}")
        import traceback
        traceback.print_exc()
        
    # Skip env.close() to prevent segfaults - let Python handle cleanup
    print("=" * 60)
    print("🔒 Done.")

if __name__ == "__main__":
    main()
