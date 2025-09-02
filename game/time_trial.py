"""
Time Trial Racing Mode.

Each car gets 3 attempts of 2 laps each (6 laps total).
The car with the fastest single lap time wins.
Environment resets between attempts for fair conditions.
"""

import sys
import os
import numpy as np
import signal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv
from game.control.base_controller import BaseController
from game.control.td3_control_class import TD3Controller
from game.control.ppo_control_class import PPOController
from game.control.sac_control_class import SACController
from game.control.a2c_control_class import A2CController

# Time trial configuration constants
LAPS_PER_ATTEMPT = 2  # Number of laps in each attempt
TOTAL_ATTEMPTS = 3     # Total number of attempts (3 attempts x 2 laps = 6 laps total)


def calculate_time_trial_results(num_cars, car_names, all_lap_times, best_lap_times, disabled_cars):
    """
    Calculate the final time trial results.
    
    Ranking criteria:
    1. Fastest single lap time across all attempts
    2. Disabled cars ranked last
    
    Returns:
        List of tuples: (car_idx, car_name, best_time, total_laps, status)
    """
    results = []
    
    for car_idx in range(num_cars):
        car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
        best_time = best_lap_times.get(car_idx, None)
        total_laps = sum(len(times) for times in all_lap_times.get(car_idx, {}).values())
        is_disabled = car_idx in disabled_cars
        
        # Format best time
        if best_time:
            minutes = int(best_time // 60)
            seconds = best_time % 60
            time_str = f"{minutes}:{seconds:06.3f}"
        else:
            time_str = "No valid lap"
        
        # Create status
        if is_disabled:
            status = "DISABLED"
        elif total_laps > 0:
            status = f"{total_laps} laps completed"
        else:
            status = "No laps completed"
        
        results.append((car_idx, car_name, best_time, total_laps, time_str, status))
    
    # Sort by best lap time (None goes to end)
    def sort_key(result):
        _, _, best_time, _, _, status = result
        if best_time is None:
            return 999999  # No valid laps go to the end
        else:
            return best_time  # Sort by fastest time regardless of status
    
    results.sort(key=sort_key)
    return results


def signal_handler(signum, frame):
    """Immediately exit on interrupt to avoid segfault"""
    print("\n⚠️  Interrupt received...")
    os._exit(0)

def main():
    # Install signal handler for immediate exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🏁 TIME TRIAL MODE")
    print("=" * 60)
    print(f"📋 RULES:")
    print(f"   • {TOTAL_ATTEMPTS} attempts of {LAPS_PER_ATTEMPT} laps each")
    print(f"   • Total of {TOTAL_ATTEMPTS * LAPS_PER_ATTEMPT} laps per car")
    print(f"   • Fastest single lap time wins")
    print(f"   • Random track selected for each attempt")
    print(f"   • Environment resets between attempts")
    print("=" * 60)
    
    # Configure which models to compete
    # You can modify this list to include any models you want to test
    model_configs = [
        ("game/control/models/a2c_best_model_opt_1.zip", "A2C-O-1"),
        ("game/control/models/td3_best_model1.zip", "TD3-B-1"),
        ("game/control/models/a2c_best_model3.zip", "A2C-B-3"),
        (None, "BC"),
        ("game/control/models/ppo_284.zip", "PPO-284"),
        ("game/control/models/td3_best_model3.zip", "TD3-B-3"),
        ("game/control/models/ppo_best_model.zip", "PPO-B"),
    ]
    
    # Take only the first 10 models (environment supports max 10 cars)
    model_configs = model_configs[:10]
    num_cars = len(model_configs)
    
    # Extract car names from configs
    car_names = [config[1] for config in model_configs]
    
    print(f"📊 Time Trial Setup: {num_cars} cars")
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
        elif "td3" in model_path.lower():
            print(f"   → Loading TD3 model: {model_path}")
            controller = TD3Controller(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Model loaded successfully")
            else:
                print(f"   ⚠ Using fallback control")
        else:
            controller = BaseController(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Base control")
            else:
                print(f"   ⚠ Using fallback control")
    
    print("=" * 60)
    
    # Initialize environment with random tracks
    env = CarEnv(track_file='tracks/nascar_banked.track',  # No fixed track (automatic random selection)
                 num_cars=num_cars,
                 reset_on_lap=False,  # We manage resets manually
                 render_mode=None, #"human",
                 discrete_action_space=False,
                 car_names=car_names)
    env.seed(42)
    
    print(f"🎮 CONTROLS:")
    print(f"   Keys 0-{min(num_cars-1, 9)}: Switch camera between cars")
    print(f"   R: Toggle reward display")
    print(f"   D: Toggle debug display")
    print(f"   I: Toggle track info display")
    print(f"   C: Change camera mode")
    print(f"   F: Full screen")
    print(f"   O: Toggle observation graphs")
    print(f"   ESC: Exit")
    print("=" * 60)
    
    # Time trial data tracking
    all_lap_times = {}  # {car_idx: {attempt_num: [lap_times]}}
    best_lap_times = {}  # {car_idx: best_time}
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = {}
        best_lap_times[i] = None
    
    # Overall tracking
    overall_best_lap_time = None
    
    try:
        # Main time trial loop - run multiple attempts
        for attempt_num in range(1, TOTAL_ATTEMPTS + 1):
            print(f"\n{'=' * 60}")
            print(f"🏁 ATTEMPT {attempt_num} OF {TOTAL_ATTEMPTS}")
            print(f"{'=' * 60}")
            
            # Reset environment for new attempt (selects new random track)
            obs, info = env.reset(attempt_num * 42)
            print(f"🏁 Track for attempt {attempt_num}: {env.track_file}")
            print(f"{'=' * 60}")
            
            # Per-attempt tracking
            attempt_lap_counts = {}  # Lap count in current attempt
            attempt_lap_times = {}  # Lap times in current attempt
            cars_finished_attempt = set()  # Cars that completed the attempt
            previous_lap_counts = {}  # Previous lap count for change detection
            lap_timing_started = {}  # Track if lap timing has started
            lap_start_rewards = {}  # Reward when lap started
            car_rewards = {}  # Total rewards per car
            
            # Initialize per-attempt tracking
            for i in range(num_cars):
                attempt_lap_counts[i] = 0
                attempt_lap_times[i] = []
                previous_lap_counts[i] = 0
                lap_timing_started[i] = False
                lap_start_rewards[i] = 0.0
                car_rewards[i] = 0.0
                all_lap_times[i][attempt_num] = []
            
            current_followed_car = 0
            simulation_step = 0
            
            # Run current attempt until all cars complete LAPS_PER_ATTEMPT or disable
            while True:
                if env.check_quit_requested():
                    print(f"   User requested quit")
                    return
                
                # Calculate controls for each car
                car_actions = []
                for car_idx in range(num_cars):
                    # Skip disabled cars
                    if car_idx in env.disabled_cars or car_idx in cars_finished_attempt:
                        car_actions.append([0.0, 0.0])  # [throttle_brake, steering]
                        continue
                    
                    # Get observation for this car
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
                
                # Step environment
                obs, reward, _, _, info = env.step(action)
                simulation_step += 1
                
                # Track rewards
                if isinstance(reward, np.ndarray):
                    for car_idx in range(min(num_cars, len(reward))):
                        car_rewards[car_idx] += reward[car_idx]
                else:
                    car_rewards[0] += reward
                
                # Check lap completions for all cars
                if isinstance(info, dict) and "cars" in info:
                    # Multi-car mode: info is a dict with cars list
                    current_followed_car = info.get('followed_car_index', current_followed_car)
                    
                    for car_idx in range(min(num_cars, len(info["cars"]))):
                        # Skip cars that have finished this attempt
                        if car_idx in cars_finished_attempt:
                            continue
                            
                        car_info = info["cars"][car_idx]
                        lap_timing = car_info.get('lap_timing', {})
                        current_lap_count = lap_timing.get('lap_count', 0)
                        is_timing = lap_timing.get('is_timing', False)
                        car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                        
                        # Track lap timing start
                        if is_timing and not lap_timing_started[car_idx]:
                            lap_timing_started[car_idx] = True
                            lap_start_rewards[car_idx] = car_rewards[car_idx]
                        
                        # Check for lap completion
                        if current_lap_count > previous_lap_counts[car_idx]:
                            last_lap_time = lap_timing.get('last_lap_time', None)
                            if last_lap_time:
                                # Record lap time
                                attempt_lap_counts[car_idx] += 1
                                attempt_lap_times[car_idx].append(last_lap_time)
                                all_lap_times[car_idx][attempt_num].append(last_lap_time)
                                
                                # Update lap start rewards for next lap
                                lap_start_rewards[car_idx] = car_rewards[car_idx]
                                
                                # Format lap time
                                minutes = int(last_lap_time // 60)
                                seconds = last_lap_time % 60
                                lap_time_str = f"{minutes}:{seconds:06.3f}"
                                
                                # Update best lap for this car
                                if best_lap_times[car_idx] is None or last_lap_time < best_lap_times[car_idx]:
                                    best_lap_times[car_idx] = last_lap_time
                                    print(f"🏆 {car_name} NEW PERSONAL BEST! Lap {attempt_lap_counts[car_idx]}/{LAPS_PER_ATTEMPT}: {lap_time_str}")
                                    
                                    # Check for overall best
                                    if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                                        overall_best_lap_time = last_lap_time
                                        print(f"   ⭐ NEW OVERALL BEST LAP!")
                                else:
                                    print(f"✓ {car_name} Lap {attempt_lap_counts[car_idx]}/{LAPS_PER_ATTEMPT}: {lap_time_str}")
                                    if best_lap_times[car_idx]:
                                        gap = last_lap_time - best_lap_times[car_idx]
                                        print(f"   Gap to PB: +{gap:.3f}s")
                                
                                # Check if car finished the attempt
                                if attempt_lap_counts[car_idx] >= LAPS_PER_ATTEMPT:
                                    cars_finished_attempt.add(car_idx)
                                    print(f"🏁 {car_name} completed attempt {attempt_num}!")
                            
                            previous_lap_counts[car_idx] = current_lap_count
                
                # Check for disabled cars
                for car_idx in env.disabled_cars:
                    if car_idx not in cars_finished_attempt:
                        cars_finished_attempt.add(car_idx)
                        car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                        print(f"❌ {car_name} disabled in attempt {attempt_num}")
                
                # Render
                env.render()
                
                # Check if all cars have finished or disabled
                if len(cars_finished_attempt) >= num_cars:
                    print(f"\n📊 ATTEMPT {attempt_num} COMPLETE")
                    break
                
                # Safety timeout
                if simulation_step > 50000:
                    print(f"⚠️ Attempt timeout reached")
                    break
            
            # Display attempt summary
            print(f"\n📈 ATTEMPT {attempt_num} SUMMARY:")
            for car_idx in range(num_cars):
                car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                times = attempt_lap_times[car_idx]
                if times:
                    best_attempt_time = min(times)
                    minutes = int(best_attempt_time // 60)
                    seconds = best_attempt_time % 60
                    print(f"   {car_name}: {len(times)} laps, Best: {minutes}:{seconds:06.3f}")
                else:
                    print(f"   {car_name}: No laps completed")
        
        # Final results
        print(f"\n{'=' * 60}")
        print(f"🏆 TIME TRIAL FINAL RESULTS")
        print(f"{'=' * 60}")
        
        results = calculate_time_trial_results(num_cars, car_names, all_lap_times, best_lap_times, env.disabled_cars)
        
        print("\n🥇 LEADERBOARD (by fastest lap):")
        for position, (car_idx, car_name, _, _, time_str, status) in enumerate(results, 1):
            position_emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            print(f"   {position_emoji} {car_name}: {time_str} - {status}")
        
        # Detailed statistics
        print(f"\n📊 DETAILED STATISTICS:")
        for car_idx in range(num_cars):
            car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
            print(f"\n   {car_name}:")
            
            for attempt_num in range(1, TOTAL_ATTEMPTS + 1):
                times = all_lap_times[car_idx].get(attempt_num, [])
                if times:
                    print(f"      Attempt {attempt_num}: ", end="")
                    for i, lap_time in enumerate(times, 1):
                        minutes = int(lap_time // 60)
                        seconds = lap_time % 60
                        if lap_time == best_lap_times.get(car_idx):
                            print(f"Lap {i}: {minutes}:{seconds:06.3f}⭐ ", end="")
                        else:
                            print(f"Lap {i}: {minutes}:{seconds:06.3f} ", end="")
                    print()
                else:
                    print(f"      Attempt {attempt_num}: No laps completed")
        
        print(f"\n{'=' * 60}")
        print("🏁 TIME TRIAL COMPLETE!")
        print(f"{'=' * 60}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Time trial interrupted by user (Ctrl+C)")
        print("🔄 Skipping...")
        # Don't call env.close() during KeyboardInterrupt as it causes segfaults
        # Use os._exit to avoid any cleanup that might cause segfaults
        import os
        os._exit(0)
    except Exception as e:
        print(f"❌ Error during time trial: {e}")
        import traceback
        traceback.print_exc()
        
    # Skip env.close() to prevent segfaults - let Python handle cleanup
    print("=" * 60)
    print("🔒 Done.")


if __name__ == "__main__":
    main()