"""
Car physics demonstration.

This demo shows the complete car physics system in action.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv
import time as ptime


def main():
    print("=" * 50)
    
    # Multi-car demo configuration with custom names
    num_cars = 10  # Change this to test different numbers of cars (1-10)
    car_names = [
        "Lightning",
        "Thunder",
        "Blaze",
        "Storm",
        "Phoenix",
        "Cyclone",
        "Tornado",
        "Inferno",
        "Viper",
        "Shadow"
    ][:num_cars]
    
    print(f"🚗 MULTI-CAR RACING DEMO - {num_cars} Named Cars")
    print("=" * 50)
    
    #env = CarEnv(render_mode="human", track_file="tracks/nascar.track", reset_on_lap=False, num_cars=num_cars, car_names=car_names)
    env = CarEnv(track_file="tracks/nascar.track", 
                 num_cars=num_cars, 
                 reset_on_lap=False, 
                 render_mode="human",
                 enable_fps_limit=True,
                 disable_cars_on_high_impact=True,
                 car_names=car_names)
    
    print(f"🎮 CONTROLS:")
    print(f"   Keys 0-{min(num_cars-1, 9)}: Switch camera between cars")
    print(f"   R: Toggle reward display")
    print(f"   D: Toggle debug display")
    print(f"   I: Toggle track info display")
    print(f"   C: Change camera")
    print(f"   ESC: Exit")
    print(f"📺 Following {car_names[0]} - Press 0-{min(num_cars-1, 9)} to switch cars")
    
    # Display car assignments
    print("🏁 CAR LINEUP:")
    for i, name in enumerate(car_names):
        print(f"   {i}: {name}")
    
    # Lap time tracking (per car)
    all_lap_times = {}  # Dictionary with car_index as key
    best_lap_time = {}  # Best lap time per car
    total_laps = {}     # Total laps per car
    previous_lap_count = {}  # Previous lap count per car
    
    # Reward tracking (per car)
    car_rewards = {}    # Total reward per car
    lap_start_rewards = {}  # Reward when lap started (to calculate lap reward)
    lap_timing_started = {}  # Track if lap timing has started for each car
    
    # Overall best lap tracking (across all cars)
    overall_best_lap_time = None
    overall_best_lap_car = None
    
    # Cumulative impact force tracking across all episodes
    cumulative_impact_across_episodes = {}  # Track impact across all episodes
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = []
        best_lap_time[i] = None
        total_laps[i] = 0
        previous_lap_count[i] = 0
        car_rewards[i] = 0.0
        lap_start_rewards[i] = 0.0
        lap_timing_started[i] = False
        cumulative_impact_across_episodes[i] = 0.0
    
    # Current followed car info
    current_followed_car = 0
    
    try:
        # Reset environment first
        obs, info = env.reset()
        print("🚗 Running simulation...")
        total_reward = 0.0
        
        # Initialize control variables for each car
        car_throttles = [0.0] * num_cars
        car_brakes = [0.0] * num_cars
        car_steerings = [0.0] * num_cars
        
        for step in range(100000):
            if env.check_quit_requested():
                print(f"   User requested quit at step {step}")
                break
            # Calculate individual controls for each car
            car_actions = []
            
            for car_idx in range(num_cars):
                # Handle multi-car observations
                if isinstance(obs, np.ndarray) and len(obs.shape) == 2:
                    # Multi-car observations: use each car's own observation
                    car_obs = obs[car_idx]
                else:
                    # Single car observation (fallback)
                    car_obs = obs
                    
                sensors = car_obs[21:29]  # All 8 sensor distances
                forward = sensors[0]     # Forward sensor (21)
                front_left = sensors[1] # Front-right sensor (22) 
                front_right = sensors[7]  # Front-left sensor (28)
                current_speed = car_obs[4]  # Speed from observation
                
                # Use working control logic from previous version
                speed_limit = ((forward * 465) + (car_idx*2))  / 3.6

                if current_speed * 200 < speed_limit: 
                    car_throttles[car_idx] += 0.1
                if current_speed * 200 > speed_limit: 
                    car_throttles[car_idx] -= 0.1

                if current_speed * 200 < speed_limit: 
                    car_brakes[car_idx] -= 0.01
                if current_speed * 200 > speed_limit: 
                    car_brakes[car_idx] += 0.01

                if front_right > front_left:
                    car_steerings[car_idx] = front_right / front_left - forward
                elif front_right < front_left:
                    car_steerings[car_idx] = front_left / front_right - forward
                    car_steerings[car_idx] *= -1
                else:
                    car_steerings[car_idx] = 0

                # Apply limits and adjustments
                if car_idx != 11:
                    car_brakes[car_idx] = max(min(car_brakes[car_idx], 1), 0)
                    car_steerings[car_idx] = max(min(car_steerings[car_idx], 1), -1)
                    if abs(car_steerings[car_idx]) > 0.1:
                        car_throttles[car_idx] -= 0.05
                    car_throttles[car_idx] = max(min(car_throttles[car_idx], 1), 0)
                else:
                    car_brakes[car_idx] = 0
                    car_steerings[car_idx] = 0
                    car_throttles[car_idx] = 1
                
                car_actions.append([car_throttles[car_idx], car_brakes[car_idx], car_steerings[car_idx]])
            
            # Create actions array
            if num_cars == 1:
                action = np.array(car_actions[0], dtype=np.float32)
                #action = np.array([1, 0, 0], dtype=np.float32)
            else:
                action = np.array(car_actions, dtype=np.float32)
                #action = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Handle multi-car rewards - track ALL cars' rewards
            if isinstance(reward, np.ndarray):
                # Multi-car rewards: update each car's total
                for car_idx in range(min(num_cars, len(reward))):
                    car_rewards[car_idx] += reward[car_idx]
                total_reward += reward[current_followed_car]  # Use followed car's reward for display
            else:
                # Single car reward: update only the followed car
                car_rewards[current_followed_car] += reward
                total_reward += reward

            # Handle multi-car info and check lap completions for ALL cars
            if isinstance(info, list):
                # Multi-car info: get followed car's info for camera tracking
                followed_car_info = info[current_followed_car] if current_followed_car < len(info) else info[0]
                current_followed_car = followed_car_info.get('followed_car_index', current_followed_car)
                
                # Check lap completions for ALL cars
                for car_idx in range(min(num_cars, len(info))):
                    car_info = info[car_idx]
                    lap_timing = car_info.get('lap_timing', {})
                    current_lap_count = lap_timing.get('lap_count', 0)
                    is_timing = lap_timing.get('is_timing', False)
                    car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                    
                    # Check if lap timing just started for this car (car crossed start line for first time)
                    if is_timing and not lap_timing_started[car_idx]:
                        lap_timing_started[car_idx] = True
                        lap_start_rewards[car_idx] = car_rewards[car_idx]  # Record reward when lap timing starts
                    
                    # Check if this car completed a lap
                    if current_lap_count > previous_lap_count[car_idx]:
                        # Lap completed for this car!
                        last_lap_time = lap_timing.get('last_lap_time', None)
                        if last_lap_time:
                            all_lap_times[car_idx].append(last_lap_time)
                            total_laps[car_idx] += 1
                            
                            # Calculate lap reward (reward earned during this lap)
                            lap_reward = car_rewards[car_idx] - lap_start_rewards[car_idx]
                            lap_start_rewards[car_idx] = car_rewards[car_idx]  # Reset for next lap
                            
                            # Format lap time for display
                            minutes = int(last_lap_time // 60)
                            seconds = last_lap_time % 60
                            lap_time_str = f"{minutes}:{seconds:06.3f}"
                            
                            # Update best lap for this car
                            if best_lap_time[car_idx] is None or last_lap_time < best_lap_time[car_idx]:
                                best_lap_time[car_idx] = last_lap_time
                                print(f"🏁 {car_name} NEW BEST LAP! Time: {lap_time_str} | Reward: {lap_reward:.1f}")
                                if total_laps[car_idx] > 1:
                                    car_times = all_lap_times[car_idx]
                                    improvement = (car_times[-2] if len(car_times) > 1 else last_lap_time) - last_lap_time
                                    print(f"   ⚡ Improved by {improvement:.3f} seconds!")
                            else:
                                print(f"🏁 {car_name} Lap {total_laps[car_idx]} completed: {lap_time_str} | Reward: {lap_reward:.1f}")
                                if best_lap_time[car_idx]:
                                    gap = last_lap_time - best_lap_time[car_idx]
                                    print(f"   📊 Gap to best: +{gap:.3f}s")
                            
                            # Check and update overall best lap across all cars
                            if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                                # New overall best lap!
                                overall_best_lap_time = last_lap_time
                                overall_best_lap_car = car_idx
                                overall_best_car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                                print(f"   🌟 NEW OVERALL BEST LAP! {overall_best_car_name} set the pace: {lap_time_str}")
                            else:
                                # Show gap to overall best
                                gap_to_overall = last_lap_time - overall_best_lap_time
                                overall_best_car_name = car_names[overall_best_lap_car] if overall_best_lap_car < len(car_names) else f"Car {overall_best_lap_car}"
                                print(f"   🏆 Gap to overall best ({overall_best_car_name}): +{gap_to_overall:.3f}s")
                            
                        previous_lap_count[car_idx] = current_lap_count
            else:
                # Single car info
                current_followed_car = info.get('followed_car_index', 0)
                lap_timing = info.get('lap_timing', {})
                current_lap_count = lap_timing.get('lap_count', 0)
                is_timing = lap_timing.get('is_timing', False)
                current_car_name = car_names[current_followed_car] if current_followed_car < len(car_names) else f"Car {current_followed_car}"
                
                # Check if lap timing just started for this car (car crossed start line for first time)
                if is_timing and not lap_timing_started[current_followed_car]:
                    lap_timing_started[current_followed_car] = True
                    lap_start_rewards[current_followed_car] = car_rewards[current_followed_car]  # Record reward when lap timing starts
                
                # Check if followed car completed a lap (single car mode)
                if current_lap_count > previous_lap_count[current_followed_car]:
                    # Lap completed for followed car!
                    last_lap_time = lap_timing.get('last_lap_time', None)
                    if last_lap_time:
                        all_lap_times[current_followed_car].append(last_lap_time)
                        total_laps[current_followed_car] += 1
                        
                        # Calculate lap reward (reward earned during this lap)
                        lap_reward = car_rewards[current_followed_car] - lap_start_rewards[current_followed_car]
                        lap_start_rewards[current_followed_car] = car_rewards[current_followed_car]  # Reset for next lap
                        
                        # Format lap time for display
                        minutes = int(last_lap_time // 60)
                        seconds = last_lap_time % 60
                        lap_time_str = f"{minutes}:{seconds:06.3f}"
                        
                        # Update best lap for this car
                        if best_lap_time[current_followed_car] is None or last_lap_time < best_lap_time[current_followed_car]:
                            best_lap_time[current_followed_car] = last_lap_time
                            print(f"🏁 {current_car_name} NEW BEST LAP! Time: {lap_time_str} | Reward: {lap_reward:.1f}")
                            if total_laps[current_followed_car] > 1:
                                car_times = all_lap_times[current_followed_car]
                                improvement = (car_times[-2] if len(car_times) > 1 else last_lap_time) - last_lap_time
                                print(f"   ⚡ Improved by {improvement:.3f} seconds!")
                        else:
                            print(f"🏁 {current_car_name} Lap {total_laps[current_followed_car]} completed: {lap_time_str} | Reward: {lap_reward:.1f}")
                            if best_lap_time[current_followed_car]:
                                gap = last_lap_time - best_lap_time[current_followed_car]
                                print(f"   📊 Gap to best: +{gap:.3f}s")
                        
                        # Check and update overall best lap across all cars (single car mode)
                        if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                            # New overall best lap!
                            overall_best_lap_time = last_lap_time
                            overall_best_lap_car = current_followed_car
                            overall_best_car_name = car_names[current_followed_car] if current_followed_car < len(car_names) else f"Car {current_followed_car}"
                            print(f"   🌟 NEW OVERALL BEST LAP! {overall_best_car_name} set the pace: {lap_time_str}")
                        else:
                            # Show gap to overall best
                            gap_to_overall = last_lap_time - overall_best_lap_time
                            overall_best_car_name = car_names[overall_best_lap_car] if overall_best_lap_car < len(car_names) else f"Car {overall_best_lap_car}"
                            print(f"   🏆 Gap to overall best ({overall_best_car_name}): +{gap_to_overall:.3f}s")
                        
                    previous_lap_count[current_followed_car] = current_lap_count

            env.render()

            if terminated or truncated:
                # Display termination reason
                termination_type = "terminated" if terminated else "truncated"
                reason = info.get("termination_reason", "unknown") if not isinstance(info, list) else info[0].get("termination_reason", "unknown")
                sim_time = info.get("simulation_time", 0) if not isinstance(info, list) else info[0].get("simulation_time", 0)
                
                print(f"   ⚠️  Episode {termination_type} at step {step}")
                print(f"   📊 Reason: {reason}")
                print(f"   ⏱️  Simulation time: {sim_time:.1f}s")
                
                # Display cumulative rewards for each car
                print(f"   💰 Cumulative rewards per car:")
                for car_idx in range(num_cars):
                    car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                    car_cumulative_reward = car_rewards[car_idx]
                    print(f"      🚗 {car_name}: {car_cumulative_reward:.2f}")
                
                # Also show total across all cars for reference
                total_all_cars_reward = sum(car_rewards.values())
                print(f"   🏆 Total (all cars): {total_all_cars_reward:.2f}")
                
                # Display cumulative impact forces for each car
                print(f"   💥 Cumulative impact forces per car:")
                for car_idx in range(num_cars):
                    car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car {car_idx}"
                    # Get cumulative impact from info
                    if isinstance(info, list) and car_idx < len(info):
                        car_cumulative_impact = info[car_idx].get("cumulative_impact_force", 0.0)
                    elif not isinstance(info, list):
                        car_cumulative_impact = info.get("cumulative_impact_force", 0.0) if car_idx == current_followed_car else 0.0
                    else:
                        car_cumulative_impact = 0.0
                    print(f"      🚗 {car_name}: {car_cumulative_impact:.2f} N⋅s")
                
                # Calculate total impact across all cars
                total_all_cars_impact = 0.0
                if isinstance(info, list):
                    total_all_cars_impact = sum(car_info.get("cumulative_impact_force", 0.0) for car_info in info)
                elif not isinstance(info, list):
                    total_all_cars_impact = info.get("cumulative_impact_force", 0.0)
                print(f"   🏆 Total impact (all cars): {total_all_cars_impact:.2f} N⋅s")
                
                # Store impact data for cumulative tracking across episodes before reset
                for car_idx in range(num_cars):
                    if isinstance(info, list) and car_idx < len(info):
                        episode_impact = info[car_idx].get("cumulative_impact_force", 0.0)
                    elif not isinstance(info, list):
                        episode_impact = info.get("cumulative_impact_force", 0.0) if car_idx == current_followed_car else 0.0
                    else:
                        episode_impact = 0.0
                    cumulative_impact_across_episodes[car_idx] += episode_impact
                
                # Show collision stats if available
                collision_stats = info.get("collisions", {}) if not isinstance(info, list) else info[0].get("collisions", {})
                if collision_stats and collision_stats.get("total_collisions", 0) > 0:
                    print(f"   💥 Total collisions: {collision_stats['total_collisions']}")
                    if collision_stats.get("max_impulse", 0) > 0:
                        print(f"   💥 Max collision impulse: {collision_stats['max_impulse']:.0f}")
                
                # Reset for next episode
                obs, info = env.reset()
                total_reward = 0.0
                
                # Reset control variables for all cars
                car_throttles = [0.0] * num_cars
                car_brakes = [0.0] * num_cars
                car_steerings = [0.0] * num_cars
                
                # Reset previous lap count tracking and reward tracking for all cars
                for i in range(num_cars):
                    previous_lap_count[i] = 0
                    car_rewards[i] = 0.0
                    lap_start_rewards[i] = 0.0
                    lap_timing_started[i] = False
                
                # Reset overall best lap tracking
                overall_best_lap_time = None
                overall_best_lap_car = None
                    
                print(f"   🔄 Environment reset, continuing simulation...")
        
        if env.render_mode == "human":
            env.render()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted by user (Ctrl+C)")
        print("🔄 Performing cleanup...")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Display lap time summary for all cars
        print("" + "=" * 60)
        print("📊 MULTI-CAR LAP TIME SUMMARY")
        print("=" * 60)
        
        # Check if any car completed laps
        any_laps_completed = any(len(times) > 0 for times in all_lap_times.values())
        
        if any_laps_completed:
            # Overall summary
            total_all_laps = sum(total_laps.values())
            print(f"🏁 Total laps completed (all cars): {total_all_laps}")
            
            # Find overall best lap time
            all_best_times = [time for time in best_lap_time.values() if time is not None]
            if all_best_times:
                overall_best = min(all_best_times)
                best_car = None
                for car_idx, time in best_lap_time.items():
                    if time == overall_best:
                        best_car = car_idx
                        break
                
                minutes = int(overall_best // 60)
                seconds = overall_best % 60
                best_car_name = car_names[best_car] if best_car < len(car_names) else f"Car #{best_car}"
                print(f"🏆 Overall best lap time: {minutes}:{seconds:06.3f} ({best_car_name})")
            
            # Per-car summary
            print(f"📋 PER-CAR RESULTS:")
            for car_idx in range(num_cars):
                car_times = all_lap_times[car_idx]
                car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car #{car_idx}"
                if car_times:
                    print(f"   🚗 {car_name}:")
                    print(f"      Laps completed: {total_laps[car_idx]}")
                    
                    # Best lap for this car
                    if best_lap_time[car_idx]:
                        minutes = int(best_lap_time[car_idx] // 60)
                        seconds = best_lap_time[car_idx] % 60
                        print(f"      Best lap: {minutes}:{seconds:06.3f}")
                    
                    # Average lap time for this car
                    if car_times:
                        avg_time = sum(car_times) / len(car_times)
                        minutes = int(avg_time // 60)
                        seconds = avg_time % 60
                        print(f"      Average: {minutes}:{seconds:06.3f}")
                    
                    # All lap times for this car
                    print(f"      All times:", end="")
                    for i, lap_time in enumerate(car_times):
                        minutes = int(lap_time // 60)
                        seconds = lap_time % 60
                        if lap_time == best_lap_time[car_idx]:
                            print(f" {minutes}:{seconds:06.3f}⭐", end="")
                        else:
                            print(f" {minutes}:{seconds:06.3f}", end="")
                    print()  # New line
                else:
                    print(f"   🚗 {car_name}: No laps completed")
        else:
            print("❌ No laps completed by any car")
        
        # Display final cumulative impact summary
        print("" + "=" * 60)
        print("💥 CUMULATIVE IMPACT FORCE SUMMARY")
        print("=" * 60)
        
        # Get final cumulative impact forces across all episodes
        if cumulative_impact_across_episodes:
            total_impact = sum(cumulative_impact_across_episodes.values())
            print(f"🏆 Total cumulative impact (all cars): {total_impact:.2f} N⋅s")
            
            # Per-car impact summary
            print(f"📋 PER-CAR IMPACT SUMMARY:")
            for car_idx in range(num_cars):
                car_name = car_names[car_idx] if car_idx < len(car_names) else f"Car #{car_idx}"
                car_impact = cumulative_impact_across_episodes.get(car_idx, 0.0)
                print(f"   🚗 {car_name}: {car_impact:.2f} N⋅s")
        else:
            print("❌ No impact data available")
        
        print("" + "=" * 50)
        
        # Safe environment cleanup
        try:
            env.close()
            print("🔒 Environment closed")
        except Exception as cleanup_error:
            print(f"⚠️ Warning during cleanup: {cleanup_error}")
        
        # Note: We intentionally don't call pygame.quit() here to avoid segfaults
        # The renderer and environment cleanup handle pygame display shutdown
        # pygame.quit() can cause segfaults when called after signal interrupts



if __name__ == "__main__":
    main()