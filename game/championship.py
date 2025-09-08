"""
Championship Racing Mode.

Multi-track championship with two-stage racing:
1. Time Trial stage: Best lap determines positions and points (15/7/3 points)
2. Competition stage: Race positions determine points (0,1,2,3,5,8,13,21,24,45) + fastest lap bonus (15 points)

Championship tracks multiple race stats and overall standings.
"""

import sys
import os
import numpy as np
import signal
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv
from game.control.base_controller import BaseController
from game.control.td3_control_class import TD3Controller
from game.control.ppo_control_class import PPOController
from game.control.sac_control_class import SACController
from game.control.a2c_control_class import A2CController
from game.control.genetic_controller import GeneticController
from game.control.regression_controller import RegressionController

# Championship configuration constants
LAPS_PER_ATTEMPT = 2  # Time trial laps per attempt
TOTAL_ATTEMPTS = 3    # Time trial attempts
TIME_TRIAL_POINTS = [15, 7, 3]  # 1st, 2nd, 3rd place points
COMPETITION_POINTS = [0, 1, 2, 3, 5, 8, 13, 21, 24, 45]  # Last to first place points
FASTEST_LAP_BONUS = 15  # Bonus points for fastest lap in competition

# Predefined championship tracks
CHAMPIONSHIP_TRACKS = [
    'tracks/daytona.track',
    'tracks/talladega.track', 
    'tracks/martinsville.track',
    'tracks/michigan.track',
    'tracks/trioval.track',
    'tracks/nascar_banked.track',
    'tracks/nascar2.track',
]

class ChampionshipManager:
    """Manages championship state, points, and statistics."""
    
    def __init__(self, car_names):
        self.car_names = car_names
        self.num_cars = len(car_names)
        self.current_track = 0
        
        # Championship standings
        self.championship_points = {i: 0 for i in range(self.num_cars)}
        
        # Statistics tracking
        self.stats = {
            'time_trial_wins': {i: 0 for i in range(self.num_cars)},
            'competition_wins': {i: 0 for i in range(self.num_cars)},
            'time_trial_podiums': {i: {'2nd': 0, '3rd': 0} for i in range(self.num_cars)},
            'competition_podiums': {i: {'2nd': 0, '3rd': 0} for i in range(self.num_cars)},
            'fastest_laps': {i: 0 for i in range(self.num_cars)},
            'track_results': []
        }
        
    def calculate_time_trial_points(self, results):
        """Calculate points for time trial stage based on fastest lap times."""
        # Sort by fastest lap time (None goes to end)
        sorted_results = sorted(results, key=lambda x: x[2] if x[2] is not None else 999999)
        
        points_awarded = {}
        for position, (car_idx, car_name, best_time, _, _, _) in enumerate(sorted_results):
            if position < len(TIME_TRIAL_POINTS) and best_time is not None:
                points = TIME_TRIAL_POINTS[position]
                self.championship_points[car_idx] += points
                points_awarded[car_idx] = points
                
                # Update statistics
                if position == 0:
                    self.stats['time_trial_wins'][car_idx] += 1
                elif position == 1:
                    self.stats['time_trial_podiums'][car_idx]['2nd'] += 1
                elif position == 2:
                    self.stats['time_trial_podiums'][car_idx]['3rd'] += 1
            else:
                points_awarded[car_idx] = 0
                
        return points_awarded, sorted_results
    
    def calculate_competition_points(self, finishing_order, fastest_lap_car):
        """Calculate points for competition stage."""
        points_awarded = {}
        
        # Award finishing position points
        for position, (car_idx, car_name, _, _) in enumerate(finishing_order):
            # Points array is indexed from last place (0) to first place
            # Position 0 = 1st place = highest points
            points_idx = min(position, len(COMPETITION_POINTS) - 1)
            points = COMPETITION_POINTS[-(points_idx + 1)]  # Reverse index
            
            self.championship_points[car_idx] += points
            points_awarded[car_idx] = points
            
            # Update statistics  
            if position == 0:
                self.stats['competition_wins'][car_idx] += 1
            elif position == 1:
                self.stats['competition_podiums'][car_idx]['2nd'] += 1
            elif position == 2:
                self.stats['competition_podiums'][car_idx]['3rd'] += 1
        
        # Award fastest lap bonus
        if fastest_lap_car is not None:
            self.championship_points[fastest_lap_car] += FASTEST_LAP_BONUS
            if fastest_lap_car in points_awarded:
                points_awarded[fastest_lap_car] += FASTEST_LAP_BONUS
            else:
                points_awarded[fastest_lap_car] = FASTEST_LAP_BONUS
            self.stats['fastest_laps'][fastest_lap_car] += 1
            
        return points_awarded
    
    def get_championship_standings(self):
        """Get current championship standings sorted by points."""
        standings = []
        for car_idx in range(self.num_cars):
            standings.append((
                car_idx,
                self.car_names[car_idx], 
                self.championship_points[car_idx]
            ))
        
        # Sort by points (descending)
        standings.sort(key=lambda x: x[2], reverse=True)
        return standings
    
    def display_standings(self, title="CHAMPIONSHIP STANDINGS"):
        """Display current championship standings."""
        print(f"\n{'=' * 60}")
        print(f"🏆 {title}")
        print(f"{'=' * 60}")
        
        standings = self.get_championship_standings()
        for position, (car_idx, car_name, points) in enumerate(standings, 1):
            position_emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            print(f"   {position_emoji} {car_name}: {points} points")
    
    def display_final_statistics(self):
        """Display comprehensive championship statistics."""
        print(f"\n{'=' * 60}")
        print(f"📊 CHAMPIONSHIP STATISTICS")
        print(f"{'=' * 60}")
        
        standings = self.get_championship_standings()
        
        for position, (car_idx, car_name, points) in enumerate(standings, 1):
            print(f"\n{position}. {car_name} - {points} points")
            
            tt_wins = self.stats['time_trial_wins'][car_idx]
            comp_wins = self.stats['competition_wins'][car_idx]
            tt_2nd = self.stats['time_trial_podiums'][car_idx]['2nd']
            tt_3rd = self.stats['time_trial_podiums'][car_idx]['3rd']
            comp_2nd = self.stats['competition_podiums'][car_idx]['2nd']
            comp_3rd = self.stats['competition_podiums'][car_idx]['3rd']
            fastest_laps = self.stats['fastest_laps'][car_idx]
            
            print(f"   🏁 Time Trial: {tt_wins} wins, {tt_2nd} 2nd places, {tt_3rd} 3rd places")
            print(f"   🏁 Competition: {comp_wins} wins, {comp_2nd} 2nd places, {comp_3rd} 3rd places")
            print(f"   ⚡ Fastest Laps: {fastest_laps}")

def calculate_time_trial_results(num_cars, car_names, all_lap_times, best_lap_times, disabled_cars):
    """Calculate time trial results (adapted from time_trial.py)."""
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
    
    return results

def calculate_competition_finishing_order(num_cars, car_names, lap_counts, best_lap_time, car_rewards, disabled_cars, finishing_times):
    """Calculate competition finishing order (adapted from competition.py)."""
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

def load_genetic_controller(pkl_path, name):
    """Load a trained genetic controller from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            controller = pickle.load(f)
        controller.name = name
        return controller, True
    except Exception as e:
        print(f"⚠️ Failed to load genetic controller {pkl_path}: {e}")
        return GeneticController(name=name), False

def signal_handler(_, __):
    """Immediately exit on interrupt to avoid segfault."""
    print("\n⚠️ Interrupt received...")
    os._exit(0)

def run_time_trial_stage(env, controllers, car_names, championship_manager):
    """Run time trial stage for current track."""
    num_cars = len(car_names)
    
    print(f"\n{'=' * 60}")
    print(f"⏱️ TIME TRIAL STAGE")
    print(f"{'=' * 60}")
    print(f"📋 RULES:")
    print(f"   • {TOTAL_ATTEMPTS} attempts of {LAPS_PER_ATTEMPT} laps each")
    print(f"   • Fastest single lap time determines position")
    print(f"   • Points: 1st={TIME_TRIAL_POINTS[0]}, 2nd={TIME_TRIAL_POINTS[1]}, 3rd={TIME_TRIAL_POINTS[2]}")
    print(f"={'=' * 60}")
    
    # Time trial data tracking
    all_lap_times = {}
    best_lap_times = {}
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = {}
        best_lap_times[i] = None
    
    # Run time trial attempts
    for attempt_num in range(1, TOTAL_ATTEMPTS + 1):
        print(f"\n🏁 TIME TRIAL ATTEMPT {attempt_num} OF {TOTAL_ATTEMPTS}")
        print(f"{'=' * 40}")
        
        # Reset environment for new attempt
        obs, info = env.reset(attempt_num * 42)
        
        # Per-attempt tracking
        attempt_lap_counts = {}
        attempt_lap_times = {}
        cars_finished_attempt = set()
        previous_lap_counts = {}
        lap_timing_started = {}
        lap_start_rewards = {}
        car_rewards = {}
        
        # Initialize per-attempt tracking
        for i in range(num_cars):
            attempt_lap_counts[i] = 0
            attempt_lap_times[i] = []
            previous_lap_counts[i] = 0
            lap_timing_started[i] = False
            lap_start_rewards[i] = 0.0
            car_rewards[i] = 0.0
            all_lap_times[i][attempt_num] = []
        
        simulation_step = 0
        
        # Run current attempt
        while True:
            if env.check_quit_requested():
                print(f"   User requested quit")
                return None
            
            # Calculate controls for each car
            car_actions = []
            for car_idx in range(num_cars):
                if car_idx in env.disabled_cars or car_idx in cars_finished_attempt:
                    car_actions.append([0.0, 0.0])
                    continue
                
                # Get observation for this car
                if isinstance(obs, np.ndarray) and len(obs.shape) == 2:
                    car_obs = obs[car_idx]
                else:
                    car_obs = obs
                
                controller = controllers[car_idx]
                action = controller.control(car_obs)
                car_actions.append(action)
            
            # Create actions array
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
            
            # Check lap completions
            if isinstance(info, dict) and "cars" in info:
                for car_idx in range(min(num_cars, len(info["cars"]))):
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
                            attempt_lap_counts[car_idx] += 1
                            attempt_lap_times[car_idx].append(last_lap_time)
                            all_lap_times[car_idx][attempt_num].append(last_lap_time)
                            
                            lap_start_rewards[car_idx] = car_rewards[car_idx]
                            
                            # Format lap time
                            minutes = int(last_lap_time // 60)
                            seconds = last_lap_time % 60
                            lap_time_str = f"{minutes}:{seconds:06.3f}"
                            
                            # Update best lap for this car
                            if best_lap_times[car_idx] is None or last_lap_time < best_lap_times[car_idx]:
                                best_lap_times[car_idx] = last_lap_time
                                print(f"🏆 {car_name} NEW BEST! Lap {attempt_lap_counts[car_idx]}/{LAPS_PER_ATTEMPT}: {lap_time_str}")
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
            
            env.render()
            
            # Check if all cars have finished or disabled
            if len(cars_finished_attempt) >= num_cars:
                break
            
            # Safety timeout
            if simulation_step > 50000:
                print(f"⚠️ Attempt timeout reached")
                break
    
    # Calculate time trial results
    results = calculate_time_trial_results(num_cars, car_names, all_lap_times, best_lap_times, env.disabled_cars)
    points_awarded, sorted_results = championship_manager.calculate_time_trial_points(results)
    
    # Display time trial results
    print(f"\n{'=' * 60}")
    print(f"⏱️ TIME TRIAL RESULTS")
    print(f"{'=' * 60}")
    
    for position, (car_idx, car_name, _, _, time_str, status) in enumerate(sorted_results, 1):
        position_emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
        points = points_awarded.get(car_idx, 0)
        print(f"   {position_emoji} {car_name}: {time_str} - {status} ({points} pts)")
    
    championship_manager.display_standings("STANDINGS AFTER TIME TRIAL")
    
    return sorted_results

def run_competition_stage(env, controllers, car_names, championship_manager):
    """Run competition stage for current track."""
    num_cars = len(car_names)
    
    print(f"\n{'=' * 60}")
    print(f"🏁 COMPETITION STAGE")
    print(f"{'=' * 60}")
    print(f"📋 RULES:")
    print(f"   • Race until completion or timeout")
    print(f"   • Points by finish position: {COMPETITION_POINTS}")
    print(f"   • Fastest lap bonus: +{FASTEST_LAP_BONUS} points")
    print(f"{'=' * 60}")
    
    # Competition tracking
    all_lap_times = {}
    best_lap_time = {}
    previous_lap_count = {}
    finishing_times = {}
    car_rewards = {}
    lap_start_rewards = {}
    lap_timing_started = {}
    
    # Initialize tracking for all cars
    for i in range(num_cars):
        all_lap_times[i] = []
        best_lap_time[i] = None
        previous_lap_count[i] = 0
        finishing_times[i] = None
        car_rewards[i] = 0.0
        lap_start_rewards[i] = 0.0
        lap_timing_started[i] = False
    
    # Reset environment for competition
    obs, info = env.reset(seed=999)
    print(f"🏁 COMPETITION STARTED!")
    print("=" * 60)
    
    overall_best_lap_time = None
    fastest_lap_car = None
    
    # Competition main loop
    for step in range(100000):
        if env.check_quit_requested():
            print(f"   User requested quit at step {step}")
            break
        
        # Calculate individual controls for each car
        car_actions = []
        
        for car_idx in range(num_cars):
            if isinstance(obs, np.ndarray) and len(obs.shape) == 2:
                car_obs = obs[car_idx]
            else:
                car_obs = obs
            
            controller = controllers[car_idx]
            action = controller.control(car_obs)
            car_actions.append(action)
        
        # Create actions array
        if num_cars > 1:
            action = np.array(car_actions, dtype=np.float32)
        else:
            action = np.array(car_actions[0], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Handle multi-car rewards
        if isinstance(reward, np.ndarray):
            for car_idx in range(min(num_cars, len(reward))):
                car_rewards[car_idx] += reward[car_idx]
        else:
            car_rewards[0] += reward
        
        # Handle lap completions
        if isinstance(info, dict) and "cars" in info:
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
                        if best_lap_time[car_idx] is None or last_lap_time < best_lap_time[car_idx]:
                            best_lap_time[car_idx] = last_lap_time
                            print(f"🏁 {car_name} Lap {current_lap_count}: {lap_time_str} (NEW PB!) | Reward: {lap_reward:.1f}")
                        else:
                            print(f"🏁 {car_name} Lap {current_lap_count}: {lap_time_str} | Reward: {lap_reward:.1f}")
                        
                        # Check overall best lap
                        if overall_best_lap_time is None or last_lap_time < overall_best_lap_time:
                            overall_best_lap_time = last_lap_time
                            fastest_lap_car = car_idx
                            print(f"   🌟 NEW FASTEST LAP by {car_name}!")
                    
                    previous_lap_count[car_idx] = current_lap_count
        
        env.render()
        
        if terminated or truncated:
            # Competition finished
            break
    
    # Calculate competition results
    final_lap_counts = {}
    if isinstance(info, dict) and "cars" in info:
        for car_idx in range(min(num_cars, len(info["cars"]))):
            lap_timing = info["cars"][car_idx].get('lap_timing', {})
            final_lap_counts[car_idx] = lap_timing.get('lap_count', 0)
    else:
        lap_timing = info.get('lap_timing', {})
        final_lap_counts[0] = lap_timing.get('lap_count', 0)
    
    finishing_order = calculate_competition_finishing_order(
        num_cars, car_names, final_lap_counts, best_lap_time, 
        car_rewards, env.disabled_cars, finishing_times
    )
    
    points_awarded = championship_manager.calculate_competition_points(finishing_order, fastest_lap_car)
    
    # Display competition results
    print(f"\n{'=' * 60}")
    print(f"🏁 COMPETITION RESULTS")
    print(f"{'=' * 60}")
    
    for position, (car_idx, car_name, status, details) in enumerate(finishing_order, 1):
        position_emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
        points = points_awarded.get(car_idx, 0)
        print(f"   {position_emoji} {car_name} - {status} ({points} pts)")
        if details:
            print(f"      {details}")
    
    # Show fastest lap bonus
    if fastest_lap_car is not None:
        fastest_name = car_names[fastest_lap_car]
        minutes = int(overall_best_lap_time // 60)
        seconds = overall_best_lap_time % 60
        fastest_time_str = f"{minutes}:{seconds:06.3f}"
        print(f"\n⚡ Fastest Lap: {fastest_name} - {fastest_time_str} (+{FASTEST_LAP_BONUS} pts)")
    
    championship_manager.display_standings("STANDINGS AFTER COMPETITION")
    
    return finishing_order

def main():
    # Install signal handler for immediate exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🏆 CHAMPIONSHIP MODE")
    print("=" * 60)
    print(f"📋 CHAMPIONSHIP FORMAT:")
    print(f"   • {len(CHAMPIONSHIP_TRACKS)} tracks in championship")
    print(f"   • 2 stages per track: Time Trial + Competition")
    print(f"   • Time Trial points: {TIME_TRIAL_POINTS}")
    print(f"   • Competition points: ascending from 0 to {max(COMPETITION_POINTS)}")
    print(f"   • Fastest lap bonus: +{FASTEST_LAP_BONUS} points")
    print("=" * 60)
    
    # Configure which models to compete
    model_configs = [
        ("game/control/models/a2c_best_model3.zip", "A2C-B-3"),
        ("game/control/models/td3_bm1.zip", "TD3-BM-1"),
        ("game/control/models/td3_bm2.zip", "TD3-BM-2"),
        ("game/control/models/td3_bm3.zip", "TD3-BM-3"),
        ("game/control/models/sac_bm3.zip", "SAC-BM-3"),
        ("game/control/models/sac_final.zip", "SAC-F"),
        ("game/control/models/genetic.pkl", "GA"),
        (None, "BC"),
        ("game/control/models/ppo_789.zip", "PPO-789"),
        ("game/control/models/ppo_849.zip", "PPO-849"),
    ]
    
    # Take only the first 10 models (environment supports max 10 cars)
    num_cars = len(model_configs)
    
    # Extract car names from configs
    car_names = [config[1] for config in model_configs]
    
    print(f"📊 Championship Setup: {num_cars} cars")
    print("=" * 60)
    
    # Create championship manager
    championship_manager = ChampionshipManager(car_names)
    
    # Create controllers for each car
    controllers = []
    for i, (model_path, name) in enumerate(model_configs):
        print(f"Car {i}: {name}")
        if model_path is None:
            print(f"   → Using rule-based control")
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
        elif "genetic" in model_path.lower() or "genetic" in name.lower():
            print(f"   → Loading Genetic controller: {model_path}")
            controller, loaded = load_genetic_controller(model_path, name)
            controllers.append(controller)
            if loaded:
                print(f"   ✓ Genetic controller loaded successfully")
                info = controller.get_info()
                genome_length = info.get('genome_length', 'unknown')
                print(f"   📊 Genome parameters: {genome_length}")
            else:
                print(f"   ⚠ Using fallback genetic control")
        else:
            controller = BaseController(model_path, name)
            controllers.append(controller)
            info = controller.get_info()
            if info['model_loaded']:
                print(f"   ✓ Base control")
            else:
                print(f"   ⚠ Using fallback control")
    
    print("=" * 60)
    
    try:
        # Run championship across all tracks
        for track_idx, track_file in enumerate(CHAMPIONSHIP_TRACKS, 1):
            print(f"\n{'=' * 80}")
            print(f"🏁 ROUND {track_idx} OF {len(CHAMPIONSHIP_TRACKS)}: {track_file}")
            print(f"{'=' * 80}")
            
            # Create environment for this track
            env = CarEnv(track_file=track_file,
                        num_cars=num_cars,
                        reset_on_lap=False,
                        render_mode=None,  # Set to "human" to see visualization
                        discrete_action_space=False,
                        car_names=car_names)
            env.seed(42 + track_idx)
            
            # Run time trial stage
            time_trial_results = run_time_trial_stage(env, controllers, car_names, championship_manager)
            if time_trial_results is None:  # User quit
                return
            
            # Run competition stage
            competition_results = run_competition_stage(env, controllers, car_names, championship_manager)
            
            # Store track results
            championship_manager.stats['track_results'].append({
                'track': track_file,
                'time_trial': time_trial_results,
                'competition': competition_results
            })
        
        # Display final championship results
        print(f"\n{'=' * 80}")
        print(f"🏆 CHAMPIONSHIP COMPLETE!")
        print(f"{'=' * 80}")
        
        championship_manager.display_standings("FINAL CHAMPIONSHIP STANDINGS")
        championship_manager.display_final_statistics()
        
        print(f"\n{'=' * 80}")
        print("🏆 CHAMPIONSHIP FINISHED!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Championship interrupted by user (Ctrl+C)")
        print("🔄 Skipping...")
        os._exit(0)
    except Exception as e:
        print(f"❌ Error during championship: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("🔒 Done.")

if __name__ == "__main__":
    main()