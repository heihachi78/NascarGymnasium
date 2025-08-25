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
    
    #env = CarEnv(render_mode="human", track_file="tracks/nascar.track", reset_on_lap=False, num_cars=num_cars, car_names=car_names)
    env = CarEnv(track_file="tracks/trioval.track", 
                 num_cars=num_cars, 
                 reset_on_lap=False, 
                 render_mode="human",
                 car_names=car_names)
    
    
    # Current followed car info
    current_followed_car = 0
    
    try:
        # Reset environment first
        obs, info = env.reset()
        
        # Initialize control variables for each car
        car_throttles = [0.0] * num_cars
        car_brakes = [0.0] * num_cars
        car_steerings = [0.0] * num_cars
        
        for step in range(100000):
            if env.check_quit_requested():
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
                    
                sensors = car_obs[22:30]  # All 8 sensor distances
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
                
                # Convert to new 2-element format: [throttle_brake, steering]
                throttle_brake = car_throttles[car_idx] - car_brakes[car_idx]
                car_actions.append([throttle_brake, car_steerings[car_idx]])
            
            # Create actions array - always use multi-car format
            action = np.array(car_actions, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            if terminated or truncated:
                # Reset for next episode
                obs, info = env.reset()
                
                # Reset control variables for all cars
                car_throttles = [0.0] * num_cars
                car_brakes = [0.0] * num_cars
                car_steerings = [0.0] * num_cars
                    
        
        if env.render_mode == "human":
            env.render()
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()