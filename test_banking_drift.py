#!/usr/bin/env python3
"""
Test script to compare car behavior on flat tracks vs banked tracks.

This script tests the banking system fixes to ensure:
1. Flat tracks don't apply unwanted banking forces
2. Banked tracks still work correctly 
3. Drift behavior is appropriate for each track type
"""

import sys
import os
import time
import math
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.car_env import CarEnv


def run_test_scenario(track_file: str, test_duration: float = 5.0, actions_per_test: int = 5):
    """Run a test scenario on a specific track and collect drift metrics."""
    print(f"\n--- Testing {track_file} ---")
    
    # Load track and create environment
    track_path = os.path.join("tracks", track_file)
    
    try:
        env = CarEnv(track_file=track_path, render_mode=None)
        # Reset environment
        obs, _ = env.reset()
    except FileNotFoundError:
        print(f"Track file not found: {track_path}")
        return None
    except Exception as e:
        print(f"Error loading track {track_path}: {e}")
        return None
    
    # Test different scenarios
    results = {}
    test_scenarios = [
        ("straight_line", np.array([0.8, 0.0], dtype=np.float32)),   # Throttle, no steering
        ("gentle_turn", np.array([0.5, 0.3], dtype=np.float32)),     # Throttle with gentle turn
        ("sharp_turn", np.array([0.3, 0.8], dtype=np.float32)),      # Light throttle with sharp turn
        ("braking_turn", np.array([-0.5, 0.5], dtype=np.float32)),   # Braking while turning (negative throttle_brake)
    ]
    
    for scenario_name, base_action in test_scenarios:
        print(f"  Testing scenario: {scenario_name}")
        
        # Reset for each scenario
        obs, _ = env.reset()
        
        # Collect metrics
        positions = []
        velocities = []
        slip_angles = []
        banking_angles = []
        
        steps = int(test_duration / 0.016)  # ~60 FPS
        
        for step in range(steps):
            # Use base action with some variation
            action = base_action.copy()
            
            # Apply action and step
            obs, reward, done, truncated, info = env.step(action)
            
            # Collect data (using first car's physics world)
            car_state = env.car_physics_worlds[0].get_car_state()
            if car_state:
                x, y, angle, vx, vy, angular_vel = car_state
                positions.append((x, y))
                velocities.append((vx, vy))
                
                # Calculate slip angle
                speed = math.sqrt(vx*vx + vy*vy)
                if speed > 1.0:  # Only calculate if moving fast enough
                    velocity_angle = math.atan2(vy, vx)
                    slip_angle = abs(velocity_angle - angle)
                    if slip_angle > math.pi:
                        slip_angle = 2*math.pi - slip_angle
                    slip_angles.append(math.degrees(slip_angle))
                
                # Get banking angle if available
                car_position = (x, y)
                physics_world = env.car_physics_worlds[0]
                if hasattr(physics_world, 'get_banking_angle_at_position'):
                    banking_angle = physics_world.get_banking_angle_at_position(car_position)
                    banking_angles.append(banking_angle)
            
            if done or truncated:
                break
        
        # Calculate metrics
        results[scenario_name] = {
            "avg_slip_angle": sum(slip_angles) / len(slip_angles) if slip_angles else 0.0,
            "max_slip_angle": max(slip_angles) if slip_angles else 0.0,
            "avg_banking_angle": sum(banking_angles) / len(banking_angles) if banking_angles else 0.0,
            "max_banking_angle": max(banking_angles, key=abs) if banking_angles else 0.0,
            "position_drift": calculate_position_drift(positions) if len(positions) > 10 else 0.0,
            "samples": len(positions)
        }
        
        print(f"    Avg slip angle: {results[scenario_name]['avg_slip_angle']:.2f}°")
        print(f"    Max slip angle: {results[scenario_name]['max_slip_angle']:.2f}°")
        print(f"    Avg banking: {results[scenario_name]['avg_banking_angle']:.2f}°")
        print(f"    Position drift: {results[scenario_name]['position_drift']:.2f}m")
    
    env.close()
    return results


def calculate_position_drift(positions):
    """Calculate how much the car drifted from its intended path."""
    if len(positions) < 10:
        return 0.0
    
    # Calculate the intended path (start to end)
    start_pos = positions[0]
    end_pos = positions[-1]
    
    # Calculate perpendicular distance from each point to the straight line
    total_drift = 0.0
    
    for pos in positions[1:-1]:  # Skip start and end points
        # Distance from point to line formula
        x0, y0 = pos
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Line segment length
        line_length_sq = (x2-x1)**2 + (y2-y1)**2
        
        if line_length_sq == 0:
            drift = math.sqrt((x0-x1)**2 + (y0-y1)**2)
        else:
            # Project point onto line
            t = max(0, min(1, ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1)) / line_length_sq))
            proj_x = x1 + t*(x2-x1)
            proj_y = y1 + t*(y2-y1)
            drift = math.sqrt((x0-proj_x)**2 + (y0-proj_y)**2)
        
        total_drift += drift
    
    return total_drift / max(1, len(positions) - 2)


def main():
    """Main test function."""
    print("Banking Drift Test - Comparing flat vs banked tracks")
    print("=" * 60)
    
    # Test tracks
    test_tracks = [
        "nascar.track",      # Flat track (no banking defined)
        "banked_oval.track"  # Banked track (with banking defined)
    ]
    
    all_results = {}
    
    for track in test_tracks:
        results = run_test_scenario(track, test_duration=3.0)
        if results:
            all_results[track] = results
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if len(all_results) >= 2:
        flat_track = "nascar.track"
        banked_track = "banked_oval.track"
        
        if flat_track in all_results and banked_track in all_results:
            print(f"\nComparing {flat_track} (flat) vs {banked_track} (banked):")
            
            for scenario in ["straight_line", "gentle_turn", "sharp_turn", "braking_turn"]:
                if scenario in all_results[flat_track] and scenario in all_results[banked_track]:
                    flat_slip = all_results[flat_track][scenario]["avg_slip_angle"]
                    banked_slip = all_results[banked_track][scenario]["avg_slip_angle"]
                    flat_drift = all_results[flat_track][scenario]["position_drift"]
                    banked_drift = all_results[banked_track][scenario]["position_drift"]
                    
                    print(f"\n  {scenario.replace('_', ' ').title()}:")
                    print(f"    Flat track  - Slip: {flat_slip:.2f}°, Drift: {flat_drift:.2f}m")
                    print(f"    Banked track - Slip: {banked_slip:.2f}°, Drift: {banked_drift:.2f}m")
                    
                    # Analysis
                    slip_diff = banked_slip - flat_slip
                    drift_diff = banked_drift - flat_drift
                    print(f"    Difference  - Slip: {slip_diff:+.2f}°, Drift: {drift_diff:+.2f}m")
                    
                    # Expected behavior
                    if "turn" in scenario:
                        if slip_diff < -1.0:  # Banking should reduce slip in turns
                            print("    ✓ Banking correctly reduces slip angle in turns")
                        else:
                            print("    ⚠ Banking may not be working effectively")
    else:
        print("Insufficient data for comparison")
    
    print(f"\n{'='*60}")
    print("Test completed. If flat tracks show significant unwanted drift,")
    print("there may still be banking force issues to address.")


if __name__ == "__main__":
    main()