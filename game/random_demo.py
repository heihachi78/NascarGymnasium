"""
Random action demonstration.

This demo runs the car with random actions within proper ranges.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv


discrete_action_space = False


def main():
    print("=" * 50)
    
    env = CarEnv(render_mode="human", 
                 track_file="tracks/nascar.track", 
                 reset_on_lap=False,
                 discrete_action_space=discrete_action_space,
                 disable_cars_on_high_impact=True,
                 enable_fps_limit=False)
    
    try:
        # Reset environment first
        obs, info = env.reset()
        print("\n🚗 Running simulation with random actions...")
        total_reward = 0.0
        
        for step in range(100000):
            if env.check_quit_requested():
                print(f"   User requested quit at step {step}")
                break
            
            if discrete_action_space:
                action = env.action_space.sample()
            else:
                action = np.array(env.action_space.sample(), dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                print(f"   Episode terminated at step {step}, total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0.0
        
        if env.render_mode == "human":
            env.render()
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted by user (Ctrl+C)")
        print("🔄 Performing cleanup...")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
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
