import numpy as np
import sys
import os
import torch
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

num_envs = 8
base_path = "learn/"
verbose = 1
total_timesteps = 25_000_000
eval_freq = 12_500
log_interval = 1
stats_window_size = 25
model_name = "ppo"

log_dir = f"./{base_path}logs/{model_name}"
checkpoint_dir = f"./{base_path}checkpoints/{model_name}"
tensorboard_log = f"./{base_path}tensorboard/{model_name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BestModelCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_best_mean_reward = -np.inf
        self.best_model_counter = 0
        
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        # Check if evaluation just happened (when n_calls is divisible by eval_freq)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            logger.info("BestModelCallback: Evaluation completed, processing results")
            
            # Use the last_mean_reward from the parent class which is set after each evaluation
            mean_reward = self.last_mean_reward
            logger.info(f"Current evaluation mean reward: {mean_reward:.4f}")
            logger.info(f"Current best mean reward: {self.custom_best_mean_reward:.4f}")
            
            if mean_reward > self.custom_best_mean_reward:
                logger.info(f"NEW BEST! {mean_reward:.4f} > {self.custom_best_mean_reward:.4f}")
                self.custom_best_mean_reward = mean_reward
                self.best_model_counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"{checkpoint_dir}/{model_name}_best_{self.best_model_counter:03d}_{timestamp}_{mean_reward:.4f}.zip"
                logger.info(f"Saving model to {model_path}")
                self.model.save(model_path)
                logger.info(f"Model saved successfully!")
            else:
                logger.info(f"No improvement: {mean_reward:.4f} <= {self.custom_best_mean_reward:.4f}")
        
        return continue_training

def make_env(rank, track_file=None):
    def _init():
        env = CarEnv(
            render_mode=None,
            track_file=track_file,
            discrete_action_space=False,
            reset_on_lap=False,
        )
        return Monitor(env, filename=os.path.join(log_dir, f"{model_name}_{rank}"))
    return _init

if __name__ == "__main__":
    train_subproc = SubprocVecEnv([make_env(i, None) for i in range(num_envs)])
    eval_dummy = DummyVecEnv([make_env("eval", None)])

    eval_callback = BestModelCallback(
        eval_env=eval_dummy,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=verbose,
    )

    model = PPO(
        "MlpPolicy",
        train_subproc,
        tensorboard_log=tensorboard_log,
        stats_window_size=stats_window_size,
        verbose=verbose,
        device='cuda',
    )

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=[eval_callback],
    )

    model.save(f"{checkpoint_dir}/{model_name}_final")
    
    train_subproc.close()
    eval_dummy.close()