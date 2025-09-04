import numpy as np
import sys
import os
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from collections import deque
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

num_envs = 8
base_path = "learn/"
verbose = 1
total_timesteps = 25_000_000
eval_freq = 12_500
log_interval = 1
learning_rate_initial_value = 3e-5
learning_rate_final_value = 1e-5
stats_window_size = 25
model_name = "td3"

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
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if len(self.evaluations_results) > 0:
            mean_reward = np.mean(self.evaluations_results[-1])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = f"{checkpoint_dir}/{model_name}_best_reward_{mean_reward:.2f}"
                logger.info(f"New best mean reward: {mean_reward:.2f}, saving model to {model_path}")
                self.model.save(model_path)
        
        return result

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

def linear_schedule(initial_value=1e-4, final_value=1e-5):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule

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
        n_eval_episodes=5,
        verbose=verbose,
    )

    n_actions = train_subproc.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.5 * np.ones(n_actions)
    )

    policy_kwargs = dict(
        net_arch=[1024, 512],
        activation_fn=torch.nn.ReLU,
    )

    model = TD3(
        "MlpPolicy",
        train_subproc,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(learning_rate_initial_value, learning_rate_final_value),
        action_noise=action_noise,
        learning_starts=25_000,
        buffer_size=250_000,
        batch_size=512,
        train_freq=(2, "step"),
        gradient_steps=4,
        policy_delay=4,
        gamma=0.999,
        stats_window_size=stats_window_size,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
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