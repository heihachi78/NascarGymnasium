import numpy as np
import sys
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# project src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

# ---------- konfiguráció ----------
num_envs = 8
base_path = "learn/"
verbose = 1
total_timesteps = 10_000_000
eval_freq = 50_000
log_interval = 1
learning_rate_initial_value = 1e-4
learning_rate_final_value = 3e-5
stats_window_size = 25
model_name = "td3_simple"

log_dir = f"./{base_path}logs/{model_name}"
checkpoint_dir = f"./{base_path}checkpoints/{model_name}"
tensorboard_log = f"./{base_path}tensorboard/{model_name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)


# ---------- environment létrehozása ----------
def make_env(rank):
    """
    Függvény, ami visszaadja a CarEnv-et Monitorral,
    a SubprocVecEnv-hez szükséges formátumban.
    """
    def _init():
        env = CarEnv(
            render_mode=None,
            track_file="tracks/nascar.track",
            discrete_action_space=False,
            enable_fps_limit=False,
            reset_on_lap=True,
            disable_cars_on_high_impact=False,
        )
        return Monitor(env, filename=os.path.join(log_dir, f"{model_name}_{rank}"))
    return _init


# ---------- lineáris tanulási ráta ----------
def linear_schedule(initial_value=1e-3, final_value=1e-4):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


# ---------- fő futtatás ----------
if __name__ == "__main__":
    # párhuzamos környezetek
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env("eval")])  # egyszemélyes eval környezet

    # callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=verbose,
    )

    # TD3 modell
    n_actions = env.action_space.shape[-1]

    # single-env noise
    base_action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.25 * np.ones(n_actions)
    )

    # vectorized noise (applies one copy per environment)
    action_noise = VectorizedActionNoise(base_action_noise, n_envs=num_envs)

    model = TD3(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(learning_rate_initial_value, learning_rate_final_value),
        action_noise=action_noise,
        stats_window_size=stats_window_size,
        verbose=verbose,
        policy_delay=4,
        tau=0.0025,
        batch_size=512,
    )

    # tanulás
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=eval_callback,
    )

    # mentés
    model.save(f"{checkpoint_dir}{model_name}_final")
