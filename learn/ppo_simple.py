import sys
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# project src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

# ---------- konfiguráció ----------
num_envs = 8
base_path = "learn/"
verbose = 1
total_timesteps = 25_000_000
eval_freq = 50_000
log_interval = 1
learning_rate_initial_value = 1e-3
learning_rate_final_value = 1e-4
stats_window_size = 25
model_name = "ppo_simple"

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

    policy_kwargs = dict(
        net_arch=[512, 512]
    )

    # PPO modell
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(learning_rate_initial_value, learning_rate_final_value),
        stats_window_size=stats_window_size,
        verbose=verbose,
        batch_size = 128,
        sde_sample_freq = 4,
        use_sde = True,
        policy_kwargs=policy_kwargs,
    )

    # tanulás
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=eval_callback,
    )

    # mentés
    model.save(f"{checkpoint_dir}/{model_name}_final")

'''
CarRacing-v3:
  env_wrapper:
    - rl_zoo3.wrappers.FrameSkip:
        skip: 2
    - rl_zoo3.wrappers.YAMLCompatResizeObservation:
        shape: [64, 64]
    - gymnasium.wrappers.transform_observation.GrayscaleObservation:
        keep_dim: true
  frame_stack: 2
  normalize: "{'norm_obs': False, 'norm_reward': True}"
  n_envs: 8
  n_timesteps: !!float 4e6
  policy: 'CnnPolicy'
  batch_size: 128
  n_steps: 512
  gamma: 0.99
  gae_lambda: 0.95
  n_epochs: 10
  ent_coef: 0.0
  sde_sample_freq: 4
  max_grad_norm: 0.5
  vf_coef: 0.5
  learning_rate: lin_1e-4
  use_sde: True
  clip_range: 0.2
  policy_kwargs: "dict(log_std_init=-2,
                       ortho_init=False,
                       activation_fn=nn.GELU,
                       net_arch=dict(pi=[256], vf=[256]),
                       )"
'''