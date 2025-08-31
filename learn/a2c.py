# optimized_a2c_curriculum.py
import os
import sys
import logging
from collections import deque
from typing import Optional, List

import numpy as np
import torch

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# project src import - ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv  # your environment implementation

# ---------- configuration ----------
num_envs = 8
base_path = "learn"
verbose = 1
total_timesteps = 25_000_000
eval_freq = 25_000
log_interval = 1_000
learning_rate_initial_value = 7.5e-4
learning_rate_final_value = 2.5e-4
stats_window_size = 25
model_name = "a2c_optimized"
seed = 12345

# curriculum
curriculum_reward_threshold = 200.0
curriculum_eval_window = 50
nascar_track = "tracks/nascar.track"

# directories
log_dir = f"./{base_path}/logs/{model_name}"
checkpoint_dir = f"./{base_path}/checkpoints/{model_name}"
tensorboard_log = f"./{base_path}/tensorboard/{model_name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# reproducibility
np.random.seed(seed)
torch.manual_seed(seed)


# ---------- utility: environment factory ----------
def make_env(rank, track_file=None, seed_offset: int = 0):
    """
    Return a callable that creates a Monitor-wrapped CarEnv.
    SubprocVecEnv requires a function with zero args.
    """
    def _init():
        env = CarEnv(
            render_mode=None,
            track_file=track_file,
            discrete_action_space=False,
            reset_on_lap=True,
        )
        # set environment seed if available
        try:
            env.seed(seed + rank + seed_offset)
        except Exception:
            pass
        return Monitor(env, filename=os.path.join(log_dir, f"{model_name}_{rank}"))
    return _init


def create_subproc_vecenv(phase: str, num_envs: int) -> SubprocVecEnv:
    if phase == "nascar":
        return SubprocVecEnv([make_env(i, nascar_track) for i in range(num_envs)])
    else:
        return SubprocVecEnv([make_env(i, None) for i in range(num_envs)])


# ---------- linear schedule ----------
def linear_schedule(initial_value=9e-4, final_value=3e-4):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


# ---------- Curriculum Learning Callback ----------
class CurriculumLearningCallback(BaseCallback):
    """
    Decides when to switch from NASCAR to Random tracks.
    Preferred switch: in-place via env_method('switch_to_random').
    Fallback: recreate the VecEnv but preserve VecNormalize stats.
    """
    def __init__(self,
                 reward_threshold: float = 200.0,
                 eval_window: int = 50,
                 num_envs: int = 8,
                 verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.eval_window = eval_window
        self.num_envs = num_envs
        self.reward_history = deque(maxlen=eval_window)
        self.phase = "nascar"
        self.phase_switched = False
        self.eval_callback: Optional[CurriculumEvalCallback] = None  # will be set externally
        self.post_switch_lr_factor = 0.6
        self.post_switch_steps = 0
        self._post_switch_lr_decay_steps = 2_000_000  # optional track, not enforced by SB3 schedule

    def _on_step(self) -> bool:
        # count steps after switch (for potential future scheduling/heuristics)
        if self.phase == "random" and self.phase_switched:
            self.post_switch_steps += 1
        return True

    def _on_training_start(self):
        logger.info(f"Starting curriculum - initial phase: {self.phase}")

    def update_reward_history(self, mean_reward: float):
        self.reward_history.append(mean_reward)
        # Consider decision only if we have enough samples to be stable
        min_required = 8
        if self.phase == "nascar" and len(self.reward_history) >= min_required and not self.phase_switched:
            recent = list(self.reward_history)[-min_required:]
            median = float(np.median(recent))
            std = float(np.std(recent))
            logger.info(f"Curriculum eval stats (last {min_required}): median={median:.1f}, std={std:.2f}")

            # require both median >= threshold and limited relative std
            if (median >= self.reward_threshold) and (std <= 0.18 * max(1.0, abs(median))):
                logger.info(f"🎉 Curriculum condition satisfied (median {median:.1f} >= {self.reward_threshold}); switching phases.")
                self.phase = "random"
                self.phase_switched = True
                self._switch_environments()

    def _switch_environments(self):
        """
        Try to switch environments in-place using env_method('switch_to_random').
        If not supported, recreate VecEnv but preserve VecNormalize statistics.
        """
        # Access model through eval_callback since this callback is not directly passed to model.learn()
        if self.eval_callback is None or self.eval_callback.model is None:
            logger.error("Cannot switch environments - eval_callback or eval_callback.model is None")
            logger.error(f"Callback state - has eval_callback: {self.eval_callback is not None}")
            if self.eval_callback:
                logger.error(f"Eval callback has model: {hasattr(self.eval_callback, 'model')}")
            logger.error(f"Callback type: {type(self)}")
            return
        
        model = self.eval_callback.model

        logger.info("Attempting to switch training environments to RANDOM tracks (in-place preferred)")

        train_env = model.get_env()
        eval_env = getattr(self.eval_callback, "eval_env", None)

        # If the env is VecNormalize, we want to handle its internal wrapped env
        try:
            # Attempt in-place switch across all sub-environments
            # This expects that underlying CarEnv implements switch_to_random()
            logger.info("🔄 Attempting in-place environment switch via env_method...")
            
            # For VecNormalize wrapped environments, we need to call on the underlying env
            if hasattr(train_env, 'venv'):
                # VecNormalize wraps the actual VecEnv in .venv
                result = train_env.venv.env_method("switch_to_random")
                logger.info(f"✅ Training env switch_to_random called on {len(result) if result else 0} sub-environments")
            else:
                # Direct call if not VecNormalize
                result = train_env.env_method("switch_to_random")
                logger.info(f"✅ Training env switch_to_random called on {len(result) if result else 0} sub-environments")
            
            if eval_env is not None:
                try:
                    if hasattr(eval_env, 'venv'):
                        eval_result = eval_env.venv.env_method("switch_to_random")
                        logger.info(f"✅ Eval env switch_to_random called on {len(eval_result) if eval_result else 0} sub-environments")
                    else:
                        eval_result = eval_env.env_method("switch_to_random")
                        logger.info(f"✅ Eval env switch_to_random called on {len(eval_result) if eval_result else 0} sub-environments")
                except Exception as ee:
                    logger.warning(f"⚠️  Eval env switch failed: {ee}")
                    
            # reduce LR temporarily to stabilize
            self._dampen_learning_rate(model, factor=self.post_switch_lr_factor)
            logger.info("🏁 In-place environment switch completed successfully!")
            return
        except Exception as e:
            logger.warning(f"❌ In-place env switch via env_method failed: {e}")

        # FALLBACK: recreate VecEnv but preserve VecNormalize stats if present
        logger.info("Fallback: Recreating VecEnv while preserving VecNormalize statistics if possible.")
        try:
            # If training env is VecNormalize, extract inner env and stats
            train_is_vecnorm = isinstance(train_env, VecNormalize)
            if train_is_vecnorm:
                old_vecnorm: VecNormalize = train_env
                old_stats = {
                    "obs_rms": getattr(old_vecnorm, "obs_rms", None),
                    "ret_rms": getattr(old_vecnorm, "ret_rms", None),
                    "num_timesteps": getattr(old_vecnorm, "num_timesteps", None),
                }
                inner_phase = "random"
                # create new subproc vecenv with same number of envs
                new_subproc = create_subproc_vecenv(inner_phase, self.num_envs)
                new_vecnorm = VecNormalize(new_subproc, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
                # copy stats if available
                if old_stats["obs_rms"] is not None:
                    new_vecnorm.obs_rms = old_stats["obs_rms"]
                if old_stats["ret_rms"] is not None:
                    new_vecnorm.ret_rms = old_stats["ret_rms"]
                if old_stats["num_timesteps"] is not None:
                    new_vecnorm.num_timesteps = old_stats["num_timesteps"]
                model.set_env(new_vecnorm)
                logger.info("✅ Training VecEnv recreated and VecNormalize stats restored.")
            else:
                # not VecNormalize: create and set new SubprocVecEnv
                new_subproc = create_subproc_vecenv("random", self.num_envs)
                model.set_env(new_subproc)
                logger.info("✅ Training VecEnv recreated (non-VecNormalize).")

            # Update eval env similarly (best-effort)
            if eval_env is not None:
                try:
                    eval_is_vecnorm = isinstance(eval_env, VecNormalize)
                    if eval_is_vecnorm:
                        # create new eval DummyVecEnv & VecNormalize then copy stats from train if possible
                        new_eval = DummyVecEnv([make_env("eval", None)])
                        new_eval_vecnorm = VecNormalize(new_eval, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
                        if train_is_vecnorm:
                            new_eval_vecnorm.obs_rms = new_vecnorm.obs_rms
                            new_eval_vecnorm.ret_rms = new_vecnorm.ret_rms
                        self.eval_callback.eval_env = new_eval_vecnorm
                        logger.info("✅ Eval VecEnv recreated and stats synced.")
                    else:
                        self.eval_callback.eval_env = DummyVecEnv([make_env("eval", None)])
                        logger.info("✅ Eval env recreated (non-VecNormalize).")
                except Exception as ee:
                    logger.warning(f"Could not fully recreate eval env: {ee}")

            # Dampen LR to help with value shock
            self._dampen_learning_rate(model, factor=self.post_switch_lr_factor)
            logger.info("🏁 Environment switch (fallback) completed.")
        except Exception as e:
            logger.error(f"❌ Failed to switch environments: {e}")
            logger.info("🔄 Continuing with current environments.")

    def _dampen_learning_rate(self, model, factor=0.6):
        # Multiply optimizer LR by factor (applies to policy optimizer groups)
        try:
            opt = model.policy.optimizer
            for pg in opt.param_groups:
                if "lr" in pg:
                    pg["lr"] = pg["lr"] * factor
            logger.info(f"🔧 Reduced optimizer LR by factor {factor:.2f}")
        except Exception as e:
            logger.warning(f"Could not dampen learning rate: {e}")

    def get_current_phase(self):
        return self.phase


# ---------- Eval callback that informs curriculum ----------
class CurriculumEvalCallback(EvalCallback):
    def __init__(self, curriculum_callback: CurriculumLearningCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_callback = curriculum_callback
        # link back so curriculum can update this eval env
        self.curriculum_callback.eval_callback = self

    def _on_step(self) -> bool:
        prev_eval_count = len(self.evaluations_results)
        result = super()._on_step()
        if len(self.evaluations_results) > prev_eval_count:
            # latest evaluation was recorded
            latest_eval_results = self.evaluations_results[-1]  # list of episode rewards for that eval
            latest_mean_reward = float(np.mean(latest_eval_results))
            self.curriculum_callback.update_reward_history(latest_mean_reward)

            # logging summary of recent reward history for visibility
            if len(self.curriculum_callback.reward_history) >= 5:
                recent = list(self.curriculum_callback.reward_history)[-5:]
                cur_mean = float(np.mean(recent))
                logger.info(f"Eval progress: {cur_mean:.1f} avg (last 5) | Recent: {[f'{r:.1f}' for r in recent]} | Phase: {self.curriculum_callback.get_current_phase()}")

        return result


# ---------- main ----------
if __name__ == "__main__":
    logger.info("A2C Curriculum training (optimized) starting")
    logger.info(f"Model: {model_name} | total_timesteps: {total_timesteps:,} | envs: {num_envs}")

    # Instantiate curriculum callback
    curriculum_cb = CurriculumLearningCallback(
        reward_threshold=curriculum_reward_threshold,
        eval_window=curriculum_eval_window,
        num_envs=num_envs,
        verbose=verbose,
    )

    # Create training & eval envs
    train_subproc = create_subproc_vecenv("nascar", num_envs)
    # Wrap with VecNormalize (norm obs & reward). Keep a single VecNormalize during the whole run.
    venv = VecNormalize(train_subproc, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # eval env (use DummyVecEnv wrapped by VecNormalize but not updating stats)
    eval_dummy = DummyVecEnv([make_env("eval", nascar_track)])
    eval_vec = VecNormalize(eval_dummy, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    # share normalization stats initially
    eval_vec.obs_rms = venv.obs_rms
    eval_vec.ret_rms = venv.ret_rms

    # Eval callback with more episodes and robust check
    eval_callback = CurriculumEvalCallback(
        curriculum_cb,
        eval_env=eval_vec,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=12,  # increased for lower variance
        verbose=verbose,
    )

    # Policy kwargs and stabilized A2C hyperparams for continuous control
    policy_kwargs = dict(
        net_arch=[256, 128],
        activation_fn=torch.nn.ReLU,
    )

    model = A2C(
        policy="MlpPolicy",
        env=venv,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(learning_rate_initial_value, learning_rate_final_value),
        n_steps=64,  # 64 steps * 8 envs => batch 512, lower variance
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0005,  # small entropy to keep exploration stable
        vf_coef=0.5,
        max_grad_norm=0.35,
        stats_window_size=stats_window_size,
        verbose=verbose,
        device="cpu",
        policy_kwargs=policy_kwargs,
    )

    logger.info("A2C model initialized with stabilized hyperparameters.")
    logger.info(f"Learning rate schedule: {learning_rate_initial_value} -> {learning_rate_final_value}")

    # Start training: single continuous call. Curriculum switching occurs via callback.
    logger.info("Beginning training loop (single model.learn call).")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=eval_callback,
    )

    # Save final model
    final_path = f"{checkpoint_dir}/{model_name}_final"
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Clean up envs
    try:
        model.get_env().close()
    except Exception:
        pass
    try:
        eval_callback.eval_env.close()
    except Exception:
        pass

    logger.info("Training complete.")
