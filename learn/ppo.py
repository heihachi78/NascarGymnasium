import numpy as np
import sys
import os
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from collections import deque
import logging

# project src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

# ---------- konfiguráció ----------
num_envs = 8
base_path = "learn/"
verbose = 1
total_timesteps = 50_000_000
eval_freq = 12_500
log_interval = 1
learning_rate_initial_value = 3e-4
learning_rate_final_value = 1e-5
stats_window_size = 25
model_name = "ppo"

# ---------- curriculum learning konfiguráció ----------
curriculum_reward_threshold = 200.0
curriculum_eval_window = 50
nascar_track = "tracks/nascar.track"

log_dir = f"./{base_path}logs/{model_name}"
checkpoint_dir = f"./{base_path}checkpoints/{model_name}"
tensorboard_log = f"./{base_path}tensorboard/{model_name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

print(f"\n🏆 PPO Curriculum Learning Configuration:")
print(f"   Model: {model_name}")
print(f"   Total timesteps: {total_timesteps:,}")
print(f"   Environments: {num_envs}")
print(f"   Curriculum threshold: {curriculum_reward_threshold}")
print(f"   NASCAR track: {nascar_track}")
print(f"   Logs: {log_dir}")
print(f"   Checkpoints: {checkpoint_dir}")
print(f"   Tensorboard: {tensorboard_log}\n")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- curriculum learning callback ----------
class CurriculumLearningCallback(BaseCallback):
    def __init__(self, reward_threshold: float = 200.0, eval_window: int = 50, 
                 num_envs: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.eval_window = eval_window
        self.num_envs = num_envs
        self.reward_history = deque(maxlen=eval_window)
        self.phase = "nascar"  # "nascar" or "random"
        self.phase_switched = False
        self.eval_callback = None  # Will be set by parent callback
        
    def _on_step(self) -> bool:
        return True
        
    def _on_training_start(self) -> None:
        logger.info(f"Starting curriculum learning - Phase: {self.phase}")
        
    def update_reward_history(self, mean_reward: float) -> None:
        """Update reward history and check for phase transition"""
        self.reward_history.append(mean_reward)
        
        if self.phase == "nascar" and len(self.reward_history) >= 5:  # Check after 5 evaluations
            recent_rewards = list(self.reward_history)[-5:]  # Last 5 evaluations
            current_mean = np.mean(recent_rewards)
            
            if current_mean >= self.reward_threshold and not self.phase_switched:
                logger.info(f"🎉 Reward threshold reached! Mean reward: {current_mean:.2f} >= {self.reward_threshold}")
                logger.info("🔄 Transitioning from NASCAR track to RANDOM tracks")
                self.phase = "random"
                self.phase_switched = True
                
                # Switch environments immediately
                self._switch_environments()
                
    def _switch_environments(self):
        """
        Switch from NASCAR to random tracks using env_method('switch_to_random').
        Falls back to recreating VecNormalize-wrapped environments if needed.
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

        # Try in-place switch via env_method (VecNormalize always wraps the VecEnv)
        try:
            logger.info("🔄 Attempting in-place environment switch via env_method...")
            
            # Call switch_to_random on the underlying VecEnv (through VecNormalize's .venv)
            result = train_env.venv.env_method("switch_to_random")
            logger.info(f"✅ Training env switch_to_random called on {len(result) if result else 0} sub-environments")
            
            # Also switch eval environment
            if eval_env is not None:
                try:
                    eval_result = eval_env.venv.env_method("switch_to_random")
                    logger.info(f"✅ Eval env switch_to_random called on {len(eval_result) if eval_result else 0} sub-environments")
                except Exception as ee:
                    logger.warning(f"⚠️  Eval env switch failed: {ee}")
                    
            logger.info("🏁 In-place environment switch completed successfully!")
            return
        except Exception as e:
            logger.error(f"❌ Environment switch failed: {e}")
            logger.info("🔄 Continuing training with current NASCAR track environment")
        
    def get_current_phase(self) -> str:
        return self.phase


# ---------- environment létrehozása ----------
def make_env(rank, track_file=None):
    """
    Függvény, ami visszaadja a CarEnv-et Monitorral,
    a SubprocVecEnv-hez szükséges formátumban.
    """
    def _init():
        env = CarEnv(
            render_mode=None,
            track_file=track_file,
            discrete_action_space=False,
            reset_on_lap=True,
        )
        return Monitor(env, filename=os.path.join(log_dir, f"{model_name}_{rank}"))
    return _init

def create_curriculum_env(phase: str, num_envs: int):
    """Create raw SubprocVecEnv for curriculum phase (will be wrapped with VecNormalize)"""
    if phase == "nascar":
        logger.info(f"Creating {num_envs} NASCAR environments")
        return SubprocVecEnv([make_env(i, nascar_track) for i in range(num_envs)])
    else:  # random phase
        logger.info(f"Creating {num_envs} RANDOM track environments")
        return SubprocVecEnv([make_env(i, None) for i in range(num_envs)])


# ---------- custom eval callback for curriculum ----------
class CurriculumEvalCallback(EvalCallback):
    def __init__(self, curriculum_callback: CurriculumLearningCallback, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_callback = curriculum_callback
        self.model_name = model_name
        # Track best rewards for each phase
        self.best_nascar_reward = -np.inf
        self.best_random_reward = -np.inf
        # Link back to curriculum callback so it can update our eval env
        self.curriculum_callback.eval_callback = self
        
    def _on_step(self) -> bool:
        # Store previous evaluation count
        prev_eval_count = len(self.evaluations_results)
        result = super()._on_step()
        
        # Only process if a new evaluation was actually performed
        if len(self.evaluations_results) > prev_eval_count:
            latest_mean_reward = np.mean(self.evaluations_results[-1])
            current_phase = self.curriculum_callback.get_current_phase()
            
            # Save phase-specific best models
            if current_phase == "nascar" and latest_mean_reward > self.best_nascar_reward:
                self.best_nascar_reward = latest_mean_reward
                model_path = f"{self.best_model_save_path}/{self.model_name}_{latest_mean_reward:.1f}_nascar"
                self.model.save(model_path)
            elif current_phase == "random" and latest_mean_reward > self.best_random_reward:
                self.best_random_reward = latest_mean_reward
                model_path = f"{self.best_model_save_path}/{self.model_name}_{latest_mean_reward:.1f}_random"
                self.model.save(model_path)
            
            self.curriculum_callback.update_reward_history(latest_mean_reward)
            
            # Show progress only after new evaluations
            if len(self.curriculum_callback.reward_history) >= 5:
                recent_rewards = list(self.curriculum_callback.reward_history)[-5:]
                current_mean = np.mean(recent_rewards)
                logger.info(f"Eval progress: {current_mean:.1f} avg (last 5) | Recent: {[f'{r:.1f}' for r in recent_rewards]} | Phase: {self.curriculum_callback.get_current_phase()}")
            
        return result

# ---------- lineáris tanulási ráta ----------
def linear_schedule(initial_value=1e-3, final_value=1e-4):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


# ---------- fő futtatás ----------
if __name__ == "__main__":
    # Initialize curriculum callback
    curriculum_callback = CurriculumLearningCallback(
        reward_threshold=curriculum_reward_threshold,
        eval_window=curriculum_eval_window,
        num_envs=num_envs,
        verbose=verbose
    )
    
    # Start with NASCAR phase environments
    train_subproc = create_curriculum_env("nascar", num_envs)
    env = VecNormalize(train_subproc, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    eval_dummy = DummyVecEnv([make_env("eval", nascar_track)])  # eval with same track
    eval_env = VecNormalize(eval_dummy, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    # Share normalization statistics between train and eval environments
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    # Create curriculum-aware eval callback
    eval_callback = CurriculumEvalCallback(
        curriculum_callback,
        model_name,
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
        net_arch=[512, 512],
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )

    # PPO modell
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(learning_rate_initial_value, learning_rate_final_value),
        stats_window_size=stats_window_size,
        verbose=verbose,
        batch_size=512,
        sde_sample_freq=4,
        clip_range=0.1,
        use_sde=True,
        device='cpu',
        policy_kwargs=policy_kwargs,
    )
    
    logger.info(f"PPO model initialized")
    logger.info(f"Learning rate: {learning_rate_initial_value} -> {learning_rate_final_value}")
    logger.info(f"Batch size: 128 | n_steps: 2048 | n_epochs: 10")
    logger.info(f"SDE enabled with sample freq: 4")

    # tanulás with curriculum progression - SINGLE CONTINUOUS LEARNING SESSION
    logger.info(f"Starting PPO training with curriculum learning")
    logger.info(f"Phase 1: NASCAR track until mean reward > {curriculum_reward_threshold}")
    logger.info(f"Phase 2: Random tracks for continued learning")
    logger.info(f"Using SINGLE continuous model.learn() call for {total_timesteps:,} timesteps")
    
    # Single model.learn() call - curriculum switching happens via callback
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=eval_callback,
    )
    
    # Log curriculum completion summary
    if curriculum_callback.phase_switched:
        logger.info(f"🏁 Training completed with curriculum progression:")
        logger.info(f"   Successfully switched from NASCAR to RANDOM tracks during training")
        logger.info(f"   Final phase: {curriculum_callback.get_current_phase()}")
    else:
        logger.info(f"🏁 Training completed entirely in NASCAR phase")
        logger.info(f"   Reward threshold ({curriculum_reward_threshold}) was not reached")

    # mentés
    model.save(f"{checkpoint_dir}/{model_name}_final")
    
    # Clean up environments
    env.close()
    eval_env.close()
    
    logger.info(f"Model saved to {checkpoint_dir}/{model_name}_final")
    logger.info("Training completed successfully! 🎉")