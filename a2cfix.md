Below is a targeted plan to stop the post-improvement “fall back” and make the curriculum transition stable. I’ve included concrete code changes you can use, but you need to validate the code and make sure it works with the current implementation.

# 1) Keep the same vectorized wrapper; flip tracks in-place

Right now you replace the whole `VecEnv`. That throws away any running statistics/warm-up the optimizer had (and if you later add normalization, it will also reset normalization stats), which often causes a reward drop.

**Action**

* Add a method to `CarEnv` that switches from the NASCAR track to random tracks.
* Call that method for every sub-env via `env.env_method(...)` rather than rebuilding the env.

```python
# in CarEnv
class CarEnv(gym.Env):
    # ...
    def switch_to_random(self):
        self.track_file = None
        self._load_random_track()  # your internal method
        # optionally also widen noise/exploration specific to phase 2

    def switch_to_fixed(self, track_path):
        self.track_file = track_path
        self._load_track(track_path)
```

```python
# In the callback: replace _switch_environments()
def _switch_environments(self):
    model = getattr(self, 'model', None) or self.locals.get('self', None)
    if model is None:
        logger.warning("Cannot switch environments - model not available")
        return
    try:
        # flip train envs in-place
        model.get_env().env_method("switch_to_random")
        logger.info("✅ Training environments switched in-place to RANDOM")

        # flip eval env in-place too
        if self.eval_callback is not None:
            self.eval_callback.eval_env.env_method("switch_to_random")
            logger.info("✅ Eval environment switched in-place to RANDOM")
    except Exception as e:
        logger.error(f"❌ Failed to switch environments in-place: {e}")
```

This avoids optimizer/normalization resets and typically eliminates sudden regressions.

# 2) Normalize observations **and** rewards (critical for curriculum)

Different tracks often shift reward scale and observation distribution. Without normalization the value baseline becomes mis-calibrated after switching.

**Action**
Wrap both train and eval envs with `VecNormalize` and **do not** recreate it during training.

```python
from stable_baselines3.common.vec_env import VecNormalize

# creation
venv = create_curriculum_env("nascar", num_envs)
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

eval_env = DummyVecEnv([make_env("eval", nascar_track)])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
eval_env.obs_rms = venv.obs_rms  # share stats
eval_env.ret_rms = venv.ret_rms  # share stats
```

When you flip to random tracks in-place (section 1), stats continue; no shock.

> If you must rebuild envs for any reason, **copy** `obs_rms` and `ret_rms` from the old `VecNormalize` onto the new one before swapping.

# 3) Stabilize A2C hyperparameters for continuous control

Your defaults are close to Atari-style settings. For continuous driving tasks the following are much more stable:

```python
model = A2C(
    "MlpPolicy",
    venv,
    tensorboard_log=tensorboard_log,
    learning_rate=linear_schedule(7.5e-4, 2.5e-4),   # slightly lower + longer tail
    n_steps=64,           # ↑ rollout length (total batch = n_steps * n_envs = 512)
    gamma=0.99,
    gae_lambda=0.95,      # reduces variance after the switch
    ent_coef=0.005,       # small but non-zero exploration pressure
    vf_coef=0.5,
    max_grad_norm=0.5,    # gradient clipping; very helpful after the switch
    stats_window_size=stats_window_size,
    verbose=verbose,
    device="cpu",
    policy_kwargs=dict(
        net_arch=[256, 128],        # a bit larger stabilizes value estimates
        activation_fn=torch.nn.ReLU,
        ortho_init=True             # default True; keep it
    ),
)
```

Notes:

* `n_steps` of 64–128 for 8 envs usually outperforms `5–20` on continuous tasks.
* Non-zero `ent_coef` prevents late training “over-confidence” (your `train/std` rising indicates the opposite dynamics: the log std may drift; a small entropy cost helps keep it bounded).
* Tighten `max_grad_norm` to 0.5 to avoid post-switch spikes in `value_loss`.

# 4) Make the curriculum gate more robust (reduce variance)

Your switch triggers off the average of the last five evals with just 5 episodes per eval; that’s noisy. Before switching, ensure performance is both **sustained** and **stable**.

**Action**

* Increase `n_eval_episodes` to 10–15.
* Require the **median** of the last `k` eval means to exceed the threshold and the std to be small.

```python
# in CurriculumEvalCallback._on_step after computing latest_mean_reward
recent = list(self.curriculum_callback.reward_history)[-8:]  # last 8 evals
if len(recent) >= 8:
    med = float(np.median(recent))
    std = float(np.std(recent))
    if (med >= self.curriculum_callback.reward_threshold) and (std <= 0.15 * abs(med)):
        self.curriculum_callback.phase_switched = True
        self.curriculum_callback.phase = "random"
        self.curriculum_callback._switch_environments()
```

* Alternatively, reduce the threshold (e.g., `150–180`) but insist on **N consecutive** passes (e.g., 3) to avoid premature switching.

# 5) Learning-rate dampening at the moment of switch

Even with normalization, a small LR dip at switch prevents value-function shocks.

**Action**
Inside `_switch_environments`, reduce the optimizer LR by 30–50% for \~1–2 million steps.

```python
def _set_lr(model, factor=0.5):
    opt = model.policy.optimizer
    for pg in opt.param_groups:
        pg["lr"] *= factor

# inside _switch_environments after env flip
_set_lr(model, factor=0.6)
logger.info("🔧 Reduced LR by 40% for post-switch stabilization")
```

Optionally, implement a tiny “warm-up” entropy bump for the first few thousand updates post-switch:

```python
# pseudo: in callback on each step after switch
if self.phase == "random" and self.post_switch_steps < 20000:
    for pg in model.policy.optimizer.param_groups:
        model.ent_coef = 0.01  # SB3 A2C exposes ent_coef; set then restore later
```

(If you prefer not to modify internals during training, start with `ent_coef=0.01` globally and don’t do this.)

# 6) Reward shaping & clipping sanity

A2C is sensitive to outliers. If your per-step reward occasionally spikes (collisions, laps, etc.), clip or scale.

**Action**

* Clip shaped reward to `[-5, 5]` **before** returning it from the env, or
* Keep raw reward, but let `VecNormalize(norm_reward=True)` handle scaling; then **do not** additionally multiply reward by large constants in the env.

Also ensure termination isn’t overly punitive at the switch (e.g., returning a single large negative at episode end).

# 7) Evaluation parity and determinism

You evaluate with a single fixed NASCAR track at the beginning and then with random tracks after switching. Ensure determinism to reduce variance in the decision to switch.

**Action**

* Seed all envs and the RNGs.
* For evaluation, use a **fixed set of K random tracks** sampled once per run and cycled deterministically. That keeps eval comparable across time.

```python
np.random.seed(123); torch.manual_seed(123)
venv.seed(123); eval_env.seed(123)
```

# 8) What to watch in TensorBoard

* `explained_variance` near 1.0 but falling reward → almost always scaling/normalization or evaluation-mismatch. Section 2 fixes this.
* `train/std` rising late → increase `ent_coef` slightly and set `max_grad_norm`.
* Spiky `value_loss` at/after the switch → apply LR dampening and increase `n_steps`.
