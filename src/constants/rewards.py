# Reward System Constants

# Positive Rewards
REWARD_DISTANCE_MULTIPLIER = 0.15  # Bonus per meter traveled
REWARD_LAP_COMPLETION = 0.0  # Bonus per completed lap
REWARD_FAST_LAP_TIME = 0  # Time threshold (seconds) for fast lap bonus
REWARD_FAST_LAP_BONUS = 0  # Bonus for completing lap under threshold time

# Negative Rewards (Penalties)
PENALTY_PER_STEP = 0.05  # penalty applied every physics step to encourage speed
PENALTY_BACKWARD_PER_METER = 0.05  # penalty per meter of backward movement beyond threshold
PENALTY_WALL_COLLISION_PER_STEP = 0.2  # penalty for collisions per step
PENALTY_DISABLED = -1 #penalty when a car gets disabled