# Reward System Constants

# Positive Rewards
REWARD_DISTANCE_MULTIPLIER = 0.1  # Bonus per meter traveled
REWARD_LAP_COMPLETION = 0.0  # Bonus per completed lap
REWARD_FAST_LAP_TIME = 120  # Time threshold (seconds) for fast lap bonus
REWARD_FAST_LAP_BONUS = 0.1  # Bonus for completing lap under threshold time

# Negative Rewards (Penalties)
PENALTY_PER_STEP = 0.05  # penalty applied every physics step to encourage speed
PENALTY_BACKWARD_PER_METER = 0.05  # penalty per meter of backward movement beyond threshold
PENALTY_WALL_COLLISION_RATIO_MULTIPLIER = 0.15  # penalty multiplier for accumulated impulse ratio (0-1 scale)