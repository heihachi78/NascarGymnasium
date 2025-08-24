# Car Racing RL Environment

A comprehensive reinforcement learning environment for car racing simulation built with Gymnasium, Box2D physics, and PyGame rendering. Features realistic car physics, multi-car support, and flexible control systems for training and evaluating RL agents.

## Features

- **Realistic Physics**: Box2D-based car physics with tire simulation, collision detection, and track boundaries
- **Multi-Car Support**: Simultaneous racing with up to 10 cars
- **Flexible Control**: Both discrete and continuous action spaces
- **Multiple RL Algorithms**: Pre-configured support for PPO, SAC, and TD3
- **Rich Observation Space**: 29-dimensional state vector with comprehensive car and environment data
- **Game Modes**: Competition racing and time trial modes
- **Visual Feedback**: Real-time rendering with debug information and camera controls

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd claude_car4

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Train a PPO model
python learn/ppo.py

# Run model competition
python game/competition.py

# Run time trial mode
python game/time_trial.py

# Run tests
pytest
```

## Observation Structure

The environment provides a 29-dimensional observation vector for each car with normalized values:

### Position and Motion (7 dimensions)
- **Position** (indices 0-1): Car position (x, y) normalized to [-1, 1]
- **Velocity** (indices 2-4): Velocity components (vx, vy) normalized to [-1, 1], and speed magnitude normalized to [0, 1]
- **Orientation** (indices 5-6): Car orientation angle and angular velocity normalized to [-1, 1]

### Tire System (12 dimensions)
- **Tire Loads** (indices 7-10): Normal force on each tire (FL, FR, RL, RR) normalized to [0, 1]
- **Tire Temperatures** (indices 11-14): Temperature of each tire normalized to [0, 1]
- **Tire Wear** (indices 15-18): Wear level of each tire normalized to [0, 1]

### Collision Data (2 dimensions)
- **Collision Impulse** (index 19): Impact force from collisions normalized to [0, 1]
- **Collision Angle** (index 20): Relative angle of collision normalized to [-1, 1]

### Distance Sensors (8 dimensions)
- **Sensor Distances** (indices 21-28): 8-directional distance sensors measuring track boundaries
  - Index 21: Forward
  - Index 22: Front-left diagonal
  - Index 23: Left
  - Index 24: Rear-left diagonal
  - Index 25: Rear
  - Index 26: Rear-right diagonal
  - Index 27: Right
  - Index 28: Front-right diagonal
  - All normalized to [0, 1] where 1.0 = maximum sensor range

### Observation Access Example
```python
# Extract key components from observation
position = observation[0:2]           # Car position (x, y)
velocity = observation[2:5]           # Velocity (vx, vy, speed)
orientation = observation[5:7]        # Angle, angular velocity
tire_loads = observation[7:11]        # Tire loads (FL, FR, RL, RR)
tire_temps = observation[11:15]       # Tire temperatures
tire_wear = observation[15:19]        # Tire wear
collision_data = observation[19:21]   # Collision impulse and angle
sensors = observation[21:29]          # 8 distance sensors
forward_distance = observation[21]    # Forward sensor only
```

## Reward Calculation

The reward system encourages fast, smooth racing while penalizing collisions and poor driving:

### Positive Rewards
- **Distance Traveled**: `+0.03` points per meter of forward progress along the track
- **Lap Completion**: `+100.0` points for completing a full lap
- **Fast Lap Bonus**: Additional reward for laps completed under 120 seconds:
  - Formula: `+0.1 × (120 - lap_time)²`
  - Encourages setting faster lap times

### Penalties
- **Collision Penalty**: `-0.01` points per second while colliding with track boundaries
- **Backward Movement**: `-0.05` points per meter when moving backward beyond the threshold
- **Severe Collision**: Large impact forces can disable cars entirely

### Survival and Speed Rewards
- **Base Survival**: Small positive reward for staying active
- **Speed Bonus**: Time-based reward proportional to forward velocity (only when not colliding)

### Reward Calculation Example
```python
# Per-step reward calculation
reward = 0.0

# Positive rewards
reward += distance_traveled * 0.03        # Distance progress
reward += 100.0 * laps_completed          # Lap bonuses
if lap_time < 120:
    reward += 0.1 * (120 - lap_time)**2   # Fast lap bonus

# Penalties
reward -= 0.01 * collision_duration       # Collision time
reward -= 0.05 * backward_meters          # Backward movement

# Speed and survival (small baseline rewards)
reward += small_survival_bonus
reward += speed_bonus * (1 - collision_factor)
```

## Action Spaces

### Discrete Action Space
When `discrete_action_space=True`, the environment uses 5 discrete actions:

| Action | Value | Description |
|--------|--------|-------------|
| 0 | Do Nothing | `[0.0, 0.0, 0.0]` - Coast |
| 1 | Accelerate | `[1.0, 0.0, 0.0]` - Full throttle |
| 2 | Brake | `[0.0, 1.0, 0.0]` - Full brake |
| 3 | Turn Left | `[0.0, 0.0, -1.0]` - Full left steering |
| 4 | Turn Right | `[0.0, 0.0, 1.0]` - Full right steering |

```python
# Using discrete actions
env = CarEnv(discrete_action_space=True)
action = 1  # Accelerate
obs, reward, terminated, truncated, info = env.step(action)
```

### Continuous Action Space
When `discrete_action_space=False` (default), the environment uses continuous control:

| Index | Range | Description |
|-------|--------|-------------|
| 0 | [0.0, 1.0] | **Throttle** - Accelerator pedal position |
| 1 | [0.0, 1.0] | **Brake** - Brake pedal position |
| 2 | [-1.0, 1.0] | **Steering** - Steering wheel angle (-1=left, +1=right) |

```python
# Using continuous actions
env = CarEnv(discrete_action_space=False)
action = [0.8, 0.0, -0.3]  # 80% throttle, no brake, slight left turn
obs, reward, terminated, truncated, info = env.step(action)
```

### Multi-Car Actions
For multi-car environments, actions are provided as arrays:

```python
# Multi-car discrete (each car gets one discrete action)
action = [1, 2, 0, 4]  # Car 0: accelerate, Car 1: brake, Car 2: coast, Car 3: turn right

# Multi-car continuous (each car gets [throttle, brake, steering])
action = [
    [0.8, 0.0, -0.2],  # Car 0: throttle + left
    [0.0, 0.5, 0.0],   # Car 1: brake + straight
    [1.0, 0.0, 0.5],   # Car 2: throttle + right
    [0.3, 0.0, 0.0]    # Car 3: light throttle + straight
]
```

## Base Control Class

The `BaseController` class provides the foundation for all control systems with built-in fallback logic:

### Key Features
- **Per-Instance State**: Maintains separate control state for multi-car scenarios
- **Fallback Algorithm**: Rule-based control when AI models are unavailable
- **Sensor-Based Logic**: Uses 8-directional distance sensors for navigation

### Fallback Control Algorithm
The rule-based fallback implements a sophisticated control strategy:

```python
class BaseController:
    def __init__(self, name=None):
        self.control_state = {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0
        }
    
    def _fallback_control(self, observation):
        # Extract sensor data
        sensors = observation[21:29]
        forward = sensors[0]          # Forward distance
        front_left = sensors[1]       # Front-left distance  
        front_right = sensors[7]      # Front-right distance
        current_speed = observation[4] # Current speed
        
        # Speed control based on forward distance
        speed_limit = forward * 500 / 3.6  # Convert to appropriate units
        
        # Accumulative throttle/brake control
        if current_speed * 200 < speed_limit:
            self.control_state['throttle'] += 0.1
            self.control_state['brake'] -= 0.01
        else:
            self.control_state['throttle'] -= 0.1
            self.control_state['brake'] += 0.01
        
        # Steering based on left/right sensor comparison
        if front_right > front_left:
            self.control_state['steering'] = front_right / front_left - forward
        elif front_right < front_left:
            self.control_state['steering'] = -(front_left / front_right - forward)
        else:
            self.control_state['steering'] = 0
        
        # Reduce throttle when turning hard
        if abs(self.control_state['steering']) > 0.1:
            self.control_state['throttle'] -= 0.05
        
        # Apply limits
        self.control_state['throttle'] = np.clip(self.control_state['throttle'], 0, 1)
        self.control_state['brake'] = np.clip(self.control_state['brake'], 0, 1)
        self.control_state['steering'] = np.clip(self.control_state['steering'], -1, 1)
        
        return np.array([
            self.control_state['throttle'], 
            self.control_state['brake'], 
            self.control_state['steering']
        ])
```

### Creating Custom Controllers
```python
from game.control.base_controller import BaseController

class MyController(BaseController):
    def __init__(self, model_path, name="MyController"):
        super().__init__(name)
        self.model = self.load_model(model_path)
        
    def control(self, observation):
        try:
            # Try to use your model
            action = self.model.predict(observation)
            return action
        except:
            # Fall back to rule-based control
            self.use_fallback = True
            return self._fallback_control(observation)
```

## Game Modes

### Competition Mode (`game/competition.py`)
Multi-car racing where different AI models and control strategies compete directly:

#### Features
- **Mixed Controllers**: Combine PPO, SAC, TD3, and rule-based controllers
- **Live Camera Switching**: Press 0-9 to follow different cars
- **Real-Time Statistics**: Lap times, rewards, and finishing positions
- **Performance Tracking**: Best lap times and cumulative rewards
- **Collision Management**: Cars can be disabled on high-impact collisions

#### Usage
```python
# Configure models to compete
model_configs = [
    ("game/control/models/ppo_model.zip", "PPO"),
    ("game/control/models/td3_model.zip", "TD3"),
    (None, "Rule-Based"),  # Rule-based fallback
]

# Environment automatically handles multi-car setup
env = CarEnv(
    track_file="tracks/nascar.track", 
    num_cars=len(model_configs),
    reset_on_lap=False,                    # Don't auto-reset
    disable_cars_on_high_impact=True,      # Disable on crashes
    car_names=[config[1] for config in model_configs]
)
```

#### Competition Results
The system tracks and ranks cars by:
1. **Laps Completed** (primary ranking)
2. **Finishing Time** (for same lap count)
3. **Best Lap Time** (tiebreaker)
4. **Cumulative Reward** (final tiebreaker)

### Time Trial Mode (`game/time_trial.py`)
Individual performance testing with multiple attempts:

#### Features
- **Multiple Attempts**: 3 attempts × 2 laps = 6 total laps per car
- **Environment Reset**: Fresh conditions between attempts
- **Personal Bests**: Track fastest lap time for each car
- **Detailed Statistics**: Per-attempt breakdown of all lap times
- **Fair Comparison**: All cars get identical track conditions

#### Configuration
```python
LAPS_PER_ATTEMPT = 2   # Laps in each attempt
TOTAL_ATTEMPTS = 3     # Number of attempts per car
```

#### Results Tracking
- **Overall Winner**: Car with fastest single lap across all attempts
- **Attempt Summary**: Performance breakdown per attempt
- **Personal Records**: Each car's best lap time highlighted
- **Completion Status**: Laps completed vs. disabled cars

#### Time Trial Flow
```python
for attempt in range(1, TOTAL_ATTEMPTS + 1):
    env.reset()  # Fresh start for each attempt
    
    # Run until all cars complete LAPS_PER_ATTEMPT or get disabled
    while not all_cars_finished:
        # Get actions from controllers
        # Step environment
        # Track lap completions
        # Check for attempt completion
    
    # Display attempt summary
    # Update personal bests
```

## Advanced Usage

### Custom Environment Configuration
```python
env = CarEnv(
    render_mode="human",                    # "human" or None
    track_file="tracks/nascar.track",       # Track definition
    num_cars=4,                            # 1-10 cars
    discrete_action_space=False,            # Continuous control
    reset_on_lap=True,                     # Auto-reset after lap
    disable_cars_on_high_impact=True,      # Crash damage
    car_names=["PPO", "TD3", "SAC", "Rule"] # Custom names
)
```

### Multi-Car Training
```python
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return CarEnv(
        num_cars=1,                    # Train single car
        discrete_action_space=True,    # Use discrete actions
        reset_on_lap=True             # Reset after each lap
    )

# Create vectorized environments for parallel training
vec_env = DummyVecEnv([make_env for _ in range(8)])
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
```

### Real-Time Controls
During rendering (`render_mode="human"`):
- **0-9**: Switch camera between cars
- **R**: Toggle reward display
- **D**: Toggle debug information
- **I**: Toggle track info display
- **C**: Change camera mode
- **ESC**: Exit simulation

## Architecture

### Core Components
- `src/car_env.py`: Main RL environment with full physics
- `src/base_env.py`: Base Gymnasium interface
- `src/car_physics.py`: Box2D physics simulation
- `src/constants/`: Modular configuration system

### Training System
- `learn/ppo.py`, `learn/sac.py`, `learn/td3.py`: Algorithm-specific training
- Automatic checkpointing and TensorBoard logging
- 8 parallel environments for efficient training

### Control System
- `game/control/base_controller.py`: Base controller with fallback
- Algorithm-specific controllers with model loading
- Graceful degradation when models fail

## Performance Tips

- Use `render_mode=None` for training (headless mode for faster simulation)
- Use `reset_on_lap=True` for training, `False` for demos
- Monitor TensorBoard logs during training: `tensorboard --logdir=./tensorboard/`
- Save checkpoints frequently during long training runs
- Test models in competition mode to compare performance

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation for new features
4. Test with both single and multi-car configurations

## License

[Add your license information here]