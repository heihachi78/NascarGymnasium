# Car Racing Environment - Reinforcement Learning Simulation

A sophisticated car racing simulation environment built with realistic physics, reinforcement learning capabilities, and comprehensive game modes. This environment provides a complete gymnasium-compatible interface for training and testing autonomous racing agents.

## Table of Contents

- [Environment Overview](#environment-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Spaces](#action-spaces)
  - [Reward System](#reward-system)
  - [Car Disabling System](#car-disabling-system)
  - [Track System](#track-system)
- [Game Modes](#game-modes)
- [Reinforcement Learning](#reinforcement-learning)
- [Tools and Utilities](#tools-and-utilities)
- [Architecture](#architecture)
- [Contributing](#contributing)

## Environment Overview

This racing simulation features:

- **Realistic Physics**: Box2D-based physics with tire temperature, collision detection, and aerodynamics
- **Multi-car Support**: Up to 10 cars racing simultaneously
- **Gymnasium Interface**: Standard RL environment with continuous and discrete action spaces
- **Multiple Game Modes**: Time trials, competitions, and training scenarios
- **Rich Observation Space**: 39-dimensional normalized observations including sensors, car state, and physics
- **Comprehensive Reward System**: Distance-based rewards with collision penalties and lap bonuses
- **Dynamic Track System**: Random track selection or specific track loading
- **Advanced Tools**: Track builder, analyzer, validator, and visualization tools

## Installation

### Prerequisites

```bash
python >= 3.8
```

### Dependencies

```bash
pip install -r requirements.txt
```

### Quick Setup

```bash
# Clone or download the project
cd claude_car4

# Install dependencies
pip install -r requirements.txt

# Test the installation
python game/random_demo.py
```

## Quick Start

### Basic Usage

```python
from src.car_env import CarEnv

# Create environment with default settings
env = CarEnv(render_mode="human")

# Reset environment
obs, info = env.reset()

# Take random actions
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training a Reinforcement Learning Model

```bash
# Train TD3 agent
python learn/td3_simple.py

# Train PPO agent  
python learn/ppo_simple.py

# Train SAC agent
python learn/sac_simple.py
```

### Running Game Modes

```bash
# Time trial mode (3 attempts × 2 laps)
python game/time_trial.py

# Competition mode (multi-car racing)
python game/competition.py

# Random demonstration
python game/random_demo.py
```

## Environment Details

### Observation Space

The environment provides a **39-dimensional continuous observation space** with normalized values:

#### Observation Vector (39 elements):

| Index | Component | Range | Description |
|-------|-----------|--------|-------------|
| 0-1 | `pos_x, pos_y` | [-1, 1] | Car position normalized to track bounds |
| 2-3 | `vel_x, vel_y` | [-1, 1] | Car velocity components |
| 4 | `speed_magnitude` | [0, 1] | Speed magnitude normalized to max velocity |
| 5-6 | `orientation, angular_vel` | [-1, 1] | Car rotation and angular velocity |
| 7-10 | `tyre_load_fl/fr/rl/rr` | [0, 1] | Tire load for each wheel (normalized) |
| 11-14 | `tyre_temp_fl/fr/rl/rr` | [0, 1] | Tire temperature for each wheel |
| 15-18 | `tyre_wear_fl/fr/rl/rr` | [0, 1] | Tire wear for each wheel |
| 19-20 | `collision_impulse, collision_angle` | [0, 1], [-1, 1] | Collision data |
| 21 | `cumulative_impact_percentage` | [0, 1] | Accumulated collision damage |
| 22-37 | `sensor_dist_0` to `sensor_dist_15` | [0, 1] | 16 distance sensors around the car |

#### Normalization Details

- **Position**: Normalized using `NORM_MAX_POSITION = 1000.0` meters
- **Velocity**: Normalized using `NORM_MAX_VELOCITY = 50.0` m/s  
- **Tire Load**: Normalized using `NORM_MAX_TYRE_LOAD` based on car weight distribution
- **Tire Temperature**: Normalized using `NORM_MAX_TYRE_TEMP = 120.0°C`
- **Tire Wear**: Normalized using `NORM_MAX_TYRE_WEAR = 100.0`
- **Sensors**: Normalized using `SENSOR_MAX_DISTANCE = 100.0` meters
- **Angular Velocity**: Normalized using `NORM_MAX_ANGULAR_VEL = 5.0` rad/s

### Action Spaces

The environment supports both **continuous** and **discrete** action spaces:

#### Continuous Action Space (Default)

**Single Car**: `Box(low=[-1, -1], high=[1, 1], shape=(2,))`
**Multi-Car**: `Box(low=[[-1, -1], ...], high=[[1, 1], ...], shape=(num_cars, 2))`

- `action[0]`: **Throttle/Brake** [-1.0 to 1.0]
  - `> 0`: Throttle (acceleration)
  - `< 0`: Brake (absolute value used as brake force)
- `action[1]`: **Steering** [-1.0 to 1.0]
  - `< 0`: Turn left
  - `> 0`: Turn right

#### Discrete Action Space

**Single Car**: `Discrete(5)`
**Multi-Car**: `MultiDiscrete([5, 5, ...])`

- `0`: No action
- `1`: Accelerate (throttle = 1.0)
- `2`: Brake (brake = 1.0)
- `3`: Turn left (steering = -1.0)
- `4`: Turn right (steering = 1.0)

### Reward System

The reward system encourages fast, clean racing with multiple components:

#### Positive Rewards
- **Distance Reward**: `+0.15` per meter traveled forward
- **Lap Completion**: Configurable bonus for completing laps
- **Fast Lap Bonus**: Extra reward for laps under threshold time

#### Negative Rewards (Penalties)
- **Per-Step Penalty**: `-0.05` per simulation step (encourages speed)
- **Wall Collision**: `-0.25` per step while colliding with walls
- **Backward Movement**: `-0.05` per meter of backward driving
- **Car Disabled**: `-100.0` when car becomes disabled

#### Reward Configuration

```python
# Located in src/constants/rewards.py
REWARD_DISTANCE_MULTIPLIER = 0.15
PENALTY_PER_STEP = 0.05
PENALTY_WALL_COLLISION_PER_STEP = 0.25
PENALTY_BACKWARD_PER_METER = 0.05
PENALTY_DISABLED = 100.0
```

### Car Disabling System

Cars can be disabled due to various conditions to simulate realistic racing consequences:

#### Disabling Conditions

1. **Catastrophic Impact**: Single collision > 5000 N⋅s instant disable
2. **Cumulative Damage**: Total collision damage > 15000 N⋅s
3. **Stuck Detection**: Car moving < 1.0 m/s for > 10 seconds with < 5m movement
4. **Excessive Backward Driving**: > 100 meters of backward movement

#### Disabled Car Behavior
- Receives zero control input (throttle=0, brake=0, steering=0)
- No longer accumulates rewards (except final disable penalty)
- Excluded from race position calculations
- Visually distinguished in rendering

### Track System

#### Track File Format

Tracks are defined using simple command files with `.track` extension:

```
# Example: nascar.track
WIDTH 25           # Track width in meters
GRID              # Starting grid segment  
STARTLINE         # Start/finish line
STRAIGHT 200      # 200m straight section
LEFT 180 300      # 180° left turn with 300m radius
STRAIGHT 400      # 400m straight section
LEFT 180 300      # Another 180° left turn
STRAIGHT 95       # Final straight to complete oval
```

#### Available Track Commands

| Command | Parameters | Description |
|---------|------------|-------------|
| `WIDTH` | `<meters>` | Set track width |
| `GRID` | - | Starting grid position |
| `STARTLINE` | - | Start/finish line marker |
| `STRAIGHT` | `<length>` | Straight section in meters |
| `LEFT` | `<degrees> <radius>` | Left turn with angle and radius |
| `RIGHT` | `<degrees> <radius>` | Right turn with angle and radius |

#### Track Parameters

- **Width**: Typically 15-30 meters for realistic racing
- **Length**: Can range from 500m (short) to 5000m+ (endurance)
- **Complexity**: Measured by turn count and radius variations
- **Banking**: Supported through track generation parameters

#### Available Tracks

The environment includes several pre-built tracks:

- `nascar.track` - Simple NASCAR-style oval
- `daytona.track` - Daytona International Speedway inspired
- `martinsville.track` - Short track configuration  
- `trioval.track` - Triangular superspeedway
- `nascar_banked.track` - Banked oval configuration

## Game Modes

### Time Trial Mode

**File**: `game/time_trial.py`

- **Format**: 3 attempts × 2 laps each (6 total laps)
- **Objective**: Fastest single lap time wins
- **Features**:
  - Random track selection for each attempt
  - Environment resets between attempts
  - Personal best tracking
  - Overall leaderboard

```bash
python game/time_trial.py
```

### Competition Mode  

**File**: `game/competition.py`

- **Format**: Multi-car simultaneous racing
- **Objective**: Complete most laps, fastest times as tiebreaker
- **Features**:
  - Up to 10 cars racing
  - Real-time leaderboards
  - Collision tracking
  - Performance statistics

```bash
python game/competition.py
```

### Controls (Rendering Mode)

| Key | Action |
|-----|--------|
| `0-9` | Switch camera between cars |
| `R` | Toggle reward display |
| `O` | Toggle observation visualization |
| `ESC` | Exit simulation |

## Reinforcement Learning

### Supported Algorithms

The environment is compatible with popular RL libraries and includes training scripts for:

- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **PPO** (Proximal Policy Optimization)  
- **SAC** (Soft Actor-Critic)
- **A2C** (Advantage Actor-Critic)
- **DDPG** (Deep Deterministic Policy Gradient)

### Training Configuration

#### Example: TD3 Training

```python
# learn/td3_simple.py configuration
num_envs = 8                    # Parallel environments
total_timesteps = 25_000_000    # Training steps
learning_rate = 1e-4            # Initial learning rate
eval_freq = 25_000              # Evaluation frequency
```

#### Training Commands

```bash
# Train different algorithms
python learn/td3_simple.py      # TD3 training
python learn/ppo_simple.py      # PPO training  
python learn/sac_simple.py      # SAC training
python learn/a2c_simple.py      # A2C training
python learn/ddpg_simple.py     # DDPG training
```

#### Pre-trained Models

The environment includes pre-trained models in `game/control/models/`:
- `td3_best_model1.zip` - Best performing TD3 model
- `a2c_best_model1.zip` - Best performing A2C model
- Additional model checkpoints for comparison

### Environment Info Dictionary

The `step()` method returns comprehensive information:

```python
info = {
    "simulation_time": 45.6,           # Current simulation time
    "num_cars": 1,                     # Number of cars
    "followed_car_index": 0,           # Camera-followed car
    "termination_reason": None,        # Why episode ended
    "cars": [                          # Per-car information
        {
            "car_index": 0,
            "disabled": False,
            "car_position": (100.5, 25.3),
            "car_speed_kmh": 180.4,
            "car_speed_ms": 50.1,
            "on_track": True,
            "lap_timing": {
                "lap_count": 2,
                "current_lap_time": 45.67,
                "last_lap_time": 87.23,
                "best_lap_time": 85.1,
                "is_timing": True
            },
            "cumulative_reward": 156.7,
            "cumulative_impact_force": 245.8
        }
    ]
}
```

## Tools and Utilities

### Track Tool CLI

**File**: `tools/track_tool.py`

Comprehensive track analysis and validation:

```bash
# List available tracks
python tools/track_tool.py list

# Analyze specific track
python tools/track_tool.py analyze tracks/nascar.track

# Validate track files
python tools/track_tool.py validate tracks/*.track

# Launch GUI track builder
python tools/track_tool.py gui

# Generate JSON analysis report
python tools/track_tool.py analyze tracks/nascar.track --json report.json
```

#### Track Analysis Features

- **Length Calculation**: Total track distance
- **Segment Analysis**: Count of straights vs curves
- **Difficulty Rating**: Technical complexity assessment
- **Lap Time Estimation**: Predicted lap times
- **Bounds Calculation**: Track dimensional analysis
- **Validation**: Syntax and logical consistency checking

### Track Builder GUI

**File**: `tools/track_builder.py`

Visual track design interface:

- Real-time track visualization
- Interactive segment placement
- Validation feedback
- Export to `.track` format
- Import existing tracks for editing

```bash
python tools/track_tool.py gui [track_file.track]
```

### Analysis Tools

**Files**: `tools/track_analyzer.py`, `tools/track_validator.py`

- **TrackAnalyzer**: Computational track metrics
- **TrackValidator**: Track file validation and error detection
- **Performance Profiling**: Physics and rendering performance analysis

## Architecture

### Core Components

#### Environment Hierarchy
```
BaseEnv (gymnasium.Env)
  ├── Action/Observation space definitions
  └── CarEnv (BaseEnv)
      ├── Physics simulation (Box2D)
      ├── Multi-car management
      ├── Rendering system
      └── Reward calculation
```

#### Key Modules

- **`src/car_env.py`** - Main environment implementation
- **`src/base_env.py`** - Base environment with space definitions
- **`src/car.py`** & **`src/car_physics.py`** - Vehicle simulation
- **`src/track_generator.py`** - Track loading and management
- **`src/renderer.py`** - Pygame-based visualization
- **`src/constants/`** - Configuration constants organized by category

#### Physics System

- **Box2D Integration**: Realistic collision and dynamics
- **Tire Model**: Temperature, wear, and grip simulation  
- **Aerodynamics**: Drag force calculation
- **Collision Detection**: Wall and car-to-car collision handling
- **Sensor System**: 16-directional distance sensors

#### Multi-car Architecture

- **Parallel Physics**: Separate physics worlds per car
- **Collision Isolation**: Independent collision tracking
- **Camera System**: Switchable car following
- **Performance Scaling**: Optimized for up to 10 simultaneous cars

### Constants System

Configuration is centralized in `src/constants/` by category:

- **Physics**: `CAR_MASS`, `GRAVITY_MS2`, tire parameters
- **Rendering**: Display settings, colors, UI layout
- **Rewards**: All reward/penalty values
- **Environment**: Termination conditions, normalization ranges
- **Track**: Default dimensions, validation rules

## Development Commands

### Testing
```bash
pytest                          # Run all tests
pytest tests/                   # Run specific test directory
```

### Environment Setup
```bash
pip install -r requirements.txt    # Install dependencies
python -m venv .venv               # Create virtual environment
source .venv/bin/activate          # Activate (Linux/Mac)
```

### Running Applications
```bash
python game/time_trial.py         # Time trial mode
python game/competition.py        # Competition mode
python game/random_demo.py        # Random demonstration
python learn/ppo_simple.py        # Train PPO model
python tools/track_tool.py gui    # Launch track builder
```

## Performance Characteristics

### Environment Performance
- **Physics Rate**: 60 Hz fixed timestep
- **Rendering Rate**: Variable (30-120 FPS typical)
- **Multi-car Scaling**: Linear performance degradation
- **Memory Usage**: ~100MB base + ~50MB per additional car

### Training Performance
- **Single Environment**: ~1000 steps/second
- **Vectorized (8 envs)**: ~6000 steps/second  
- **Typical Training Time**: 2-6 hours for 25M timesteps (TD3)

### Observation Processing
- **Sensor Calculation**: Ray casting with Box2D physics
- **Normalization**: Pre-computed constants for efficiency
- **Memory Layout**: Contiguous numpy arrays for ML frameworks

## Troubleshooting

### Common Issues

**Environment Not Rendering**
```python
# Ensure pygame is installed
pip install pygame

# Check render_mode parameter
env = CarEnv(render_mode="human")
```

**Physics Simulation Slow**
```python
# Disable rendering for training
env = CarEnv(render_mode=None)

# Reduce number of cars
env = CarEnv(num_cars=1)
```

**Track Loading Errors**
```bash
# Validate track file
python tools/track_tool.py validate tracks/your_track.track

# Check file path and format
```

### Performance Optimization

- Use `render_mode=None` for training
- Vectorize environments with `stable-baselines3`
- Limit `num_cars` for complex scenarios
- Monitor memory usage with multiple cars
- Use pre-compiled Box2D for better physics performance

## Contributing

### Code Style
- Follow PEP 8 standards
- Use type hints where possible
- Document complex physics calculations
- Test multi-car scenarios

### Adding New Features
1. **New Track Commands**: Extend `TrackLoader` in `src/track_generator.py`
2. **Reward Components**: Add to `src/constants/rewards.py`
3. **Observation Elements**: Update `_get_multi_obs()` in `CarEnv`
4. **Game Modes**: Create new scripts in `game/` directory

### Testing Guidelines
- Test both single and multi-car scenarios
- Verify observation/action space consistency  
- Test with different track configurations
- Performance test with maximum cars (10)

---

## License

This project is available for educational and research purposes. See the repository for specific licensing terms.

## Acknowledgments

- **Box2D Physics Engine** - Realistic collision detection and dynamics
- **Gymnasium** - Standard RL environment interface
- **Stable Baselines3** - Reference RL algorithm implementations  
- **Pygame** - Graphics rendering and user interface
- **NumPy** - Efficient numerical computations

---

*For additional documentation, examples, and tutorials, see the individual module docstrings and the `game/` directory for practical usage examples.*