# Car Racing Simulation Environment

A comprehensive car racing simulation environment built with realistic physics, reinforcement learning capabilities, and multiple game modes. This system provides a complete racing ecosystem with Gymnasium-compatible environment interface, advanced car physics, track management, and AI training capabilities.

## Table of Contents

- [Environment Overview](#environment-overview)
- [Environment Observation Space](#environment-observation-space)
- [Environment Action Space](#environment-action-space)
- [Environment Info Dictionary](#environment-info-dictionary)
- [Reward Structure](#reward-structure)
- [Game Modes](#game-modes)
- [Controllers](#controllers)
- [Learning Methods](#learning-methods)
- [Tools](#tools)
- [Track Files](#track-files)
- [Creating New Components](#creating-new-components)
- [Installation and Usage](#installation-and-usage)

## Environment Overview

The car racing environment (`CarEnv`) is built on top of Gymnasium and provides a realistic racing simulation with:

- **Realistic Physics**: Box2D-based physics simulation with tire temperature, wear, and collision detection
- **Distance Sensors**: 16-directional distance sensors for AI navigation
- **Multi-car Support**: Support for up to 10 cars simultaneously
- **Track System**: Comprehensive track loading and management system
- **Lap Timing**: Precise lap timing and progress tracking
- **Flexible Action Spaces**: Both discrete and continuous control modes

### Key Features

- **Environment Hierarchy**: `BaseEnv` → `CarEnv` structure with Gymnasium compatibility
- **Physics Integration**: Realistic car dynamics with tire physics, aerodynamics, and collision detection
- **Sensor System**: 16-directional distance sensors at 22.5° intervals
- **Track Integration**: Dynamic track loading from `.track` files
- **Rendering**: Pygame-based real-time visualization

## Environment Observation Space

The observation space is a **38-dimensional vector** containing comprehensive car state information, all normalized to `[-1, 1]` or `[0, 1]` ranges for optimal neural network training.

### Observation Vector Structure (38 elements total)

| Index | Element | Range | Description |
|-------|---------|-------|-------------|
| 0-1 | `pos_x`, `pos_y` | [-1, 1] | Car position coordinates (normalized to world bounds) |
| 2-3 | `vel_x`, `vel_y` | [-1, 1] | Car velocity components (normalized to max velocity) |
| 4 | `speed_magnitude` | [0, 1] | Speed magnitude (normalized to max realistic speed) |
| 5-6 | `orientation`, `angular_vel` | [-1, 1] | Car orientation and angular velocity |
| 7-10 | `tyre_load_fl/fr/rl/rr` | [0, 1] | Tire load forces (front-left, front-right, rear-left, rear-right) |
| 11-14 | `tyre_temp_fl/fr/rl/rr` | [0, 1] | Tire temperatures (affects grip and performance) |
| 15-18 | `tyre_wear_fl/fr/rl/rr` | [0, 1] | Tire wear percentages (affects performance over time) |
| 19-20 | `collision_impulse`, `collision_angle` | [0, 1], [-1, 1] | Collision impact force and relative angle |
| 21 | `cumulative_impact_percentage` | [0, 1] | Accumulated collision damage |
| 22-37 | `sensor_dist_0` to `sensor_dist_15` | [0, 1] | Distance sensor readings (16 directions) |

### Normalization Details

- **Position**: Normalized by `NORM_MAX_POSITION = 10000.0` meters
- **Velocity**: Normalized by `NORM_MAX_VELOCITY = 111.1` m/s (≈400 km/h)
- **Angular Velocity**: Normalized by `NORM_MAX_ANGULAR_VEL = 10.0` rad/s
- **Tire Temperature**: Normalized by `NORM_MAX_TYRE_TEMP = 200.0`°C
- **Tire Load**: Normalized by `MAX_TYRE_LOAD = CAR_MASS * GRAVITY * 2.0`
- **Sensor Distances**: Normalized by `SENSOR_MAX_DISTANCE = 250.0` meters

### Distance Sensors Configuration

The 16 distance sensors are positioned around the car at 22.5° intervals:
- **Sensor 0**: 0° (forward)
- **Sensor 1**: 22.5° (forward-right)
- **Sensor 4**: 90° (right)
- **Sensor 8**: 180° (backward)
- **Sensor 12**: 270° (left)
- **etc.**

Each sensor returns normalized distance (0.0 = obstacle at car position, 1.0 = no obstacle within max range).

## Environment Action Space

The environment supports both **discrete** and **continuous** action spaces, configurable during initialization.

### Continuous Action Space (Default)

**Shape**: `(2,)` - Two-dimensional continuous control
**Range**: `[-1.0, 1.0]` for both dimensions

| Index | Action | Range | Description |
|-------|---------|-------|-------------|
| 0 | `throttle_brake` | [-1.0, 1.0] | Combined throttle/brake axis: <br/>• `1.0` = Full throttle<br/>• `0.0` = Neutral<br/>• `-1.0` = Full brake |
| 1 | `steering` | [-1.0, 1.0] | Steering angle: <br/>• `-1.0` = Full left turn<br/>• `0.0` = Straight<br/>• `1.0` = Full right turn |

### Discrete Action Space

**Shape**: `Discrete(5)` - Five discrete actions

| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | Do Nothing | No throttle, no brake, no steering |
| 1 | Accelerate | Full throttle, no steering |
| 2 | Brake | Full brake, no steering |
| 3 | Turn Left | No throttle/brake, full left steering |
| 4 | Turn Right | No throttle/brake, full right steering |

### Multi-Car Action Spaces

For multi-car environments (`num_cars > 1`):
- **Continuous**: `Box(shape=(num_cars, 2))` - Array of 2D actions
- **Discrete**: `MultiDiscrete([5] * num_cars)` - Array of discrete actions

### Internal Action Conversion

Actions are internally converted to a 3-element format `[throttle, brake, steering]`:
- Continuous `throttle_brake` ≥ 0 → `throttle = throttle_brake, brake = 0`
- Continuous `throttle_brake` < 0 → `throttle = 0, brake = -throttle_brake`

## Environment Info Dictionary

The `info` dictionary returned by `step()` and `reset()` contains comprehensive environment state information:

### Structure Overview

```python
info = {
    "elapsed_time": float,          # Episode time in seconds
    "last_action": [float, float, float],  # [throttle, brake, steering]
    "throttle": float,              # Current throttle value [0, 1]
    "brake": float,                 # Current brake value [0, 1] 
    "steering": float,              # Current steering value [-1, 1]
    "termination_reason": str,      # Why episode ended (if terminated)
    "cars": [...],                  # Per-car detailed information
    "physics": {...}                # Physics simulation statistics
}
```

### Per-Car Information (`info["cars"][i]`)

Each car's info contains:

```python
car_info = {
    "car_index": int,               # Car identifier
    "car_name": str,                # Car name
    "disabled": bool,               # Whether car is disabled
    "position": [float, float],     # World coordinates [x, y]
    "velocity": [float, float],     # Velocity components [vx, vy]
    "speed": float,                 # Speed magnitude (m/s)
    "orientation": float,           # Car heading (radians)
    "angular_velocity": float,      # Rotation rate (rad/s)
    "lap_count": int,              # Completed laps
    "lap_times": [float, ...],     # All lap times
    "best_lap_time": float,        # Best lap time (or None)
    "current_lap_time": float,     # Current lap progress time
    "track_progress": float,       # Progress around track [0, 1]
    "distance_traveled": float,    # Total distance traveled
    "tire_data": {...},            # Tire temperatures, loads, wear
    "collision_data": {...},       # Collision history and damage
    "reward": float,               # Current step reward
    "cumulative_reward": float     # Total episode reward
}
```

### Physics Statistics (`info["physics"]`)

Contains simulation performance metrics:

```python
physics_info = {
    "fps": float,                   # Current simulation FPS
    "physics_steps_per_frame": int, # Physics substeps
    "total_physics_steps": int,     # Cumulative physics steps
    "simulation_time": float,       # Total simulation time
    "real_time_factor": float      # Simulation speed relative to real-time
}
```

## Reward Structure

The reward system encourages fast, safe driving with multiple components:

### Positive Rewards

| Component | Value | Description |
|-----------|--------|-------------|
| `REWARD_DISTANCE_MULTIPLIER` | 0.15 | Bonus per meter of forward progress |
| `REWARD_LAP_COMPLETION` | 0.0 | Bonus for completing a lap (currently disabled) |
| `REWARD_FAST_LAP_BONUS` | 0.0 | Bonus for fast lap times (currently disabled) |

### Negative Rewards (Penalties)

| Component | Value | Description |
|-----------|--------|-------------|
| `PENALTY_PER_STEP` | -0.05 | Small penalty each step (encourages speed) |
| `PENALTY_BACKWARD_PER_METER` | -0.05 | Penalty for backward movement |
| `PENALTY_WALL_COLLISION_PER_STEP` | -0.5 | Penalty during wall collisions |
| `PENALTY_DISABLED` | -1.0 | Large penalty when car is disabled |

### Reward Calculation Logic

1. **Distance Reward**: `REWARD_DISTANCE_MULTIPLIER × meters_forward_progress`
2. **Time Penalty**: `-PENALTY_PER_STEP` (encourages faster completion)
3. **Collision Penalty**: `-PENALTY_WALL_COLLISION_PER_STEP` while colliding
4. **Backward Movement**: `-PENALTY_BACKWARD_PER_METER × backward_distance`
5. **Disability**: `-PENALTY_DISABLED` when car becomes disabled

### Termination Conditions

- **Reward Termination**: Episode ends if cumulative reward < `TERMINATION_MIN_REWARD = -250.0`
- **Time Termination**: Episode ends after `TERMINATION_MAX_TIME = 60.0` seconds
- **Time Truncation**: Hard limit at `TRUNCATION_MAX_TIME = 180.0` seconds

## Game Modes

The system provides multiple game modes for different racing scenarios:

### 1. Time Trial Mode (`game/time_trial.py`)

**Objective**: Achieve the fastest single lap time across multiple attempts.

**Rules**:
- Each car gets 3 attempts
- Each attempt consists of 2 laps
- Environment resets between attempts for fair conditions
- Winner determined by fastest single lap time
- Disabled cars ranked last

**Usage**:
```bash
python game/time_trial.py
```

### 2. Competition Mode (`game/competition.py`)

**Objective**: Multi-car racing competition with position-based scoring.

**Rules**:
- Multiple cars race simultaneously
- Ranking based on laps completed and finishing times
- Real-time collision and interaction between cars
- Final positions determine winner

**Usage**:
```bash
python game/competition.py
```

### 3. Championship Mode (`game/championship.py`)

**Objective**: Multi-track championship with comprehensive points system.

**Rules**:
- **Two-stage racing per track**:
  - **Time Trial Stage**: 3 attempts × 2 laps, points: [15, 7, 3] for top 3
  - **Competition Stage**: Race positions, points: [0, 1, 2, 3, 5, 8, 13, 21, 24, 45]
- **Fastest lap bonus**: 15 additional points in competition stage
- **Multi-track campaign**: Race across all available tracks
- **Comprehensive statistics**: Track performance, overall standings

**Usage**:
```bash
python game/championship.py
```

## Controllers

The system features a modular controller architecture with multiple AI implementations:

### Base Controller Architecture

All controllers inherit from `BaseController` (`game/control/base_controller.py`):

```python
class BaseController:
    def __init__(self, name=None):
        self.name = name
        self.control_state = {...}  # Per-instance state
    
    def control(self, observation):
        """Main control method - returns [throttle_brake, steering]"""
        pass
    
    def _fallback_control(self, observation):
        """Rule-based fallback control logic"""
        pass
```

### Available Controllers

#### 1. **Rule-Based Controller** (`BaseController`)
- **Type**: Hand-crafted rule-based logic  
- **Strategy**: Sensor-based speed and steering control
- **Features**: 
  - Speed adaptation based on forward distance sensor
  - Steering based on left/right sensor comparison
  - Throttle reduction during steering for stability

#### 2. **PPO Controller** (`PPOController`)
- **Type**: Proximal Policy Optimization reinforcement learning
- **Features**:
  - Pre-trained model loading from checkpoints
  - Fallback to rule-based control if model unavailable
  - Per-instance model management for multi-car scenarios

#### 3. **TD3 Controller** (`TD3Controller`) 
- **Type**: Twin Delayed Deep Deterministic Policy Gradient
- **Features**:
  - Deterministic policy for precise control
  - Model checkpoint loading
  - Robust fallback mechanism

#### 4. **SAC Controller** (`SACController`)
- **Type**: Soft Actor-Critic reinforcement learning
- **Features**:
  - Maximum entropy policy optimization
  - Continuous action space specialization
  - Temperature parameter learning

#### 5. **A2C Controller** (`A2CController`)
- **Type**: Advantage Actor-Critic
- **Features**: 
  - Actor-critic architecture
  - Value function approximation
  - Policy gradient optimization

#### 6. **Genetic Controller** (`GeneticController`)
- **Type**: Genetically evolved rule-based parameters
- **Features**:
  - Evolved parameter optimization
  - Rule-based structure with optimized constants
  - Population-based parameter search

#### 7. **Regression Controller** (`RegressionController`)
- **Type**: Supervised learning from expert demonstrations
- **Features**:
  - Neural network trained on expert data
  - Behavior cloning approach
  - Fast inference for real-time control

### Controller Selection

Controllers can be dynamically selected and instantiated:

```python
controllers = [
    PPOController("game/control/models/ppo_model.zip", "PPO-V1"),
    TD3Controller("game/control/models/td3_model.zip", "TD3-V1"),
    GeneticController("Genetic-Optimized"),
    BaseController("Rule-Based-Fallback")
]
```

## Learning Methods

The system supports multiple reinforcement learning algorithms with comprehensive training infrastructure:

### Supported Algorithms

#### 1. **Proximal Policy Optimization (PPO)** (`learn/ppo.py`)
- **Type**: On-policy actor-critic method
- **Strengths**: Stable training, good sample efficiency
- **Configuration**:
  - 8 parallel environments
  - 25M timesteps training
  - Continuous action space optimization

#### 2. **Soft Actor-Critic (SAC)** (`learn/sac.py`)
- **Type**: Off-policy maximum entropy method
- **Strengths**: Sample efficient, robust exploration
- **Configuration**:
  - Experience replay buffer
  - Automatic temperature tuning
  - Twin critic networks

#### 3. **Twin Delayed DDPG (TD3)** (`learn/td3.py`)
- **Type**: Off-policy deterministic policy gradient
- **Strengths**: Low variance, deterministic policies
- **Configuration**:
  - Twin critic networks
  - Target policy smoothing
  - Delayed policy updates

#### 4. **Advantage Actor-Critic (A2C)** (`learn/a2c.py`)
- **Type**: On-policy actor-critic method
- **Strengths**: Simple, fast training
- **Configuration**:
  - Advantage estimation
  - Policy and value function optimization

#### 5. **Genetic Algorithm** (`learn/genetic_trainer.py`)
- **Type**: Evolutionary optimization
- **Strengths**: Parameter space exploration
- **Configuration**:
  - Population-based parameter evolution
  - Rule-based controller optimization

#### 6. **Regression Learning** (`learn/regression_trainer.py`)
- **Type**: Supervised learning from demonstrations
- **Strengths**: Fast training from expert data
- **Configuration**:
  - Expert trajectory collection
  - Behavior cloning neural networks

### Training Infrastructure

#### Vectorized Environments
```python
# Multi-environment training for sample efficiency
num_envs = 8
env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
```

#### Monitoring and Logging
- **Tensorboard Integration**: Real-time training metrics
- **Model Checkpointing**: Automatic best model saving  
- **Evaluation Callbacks**: Periodic performance assessment
- **Logging Systems**: Comprehensive training logs

#### Training Configuration
```python
# Common training parameters
total_timesteps = 25_000_000
eval_freq = 12_500
log_interval = 1
```

### Model Management

Trained models are saved in structured directories:
```
game/control/models/
├── ppo_model.zip
├── td3_model.zip  
├── sac_model.zip
└── a2c_model.zip
```

## Tools

The system includes comprehensive tools for track management and analysis:

### Track Tool CLI (`tools/track_tool.py`)

**Command-line interface for track operations**:

```bash
# List available tracks
python tools/track_tool.py list

# Analyze a specific track
python tools/track_tool.py analyze nascar.track

# Validate track file format
python tools/track_tool.py validate daytona.track

# Launch interactive GUI
python tools/track_tool.py gui

# Load specific track in GUI
python tools/track_tool.py gui talladega.track
```

### Track Builder GUI (`tools/track_builder.py`)

**Interactive track visualization and analysis**:

**Features**:
- Real-time track rendering
- Multiple view modes (overhead, centerline, boundaries, segments)
- Track statistics display
- Validation feedback
- Analysis metrics

**View Modes**:
- **Overhead**: Complete track overview
- **Centerline**: Racing line visualization
- **Boundaries**: Track boundary display
- **Segments**: Individual segment breakdown

### Track Analyzer (`tools/track_analyzer.py`)

**Comprehensive track analysis**:

**Capabilities**:
- Track length calculation
- Segment analysis
- Curvature metrics
- Banking angle analysis
- Difficulty assessment

**Output Statistics**:
```python
TrackStatistics = {
    "total_length": float,          # Total track length (meters)
    "segment_count": int,           # Number of track segments
    "straight_length": float,       # Total straight sections
    "curve_length": float,          # Total curved sections
    "max_banking": float,           # Maximum banking angle
    "avg_curvature": float,         # Average curvature
    "difficulty_score": float       # Calculated difficulty rating
}
```

### Track Validator (`tools/track_validator.py`)

**Track file format validation**:

**Validation Checks**:
- File format correctness
- Segment continuity
- Track closure verification
- Banking angle limits
- Minimum segment lengths

**Validation Result**:
```python
ValidationResult = {
    "is_valid": bool,               # Overall validity
    "errors": [str, ...],           # Error messages
    "warnings": [str, ...],         # Warning messages  
    "suggestions": [str, ...]       # Improvement suggestions
}
```

## Track Files

Track files use a simple text-based format for defining racing circuits.

### Track File Format

Track files (`.track` extension) contain sequential commands defining track geometry:

```
WIDTH <width_in_meters>
GRID                           # Starting grid area
STARTLINE                      # Start/finish line
STRAIGHT <length>              # Straight section
LEFT <angle> <radius> [banking] # Left curve
RIGHT <angle> <radius> [banking] # Right curve
FINISHLINE                     # Finish line (optional)
```

### Example Track File (`nascar.track`)

```
WIDTH 25
GRID
STARTLINE
STRAIGHT 200
LEFT 180 300
STRAIGHT 400  
LEFT 180 300
STRAIGHT 95
```

### Track Commands Reference

| Command | Parameters | Description |
|---------|------------|-------------|
| `WIDTH` | `<meters>` | Sets default track width |
| `GRID` | None | Creates starting grid area |
| `STARTLINE` | None | Places start/finish line |
| `FINISHLINE` | None | Places separate finish line |
| `STRAIGHT` | `<length>` | Straight section in meters |
| `LEFT` | `<angle> <radius> [banking]` | Left turn: degrees, radius(m), banking(°) |
| `RIGHT` | `<angle> <radius> [banking]` | Right turn: degrees, radius(m), banking(°) |

### Advanced Track Features

#### Banking Angles
```
LEFT 180 300 31    # 31° banked left turn
RIGHT 90 150 15    # 15° banked right turn
```

#### Variable Width (Future Enhancement)
```
WIDTH 20           # Change width mid-track
STRAIGHT 100       # Applies new width
```

### Available Tracks

| Track | Type | Description |
|-------|------|-------------|
| `nascar.track` | Oval | Basic NASCAR-style oval |
| `daytona.track` | Superspeedway | High-speed banked oval |
| `talladega.track` | Superspeedway | Large banked oval |
| `martinsville.track` | Short Track | Tight, flat turns |
| `michigan.track` | Intermediate | Medium-speed oval |
| `trioval.track` | Trioval | Three-turn configuration |
| `nascar2.track` | Modified Oval | Alternative oval design |
| `nascar_banked.track` | Banked Oval | Heavily banked turns |

## Creating New Components

### Creating a New Track

#### Method 1: Manual Track File Creation

1. **Create track file** (`tracks/mytrack.track`):
```
WIDTH 20
GRID
STARTLINE
STRAIGHT 300
RIGHT 90 200 10
STRAIGHT 500
LEFT 180 250 15
STRAIGHT 400
RIGHT 90 200 10
STRAIGHT 195
```

2. **Test track validity**:
```bash
python tools/track_tool.py validate mytrack.track
```

3. **Analyze track**:
```bash
python tools/track_tool.py analyze mytrack.track
```

#### Method 2: Interactive GUI Creation

1. **Launch track builder**:
```bash
python tools/track_tool.py gui
```

2. **Load and modify existing track**:
```bash
python tools/track_tool.py gui nascar.track
```

### Creating a New Game Mode

1. **Create game mode file** (`game/my_game_mode.py`):

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.car_env import CarEnv
from game.control.base_controller import BaseController

def run_my_game_mode():
    # Initialize environment
    env = CarEnv(
        render_mode="human",
        num_cars=2,
        discrete_action_space=False
    )
    
    # Create controllers
    controllers = [
        BaseController("Player 1"),
        BaseController("Player 2") 
    ]
    
    # Game loop
    observation, info = env.reset()
    done = False
    
    while not done:
        # Get actions from controllers
        actions = []
        for i, controller in enumerate(controllers):
            action = controller.control(observation[i] if env.num_cars > 1 else observation)
            actions.append(action)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        
        # Custom game logic here
        process_game_state(info)
    
    env.close()

def process_game_state(info):
    """Custom game state processing logic"""
    # Implement scoring, conditions, etc.
    pass

if __name__ == "__main__":
    run_my_game_mode()
```

2. **Add to CLAUDE.md**:
```bash
python game/my_game_mode.py    # My custom game mode
```

### Creating a New Controller

1. **Create controller class** (`game/control/my_controller.py`):

```python
import numpy as np
from .base_controller import BaseController

class MyController(BaseController):
    """Custom controller implementation"""
    
    def __init__(self, name=None, custom_param=1.0):
        super().__init__(name or "MyController")
        self.custom_param = custom_param
    
    def control(self, observation):
        """
        Custom control logic
        
        Args:
            observation: numpy array of shape (38,) containing car state
            
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
        """
        # Extract sensor data
        sensors = observation[22:38]  # 16 distance sensors
        speed = observation[4]        # Current speed
        
        # Custom control logic
        throttle_brake = self.calculate_throttle(sensors, speed)
        steering = self.calculate_steering(sensors)
        
        return np.array([throttle_brake, steering], dtype=np.float32)
    
    def calculate_throttle(self, sensors, speed):
        """Custom throttle calculation"""
        forward_distance = sensors[0]
        target_speed = forward_distance * self.custom_param
        
        if speed < target_speed:
            return 0.5  # Accelerate
        else:
            return -0.2  # Light braking
    
    def calculate_steering(self, sensors):
        """Custom steering calculation"""  
        left_distance = sensors[12]   # Left sensor
        right_distance = sensors[4]   # Right sensor
        
        if left_distance > right_distance:
            return -0.3  # Turn left
        elif right_distance > left_distance:
            return 0.3   # Turn right
        else:
            return 0.0   # Go straight
```

2. **Use in game modes**:

```python
from game.control.my_controller import MyController

# Create controller instance
controller = MyController("Custom-AI", custom_param=1.5)

# Use in environment
action = controller.control(observation)
```

## Installation and Usage

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `gymnasium>=0.29.1` - RL environment interface
- `stable_baselines3>=2.7.0` - RL algorithms
- `pygame>=2.5.2` - Graphics and rendering
- `numpy>=2.2.6` - Numerical computations  
- `box2d-py>=2.3.8` - Physics simulation
- `torch>=2.8.0` - Neural networks
- `matplotlib>=3.10.5` - Plotting and visualization

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment  
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### Run Time Trial Mode
```bash
python game/time_trial.py
```

#### Run Competition Mode  
```bash
python game/competition.py
```

#### Run Championship Mode
```bash
python game/championship.py
```

#### Train RL Model
```bash
python learn/ppo.py        # Train PPO model
python learn/sac.py        # Train SAC model
python learn/td3.py        # Train TD3 model
```

#### Analyze Tracks
```bash
python tools/track_tool.py list                    # List tracks
python tools/track_tool.py analyze nascar.track    # Analyze track
python tools/track_tool.py gui                     # Launch GUI
```

### Testing

```bash
pytest                     # Run all tests
pytest tests/              # Run specific test directory
```

### Development

The codebase follows a modular architecture with clear separation of concerns:

- `src/` - Core environment and physics
- `game/` - Game modes and controllers  
- `learn/` - Training scripts and algorithms
- `tools/` - Track management utilities
- `tracks/` - Track definition files

For detailed development guidance, see `CLAUDE.md`.

## License

This project is open source. Please check the repository for license details.

## Contributing

Contributions are welcome! Please follow the established code structure and add appropriate tests for new features.