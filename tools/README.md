# Track Builder & Analysis Tools

This directory contains comprehensive tools for analyzing, validating, and visualizing race track files.

## Components

### Core Modules

- **`track_analyzer.py`** - Comprehensive track analysis with statistics and performance metrics
- **`track_validator.py`** - Enhanced validation system with loop closure and self-intersection checks
- **`track_builder.py`** - Interactive GUI for track visualization and analysis
- **`track_tool.py`** - Command-line interface for all track operations

## Usage

### Command Line Interface

The main entry point is `track_tool.py` which provides several commands:

#### List Available Tracks
```bash
python tools/track_tool.py list
python tools/track_tool.py list --directory path/to/tracks
```

#### Analyze Tracks
```bash
# Analyze single track
python tools/track_tool.py analyze tracks/nascar.track

# Detailed analysis
python tools/track_tool.py analyze tracks/nascar.track --verbose

# Batch analysis
python tools/track_tool.py analyze tracks/*.track --batch

# Generate JSON report
python tools/track_tool.py analyze tracks/nascar.track --json report.json
```

#### Validate Tracks
```bash
# Validate single track
python tools/track_tool.py validate tracks/nascar.track

# Batch validation
python tools/track_tool.py validate tracks/*.track --batch

# Strict validation (warnings as errors)
python tools/track_tool.py validate tracks/*.track --strict
```

#### Launch GUI
```bash
# Launch GUI
python tools/track_tool.py gui

# Launch with specific track
python tools/track_tool.py gui tracks/nascar.track
```

### GUI Controls

When using the graphical interface:

- **O** - Load track file (cycles through available tracks)
- **M** - Change view mode (overhead/centerline/boundaries/segments)  
- **R** - Reset view to fit track
- **ESC** - Exit
- **Mouse Wheel** - Zoom in/out
- **Click & Drag** - Pan the view

### View Modes

1. **Overhead** - Shows complete track with boundaries filled
2. **Centerline** - Shows only the track centerline
3. **Boundaries** - Shows left (blue) and right (red) track boundaries  
4. **Segments** - Shows individual segments color-coded by type

## Analysis Features

### Track Statistics
- Total length and segment count
- Curve analysis (count, angles, radii)
- Track geometry (width, area, bounds)
- Performance estimates (lap time, difficulty rating)

### Validation Checks
- **Basic Structure** - Required segments, proper ordering
- **Loop Closure** - Position and heading continuity  
- **Geometry** - Segment lengths, widths, curve parameters
- **Safety** - Minimum radii, turn rates, track width
- **Self-Intersection** - Boundary collision detection

### Output Formats
- **Console** - Human-readable text reports
- **JSON** - Machine-readable structured data
- **Interactive** - Real-time GUI visualization

## Track File Format

Track files use a simple command-based format:

```
WIDTH 25
GRID
STARTLINE  
STRAIGHT 200 5      # 200m straight with 5° banking
LEFT 180 300 15     # 180° left turn, 300m radius, 15° banking
STRAIGHT 400 8      # 400m straight with 8° banking  
RIGHT 90 250        # 90° right turn, 250m radius (no banking)
STRAIGHT 95 5       # 95m straight with 5° banking
```

### Commands
- `WIDTH <meters>` - Set track width
- `GRID` - Add grid/staging area
- `STARTLINE` - Add start line
- `FINISHLINE` - Add finish line (optional)
- `STRAIGHT <length> [banking]` - Add straight segment (optional banking in degrees)
- `LEFT <angle> <radius> [banking]` - Add left turn (optional banking in degrees)
- `RIGHT <angle> <radius> [banking]` - Add right turn (optional banking in degrees)

## Requirements

- Python 3.7+
- pygame (for GUI)
- Standard library modules

## Example Output

```
TRACK ANALYSIS REPORT: nascar.track
====================================

BASIC INFORMATION:
  Total Length: 2685.0 m
  Total Segments: 7
  Curve Segments: 2
  Width: 25.0 m

PERFORMANCE ESTIMATES:
  Estimated Lap Time: 60.9 seconds
  Technical Difficulty: 0.2/10

VALIDATION STATUS:
  ✓ Track is a valid closed loop
```

## Integration

These tools integrate seamlessly with the existing car racing simulation:

- Uses the same track loading system (`TrackLoader`)
- Compatible with all existing `.track` files
- Leverages physics and rendering components
- Can validate tracks before use in simulations