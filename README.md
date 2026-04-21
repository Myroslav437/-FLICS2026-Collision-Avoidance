# Benchmarking Lightweight Collision-Avoidance Pipelines for AGVs

A modular, simulation-based benchmarking framework for evaluating lightweight
collision-avoidance pipelines for Automated Guided Vehicles (AGVs) equipped
with LiDAR sensors.

> **Publication:** The results obtained with this framework are being prepared
> for publication at _(conference TBD)_. A link to the published paper will be
> added here once available.

The framework consists of three decoupled components:

- **World Generator** — procedural indoor environments (BSP partitioning +
  PRM reference paths) producing reproducible initial world states.
- **Simulation Engine** — closed-loop simulator with a swappable three-stage
  pipeline (detection → fusion → avoidance) and a stochastic perception
  function (prior-map degradation + distorted LiDAR).
- **Visualization** — static and telemetry-driven rendering of worlds and
  simulation runs.

## Project structure

```
config/
  default_world.yaml        # World-generator parameters
src/
  world_generator/          # Reproducible world generation
    models.py               # Geometric and kinematic data classes
    config.py               # Parameter dataclasses
    bsp.py                  # Recursive BSP environment carving
    prm.py                  # PRM + Dijkstra reference path planner
    obstacles.py            # Obstacle placement + dynamic trajectories
    generator.py            # Top-level generate_world(...)
    metrics.py              # Corpus-level validation metrics
  simulation_engine/        # Closed-loop simulator
    config.py               # Sigma = <Lambda, Phi, Delta> configurations
    types.py                # Data classes (PriorMap, LiDARScan, metrics, ...)
    perception/             # Prior-map degradation + LiDAR simulation
    stages/                 # Swappable detection/fusion/avoidance stages
      detection/            #   EC, DB
      fusion/               #   PT, KF
      avoidance/            #   VFH, DWA
      registry.py           # Name -> factory registry for all three
    core/                   # Main loop, metrics aggregator, telemetry sink, report generator
  simulation_visualizer/    # Post-hoc telemetry playback
    reader.py               # JSONL telemetry parser
    panels/                 # One module per visualization panel
      objective_world.py    #   True world state W_t
      perception.py         #   Perceived world W_tilde_t
      pipeline_output.py    #   Detection, fusion, avoidance outputs
      metrics.py            #   Six per-run metrics (§IV-B)
    layout.py               # 2×2 panel compositor
    playback.py             # Animation timing + file rendering
    __main__.py             # CLI driver
  visualization/
    world.py                # Static world rendering
scripts/
  generate_worlds.py        # CLI: build a flat or stratified corpus
  visualize_world.py        # CLI: render a saved world to PNG
  run_simulation.py         # CLI: run a single simulation iteration
  run_benchmark.py          # CLI: parallel benchmarking runner
  analyze_results.py        # CLI: analysis pipeline for run reports
tests/                      # pytest suite
```

## Install

```bash
python -m venv venv
./venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt   # Linux/mac
```

Dependencies: `numpy`, `scipy`, `shapely`, `networkx`, `matplotlib`, `pyyaml`,
`scikit-learn`, `filterpy`, `pytest`.

## Usage

### Generate a single world (Default)

```bash
./venv/Scripts/python scripts/generate_worlds.py \
    --config config/default_world.yaml \
    --seed 42 \
    --output data/sample_single
```

### Generate a small flat corpus

```bash
./venv/Scripts/python scripts/generate_worlds.py \
    --config config/default_world.yaml \
    --mode flat --count 10 \
    --output data/sample
```

### Generate a stratified corpus

```bash
./venv/Scripts/python scripts/generate_worlds.py \
    --mode stratified \
    --strata-config config/default_strata.yaml \
    --per-stratum 10 \
    --output data/corpus
```

By default, `--strata-config config/default_strata.yaml` is used. This dynamically generates a full cartesian product combining arrays of variables you declare matching the exact parameter dot-hierarchy.
You can stratify *ANY* parameter from `config/default_world.yaml` using this method!

For example, a custom `my_strata.yaml` might look like:
```yaml
environment.bsp_depth: [1, 2]
obstacles.n_dynamic: [0, 2]
path.prm_samples: [500, 1000]
```

#### BSP depth semantics (`d_bsp`)

| `d_bsp` | Meaning                                                             |
|---------|---------------------------------------------------------------------|
| `0`     | Empty bounded hall: only the outer perimeter walls, no carved rooms |
| `1`     | One carved inner room inside the bounded hall (0 BSP splits)        |
| `2`     | One recursive split -> 2 leaves -> 2 rooms + 1 partition wall       |
| `k`     | `k-1` levels of recursive subdivision, then a room carved per leaf  |

`d_bsp = 0` is a legitimate code path (tested in `tests/test_bsp.py`) even
though the stratified corpus uses only `{1, 2, 3}`.

#### Minimum corridor width (`c_min`)

Because `p_room` is a fractional parameter, the absolute corridor gap between
a leaf's inner room and its surrounding partition walls shrinks with partition
size. At higher BSP depths this can produce unwalkably narrow corridors.
`c_min` is an absolute floor that overrides `p_room` when the two conflict.
For a partition of width `W`:

    nominal_corridor = p_room * W
    actual_corridor  = max(nominal_corridor, c_min)
    room_size        = W - 2 * actual_corridor - 2 * t_wall

If `actual_corridor >= W / 2` (or the room size would be non-positive), room
placement in that partition is skipped cleanly — the partition stays empty.

#### Minimum reference-path length (`l_pi_min`)

Uniform start-goal sampling on the PRM can pick two nodes in the same room,
yielding a trivial intra-room hop. `l_pi_min` is the minimum allowed
shortest-path length between start and goal. Enforcement is
**rejection-resampling**:

1. Build the PRM roadmap on the `d_clear`-inflated free space.
2. Sample a start-goal pair from the largest connected component.
3. Run Dijkstra; sum the Euclidean distances between consecutive waypoints.
4. If the total is below `l_pi_min`, remove the terminals, draw a new pair,
   and try again; otherwise accept.
5. An internal cap (100 attempts; see `_REFERENCE_PATH_MAX_ATTEMPTS` in
   `src/world_generator/prm.py`) bounds the loop so infeasible
   `(Omega, seed)` combinations surface as a `PRMGenerationError` rather
   than looping forever.

The attempt count per world is surfaced through `compute_metrics` and
written to the `manifest.csv` column `prm_reference_attempts`.

### Parameter single source of truth

Every default lives in exactly one place: **`config/default_world.yaml`**.
`src/world_generator/config.py` declares the dataclasses with **no default
values** on any field — every field is required, so forgetting to populate
one fails loudly. Scripts and tests read canonical values through
`Omega.load(config/default_world.yaml)`.

Each world is saved as `world_<id>.json` plus a `manifest.csv` summarising
world-level metrics (free-space ratio, minimum passage width on `pi`,
obstacle counts, path length, generation time).

### Visualise a world

```bash
# single file
./venv/Scripts/python scripts/visualize_world.py \
    --input data/sample/world_00000.json --output data/sample/world_00000.png

# batch (directory -> directory)
./venv/Scripts/python scripts/visualize_world.py \
    --input data/sample --output data/sample/figures
```

### Run a simulation

```bash
./venv/Scripts/python scripts/run_simulation.py \
    --world data/validation_v5/world_00000.json \
    --config config/simulation_nominal.yaml \
    --detection EC --fusion PT --avoidance VFH \
    --report data/run_report.json \
    --telemetry data/run_example.jsonl
```

Available stage names (all combinations supported):

- **Detection:** `EC` (Euclidean clustering), `DB` (DBSCAN)
- **Fusion:** `PT` (pass-through), `KF` (Kalman filter + Hungarian tracker)
- **Avoidance:** `VFH` (Vector Field Histogram), `DWA` (Dynamic Window)

### Run Reports

Every simulation run automatically generates a structured JSON report containing final metrics, configuration metadata, and wall-clock execution time.

**Output Path Logic:**
1. If `--report` is specified, it uses that path.
2. If `--report` is omitted but `--telemetry` is provided, the report is saved as `<telemetry_basename>.report.json`.
3. If both are omitted, the report is saved next to the world file with a name encoding the pipeline configuration: `<world_prefix>.<det>.<fus>.<avd>.<sigma>.<seed>.report.json`.

### Simulation Configuration (`--config`)

Three physical configurations are provided natively modeling the degradation constraints found in the publication:
1. `config/simulation_nominal.yaml`
2. `config/simulation_degraded1.yaml`
3. `config/simulation_degraded2.yaml`

### Visualize a simulation run

The Simulation Visualizer plays back JSONL telemetry files produced by the
engine. It renders four panels (objective world, perception, pipeline output,
metrics) as a matplotlib animation.

```bash
# First, run a simulation with telemetry capture using the CLI runner
./venv/Scripts/python scripts/run_simulation.py \
    --world data/sample/world_00000.json \
    --config config/simulation_nominal.yaml \
    --detection EC --fusion KF --avoidance VFH \
    --seed 42 \
    --telemetry data/run_example.jsonl

# Play back at wall-clock speed (live window)
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --speed 1.0

# Play at 2× speed
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --speed 2.0

# Step-through on keypress (space bar)
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --speed 0

# Render to a directory of PNGs
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --render-to data/frames/ --dpi 150

# Render to MP4 (requires ffmpeg on PATH)
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --render-to data/run.mp4

# Render to GIF (requires Pillow)
./venv/Scripts/python -m src.simulation_visualizer play \
    --telemetry data/run_example.jsonl --render-to data/run.gif
```

#### Panel layout

```
+---------------------+---------------------+
|  Objective World    |  Perception (W̃_t)   |
|  (true W_t)         |  (prior map M_0,    |
|                     |   distorted LiDAR)  |
+---------------------+---------------------+
|  Pipeline Output    |  Metrics (§IV-B)    |
|  (detections, fused |  (μ_col, μ_dev,     |
|   tracks, control)  |   μ_goal, μ_vel,    |
|                     |   μ_comp, μ_lat)    |
+---------------------+---------------------+
```

- **Objective World**: Ground-truth W_t — walls, reference path π,
  obstacles at current positions, AGV footprint.
- **Perception**: W̃_t from Eq. (12) — prior map M_0 (dashed lines),
  distorted LiDAR scan hits (blue dots), AGV pose.
- **Pipeline Output**: Detections Ô_t (orange ×), fused tracks Ō_t
  (red circles + velocity arrows), control action (green arrow).
- **Metrics**: Running accumulation of six §IV-B metrics; final values
  shown in bold at run end.

#### Adding a new panel

1. Create `src/simulation_visualizer/panels/my_panel.py`.
2. Subclass `Panel` (from `panels.base`).
3. Implement `setup(ax, header)` and `update(step, step_index, total_steps)`.
4. Register it in `panels/__init__.py`.
5. Add an axis cell in `layout.py`.
### Run the test suite

```bash
./venv/Scripts/python -m pytest tests/ -v
```

## Validation conditions

A corpus generated by `scripts/generate_worlds.py` must satisfy:

1. **Navigability-safety:** `min_passage_width(pi) > W_agv + 2 * d_clear` for
   every retained world.
2. **Free-space ratio decreases with `d_bsp`** (more partitions carve more
   wall slabs into the same `X * Y` domain).
3. **Corridor floor:** every per-world `min_corridor_width` must be at least
   `c_min` (within floating-point tolerance).
4. **Reference-path length floor:** every per-world `path_length` must be at
   least `l_pi_min` (within floating-point tolerance).

Run `scripts/validate_corpus.py` against the manifest:

```bash
./venv/Scripts/python scripts/validate_corpus.py \
    --manifest data/corpus/manifest.csv \
    --config config/default_world.yaml
```

## Adding Custom Pipeline Algorithms

The Simulation Engine architecture is pluggable. It resolves pipeline stages using dynamically registered factory functions rather than hardcoded logic. You can easily add and experiment with your own algorithms for any of the three stages (Detection, Fusion, Avoidance).

### 1. Subclass the relevant interface
Subclass `DetectionStage`, `FusionStage`, or `AvoidanceStage` from `src.simulation_engine.stages.base`.

```python
# src/simulation_engine/stages/detection/my_detector.py
import numpy as np
from src.simulation_engine.stages.base import DetectionStage
from src.simulation_engine.types import LiDARScan, DetectedObstacle

class MyDetector(DetectionStage):
    name = "MY-DET"

    def process(self, scan: LiDARScan) -> list[DetectedObstacle]:
        # Your custom detection algorithm logic here
        return []
```

### 2. Register your algorithm
Bind your class into the engine's registry.

```python
# Typically done where your stages are initialized, or right under the class definition
from src.simulation_engine.stages.registry import register_detection

register_detection("MY-DET", MyDetector)
```

### 3. Run the simulation
You can now pass your custom string identifier natively through the simulation tools!

```bash
./venv/Scripts/python scripts/run_simulation.py \
    --world data/validation_v5/world_00000.json \
    --config config/simulation_nominal.yaml \
    --detection MY-DET --fusion PT --avoidance VFH \
    --telemetry my_custom_test.jsonl
```

## Authors

- (add authors)
