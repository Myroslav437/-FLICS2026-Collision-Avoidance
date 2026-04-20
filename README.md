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
data/                       # Generated corpora (git-ignored)
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
    core/                   # Main loop, metrics aggregator, telemetry sink
  visualization/
    world.py                # Static world rendering
scripts/
  generate_worlds.py        # CLI: build a flat or stratified corpus
  visualize_world.py        # CLI: render a saved world to PNG
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
    --config config/default_world.yaml \
    --mode stratified --per-stratum 10 \
    --output data/corpus
```

Stratification axes:

- `d_bsp` in `{1, 2, 3}`
- `n_static` in `{0, 4, 8, 16}`
- `n_dynamic` in `{0, 2, 4, 8}`

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

```python
from world_generator import Omega, generate_world
from simulation_engine import run_simulation, NOMINAL

omega = Omega.load("config/default_world.yaml")
world = generate_world(omega, world_id=0, seed=0)

metrics = run_simulation(
    world, sigma=NOMINAL,
    detection="EC", fusion="KF", avoidance="VFH",
    seed=42,
)
print(metrics.to_dict())
```

Available stage names (all combinations supported):

- **Detection:** `EC` (Euclidean clustering), `DB` (DBSCAN)
- **Fusion:** `PT` (pass-through), `KF` (Kalman filter + Hungarian tracker)
- **Avoidance:** `VFH` (Vector Field Histogram), `DWA` (Dynamic Window)

Available `Sigma` presets: `NOMINAL`, `DEGRADED_1`, `DEGRADED_2`
(plus name-based lookup via `get_sigma("degraded-1")`).

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

## Authors

- (add authors)
