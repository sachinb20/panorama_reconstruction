# IMU Orientation Estimation & Panorama Generation

## Project Structure

```
code/
├── imu_core.py
├── imu_optimization.py
├── part1.py
├── part1_optimization.py
├── experiment_alpha.py
├── run_experiments.py
├── run_opt_strat.py
└── part2/
    ├── panorama.py
    └── panorama_test.py
```

---

## Main Files

| File | Description |
|------|-------------|
| `imu_core.py` | Core classes: `IMUDataLoader`, `QuaternionOperations`, `OrientationEstimator` |
| `imu_optimization.py` | Optimizers: `TrajectoryOptimizer`, `GeodesicTrajectoryOptimizer` |
| `part1.py` | Dead-reckoning orientation estimation |
| `part1_optimization.py` | Trajectory optimization on all datasets |
| `experiment_alpha.py` | Learning rate (α) convergence analysis |
| `run_experiments.py` | Init comparison & motion weight sensitivity experiments |
| `run_opt_strat.py` | Projected GD vs Geodesic GD comparison |
| `part2/panorama.py` | Panorama generation (trainset, VICON & IMU) |
| `part2/panorama_test.py` | Panorama generation (testset, IMU only) |

---

## How to Run

```bash
pip install numpy matplotlib torch transforms3d

python part1.py
python part1_optimization.py
python part1_optimization.py testset

python experiment_alpha.py
python run_experiments.py
python run_opt_strat.py

cd part2/
python panorama.py
python panorama_test.py
```
