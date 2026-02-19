"""Optimization Strategy Comparison - Projected GD vs Geodesic GD."""

import numpy as np
import sys
import os

from imu_core import IMUDataLoader, OrientationEstimator
from imu_optimization import TrajectoryOptimizer, GeodesicTrajectoryOptimizer

MAX_ITERS = 5000
DATASETS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
MOTION_WEIGHT = 0.5
OBS_WEIGHT = 0.5

PROJ_GD_ALPHA = 0.01
GEO_GD_PHI = 0.01


def run_projected_gd(data_loader, estimator):
    optimizer = TrajectoryOptimizer(motion_weight=MOTION_WEIGHT, obs_weight=OBS_WEIGHT)
    optimizer.optimize(
        q_init=estimator.quaternions,
        omega=data_loader.calibrated_omega,
        acc=data_loader.calibrated_acc,
        timestamps=data_loader.imu_ts,
        alpha=PROJ_GD_ALPHA,
        max_iters=MAX_ITERS,
        tol=0.015,
        verbose=False
    )
    return optimizer


def run_geodesic_gd_ls(data_loader, estimator):
    optimizer = GeodesicTrajectoryOptimizer(motion_weight=MOTION_WEIGHT, obs_weight=OBS_WEIGHT)
    optimizer.optimize(
        q_init=estimator.quaternions,
        omega=data_loader.calibrated_omega,
        acc=data_loader.calibrated_acc,
        timestamps=data_loader.imu_ts,
        phi=GEO_GD_PHI,
        max_iters=MAX_ITERS,
        tol=0.015,
        use_line_search=True,
        verbose=False
    )
    return optimizer


def main():
    results = []
    
    print("=" * 80)
    print("Optimization Strategy Comparison")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Projected GD alpha: {PROJ_GD_ALPHA}, Geodesic GD phi: {GEO_GD_PHI}")
    print("=" * 80)
    
    for dataset in DATASETS:
        print(f"\n>>> Processing Dataset {dataset}...")
        
        imu_file = f"../data/trainset/imu/imuRaw{dataset}.p"
        vicon_file = f"../data/trainset/vicon/viconRot{dataset}.p"
        
        try:
            data_loader = IMUDataLoader(imu_file, vicon_file)
            estimator = OrientationEstimator(data_loader)
            estimator.integrate()
        except Exception as e:
            print(f"  Error loading dataset {dataset}: {e}")
            continue
        
        row = {'dataset': dataset}
        
        print(f"  Running Projected GD...")
        try:
            opt_proj = run_projected_gd(data_loader, estimator)
            row['proj_init'] = opt_proj.initial_cost
            row['proj_cost'] = opt_proj.final_cost
            row['proj_iter'] = opt_proj.converged_iter
        except Exception as e:
            print(f"    Error: {e}")
            row['proj_init'] = float('nan')
            row['proj_cost'] = float('nan')
            row['proj_iter'] = -1
        
        print(f"  Running Geodesic GD + Line Search...")
        try:
            opt_geo_ls = run_geodesic_gd_ls(data_loader, estimator)
            row['geo_ls_init'] = opt_geo_ls.initial_cost
            row['geo_ls_cost'] = opt_geo_ls.final_cost
            row['geo_ls_iter'] = opt_geo_ls.converged_iter
        except Exception as e:
            print(f"    Error: {e}")
            row['geo_ls_init'] = float('nan')
            row['geo_ls_cost'] = float('nan')
            row['geo_ls_iter'] = -1
        
        results.append(row)
        print(f"  Done! Proj: {row.get('proj_cost', 0):.4f} (iter {row.get('proj_iter', 0)}), "
              f"Geo+LS: {row.get('geo_ls_cost', 0):.4f} (iter {row.get('geo_ls_iter', 0)})")
    
    print("\n")
    print("=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    
    print(f"{'Dataset':^8} | {'Projected GD':^35} | {'Geodesic GD + Line Search':^35}")
    print(f"{'':^8} | {'Init':^10} {'Final':^10} {'Conv Iter':^12} | {'Init':^10} {'Final':^10} {'Conv Iter':^12}")
    print("-" * 90)
    
    for row in results:
        ds = row['dataset']
        proj_iter = row.get('proj_iter', 0)
        geo_iter = row.get('geo_ls_iter', 0)
        proj_conv = f"{proj_iter}" if proj_iter < MAX_ITERS else "NC"
        geo_conv = f"{geo_iter}" if geo_iter < MAX_ITERS else "NC"
        
        print(f"{ds:^8} | {row.get('proj_init', 0):>10.2f} {row.get('proj_cost', 0):>10.4f} {proj_conv:^12} | "
              f"{row.get('geo_ls_init', 0):>10.2f} {row.get('geo_ls_cost', 0):>10.4f} {geo_conv:^12}")
    
    print("=" * 90)
    print("NC = Not Converged (reached max iterations)")
    
    print("\nSUMMARY:")
    avg_proj = np.nanmean([r.get('proj_cost', np.nan) for r in results])
    avg_geo_ls = np.nanmean([r.get('geo_ls_cost', np.nan) for r in results])
    proj_converged = sum(1 for r in results if r.get('proj_iter', MAX_ITERS) < MAX_ITERS)
    geo_converged = sum(1 for r in results if r.get('geo_ls_iter', MAX_ITERS) < MAX_ITERS)
    
    print(f"  Projected GD:           avg final cost = {avg_proj:.4f}, converged = {proj_converged}/{len(results)}")
    print(f"  Geodesic GD + LS:       avg final cost = {avg_geo_ls:.4f}, converged = {geo_converged}/{len(results)}")
    
    os.makedirs("../results", exist_ok=True)
    csv_path = "../results/optimization_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("Dataset,Proj_Init,Proj_Final,Proj_ConvIter,GeoLS_Init,GeoLS_Final,GeoLS_ConvIter\n")
        for row in results:
            f.write(f"{row['dataset']},{row.get('proj_init', '')},{row.get('proj_cost', '')},{row.get('proj_iter', '')},"
                   f"{row.get('geo_ls_init', '')},{row.get('geo_ls_cost', '')},{row.get('geo_ls_iter', '')}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
