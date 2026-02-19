"""Part 1 Optimization: Quaternion trajectory optimization using geodesic gradient descent."""

import os
import numpy as np
import matplotlib.pyplot as plt

from imu_core import IMUDataLoader, OrientationEstimator
from imu_optimization import TrajectoryOptimizer, optimize_trajectory
from imu_optimization import GeodesicTrajectoryOptimizer, optimize_trajectory_geodesic

TRAINSET_DIR = "../data/trainset"
TESTSET_DIR = "../data/testset"
RESULTS_DIR = "../results"
TRAINSET_INDICES = list(range(1, 10))
TESTSET_INDICES = [10, 11]

ALPHA = 0.01
MAX_ITERS = 2000


def process_dataset(dataset_idx, show_plots=False):
    imu_file = f"{TRAINSET_DIR}/imu/imuRaw{dataset_idx}.p"
    vicon_file = f"{TRAINSET_DIR}/vicon/viconRot{dataset_idx}.p"
    
    print(f"\n{'='*60}")
    print(f"Processing Dataset {dataset_idx}")
    print(f"{'='*60}")
    
    print("Loading data...")
    data_loader = IMUDataLoader(imu_file, vicon_file)
    print(f"Loaded {data_loader.num_samples} samples")
    
    print("\nInitializing quaternion trajectory using motion model...")
    estimator = OrientationEstimator(data_loader)
    estimator.integrate()
    estimator.compute_rpy()
    print(f"Number of quaternions: {len(estimator.quaternions)}")
    
    print("\nOptimizing quaternion trajectory...")
    optimizer = optimize_trajectory(
        data_loader, estimator,
        alpha=ALPHA,
        max_iters=MAX_ITERS,
        verbose=True,
        motion_weight=0.5,
        obs_weight=0.5,
    )
    
    print(f"\nInitial cost: {optimizer.initial_cost:.6f}")
    print(f"Final cost: {optimizer.final_cost:.6f}")
    
    labels = ['Roll', 'Pitch', 'Yaw']
    
    plt.figure(figsize=(14, 10))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, estimator.rpy[:, i], 'g-', 
                 label='Initial', linewidth=1, alpha=0.7)
        plt.plot(data_loader.imu_ts, optimizer.optimized_rpy[:, i], 'r-', 
                 label='Optimized', linewidth=1.5)
        plt.plot(data_loader.vicon_ts, data_loader.vicon_rpy[:, i], 'b-', 
                 label='VICON Ground Truth', linewidth=1.5, alpha=0.8)
        plt.ylabel(f'{labels[i]} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle(f'Dataset {dataset_idx}: Quaternion Trajectory Optimization - Estimated vs Ground Truth')
    plt.tight_layout()
    rpy_path = f'{RESULTS_DIR}/optimization_rpy_comparison_{dataset_idx}.png'
    plt.savefig(rpy_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"\nPlot saved to {rpy_path}")
    
    g_pred_init = estimator.compute_predicted_gravity()
    g_pred_opt = optimizer.compute_predicted_gravity()
    
    acc_labels = ['Acc X', 'Acc Y', 'Acc Z']
    
    plt.figure(figsize=(14, 10))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, data_loader.calibrated_acc[i, :], 'b-', 
                 label='IMU Measured', linewidth=1, alpha=0.8)
        plt.plot(data_loader.imu_ts, g_pred_init[:, i], 'g-', 
                 label='Initial', linewidth=1, alpha=0.7)
        plt.plot(data_loader.imu_ts, g_pred_opt[:, i], 'r-', 
                 label='Optimized', linewidth=1.5)
        plt.ylabel(f'{acc_labels[i]} (g)')
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle(f'Dataset {dataset_idx}: Acceleration - IMU Measured vs Predicted Gravity')
    plt.tight_layout()
    acc_path = f'{RESULTS_DIR}/optimization_acceleration_comparison_{dataset_idx}.png'
    plt.savefig(acc_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"Plot saved to {acc_path}")
    
    return data_loader, estimator, optimizer


def process_testset(dataset_idx, show_plots=False):
    imu_file = f"{TESTSET_DIR}/imu/imuRaw{dataset_idx}.p"
    
    print(f"\n{'='*60}")
    print(f"Processing Testset Dataset {dataset_idx} (No Ground Truth)")
    print(f"{'='*60}")
    
    print("Loading data...")
    data_loader = IMUDataLoader(imu_file, vicon_file=None)
    print(f"Loaded {data_loader.num_samples} samples")
    
    print("\nInitializing quaternion trajectory using motion model...")
    estimator = OrientationEstimator(data_loader)
    estimator.integrate()
    estimator.compute_rpy()
    print(f"Number of quaternions: {len(estimator.quaternions)}")
    
    print("\nOptimizing quaternion trajectory...")
    optimizer = optimize_trajectory(
        data_loader, estimator,
        alpha=ALPHA,
        max_iters=MAX_ITERS,
        verbose=True,
        motion_weight=0.5,
        obs_weight=0.5,
    )
    
    print(f"\nInitial cost: {optimizer.initial_cost:.6f}")
    print(f"Final cost: {optimizer.final_cost:.6f}")
    
    labels = ['Roll', 'Pitch', 'Yaw']
    
    plt.figure(figsize=(14, 10))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, estimator.rpy[:, i], 'g-', 
                 label='Initial ', linewidth=1, alpha=0.7)
        plt.plot(data_loader.imu_ts, optimizer.optimized_rpy[:, i], 'r-', 
                 label='Optimized', linewidth=1.5)
        plt.ylabel(f'{labels[i]} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle(f'Testset Dataset {dataset_idx}: Optimized Roll, Pitch, Yaw')
    plt.tight_layout()
    rpy_path = f'{RESULTS_DIR}/testset_optimization_rpy_{dataset_idx}.png'
    plt.savefig(rpy_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"\nPlot saved to {rpy_path}")
    
    g_pred_init = estimator.compute_predicted_gravity()
    g_pred_opt = optimizer.compute_predicted_gravity()
    
    acc_labels = ['Acc X', 'Acc Y', 'Acc Z']
    
    plt.figure(figsize=(14, 10))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, data_loader.calibrated_acc[i, :], 'b-', 
                 label='IMU Measured', linewidth=1, alpha=0.8)
        plt.plot(data_loader.imu_ts, g_pred_init[:, i], 'g-', 
                 label='Initial', linewidth=1, alpha=0.7)
        plt.plot(data_loader.imu_ts, g_pred_opt[:, i], 'r-', 
                 label='Optimized', linewidth=1.5)
        plt.ylabel(f'{acc_labels[i]} (g)')
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle(f'Testset Dataset {dataset_idx}: Acceleration - IMU Measured vs Predicted Gravity')
    plt.tight_layout()
    acc_path = f'{RESULTS_DIR}/testset_optimization_acceleration_{dataset_idx}.png'
    plt.savefig(acc_path, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"Plot saved to {acc_path}")
    
    return data_loader, estimator, optimizer


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    for idx in TRAINSET_INDICES:
        data_loader, estimator, optimizer = process_dataset(idx, show_plots=False)
        results[idx] = {
            'data_loader': data_loader,
            'estimator': estimator,
            'optimizer': optimizer,
            'initial_cost': optimizer.initial_cost,
            'final_cost': optimizer.final_cost,
        }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Initial Cost':<15} {'Final Cost':<15} {'Improvement':<15}")
    print("-" * 55)
    for idx in TRAINSET_INDICES:
        r = results[idx]
        improvement = (r['initial_cost'] - r['final_cost']) / r['initial_cost'] * 100
        print(f"{idx:<10} {r['initial_cost']:<15.6f} {r['final_cost']:<15.6f} {improvement:<14.2f}%")
    
    return results


def main_testset():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    for idx in TESTSET_INDICES:
        data_loader, estimator, optimizer = process_testset(idx, show_plots=False)
        results[idx] = {
            'data_loader': data_loader,
            'estimator': estimator,
            'optimizer': optimizer,
            'initial_cost': optimizer.initial_cost,
            'final_cost': optimizer.final_cost,
        }
    
    print(f"\n{'='*60}")
    print("TESTSET SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Initial Cost':<15} {'Final Cost':<15} {'Improvement':<15}")
    print("-" * 55)
    for idx in TESTSET_INDICES:
        r = results[idx]
        improvement = (r['initial_cost'] - r['final_cost']) / r['initial_cost'] * 100
        print(f"{idx:<10} {r['initial_cost']:<15.6f} {r['final_cost']:<15.6f} {improvement:<14.2f}%")
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "testset":
        main_testset()
    else:
        main()