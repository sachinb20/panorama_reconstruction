"""Alpha Convergence Experiment - Tests how different learning rates affect optimization."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian

from imu_core import IMUDataLoader, OrientationEstimator
from imu_optimization import TrajectoryOptimizer

DATA_DIR = "../data/trainset"
RESULTS_DIR = "../results/experiments/alpha_convergence"

ALPHA_VALUES = [0.001, 0.005, 0.01]
MAX_ITERS = 1000


def run_alpha_experiment(dataset_id: int) -> dict:
    imu_file = os.path.join(DATA_DIR, f"imu/imuRaw{dataset_id}.p")
    vicon_file = os.path.join(DATA_DIR, f"vicon/viconRot{dataset_id}.p")
    
    print(f"\n{'='*50}")
    print(f"Dataset {dataset_id}")
    print(f"{'='*50}")
    
    data_loader = IMUDataLoader(imu_file, vicon_file)
    print(f"Loaded {data_loader.num_samples} samples")
    
    estimator = OrientationEstimator(data_loader)
    estimator.integrate()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for alpha in ALPHA_VALUES:
        print(f"  Testing α = {alpha}...", end=" ", flush=True)
        
        optimizer = TrajectoryOptimizer(device=device)
        optimizer._init_sensor_data(
            data_loader.calibrated_omega,
            data_loader.calibrated_acc,
            data_loader.imu_ts
        )
        
        q_array = np.array([q.copy() for q in estimator.quaternions])
        q_flat = optimizer.q_ops.numpy_to_torch(q_array.flatten())
        
        cost_history = []
        initial_cost = optimizer._cost_function(q_flat).item()
        cost_history.append(initial_cost)
        prev_cost = initial_cost
        
        for iteration in range(MAX_ITERS):
            grad = jacobian(optimizer._cost_function, q_flat)
            grad_norm = torch.norm(grad)
            if grad_norm > 100:
                grad = grad * (100 / grad_norm)
            
            q_flat = q_flat - alpha * grad
            q_flat = optimizer._project_to_unit_quaternions(q_flat)
            q_flat = q_flat.detach().clone()
            
            current_cost = optimizer._cost_function(q_flat).item()
            cost_history.append(current_cost)
            
            if np.isnan(current_cost):
                print("NaN!")
                break
            if abs(prev_cost - current_cost) < 1e-7:
                print(f"converged @ iter {iteration}")
                break
            prev_cost = current_cost
        else:
            print(f"final cost: {current_cost:.6f}")
        
        results[alpha] = {
            'cost_history': cost_history,
            'initial_cost': initial_cost,
            'final_cost': cost_history[-1]
        }
    
    return results


def plot_aggregate_summary(all_results: dict, save_dir: str):
    datasets = sorted(all_results.keys())
    
    alpha_means = []
    alpha_stds = []
    
    for alpha in ALPHA_VALUES:
        final_costs = [all_results[d][alpha]['final_cost'] for d in datasets]
        alpha_means.append(np.mean(final_costs))
        alpha_stds.append(np.std(final_costs))
    
    max_len = max(
        len(all_results[d][alpha]['cost_history'])
        for d in datasets for alpha in ALPHA_VALUES
    )
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, alpha in enumerate(ALPHA_VALUES):
        all_histories = []
        for d in datasets:
            hist = all_results[d][alpha]['cost_history']
            padded = hist + [hist[-1]] * (max_len - len(hist))
            all_histories.append(padded)
        
        histories_arr = np.array(all_histories)
        mean_curve = np.mean(histories_arr, axis=0)
        std_curve = np.std(histories_arr, axis=0)
        
        x = np.arange(len(mean_curve))
        ax1.plot(x, mean_curve, color=colors[i], label=f'α = {alpha}', linewidth=2)
        ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Mean Convergence Curves Across All Datasets (shaded = ±1 std)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'alpha_aggregate.png'), dpi=150)
    plt.close()

def print_summary_table(all_results: dict):
    datasets = sorted(all_results.keys())
    
    print("\n" + "="*60)
    print("SUMMARY: Mean and Std of Final Costs per Alpha")
    print("="*60)
    
    for alpha in ALPHA_VALUES:
        final_costs = [all_results[d][alpha]['final_cost'] for d in datasets]
        mean = np.mean(final_costs)
        std = np.std(final_costs)
        print(f"α = {alpha}: Mean = {mean:.6f}, Std = {std:.6f}")
    
    print("="*60)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Alpha Convergence Experiment")
    print(f"Testing α values: {ALPHA_VALUES}")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    all_results = {}
    
    for dataset_id in range(1, 10):
        try:
            results = run_alpha_experiment(dataset_id)
            all_results[dataset_id] = results
        except Exception as e:
            print(f"  Error on dataset {dataset_id}: {e}")
            continue
    
    if all_results:
        plot_aggregate_summary(all_results, RESULTS_DIR)
        print_summary_table(all_results)
        print(f"\nSaved: alpha_aggregate.png")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
