"""Optimization Experiments - Zero init vs motion model init, weight sensitivity."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from imu_core import IMUDataLoader, OrientationEstimator
from imu_optimization import TrajectoryOptimizer

DATA_DIR = "../data/trainset"
RESULTS_DIR = "../results/experiments"

MOTION_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_ALPHA = 0.01
DEFAULT_MAX_ITERS = 1000
QUICK_MAX_ITERS = 500


class ExperimentRunner:
    def __init__(self, dataset_id: int, results_dir: str, max_iters: int = DEFAULT_MAX_ITERS):
        self.dataset_id = dataset_id
        self.results_dir = os.path.join(results_dir, f"dataset_{dataset_id}")
        self.max_iters = max_iters
        os.makedirs(self.results_dir, exist_ok=True)
        
        imu_file = os.path.join(DATA_DIR, f"imu/imuRaw{dataset_id}.p")
        vicon_file = os.path.join(DATA_DIR, f"vicon/viconRot{dataset_id}.p")
        
        print(f"\n{'='*60}")
        print(f"Loading Dataset {dataset_id}")
        print(f"{'='*60}")
        self.data_loader = IMUDataLoader(imu_file, vicon_file)
        print(f"Loaded {self.data_loader.num_samples} samples")
        
        self.estimator = OrientationEstimator(self.data_loader)
        self.estimator.integrate()
        self.estimator.compute_rpy()
    
    def _create_zero_init(self) -> list:
        N = self.data_loader.num_samples
        return [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(N)]
    
    def _run_optimization(self, q_init: list, alpha: float, device: torch.device,
                          motion_weight: float = 0.5, max_iters: int = None,
                          track_cost: bool = False) -> dict:
        if max_iters is None:
            max_iters = self.max_iters
        
        optimizer = TrajectoryOptimizer(device=device, motion_weight=motion_weight, 
                                         obs_weight=1.0 - motion_weight)
        
        cost_history = []
        
        if track_cost:
            optimizer._init_sensor_data(
                self.data_loader.calibrated_omega,
                self.data_loader.calibrated_acc,
                self.data_loader.imu_ts
            )
            
            q_array = np.array([q.copy() for q in q_init])
            q_flat = optimizer.q_ops.numpy_to_torch(q_array.flatten())
            T = len(q_array)
            
            optimizer.initial_cost = optimizer._cost_function(q_flat).item()
            cost_history.append(optimizer.initial_cost)
            prev_cost = optimizer.initial_cost
            
            for iteration in range(max_iters):
                grad = torch.autograd.functional.jacobian(optimizer._cost_function, q_flat)
                grad_norm = torch.norm(grad)
                if grad_norm > 100:
                    grad = grad * (100 / grad_norm)
                
                q_flat = q_flat - alpha * grad
                q_flat = optimizer._project_to_unit_quaternions(q_flat)
                q_flat = q_flat.detach().clone()
                
                current_cost = optimizer._cost_function(q_flat).item()
                cost_history.append(current_cost)
                
                if np.isnan(current_cost):
                    break
                if abs(prev_cost - current_cost) < 1e-6:
                    break
                prev_cost = current_cost
            
            optimizer.final_cost = current_cost
            optimizer.optimized_quaternions = list(optimizer.q_ops.torch_to_numpy(q_flat).reshape(T, 4))
            optimizer.compute_rpy()
        else:
            optimizer.optimize(
                q_init=q_init,
                omega=self.data_loader.calibrated_omega,
                acc=self.data_loader.calibrated_acc,
                timestamps=self.data_loader.imu_ts,
                alpha=alpha,
                max_iters=max_iters,
                verbose=False
            )
            optimizer.compute_rpy()
            return {
                'optimizer': optimizer,
                'cost_history': cost_history
            }
        
        return {
            'optimizer': optimizer,
            'cost_history': cost_history
        }
    
    
    def experiment_init_comparison(self):
        print(f"\n--- Experiment 2: Init Comparison (Dataset {self.dataset_id}) ---")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("  Running with zero init...")
        zero_result = self._run_optimization(
            q_init=self._create_zero_init(),
            alpha=DEFAULT_ALPHA,
            device=device,
            track_cost=True
        )
        
        print("  Running with motion model init...")
        motion_result = self._run_optimization(
            q_init=self.estimator.quaternions.copy(),
            alpha=DEFAULT_ALPHA,
            device=device,
            track_cost=True
        )
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1 = axes[0]
        if zero_result['cost_history']:
            ax1.plot(zero_result['cost_history'], 'b-', label='Zero Init', linewidth=1.5)
        if motion_result['cost_history']:
            ax1.plot(motion_result['cost_history'], 'r-', label='Motion Model Init', linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title(f'Convergence: Zero Init vs Motion Model Init - Dataset {self.dataset_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        labels = ['Roll', 'Pitch', 'Yaw']
        ax2 = axes[1]
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        for i, label in enumerate(labels):
            ax2.plot(self.data_loader.vicon_ts, self.data_loader.vicon_rpy[:, i], 
                     color=colors[i], linestyle='-', label=f'{label} (VICON)', linewidth=1.5, alpha=0.8)
            ax2.plot(self.data_loader.imu_ts, motion_result['optimizer'].optimized_rpy[:, i],
                     color=colors[i], linestyle='--', label=f'{label} (Motion Init)', linewidth=1)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (rad)')
        ax2.set_title('Optimized RPY vs Ground Truth')
        ax2.legend(loc='upper right', ncol=2, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'init_comparison.png'), dpi=150)
        plt.close()
        print(f"  Saved: init_comparison.png")
        
        zero_iters = len(zero_result['cost_history']) - 1
        motion_iters = len(motion_result['cost_history']) - 1
        
        print(f"  Zero Init - Initial: {zero_result['optimizer'].initial_cost:.4f}, "
              f"Final: {zero_result['optimizer'].final_cost:.4f}, Iterations: {zero_iters}")
        print(f"  Motion Init - Initial: {motion_result['optimizer'].initial_cost:.4f}, "
              f"Final: {motion_result['optimizer'].final_cost:.4f}, Iterations: {motion_iters}")
        
        return {
            'zero': zero_result, 
            'motion': motion_result,
            'zero_iters': zero_iters,
            'motion_iters': motion_iters
        }
    
    
    def experiment_weight_sensitivity(self):
        print(f"\n--- Experiment 4: Weight Sensitivity (Dataset {self.dataset_id}) ---")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results_motion = {}
        results_zero = {}
        
        print("\n  [Motion Model Init]")
        for weight in MOTION_WEIGHTS:
            print(f"    Testing motion_weight = {weight}...")
            result = self._run_optimization(
                q_init=self.estimator.quaternions.copy(),
                alpha=DEFAULT_ALPHA,
                device=device,
                motion_weight=weight
            )
            results_motion[weight] = result
        
        print("\n  [Zero Init]")
        for weight in MOTION_WEIGHTS:
            print(f"    Testing motion_weight = {weight}...")
            result = self._run_optimization(
                q_init=self._create_zero_init(),
                alpha=DEFAULT_ALPHA,
                device=device,
                motion_weight=weight
            )
            results_zero[weight] = result
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1 = axes[0]
        weights = list(results_motion.keys())
        motion_final_costs = [results_motion[w]['optimizer'].final_cost for w in weights]
        zero_final_costs = [results_zero[w]['optimizer'].final_cost for w in weights]
        
        ax1.plot(weights, motion_final_costs, 'bo-', linewidth=2, markersize=8, label='Motion Init')
        ax1.plot(weights, zero_final_costs, 'rs--', linewidth=2, markersize=8, label='Zero Init')
        ax1.set_xlabel('Motion Weight')
        ax1.set_ylabel('Final Cost')
        ax1.set_title(f'Final Cost vs Motion Weight - Dataset {self.dataset_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        labels = ['Roll', 'Pitch', 'Yaw']
        
        for i, label in enumerate(labels):
            ax2.plot(self.data_loader.vicon_ts, self.data_loader.vicon_rpy[:, i],
                     'k-', linewidth=2, alpha=0.8, label=f'{label} (VICON)' if i == 0 else None)
        
        selected_weights = [0.1, 0.5, 0.9]
        linestyles = ['--', '-.', ':']
        for weight, ls in zip(selected_weights, linestyles):
            if weight in results_motion:
                rpy = results_motion[weight]['optimizer'].optimized_rpy
                ax2.plot(self.data_loader.imu_ts, rpy[:, 1],
                         linestyle=ls, linewidth=1.5, label=f'w={weight}')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pitch (rad)')
        ax2.set_title('Pitch Angle for Different Motion Weights')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'weight_sensitivity.png'), dpi=150)
        plt.close()
        print(f"\n  Saved: weight_sensitivity.png")
        
        detailed_results = {'motion_init': {}, 'zero_init': {}}
        
        print(f"\n  [Motion Model Init] Individual Costs:")
        print(f"  {'Weight':<10} {'Motion Cost':<15} {'Obs Cost':<15} {'Unweighted Sum':<18} {'Weighted Cost':<15}")
        print(f"  {'-'*73}")
        for weight in MOTION_WEIGHTS:
            optimizer = results_motion[weight]['optimizer']
            motion_cost, obs_cost = optimizer.compute_individual_costs()
            unweighted_sum = motion_cost + obs_cost
            weighted_cost = optimizer.final_cost
            print(f"  {weight:<10.1f} {motion_cost:<15.4f} {obs_cost:<15.4f} {unweighted_sum:<18.4f} {weighted_cost:<15.4f}")
            detailed_results['motion_init'][weight] = {
                'motion_cost': motion_cost,
                'obs_cost': obs_cost,
                'unweighted_sum': unweighted_sum,
                'weighted_cost': weighted_cost
            }
        
        print(f"\n  [Zero Init] Individual Costs:")
        print(f"  {'Weight':<10} {'Motion Cost':<15} {'Obs Cost':<15} {'Unweighted Sum':<18} {'Weighted Cost':<15}")
        print(f"  {'-'*73}")
        for weight in MOTION_WEIGHTS:
            optimizer = results_zero[weight]['optimizer']
            motion_cost, obs_cost = optimizer.compute_individual_costs()
            unweighted_sum = motion_cost + obs_cost
            weighted_cost = optimizer.final_cost
            print(f"  {weight:<10.1f} {motion_cost:<15.4f} {obs_cost:<15.4f} {unweighted_sum:<18.4f} {weighted_cost:<15.4f}")
            detailed_results['zero_init'][weight] = {
                'motion_cost': motion_cost,
                'obs_cost': obs_cost,
                'unweighted_sum': unweighted_sum,
                'weighted_cost': weighted_cost
            }
        
        return {'motion_init': results_motion, 'zero_init': results_zero, 'detailed': detailed_results}
    
    def run_all_experiments(self) -> dict:
        return {
            'init': self.experiment_init_comparison()
        }


def create_summary_plot(all_results: dict, results_dir: str):
    summary_dir = os.path.join(results_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    datasets = sorted(all_results.keys())
    
    zero_iters_list = []
    motion_iters_list = []
    zero_final_costs = []
    motion_final_costs = []
    
    for d in datasets:
        if 'init' in all_results[d]:
            init_result = all_results[d]['init']
            zero_iters_list.append(init_result['zero_iters'])
            motion_iters_list.append(init_result['motion_iters'])
            zero_final_costs.append(init_result['zero']['optimizer'].final_cost)
            motion_final_costs.append(init_result['motion']['optimizer'].final_cost)
    
    zero_mean = np.mean(zero_iters_list)
    zero_std = np.std(zero_iters_list)
    zero_var = np.var(zero_iters_list)
    motion_mean = np.mean(motion_iters_list)
    motion_std = np.std(motion_iters_list)
    motion_var = np.var(motion_iters_list)
    
    print("\n" + "="*80)
    print("CONVERGENCE TIME SUMMARY: ZERO INIT vs MOTION MODEL INIT")
    print("="*80)
    print(f"\n{'Dataset':<10} {'Zero Init Iters':<18} {'Motion Init Iters':<18} {'Speed-up':<10}")
    print("-" * 56)
    for i, d in enumerate(datasets):
        speedup = zero_iters_list[i] / max(motion_iters_list[i], 1)
        print(f"{d:<10} {zero_iters_list[i]:<18} {motion_iters_list[i]:<18} {speedup:<10.2f}x")
    
    print("-" * 56)
    print(f"\nZero Init:     Mean = {zero_mean:.2f}, Std = {zero_std:.2f}, Variance = {zero_var:.2f}")
    print(f"Motion Init:   Mean = {motion_mean:.2f}, Std = {motion_std:.2f}, Variance = {motion_var:.2f}")
    print(f"\nAverage Speed-up: {zero_mean/motion_mean:.2f}x faster with Motion Init")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    ax1.bar(x - width/2, zero_iters_list, width, label='Zero Init', color='tab:blue')
    ax1.bar(x + width/2, motion_iters_list, width, label='Motion Model Init', color='tab:orange')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Iterations to Converge')
    ax1.set_title('Convergence Iterations per Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    methods = ['Zero Init', 'Motion Init']
    means = [zero_mean, motion_mean]
    stds = [zero_std, motion_std]
    bars = ax2.bar(methods, means, yerr=stds, capsize=10, color=['tab:blue', 'tab:orange'], edgecolor='black')
    ax2.set_ylabel('Iterations to Converge')
    ax2.set_title(f'Mean Convergence Time (± Std Dev)\nZero: {zero_mean:.1f}±{zero_std:.1f}, Motion: {motion_mean:.1f}±{motion_std:.1f}')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, var) in enumerate(zip(bars, [zero_var, motion_var])):
        ax2.annotate(f'Var: {var:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, zero_final_costs, width, label='Zero Init', color='tab:blue')
    ax3.bar(x + width/2, motion_final_costs, width, label='Motion Model Init', color='tab:orange')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Final Cost')
    ax3.set_title('Final Cost per Dataset')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    speedups = [z / max(m, 1) for z, m in zip(zero_iters_list, motion_iters_list)]
    ax4.bar([str(d) for d in datasets], speedups, color='tab:green', edgecolor='black')
    ax4.axhline(y=np.mean(speedups), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(speedups):.2f}x')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Speed-up Factor')
    ax4.set_title('Speed-up: Motion Init vs Zero Init')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'init_comparison_summary.png'), dpi=150)
    plt.close()
    print(f"\nSaved summary plot: {summary_dir}/init_comparison_summary.png")
    
    csv_path = os.path.join(summary_dir, 'init_comparison_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Dataset,Zero_Iters,Motion_Iters,Zero_Final_Cost,Motion_Final_Cost,Speedup\n")
        for i, d in enumerate(datasets):
            speedup = zero_iters_list[i] / max(motion_iters_list[i], 1)
            f.write(f"{d},{zero_iters_list[i]},{motion_iters_list[i]},"
                    f"{zero_final_costs[i]:.6f},{motion_final_costs[i]:.6f},{speedup:.4f}\n")
    print(f"Saved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Run optimization experiments')
    parser.add_argument('--dataset', type=int, default=None,
                        help='Specific dataset to run (1-9), or all if not specified')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer iterations')
    args = parser.parse_args()
    
    max_iters = QUICK_MAX_ITERS if args.quick else DEFAULT_MAX_ITERS
    
    if args.dataset:
        dataset_ids = [args.dataset]
    else:
        dataset_ids = list(range(1, 10))
    
    print(f"Running experiments on datasets: {dataset_ids}")
    print(f"Max iterations: {max_iters}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    for dataset_id in dataset_ids:
        try:
            runner = ExperimentRunner(dataset_id, RESULTS_DIR, max_iters)
            all_results[dataset_id] = runner.run_all_experiments()
        except Exception as e:
            print(f"Error on dataset {dataset_id}: {e}")
            continue
    
    if len(all_results) > 1:
        create_summary_plot(all_results, RESULTS_DIR)
    
    print(f"\n{'='*60}")
    print(f"All experiments complete! Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
