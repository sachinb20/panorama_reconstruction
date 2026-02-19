"""Panorama Generation for Test Set (IMU only, no VICON)."""

import os
import argparse
import sys
import numpy as np

sys.path.append('..')
from imu_core import IMUDataLoader, OrientationEstimator
from imu_optimization import optimize_trajectory

try:
    from panorama import PanoramaGenerator, read_data
except ImportError:
    from part2.panorama import PanoramaGenerator, read_data


def generate_testset_panorama(dataset="10", step=5, output_dir=".", 
                               alpha=0.01, max_iters=1000,
                               motion_weight=0.5, obs_weight=0.5,
                               blend_mode='overwrite'):
    cfile = f"../../data/testset/cam/cam{dataset}.p"
    imu_file = f"../../data/testset/imu/imuRaw{dataset}.p"
    
    print(f"=" * 60)
    print(f"Generating Panorama for Testset Dataset {dataset}")
    print(f"=" * 60)
    
    if not os.path.exists(cfile):
        raise FileNotFoundError(f"Camera file not found: {cfile}")
    if not os.path.exists(imu_file):
        raise FileNotFoundError(f"IMU file not found: {imu_file}")
    
    print(f"\nLoading camera data from {cfile}...")
    camd = read_data(cfile)
    print(f"Camera: {camd['cam'].shape[3]} frames")
    
    print(f"\nLoading and calibrating IMU data from {imu_file}...")
    data_loader = IMUDataLoader(imu_file, vicon_file=None)
    
    print("\nInitializing quaternion trajectory (dead reckoning)...")
    estimator = OrientationEstimator(data_loader)
    estimator.integrate()
    print(f"Initialized {len(estimator.quaternions)} quaternions")
    
    print(f"\nRunning trajectory optimization...")
    print(f"  Alpha: {alpha}, Max iters: {max_iters}")
    print(f"  Motion weight: {motion_weight}, Obs weight: {obs_weight}")
    
    optimizer = optimize_trajectory(
        data_loader=data_loader,
        estimator=estimator,
        alpha=alpha,
        max_iters=max_iters,
        verbose=True,
        motion_weight=motion_weight,
        obs_weight=obs_weight
    )
    
    print(f"\nOptimization complete!")
    print(f"  Initial cost: {optimizer.initial_cost:.6f}")
    print(f"  Final cost: {optimizer.final_cost:.6f}")
    if optimizer.initial_cost != 0:
        improvement = (1 - optimizer.final_cost/optimizer.initial_cost)*100
        print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\nGenerating panorama...")
    generator = PanoramaGenerator(camd)
    
    panorama = generator.generate_from_rotations(
        rotations=optimizer.optimized_quaternions,
        timestamps=data_loader.imu_ts,
        step=step,
        blend_mode=blend_mode
    )
    
    os.makedirs(output_dir, exist_ok=True)
    blend_suffix = '_alpha' if blend_mode == 'alpha' else '_overwrite'
    output_path = os.path.join(output_dir, f'panorama_testset_{dataset}{blend_suffix}.png')
    generator.save_panorama(output_path, title=f"Panorama from IMU Optimization (Testset {dataset}, {blend_mode})")
    
    return panorama, optimizer


TESTSET_DATASETS = ['10', '11']
BLEND_MODES = ['overwrite', 'alpha']


def main():
    parser = argparse.ArgumentParser(description='Generate panoramas from testset data (IMU only)')
    parser.add_argument('--dataset', type=str, default=None, choices=['10', '11'],
                        help='Dataset number (default: all datasets 10 and 11)')
    parser.add_argument('--step', type=int, default=5, help='Frame sampling step')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max-iters', type=int, default=1000, help='Max optimization iterations')
    parser.add_argument('--motion-weight', type=float, default=0.5, help='Motion model weight')
    parser.add_argument('--obs-weight', type=float, default=0.5, help='Observation model weight')
    parser.add_argument('--blend-mode', type=str, default=None, choices=['overwrite', 'alpha'],
                        help='Blending mode (default: both overwrite and alpha)')
    
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else TESTSET_DATASETS
    blend_modes = [args.blend_mode] if args.blend_mode else BLEND_MODES
    
    print(f"Processing testset datasets: {datasets}")
    print(f"Blend modes: {blend_modes}")
    print("=" * 60)
    
    for dataset in datasets:
        for blend_mode in blend_modes:
            try:
                generate_testset_panorama(
                    dataset=dataset,
                    step=args.step,
                    output_dir=args.output_dir,
                    alpha=args.alpha,
                    max_iters=args.max_iters,
                    motion_weight=args.motion_weight,
                    obs_weight=args.obs_weight,
                    blend_mode=blend_mode
                )
            except Exception as e:
                print(f"Error processing dataset {dataset} with {blend_mode}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All testset panoramas generated!")


if __name__ == "__main__":
    main()
