"""Part 1: IMU Orientation Estimation using Dead Reckoning."""

import numpy as np
import matplotlib.pyplot as plt

from imu_core import IMUDataLoader, OrientationEstimator

IMU_FILE = "../data/trainset/imu/imuRaw7.p"
VICON_FILE = "../data/trainset/vicon/viconRot7.p"


def main():
    print("Loading IMU and VICON data...")
    data_loader = IMUDataLoader(IMU_FILE, VICON_FILE)
    print(f"Loaded {data_loader.num_samples} IMU samples")
    print(f"Calibrated omega shape: {data_loader.calibrated_omega.shape}")
    print(f"Calibrated acc shape: {data_loader.calibrated_acc.shape}")
    
    print("\nIntegrating quaternions...")
    estimator = OrientationEstimator(data_loader)
    estimator.integrate()
    estimator.compute_rpy()
    print(f"Computed {len(estimator.quaternions)} quaternions")
    
    labels = ['Roll', 'Pitch', 'Yaw']
    
    plt.figure(figsize=(14, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, estimator.rpy[:, i], label='IMU', linewidth=1)
        plt.plot(data_loader.vicon_ts, data_loader.vicon_rpy[:, i], label='VICON', linewidth=2)
        plt.ylabel(labels[i] + ' (rad)')
        plt.legend()
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle('IMU Orientation vs VICON Ground Truth')
    plt.tight_layout()
    plt.show()
    
    rotated_gravity = estimator.compute_predicted_gravity()
    
    acc_labels = ['Acc X', 'Acc Y', 'Acc Z']
    plt.figure(figsize=(14, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(data_loader.imu_ts, data_loader.calibrated_acc[i, :], 
                 linewidth=1, label='Measured Acc')
        plt.plot(data_loader.imu_ts, rotated_gravity[:, i], 
                 linewidth=1, label='Rotated Gravity', alpha=0.7)
        plt.ylabel(acc_labels[i] + ' (g)')
        plt.legend()
        plt.grid(True)
    
    plt.xlabel('Time (s)')
    plt.suptitle('Measured Acceleration vs Rotated Gravity (q⁻¹ ◦ g ◦ q)')
    plt.tight_layout()
    plt.show()
    
    return data_loader, estimator




if __name__ == "__main__":
    main()
