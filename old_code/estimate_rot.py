import numpy as np
import matplotlib.pyplot as plt
from load_data import read_data
import os
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import qmult, qinverse, quat2mat

# Load Data
dataset = "3" # Using dataset 8 as per user's last edit to plot_vidcon.py
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

print(f"Loading data for dataset {dataset}...")
imud = read_data(ifile)
viconrots = read_data(vfile)

# IMU Data: 7 x N
# [timestamp, ax, ay, az, wx, wy, wz]
imu_data = imud

# Extract W (indices 4, 5, 6 correspond to Wx, Wy, Wz based on inspection)
# And timestamp is row 0
wx = imu_data[4, :]
wy = imu_data[5, :]
wz = imu_data[6, :]
timestamps = imu_data[0, :]

# Bias removal
# Assuming the device is stationary for the first few samples
num_bias_samples = 50
bias_wx = np.mean(wx[:num_bias_samples])
bias_wy = np.mean(wy[:num_bias_samples])
bias_wz = np.mean(wz[:num_bias_samples])

print(f"Bias detected: Wx={bias_wx:.4f}, Wy={bias_wy:.4f}, Wz={bias_wz:.4f}")

wx = wx - bias_wx
wy = wy - bias_wy
wz = wz - bias_wz

# Initialize
n_samples = imu_data.shape[1]
q = np.zeros((n_samples, 4))
q[0] = [1, 0, 0, 0] # w, x, y, z

# Loop
print("Running motion model integration...")
for i in range(n_samples - 1):
    dt = timestamps[i+1] - timestamps[i]
    
    # Current angular velocity vector
    # Using raw values as requested by user
    w = np.array([wx[i], wy[i], wz[i]])
    
    # Calculate exponential map term: exp([0, dt * w / 2])
    # This represents the incremental rotation quaternion
    # theta_vec = dt * w
    # q_inc = [cos(|theta|/2), sin(|theta|/2) * theta/|theta|]
    
    alpha_vec = w * dt / 2.0
    alpha_norm = np.linalg.norm(alpha_vec)
    
    if alpha_norm > 0:
        dq = np.array([np.cos(alpha_norm), 
                       np.sin(alpha_norm) * alpha_vec[0] / alpha_norm,
                       np.sin(alpha_norm) * alpha_vec[1] / alpha_norm,
                       np.sin(alpha_norm) * alpha_vec[2] / alpha_norm])
    else:
        dq = np.array([1, 0, 0, 0])
        
    # Update q_{t+1} = q_t * dq
    # Assuming body frame rates, so right multiplication
    q[i+1] = qmult(q[i], dq)

# Convert to Euler
print("Converting to Euler angles...")
pred_euler = np.zeros((n_samples, 3))
for i in range(n_samples):
    # Using 'szxy' or similar? plot_vidcon used 'syxz'
    # 'sxyz' is standard static frame
    # 'sxyz' is standard static frame
    pred_euler[i] = mat2euler(quat2mat(q[i]), 'syxz')

# Plot
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
axs[0].plot(pred_euler[:, 0], label='Predicted')
axs[0].set_ylabel('Roll (rad)')
axs[0].set_title('Predicted Orientation (Motion Model)')
axs[0].legend()

axs[1].plot(pred_euler[:, 1], label='Predicted')
axs[1].set_ylabel('Pitch (rad)')

axs[2].plot(pred_euler[:, 2], label='Predicted')
axs[2].set_ylabel('Yaw (rad)')
axs[2].set_xlabel('Sample Index')

plt.tight_layout()
plt.show()

print(f"IMU Data Shape: {imu_data.shape}")
print(f"First 5 IMU samples:\n{imu_data[:, :5]}")

# Check if data looks like raw ADC (integers approx 300-600 usually for 10-bit zero-g) 
# or processed (floats near 0)
print(f"Max Val: {np.max(imu_data)}")
print(f"Min Val: {np.min(imu_data)}")

# Vicon Data
vicon_rots = viconrots['rots']
vicon_ts = viconrots['ts']
print(f"Vicon Data Shape: {vicon_rots.shape}")
