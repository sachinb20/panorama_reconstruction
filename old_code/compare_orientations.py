import numpy as np
import matplotlib.pyplot as plt
from load_data import read_data
from transforms3d.euler import mat2euler, quat2euler
from transforms3d.quaternions import qmult, quat2mat

# Dataset selection
dataset = "3"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

# Load data
print(f"Loading dataset {dataset}...")
imud = read_data(ifile)
vicd = read_data(vfile)

# Extract IMU data
imu_ts = imud[0, :]
wx, wy, wz = imud[4, :], imud[5, :], imud[6, :]

# Calibrate gyro (remove bias from first 50 samples)
bias_wx = np.mean(wx[:50])
bias_wy = np.mean(wy[:50])
bias_wz = np.mean(wz[:50])
wx -= bias_wx
wy -= bias_wy
wz -= bias_wz

# Gyro sensitivity and scaling
gyro_sens = 3.33 * 180.0 / np.pi  # mV/(rad/s)
Vref = 3300.0
adc_max = 1023.0
scale = Vref / (adc_max * gyro_sens)
wx *= scale
wy *= scale
wz *= scale

# Initialize quaternion
n_samples = len(imu_ts)
q = np.zeros((n_samples, 4))
q[0] = [1, 0, 0, 0]

# Integrate orientation
print("Integrating IMU orientation...")
for i in range(n_samples - 1):
    dt = imu_ts[i+1] - imu_ts[i]
    w = np.array([wx[i], wy[i], wz[i]])
    alpha_vec = w * dt / 2.0
    alpha_norm = np.linalg.norm(alpha_vec)
    
    if alpha_norm > 0:
        dq = np.array([np.cos(alpha_norm), 
                       np.sin(alpha_norm) * alpha_vec[0] / alpha_norm,
                       np.sin(alpha_norm) * alpha_vec[1] / alpha_norm,
                       np.sin(alpha_norm) * alpha_vec[2] / alpha_norm])
    else:
        dq = np.array([1, 0, 0, 0])
    
    q[i+1] = qmult(q[i], dq)

# Convert to Euler angles
imu_euler = np.array([mat2euler(quat2mat(q[i]), 'syxz') for i in range(n_samples)])

# Process VICON data
vicon_rots = vicd['rots']
vicon_ts = vicd['ts'].flatten()
vicon_euler = np.array([mat2euler(vicon_rots[:, :, i], 'syxz') for i in range(vicon_rots.shape[2])])

# Plot comparison
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 8))
labels = ['Roll', 'Pitch', 'Yaw']

for i in range(3):
    axs[i].plot(imu_ts, imu_euler[:, i], label='IMU Estimate', linewidth=1, alpha=0.8)
    axs[i].plot(vicon_ts, vicon_euler[:, i], label='VICON Ground Truth', linewidth=2, alpha=0.7)
    axs[i].set_ylabel(labels[i] + ' (rad)')
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)

axs[2].set_xlabel('Time (s)')
plt.suptitle(f'IMU Orientation Estimate vs VICON Ground Truth (Dataset {dataset})')
plt.tight_layout()
plt.show()

print("Done!")
