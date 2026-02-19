import pickle
from transforms3d.euler import euler2mat, mat2euler, quat2euler
from transforms3d.quaternions import qmult, qinverse, quat2mat, mat2quat


# 1. Load IMU data
imu_file = "../data/trainset/imu/imuRaw3.p"
vicon_file = "../data/trainset/vicon/viconRot3.p"

with open(imu_file, 'rb') as f:
    imu_arr = pickle.load(f)


with open(vicon_file, 'rb') as f:
    vicon_dict = pickle.load(f, encoding='latin1')

imu_ts = imu_arr[0, :]
print("shape of the imu_ts: ",imu_ts.shape)

gyro_sens = 3.33 * 180.0 / np.pi # mV/rad/sec (not degrees)
acc_sens = 330 # mV/g
gyro_raw = imu_arr[4:7, :]
acc_raw = imu_arr[1:4, :]
print("shape of the gyro_raw: ", gyro_raw.shape)
print("shape of the acc_raw: ", acc_raw.shape)

# compute bias from first 100 samples
gyro_bias = np.mean(gyro_raw[:, :100], axis=1)
#acc_bias but taking gravity into account
acc_bias = np.mean(acc_raw[:, :100], axis=1) 
Vref = 3300.0 # mV
adc_max = 1023.0
gyro_scale_factor= (Vref / (adc_max * gyro_sens)) 
acc_scale_factor= (Vref / (adc_max * acc_sens)) 
# convert to rad/sec
calibrated_omega = (gyro_raw - gyro_bias[:, None]) * gyro_scale_factor # rad/sec
calibrated_acc = (acc_raw - acc_bias[:, None]) * acc_scale_factor + np.array([0, 0, 1])[:, None] # g

print("shape of the calibrated_omega: ", calibrated_omega.shape)
print("shape of the calibrated_acc: ", calibrated_acc.shape)

N = imu_ts.shape[0]
print(f"Loaded IMU data: {N} samples")

# 2. Initialize quaternion storage
q = np.array([1.0, 0.0, 0.0, 0.0]) # initial orientation (identity)
q_calculated = [q.copy()]
prev_ts = imu_ts[0]

def q_inv(q):
    """Quaternion inverse (conjugate for unit quaternions)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def motion_model(qt, omega, dt):
    """
    Given qt, compute qt+1 using exponential map.
    qt+1 = qt ◦ exp(omega * dt)
    """
    theta_vec = omega * dt
    angle = np.linalg.norm(theta_vec)
    if angle < 1e-8:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        half_angle = angle / 2
        unit_axis = theta_vec / angle
        dq = np.array([np.cos(half_angle), 
                       unit_axis[0]*np.sin(half_angle), 
                       unit_axis[1]*np.sin(half_angle), 
                       unit_axis[2]*np.sin(half_angle)])
    qt1 = qmult(qt, dq)
    qt1 = qt1 / np.linalg.norm(qt1)
    return qt1

def observation_model(qt):
    """
    Given qt, compute q^{-1} ◦ [0, 0, 0, 1] ◦ q.
    Returns gravity vector in body frame (3D vector).
    """
    gravity_quat = np.array([0, 0, 0, 1])  # [w, x, y, z] with w=0
    qt_inv = q_inv(qt)
    temp = qmult(qt_inv, gravity_quat)
    result = qmult(temp, qt)
    return result[1:4]  # extract x, y, z components

# 3. Loop over all timesteps 
def quaternion_integration(calibrated_omega, imu_ts):
    q = np.array([1.0, 0.0, 0.0, 0.0]) # initial orientation (identity)
    q_calculated = [q.copy()]
    for t in range(1, N):
        dt = imu_ts[t] - imu_ts[t-1]
        omega = calibrated_omega[:, t-1]
        q = motion_model(q, omega, dt)
        q_calculated.append(q.copy())
    return q_calculated

q_calculated = quaternion_integration(calibrated_omega, imu_ts)

print("length of q_calculated: ", len(q_calculated))

# 4. Convert quaternions to roll-pitch-yaw
rpy_list = [quat2euler(qi, axes='sxyz') for qi in q_calculated] # radians
rpy_array = np.array(rpy_list) # shape (N, 3)

vicon_rots = vicon_dict['rots'] # shape (3,3,M)
vicon_ts = vicon_dict['ts'].flatten() # shape (M,)

# 6. Convert rotation matrices → RPY
vicon_rpy = []

for i in range(vicon_rots.shape[2]):
    R = vicon_rots[:, :, i]
    q_vicon = mat2quat(R) # [w, x, y, z]
    rpy = quat2euler(q_vicon, axes='sxyz')
    vicon_rpy.append(rpy)

vicon_rpy = np.array(vicon_rpy) # shape (M,3)
print(f"Loaded Vicon RPY: {vicon_rpy}")
print(f"Loaded IMU RPY: {rpy_array}")


##############plots


import torch
from torch.autograd.functional import jacobian
from transforms3d.euler import quat2euler

    # === Motion Model Term ===
    # q_pred = f(q[:-1], omega, dt)
    q_pred = motion_model_batch(q[:-1], omega_torch, dt_torch)
    
    # Relative rotation: q[1:]^{-1} ◦ q_pred
    q_rel = qmult_batch(q_inv_batch(q[1:]), q_pred)
    
    # Axis-angle error
    log_error = 2 * quaternion_log_batch(q_rel)
    motion_cost = 0.5 * torch.sum(log_error ** 2)
    
    # === Observation Model Term ===
    # h(q[1:]) vs acc[1:]
    g_pred = observation_model_batch(q[1:])
    obs_error = acc_torch[1:] - g_pred
    obs_cost = 0.5 * torch.sum(obs_error ** 2)
    
    return motion_cost + obs_cost


# ============= Optimization =============
def project_to_unit_quaternions(q_flat):
    """Project each quaternion in trajectory to unit sphere."""
    T = q_flat.shape[0] // 4
    q = q_flat.reshape(T, 4)
    q_normalized = q / torch.norm(q, dim=1, keepdim=True)
    return q_normalized.flatten()


def optimize_q(q_init, alpha=0.001, max_iters=100, tol=1e-6, verbose=True):
    """
    Projected gradient descent using torch.autograd.functional.jacobian on GPU.
    """
    # Initialize sensor data on GPU
    init_sensor_data()
    
    # Convert initial trajectory to GPU tensor
    q_array = np.array([q.copy() for q in q_init])
    q_flat = numpy_to_torch(q_array.flatten())
    T = len(q_array)
    
    prev_cost = cost_function(q_flat).item()
    print(f"Starting optimization with initial cost: {prev_cost:.6f}")
    
    for iteration in range(max_iters):
        # Compute gradient using torch jacobian
        grad = jacobian(cost_function, q_flat)
        
        # Clip gradient to prevent explosion
        grad_norm = torch.norm(grad)
        if grad_norm > 100:
            grad = grad * (100 / grad_norm)
        
        # Gradient descent step
        q_flat = q_flat - alpha * grad
        
        # Project each quaternion to unit sphere
        q_flat = project_to_unit_quaternions(q_flat)
        
        # Detach to prevent graph accumulation
        q_flat = q_flat.detach().clone()
        
        # Compute new cost
        current_cost = cost_function(q_flat).item()
        
        if verbose and iteration % 5 == 0:
            print(f"Iteration {iteration}: Cost = {current_cost:.6f}, Grad norm = {grad_norm.item():.6f}")
        
        # Check for NaN
        if np.isnan(current_cost):
            print(f"Warning: NaN detected at iteration {iteration}. Reducing step size.")
            alpha *= 0.5
            q_flat = numpy_to_torch(q_array.flatten())  # Reset
            continue
        
        # Check convergence
        if abs(prev_cost - current_cost) < tol:
            print(f"Converged at iteration {iteration}")
            break
        
        prev_cost = current_cost
    
    # Convert back to list of numpy quaternions
    q_opt = list(torch_to_numpy(q_flat).reshape(T, 4))
    return q_opt


# ============= Main =============
if __name__ == "__main__":
    # Create results directory if needed
    os.makedirs('../results', exist_ok=True)
    
    print("Initializing quaternion trajectory using motion model...")
    q_init = quaternion_integration(calibrated_omega, imu_ts)
    
    # Initialize sensor data
    init_sensor_data()
    
    q_flat_init = numpy_to_torch(np.array(q_init).flatten())
    print(f"Initial cost: {cost_function(q_flat_init).item():.6f}")
    print(f"Number of quaternions: {len(q_init)}")

        q_opt = optimize_q(q_init, alpha=0.001, max_iters=5000, verbose=True)
    
    q_flat_opt = numpy_to_torch(np.array(q_opt).flatten())
    print(f"\nFinal cost: {cost_function(q_flat_opt).item():.6f}")
    
    # Convert to RPY for plotting
    rpy_init = np.array([quat2euler(qi, axes='sxyz') for qi in q_init])
    rpy_opt = np.array([quat2euler(qi, axes='sxyz') for qi in q_opt])