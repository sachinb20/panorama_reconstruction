"""GPU-accelerated quaternion trajectory optimization module."""

import numpy as np
import torch
from torch.autograd.functional import jacobian
from transforms3d.euler import quat2euler


class TorchQuaternionOps:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def numpy_to_torch(self, arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float64, device=self.device, requires_grad=requires_grad)
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()
    
    def qmult_batch(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if q1.dim() == 1:
            q1 = q1.unsqueeze(0)
        if q2.dim() == 1:
            q2 = q2.unsqueeze(0)
        
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        result = torch.stack([w, x, y, z], dim=1)
        return result.squeeze(0) if result.shape[0] == 1 else result
    
    def q_inv_batch(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            return torch.stack([q[0], -q[1], -q[2], -q[3]])
        return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)
    
    def quaternion_log_batch(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(0)
        
        w = q[:, 0:1]
        v = q[:, 1:4]
        v_norm = torch.norm(v, dim=1, keepdim=True)
        
        theta = 2 * torch.atan2(v_norm, torch.abs(w))
        small_angle_mask = v_norm < 1e-6
        scale = torch.where(small_angle_mask, 2.0 * torch.ones_like(v_norm), theta / (v_norm + 1e-10))
        
        result = scale * v * torch.sign(w + 1e-10)
        return result.squeeze(0) if result.shape[0] == 1 else result
    
    def motion_model_batch(self, q: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(0)
            omega = omega.unsqueeze(0)
            dt = dt.unsqueeze(0) if isinstance(dt, torch.Tensor) else torch.tensor([dt], device=self.device, dtype=torch.float64)
            squeeze = True
        else:
            squeeze = False
            if not isinstance(dt, torch.Tensor):
                dt = torch.full((q.shape[0],), dt, device=self.device, dtype=torch.float64)
        
        theta_vec = omega * dt.unsqueeze(1)
        angle = torch.norm(theta_vec, dim=1, keepdim=True)
        half_angle = angle / 2
        
        unit_axis = torch.where(angle > 1e-8, theta_vec / (angle + 1e-10), torch.zeros_like(theta_vec))
        
        dq_w = torch.cos(half_angle)
        dq_xyz = torch.sin(half_angle) * unit_axis
        dq = torch.cat([dq_w, dq_xyz], dim=1)
        
        q_new = self.qmult_batch(q, dq)
        q_new = q_new / torch.norm(q_new, dim=1, keepdim=True)
        
        return q_new.squeeze(0) if squeeze else q_new
    
    def observation_model_batch(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        N = q.shape[0]
        gravity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float64).expand(N, 4)
        
        q_inv = self.q_inv_batch(q)
        temp = self.qmult_batch(q_inv, gravity_quat)
        result = self.qmult_batch(temp, q)
        
        g_body = result[:, 1:4]
        return g_body.squeeze(0) if squeeze else g_body


class TrajectoryOptimizer:
    def __init__(self, device: torch.device = None, motion_weight: float = 0.5, obs_weight: float = 0.5):
        self.device = device 
        self.q_ops = TorchQuaternionOps(self.device)
        self.motion_weight = motion_weight
        self.obs_weight = obs_weight
        
        self._dt_torch = None
        self._omega_torch = None
        self._acc_torch = None
        
        self.optimized_quaternions = None
        self.optimized_rpy = None
        self.final_cost = None
        self.initial_cost = None
        self.cost_history = []
        self.converged_iter = None
    
    def _init_sensor_data(self, omega: np.ndarray, acc: np.ndarray, timestamps: np.ndarray):
        self._dt_torch = self.q_ops.numpy_to_torch(np.diff(timestamps))
        self._omega_torch = self.q_ops.numpy_to_torch(omega[:, :-1].T)
        self._acc_torch = self.q_ops.numpy_to_torch(acc.T)
    
    def _cost_function(self, q_flat: torch.Tensor) -> torch.Tensor:
        T = q_flat.shape[0] // 4
        q = q_flat.reshape(T, 4)
        
        q_pred = self.q_ops.motion_model_batch(q[:-1], self._omega_torch, self._dt_torch)
        q_rel = self.q_ops.qmult_batch(self.q_ops.q_inv_batch(q[1:]), q_pred)
        log_error = 2 * self.q_ops.quaternion_log_batch(q_rel)
        motion_cost = torch.sum(log_error ** 2)
        
        g_pred = self.q_ops.observation_model_batch(q[1:])
        obs_error = self._acc_torch[1:] - g_pred
        obs_cost = torch.sum(obs_error ** 2)
        
        return (self.motion_weight * motion_cost + self.obs_weight * obs_cost) / (self.motion_weight + self.obs_weight)
    
    def compute_individual_costs(self) -> tuple:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        q_array = np.array(self.optimized_quaternions)
        q_flat = self.q_ops.numpy_to_torch(q_array.flatten())
        T = q_flat.shape[0] // 4
        q = q_flat.reshape(T, 4)
        
        q_pred = self.q_ops.motion_model_batch(q[:-1], self._omega_torch, self._dt_torch)
        q_rel = self.q_ops.qmult_batch(self.q_ops.q_inv_batch(q[1:]), q_pred)
        log_error = 2 * self.q_ops.quaternion_log_batch(q_rel)
        motion_cost = torch.sum(log_error ** 2).item()
        
        g_pred = self.q_ops.observation_model_batch(q[1:])
        obs_error = self._acc_torch[1:] - g_pred
        obs_cost = torch.sum(obs_error ** 2).item()
        
        return motion_cost, obs_cost
    
    def _project_to_unit_quaternions(self, q_flat: torch.Tensor) -> torch.Tensor:
        T = q_flat.shape[0] // 4
        q = q_flat.reshape(T, 4)
        q_normalized = q / torch.norm(q, dim=1, keepdim=True)
        return q_normalized.flatten()
    
    def optimize(self, q_init: list, omega: np.ndarray, acc: np.ndarray, timestamps: np.ndarray,
                 alpha: float = 0.001, max_iters: int = 100, tol: float = 1e-6,
                 verbose: bool = True, motion_weight: float = None, obs_weight: float = None) -> list:
        if motion_weight is not None:
            self.motion_weight = motion_weight
        if obs_weight is not None:
            self.obs_weight = obs_weight
        
        self._init_sensor_data(omega, acc, timestamps)
        
        q_array = np.array([q.copy() for q in q_init])
        q_flat = self.q_ops.numpy_to_torch(q_array.flatten())
        T = len(q_array)
        
        self.initial_cost = self._cost_function(q_flat).item()
        prev_cost = self.initial_cost
        self.cost_history = [self.initial_cost]
        self.converged_iter = max_iters
        
        if verbose:
            print(f"Starting optimization with initial cost: {prev_cost:.6f}")
        
        for iteration in range(max_iters):
            grad = jacobian(self._cost_function, q_flat)
            
            grad_norm = torch.norm(grad)
            if grad_norm > 100:
                grad = grad * (100 / grad_norm)
            
            q_flat = q_flat - alpha * grad
            q_flat = self._project_to_unit_quaternions(q_flat)
            q_flat = q_flat.detach().clone()
            
            current_cost = self._cost_function(q_flat).item()
            self.cost_history.append(current_cost)
            
            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Cost = {current_cost:.6f}, Grad norm = {grad_norm.item():.6f}")
            
            if np.isnan(current_cost):
                print(f"Warning: NaN detected at iteration {iteration}. Reducing step size.")
                alpha *= 0.5
                q_flat = self.q_ops.numpy_to_torch(q_array.flatten())
                continue
            
            if abs(prev_cost - current_cost) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                self.converged_iter = iteration
                break
            
            prev_cost = current_cost
        
        self.final_cost = current_cost
        self.optimized_quaternions = list(self.q_ops.torch_to_numpy(q_flat).reshape(T, 4))
        return self.optimized_quaternions
    
    def compute_rpy(self, unwrap=True) -> np.ndarray:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        self.optimized_rpy = np.array([
            quat2euler(qi, axes='sxyz') for qi in self.optimized_quaternions
        ])
        
        if unwrap:
            for i in range(3):
                self.optimized_rpy[:, i] = np.unwrap(self.optimized_rpy[:, i])
        
        return self.optimized_rpy
    
    def compute_predicted_gravity(self) -> np.ndarray:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        q_torch = self.q_ops.numpy_to_torch(np.array(self.optimized_quaternions))
        return self.q_ops.torch_to_numpy(self.q_ops.observation_model_batch(q_torch))


class GeodesicTrajectoryOptimizer:
    def __init__(self, device: torch.device = None, motion_weight: float = 0.5, obs_weight: float = 0.5):
        self.device = device 
        self.q_ops = TorchQuaternionOps(self.device)
        self.motion_weight = motion_weight
        self.obs_weight = obs_weight
        
        self._dt_torch = None
        self._omega_torch = None
        self._acc_torch = None
        
        self.optimized_quaternions = None
        self.optimized_rpy = None
        self.final_cost = None
        self.initial_cost = None
        self.cost_history = []
        self.converged_iter = None
    
    def _init_sensor_data(self, omega: np.ndarray, acc: np.ndarray, timestamps: np.ndarray):
        self._dt_torch = self.q_ops.numpy_to_torch(np.diff(timestamps))
        self._omega_torch = self.q_ops.numpy_to_torch(omega[:, :-1].T)
        self._acc_torch = self.q_ops.numpy_to_torch(acc.T)
    
    def _cost_function(self, q_flat: torch.Tensor) -> torch.Tensor:
        T = q_flat.shape[0] // 4
        q = q_flat.reshape(T, 4)
        
        q_pred = self.q_ops.motion_model_batch(q[:-1], self._omega_torch, self._dt_torch)
        q_rel = self.q_ops.qmult_batch(self.q_ops.q_inv_batch(q[1:]), q_pred)
        log_error = 2 * self.q_ops.quaternion_log_batch(q_rel)
        motion_cost = torch.sum(log_error ** 2)
        
        g_pred = self.q_ops.observation_model_batch(q[1:])
        obs_error = self._acc_torch[1:] - g_pred
        obs_cost = torch.sum(obs_error ** 2)
        
        return (self.motion_weight * motion_cost + self.obs_weight * obs_cost) / (self.motion_weight + self.obs_weight)
    
    def compute_individual_costs(self) -> tuple:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        q_array = np.array(self.optimized_quaternions)
        q_flat = self.q_ops.numpy_to_torch(q_array.flatten())
        T = q_flat.shape[0] // 4
        q = q_flat.reshape(T, 4)
        
        q_pred = self.q_ops.motion_model_batch(q[:-1], self._omega_torch, self._dt_torch)
        q_rel = self.q_ops.qmult_batch(self.q_ops.q_inv_batch(q[1:]), q_pred)
        log_error = 2 * self.q_ops.quaternion_log_batch(q_rel)
        motion_cost = torch.sum(log_error ** 2).item()
        
        g_pred = self.q_ops.observation_model_batch(q[1:])
        obs_error = self._acc_torch[1:] - g_pred
        obs_cost = torch.sum(obs_error ** 2).item()
        
        return motion_cost, obs_cost
    
    def _project_to_tangent_space(self, g: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        inner_product = torch.sum(g * q, dim=1, keepdim=True)
        h = g - inner_product * q
        return h
    
    def _geodesic_update(self, q: torch.Tensor, h: torch.Tensor, phi: float, eps: float = 1e-8) -> torch.Tensor:
        h_norm = torch.norm(h, dim=1, keepdim=True)
        significant_mask = (h_norm > eps).float()
        n = -h / (h_norm + eps)
        
        cos_phi = torch.cos(torch.tensor(phi, device=q.device, dtype=q.dtype))
        sin_phi = torch.sin(torch.tensor(phi, device=q.device, dtype=q.dtype))
        
        q_new = q * cos_phi + n * sin_phi * significant_mask
        q_new = torch.where(h_norm > eps, q_new, q)
        q_new = q_new / torch.norm(q_new, dim=1, keepdim=True)
        
        sign = torch.sign(q_new[:, 0:1] + 1e-10)
        q_new = q_new * sign
        
        return q_new
    
    def _backtracking_line_search(self, q: torch.Tensor, h: torch.Tensor, 
                                   current_cost: float, phi_init: float = 0.1,
                                   alpha: float = 0.5, beta: float = 0.8,
                                   max_iters: int = 10) -> float:
        phi = phi_init
        grad_norm_sq = torch.sum(h ** 2).item()
        
        for _ in range(max_iters):
            q_trial = self._geodesic_update(q, h, phi)
            q_trial_flat = q_trial.flatten()
            trial_cost = self._cost_function(q_trial_flat).item()
            
            if trial_cost <= current_cost - alpha * phi * grad_norm_sq:
                return phi
            
            phi *= beta
        
        return phi
    
    def optimize(self, q_init: list, omega: np.ndarray, acc: np.ndarray, timestamps: np.ndarray,
                 phi: float = 0.01, max_iters: int = 100, tol: float = 1e-6,
                 eps: float = 1e-8, use_line_search: bool = False,
                 verbose: bool = True, motion_weight: float = None, obs_weight: float = None) -> list:
        if motion_weight is not None:
            self.motion_weight = motion_weight
        if obs_weight is not None:
            self.obs_weight = obs_weight
        
        self._init_sensor_data(omega, acc, timestamps)
        
        q_array = np.array([q.copy() for q in q_init])
        T = len(q_array)
        q = self.q_ops.numpy_to_torch(q_array)
        q = q / torch.norm(q, dim=1, keepdim=True)
        
        q_flat = q.flatten()
        self.initial_cost = self._cost_function(q_flat).item()
        prev_cost = self.initial_cost
        self.cost_history = [self.initial_cost]
        self.converged_iter = max_iters
        
        if verbose:
            print(f"Geodesic GD - Starting optimization with initial cost: {prev_cost:.6f}")
        
        for iteration in range(max_iters):
            q_flat = q.flatten().clone().detach().requires_grad_(True)
            cost = self._cost_function(q_flat)
            cost.backward()
            g_flat = q_flat.grad.clone()
            
            g = g_flat.reshape(T, 4)
            q = q_flat.detach().reshape(T, 4)
            
            h = self._project_to_tangent_space(g, q)
            grad_norm = torch.norm(h)
            
            if use_line_search:
                current_cost = self._cost_function(q.flatten()).item()
                step_phi = self._backtracking_line_search(q, h, current_cost, phi_init=phi)
            else:
                step_phi = phi
            
            q = self._geodesic_update(q, h, step_phi, eps)
            
            q_flat = q.flatten()
            current_cost = self._cost_function(q_flat).item()
            self.cost_history.append(current_cost)
            
            if verbose and iteration % 5 == 0:
                print(f"Iteration {iteration}: Cost = {current_cost:.6f}, Grad norm = {grad_norm.item():.6f}, Step = {step_phi:.6f}")
            
            if np.isnan(current_cost):
                print(f"Warning: NaN detected at iteration {iteration}. Reducing step size.")
                phi *= 0.5
                q = self.q_ops.numpy_to_torch(q_array)
                continue
            
            if grad_norm < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} (gradient norm < {tol})")
                self.converged_iter = iteration
                break
            
            if abs(prev_cost - current_cost) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} (cost change < {tol})")
                self.converged_iter = iteration
                break
            
            prev_cost = current_cost
        
        self.final_cost = current_cost
        self.optimized_quaternions = list(self.q_ops.torch_to_numpy(q))
        return self.optimized_quaternions
    
    def compute_rpy(self) -> np.ndarray:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        self.optimized_rpy = np.array([
            quat2euler(qi, axes='sxyz') for qi in self.optimized_quaternions
        ])
        return self.optimized_rpy
    
    def compute_predicted_gravity(self) -> np.ndarray:
        if self.optimized_quaternions is None:
            raise ValueError("Run optimize() first")
        
        q_torch = self.q_ops.numpy_to_torch(np.array(self.optimized_quaternions))
        return self.q_ops.torch_to_numpy(self.q_ops.observation_model_batch(q_torch))


def optimize_trajectory_geodesic(data_loader, estimator, phi: float = 0.01,
                                  max_iters: int = 100, verbose: bool = True,
                                  motion_weight: float = 0.5, obs_weight: float = 0.5,
                                  use_line_search: bool = False) -> GeodesicTrajectoryOptimizer:
    optimizer = GeodesicTrajectoryOptimizer(motion_weight=motion_weight, obs_weight=obs_weight)
    optimizer.optimize(
        q_init=estimator.quaternions,
        omega=data_loader.calibrated_omega,
        acc=data_loader.calibrated_acc,
        timestamps=data_loader.imu_ts,
        phi=phi,
        max_iters=max_iters,
        verbose=verbose,
        use_line_search=use_line_search
    )
    optimizer.compute_rpy()
    return optimizer


def optimize_trajectory(data_loader, estimator, alpha: float = 0.001,
                        max_iters: int = 100, verbose: bool = True,
                        motion_weight: float = 0.5, obs_weight: float = 0.5) -> TrajectoryOptimizer:
    optimizer = TrajectoryOptimizer(motion_weight=motion_weight, obs_weight=obs_weight)
    optimizer.optimize(
        q_init=estimator.quaternions,
        omega=data_loader.calibrated_omega,
        acc=data_loader.calibrated_acc,
        timestamps=data_loader.imu_ts,
        alpha=alpha,
        max_iters=max_iters,
        verbose=verbose
    )
    optimizer.compute_rpy()
    return optimizer
