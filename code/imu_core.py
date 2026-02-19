"""Core IMU processing module with reusable classes for orientation estimation."""

import pickle
import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qmult, mat2quat


class QuaternionOperations:
    @staticmethod
    def q_inv(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)
    
    @staticmethod
    def exp_map(omega: np.ndarray, dt: float) -> np.ndarray:
        theta_vec = omega * dt
        angle = np.linalg.norm(theta_vec)
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        half_angle = angle / 2
        unit_axis = theta_vec / angle
        return np.array([
            np.cos(half_angle),
            unit_axis[0] * np.sin(half_angle),
            unit_axis[1] * np.sin(half_angle),
            unit_axis[2] * np.sin(half_angle)
        ])
    
    @staticmethod
    def to_euler(q: np.ndarray, axes: str = 'sxyz') -> tuple:
        return quat2euler(q, axes=axes)


class IMUDataLoader:
    GYRO_SENS = 3.33 * 180.0 / np.pi
    ACC_SENS = 330
    VREF = 3300.0
    ADC_MAX = 1023.0
    BIAS_SAMPLES = 100
    
    def __init__(self, imu_file: str, vicon_file: str = None):
        self.imu_file = imu_file
        self.vicon_file = vicon_file
        
        self.imu_ts = None
        self.calibrated_omega = None
        self.calibrated_acc = None
        self.gyro_raw = None
        self.acc_raw = None
        
        self.vicon_ts = None
        self.vicon_rpy = None
        self.vicon_rots = None
        
        self._load_imu_data()
        if vicon_file:
            self._load_vicon_data()
    
    def _load_imu_data(self):
        with open(self.imu_file, 'rb') as f:
            imu_arr = pickle.load(f)
        
        self.imu_ts = imu_arr[0, :]
        self.gyro_raw = imu_arr[4:7, :]
        self.acc_raw = imu_arr[1:4, :]
        
        gyro_bias = np.mean(self.gyro_raw[:, :self.BIAS_SAMPLES], axis=1)
        acc_bias = np.mean(self.acc_raw[:, :self.BIAS_SAMPLES], axis=1)
        
        gyro_scale = self.VREF / (self.ADC_MAX * self.GYRO_SENS)
        acc_scale = self.VREF / (self.ADC_MAX * self.ACC_SENS)
        
        self.calibrated_omega = (self.gyro_raw - gyro_bias[:, None]) * gyro_scale
        self.calibrated_acc = (self.acc_raw - acc_bias[:, None]) * acc_scale + np.array([0, 0, 1])[:, None]
    
    def _load_vicon_data(self):
        with open(self.vicon_file, 'rb') as f:
            vicon_dict = pickle.load(f, encoding='latin1')
        
        self.vicon_rots = vicon_dict['rots']
        self.vicon_ts = vicon_dict['ts'].flatten()
        
        vicon_rpy = []
        for i in range(self.vicon_rots.shape[2]):
            R = self.vicon_rots[:, :, i]
            q_vicon = mat2quat(R)
            rpy = quat2euler(q_vicon, axes='sxyz')
            vicon_rpy.append(rpy)
        self.vicon_rpy = np.array(vicon_rpy)
        
        for i in range(3):
            self.vicon_rpy[:, i] = np.unwrap(self.vicon_rpy[:, i])
    
    @property
    def num_samples(self) -> int:
        return self.imu_ts.shape[0]
    
    def get_dt(self) -> np.ndarray:
        return np.diff(self.imu_ts)


class OrientationEstimator:
    def __init__(self, data_loader: IMUDataLoader = None):
        self.data_loader = data_loader
        self.quaternions = None
        self.rpy = None
    
    def motion_model(self, qt: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        dq = QuaternionOperations.exp_map(omega, dt)
        qt1 = qmult(qt, dq)
        return QuaternionOperations.normalize(qt1)
    
    def observation_model(self, qt: np.ndarray) -> np.ndarray:
        gravity_quat = np.array([0, 0, 0, 1])
        qt_inv = QuaternionOperations.q_inv(qt)
        temp = qmult(qt_inv, gravity_quat)
        result = qmult(temp, qt)
        return result[1:4]
    
    def integrate(self, omega: np.ndarray = None, timestamps: np.ndarray = None,
                  q0: np.ndarray = None) -> list:
        if omega is None:
            omega = self.data_loader.calibrated_omega
        if timestamps is None:
            timestamps = self.data_loader.imu_ts
        
        N = timestamps.shape[0]
        q = q0 if q0 is not None else np.array([1.0, 0.0, 0.0, 0.0])
        
        self.quaternions = [q.copy()]
        for t in range(1, N):
            dt = timestamps[t] - timestamps[t - 1]
            w = omega[:, t - 1]
            q = self.motion_model(q, w, dt)
            self.quaternions.append(q.copy())
        
        return self.quaternions
    
    def compute_rpy(self, unwrap=True) -> np.ndarray:
        if self.quaternions is None:
            raise ValueError("Run integrate() first")
        
        self.rpy = np.array([
            QuaternionOperations.to_euler(qi) for qi in self.quaternions
        ])
        
        if unwrap:
            for i in range(3):
                self.rpy[:, i] = np.unwrap(self.rpy[:, i])
        
        return self.rpy
    
    def compute_predicted_gravity(self) -> np.ndarray:
        if self.quaternions is None:
            raise ValueError("Run integrate() first")
        
        return np.array([self.observation_model(qi) for qi in self.quaternions])


def load_and_estimate(imu_file: str, vicon_file: str = None) -> tuple:
    loader = IMUDataLoader(imu_file, vicon_file)
    estimator = OrientationEstimator(loader)
    estimator.integrate()
    estimator.compute_rpy()
    return loader, estimator
