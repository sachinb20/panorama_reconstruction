"""Panorama Generation - Generates panoramic images from camera data using VICON or IMU."""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from transforms3d.euler import mat2euler
from transforms3d.quaternions import quat2mat


def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')
    return d


def get_closest_past_index(target_ts, source_ts):
    idx = np.searchsorted(source_ts, target_ts, side='right') - 1
    return np.maximum(idx, 0)


class PanoramaGenerator:
    def __init__(self, cam_data, pano_h=512, pano_w=1024, fov_h_deg=60):
        self.cam_ts = cam_data['ts'].flatten()
        self.cam_imgs = cam_data['cam']
        
        self.n_frames = self.cam_imgs.shape[3]
        self.H, self.W = self.cam_imgs.shape[0], self.cam_imgs.shape[1]
        
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
        
        fov_h = fov_h_deg * np.pi / 180
        f = (self.W / 2) / np.tan(fov_h / 2)
        self.K = np.array([[f, 0, self.W / 2],
                           [0, f, self.H / 2],
                           [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        
        self._precompute_camera_rays()
        
        self.R_bc = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
    
    def _precompute_camera_rays(self):
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        pixel_coords = np.vstack([u.flatten(), v.flatten(), np.ones(self.H * self.W)])
        self.cam_rays = self.K_inv @ pixel_coords
    
    def _rotation_to_matrix(self, rotation):
        if rotation.shape == (3, 3):
            return rotation
        elif rotation.shape == (4,):
            return quat2mat(rotation)
        else:
            raise ValueError(f"Unsupported rotation format: {rotation.shape}")
    
    def _project_to_panorama(self, world_rays):
        x, y, z = world_rays[0, :], world_rays[1, :], world_rays[2, :]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)
        
        u_pano = ((lon + np.pi) / (2 * np.pi) * self.pano_w).astype(int)
        v_pano = ((lat + np.pi / 2) / np.pi * self.pano_h).astype(int)
        
        u_pano = np.clip(u_pano, 0, self.pano_w - 1)
        v_pano = np.clip(v_pano, 0, self.pano_h - 1)
        
        return u_pano, v_pano
    
    def generate_from_rotations(self, rotations, timestamps, step=5, verbose=True, 
                                  blend_mode='overwrite'):
        if blend_mode == 'alpha':
            color_sum = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.float64)
            count = np.zeros((self.pano_h, self.pano_w), dtype=np.float64)
        else:
            self.panorama = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.uint8)
        
        if verbose:
            print(f"Processing {self.n_frames} frames (step={step}, blend_mode={blend_mode})...")
        
        for i in range(0, self.n_frames, step):
            if verbose and i % 100 == 0:
                print(f"  Frame {i}/{self.n_frames}")
            
            t = self.cam_ts[i]
            curr_img = self.cam_imgs[:, :, :, i]
            
            rot_idx = get_closest_past_index(t, timestamps)
            
            if isinstance(rotations, np.ndarray) and rotations.ndim == 3:
                R = rotations[:, :, rot_idx]
            elif isinstance(rotations, list):
                R = self._rotation_to_matrix(rotations[rot_idx])
            else:
                raise ValueError("Unsupported rotation format")
            
            cam_rays_body = self.R_bc @ self.cam_rays
            world_rays = R @ cam_rays_body
            
            u_pano, v_pano = self._project_to_panorama(world_rays)
            
            flat_colors = curr_img.reshape(-1, 3)
            
            if blend_mode == 'alpha':
                np.add.at(color_sum, (v_pano, u_pano), flat_colors)
                np.add.at(count, (v_pano, u_pano), 1)
            else:
                self.panorama[v_pano, u_pano] = flat_colors
        
        if blend_mode == 'alpha':
            mask = count > 0
            self.panorama = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.uint8)
            for c in range(3):
                channel = np.zeros((self.pano_h, self.pano_w), dtype=np.float64)
                channel[mask] = color_sum[:, :, c][mask] / count[mask]
                self.panorama[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        self.panorama = np.rot90(self.panorama, k=2)
        
        if verbose:
            print("Panorama generation complete!")
        
        return self.panorama
    
    def save_panorama(self, filename, title="Panorama"):
        plt.figure(figsize=(15, 8))
        plt.imshow(self.panorama)
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")


def generate_vicon_panorama(dataset="9", step=5, output_dir=".", blend_mode='overwrite'):
    cfile = f"../../data/trainset/cam/cam{dataset}.p"
    vfile = f"../../data/trainset/vicon/viconRot{dataset}.p"
    
    print(f"=== Generating VICON Panorama (Dataset {dataset}) ===")
    print(f"Loading data from {cfile} and {vfile}...")
    
    camd = read_data(cfile)
    vicd = read_data(vfile)
    
    generator = PanoramaGenerator(camd)
    
    panorama = generator.generate_from_rotations(
        rotations=vicd['rots'],
        timestamps=vicd['ts'].flatten(),
        step=step,
        blend_mode=blend_mode
    )
    
    blend_suffix = '_alpha' if blend_mode == 'alpha' else '_overwrite'
    output_path = os.path.join(output_dir, f'panorama_vicon_{dataset}{blend_suffix}.png')
    generator.save_panorama(output_path, title=f"Panorama from VICON (Dataset {dataset}, {blend_mode})")
    
    return panorama


def generate_imu_panorama(dataset="1", step=5, output_dir=".", optimizer=None, blend_mode='overwrite'):
    import sys
    sys.path.append('..')
    from imu_core import IMUDataLoader, OrientationEstimator
    from imu_optimization import optimize_trajectory
    
    cfile = f"../../data/trainset/cam/cam{dataset}.p"
    imu_file = f"../../data/trainset/imu/imuRaw{dataset}.p"
    vicon_file = f"../../data/trainset/vicon/viconRot{dataset}.p"
    
    print(f"=== Generating IMU Panorama (Dataset {dataset}) ===")
    
    print(f"Loading camera data from {cfile}...")
    camd = read_data(cfile)
    
    if optimizer is None:
        print(f"Loading IMU data and running optimization...")
        data_loader = IMUDataLoader(imu_file, vicon_file)
        estimator = OrientationEstimator(data_loader)
        estimator.integrate()
        
        optimizer = optimize_trajectory(
            data_loader, estimator,
            alpha=0.005,
            max_iters=5000,
            verbose=True,
            motion_weight=0.5,
            obs_weight=0.5
        )
        print(f"Optimization complete: Initial cost={optimizer.initial_cost:.6f}, Final cost={optimizer.final_cost:.6f}")
    
    generator = PanoramaGenerator(camd)
    
    panorama = generator.generate_from_rotations(
        rotations=optimizer.optimized_quaternions,
        timestamps=data_loader.imu_ts,
        step=step,
        blend_mode=blend_mode
    )
    
    blend_suffix = '_alpha' if blend_mode == 'alpha' else '_overwrite'
    output_path = os.path.join(output_dir, f'panorama_imu_{dataset}{blend_suffix}.png')
    generator.save_panorama(output_path, title=f"Panorama from Optimized IMU (Dataset {dataset}, {blend_mode})")
    
    return panorama


TRAINSET_DATASETS = [1,2,8,9]
BLEND_MODES = ['overwrite', 'alpha']


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate panoramic images')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset number (default: all)')
    parser.add_argument('--step', type=int, default=5, help='Frame sampling step')
    parser.add_argument('--mode', type=str, choices=['vicon', 'imu', 'both'], default='imu',
                        help='Generation mode: vicon, imu, or both')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--blend-mode', type=str, default=None, choices=['overwrite', 'alpha'],
                        help='Blending mode (default: both overwrite and alpha)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = [args.dataset] if args.dataset else TRAINSET_DATASETS
    blend_modes = [args.blend_mode] if args.blend_mode else BLEND_MODES
    
    print(f"Processing datasets: {datasets}")
    print(f"Blend modes: {blend_modes}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    for dataset in datasets:
        for blend_mode in blend_modes:
            try:
                if args.mode in ['vicon', 'both']:
                    generate_vicon_panorama(dataset, args.step, args.output_dir, blend_mode)
                
                if args.mode in ['imu', 'both']:
                    generate_imu_panorama(dataset, args.step, args.output_dir, blend_mode=blend_mode)
            except Exception as e:
                print(f"Error processing dataset {dataset} with {blend_mode}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All panoramas generated!")


if __name__ == "__main__":
    main()
