
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from transforms3d.euler import mat2euler

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')
  return d

def get_closest_past_index(target_ts, source_ts):
    """
    Finds the index in source_ts that is closest to target_ts but in the past.
    Assumes source_ts is sorted.
    """
    # Use searchsorted to find the insertion point
    idx = np.searchsorted(source_ts, target_ts, side='right') - 1
    # Clip to verify valid index (though -1 is valid for last element in python, we want 0 to N)
    # If idx is -1, it means target_ts is before the first source_ts. 
    # We'll just use 0 in that case or handle it.
    return np.maximum(idx, 0)

def main():
    dataset = "9" # Can be made an argument
    cfile = "../../data/trainset/cam/cam" + dataset + ".p"
    vfile = "../../data/trainset/vicon/viconRot" + dataset + ".p"

    print(f"Loading data from {cfile} and {vfile}...")
    camd = read_data(cfile)
    vicd = read_data(vfile)

    cam_ts = camd['ts'].flatten()
    cam_imgs = camd['cam'] # Shape: (H, W, 3, N) or similar, need to check
    # Based on inspect output: (240, 320, 3, 1259) -> (H, W, C, N)
    
    vic_ts = vicd['ts'].flatten()
    vic_rots = vicd['rots'] # Shape: (3, 3, N)

    # Transpose images to (N, H, W, C) for easier iteration if needed, 
    # but the current format (H,W,C,N) suggests iterating over the last axis.
    n_cams = cam_imgs.shape[3]
    H, W, _, _ = cam_imgs.shape
    
    # Camera Intrinsics Assumption (FOV 60 degrees horizontal)
    fov_h = 60 * np.pi / 180
    f = (W / 2) / np.tan(fov_h / 2)
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    # Panorama Canvas
    # Equirectangular projection
    # Longitude: -pi to pi (width)
    # Latitude: -pi/2 to pi/2 (height)
    pano_h = 512
    pano_w = 1024
    panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

    print(f"Processing {n_cams} frames...")

    # Pre-compute pixel coordinates grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Homogeneous coordinates (3, H*W)
    pixel_coords = np.vstack([u.flatten(), v.flatten(), np.ones(H*W)])
    
    # Convert to normalized camera coordinates (ray directions in camera frame)
    # P_c = K_inv * p_uv
    cam_rays = K_inv @ pixel_coords # (3, H*W)
    
    # Normalize rays? Not strictly necessary for rotation but good for understanding
    # We want direction vectors generally.
    
    # Iterate over camera frames (subsampling for speed if needed)
    step = 5 
    for i in range(0, n_cams, step):
        if i % 100 == 0:
            print(f"  Frame {i}/{n_cams}")
        
        t = cam_ts[i]
        curr_img = cam_imgs[:, :, :, i] # (H, W, 3)

        # Get orientation
        v_idx = get_closest_past_index(t, vic_ts)
        R = vic_rots[:, :, v_idx]

        # Camera frame to Body frame
        # Camera: Z=Forward, X=Right, Y=Down
        # Body: X=Forward, Y=Left, Z=Up
        # R_bc maps Camera vectors to Body vectors
        # X_c (Right) -> -Y_b (Right)
        # Y_c (Down)  -> -Z_b (Down)
        # Z_c (Fwd)   ->  X_b (Fwd)
        
        R_bc = np.array([[0, 0, 1],
                         [-1, 0, 0],
                         [0, -1, 0]])
        
        cam_rays_body = R_bc @ cam_rays
        
        # Rotate rays to world frame
        # P_w = R * P_b
        world_rays = R @ cam_rays_body # (3, H*W)
        
        x = world_rays[0, :]
        y = world_rays[1, :]
        z = world_rays[2, :]

        # Convert to Spherical Coordinates (r, theta, phi)
        # Assuming Z is up?? Or Y is up? 
        # Standard physics: Theta is azimuthal (longitude), Phi is polar (latitude) from Z axis.
        # But for pano mapping usually:
        # Longitude (lambda) = atan2(y, x) -> [-pi, pi] maps to [0, W]
        # Latitude (phi) = asin(z / r) -> [-pi/2, pi/2] maps to [0, H]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x) # [-pi, pi]
        lat = np.arcsin(z / r) # [-pi/2, pi/2]

        # Map to panorama pixels
        # u_pano = (lon + pi) / (2*pi) * W_pano
        # v_pano = (lat + pi/2) / pi * H_pano
        
        u_pano = ((lon + np.pi) / (2 * np.pi) * pano_w).astype(int)
        v_pano = ((lat + np.pi / 2) / np.pi * pano_h).astype(int)

        # Clip to be safe
        u_pano = np.clip(u_pano, 0, pano_w - 1)
        v_pano = np.clip(v_pano, 0, pano_h - 1)

        # Update panorama (overwrite)
        # Using flat indexing to speed up assignment might be hard with overwrites, 
        # but pure numpy assignment works with duplicate indices (last one wins)
        
        flat_colors = curr_img.reshape(-1, 3)
        panorama[v_pano, u_pano] = flat_colors

    # Rotate panorama 180 degrees
    panorama = np.rot90(panorama, k=2)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(panorama)
    plt.title("Panorama from VICON Orientation")
    plt.axis('off')
    plt.savefig('panorama_vicon.png', bbox_inches='tight')
    print("Saved panorama_vicon.png")

if __name__ == "__main__":
    main()
