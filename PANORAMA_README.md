# Panorama Generation: Algorithm and Equations

This document provides a detailed explanation of the panorama generation algorithm used in the ECE276A project.

---

## 1. Overview

The goal is to stitch multiple camera images together into a single **spherical panorama** (equirectangular projection). The camera is mounted on a rotating platform, and we have orientation estimates (from VICON or optimized IMU data) that tell us where the camera was pointing at each timestamp.

### Key Idea
For each camera frame:
1. Compute the 3D ray direction for every pixel in camera coordinates.
2. Transform those rays into the **world frame** using the camera's orientation.
3. Project each world-frame ray onto a spherical panorama using **equirectangular mapping**.
4. Paint the corresponding pixel colors onto the panorama canvas.

---

## 2. Coordinate Frames

We have three coordinate frames:

| Frame | Convention | Usage |
|-------|------------|-------|
| **Camera** | X=Right, Y=Down, Z=Forward | Image plane coordinates |
| **Body** | X=Forward, Y=Left, Z=Up | IMU/platform coordinates |
| **World** | X=Forward, Y=Left, Z=Up (at identity rotation) | Global reference frame |

### Camera-to-Body Transformation

The camera is rigidly attached to the body. We define the rotation matrix:

$$
R_{bc} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}
$$

This transforms a vector in camera frame to body frame:
$$
\mathbf{v}_{\text{body}} = R_{bc} \cdot \mathbf{v}_{\text{cam}}
$$

---

## 3. Camera Intrinsics and Ray Computation

### 3.1 Camera Intrinsic Matrix

Given a horizontal field of view $\text{FOV}_h$ (default 60°), the focal length is:

$$
f = \frac{W/2}{\tan(\text{FOV}_h / 2)}
$$

where $W$ is the image width in pixels.

The intrinsic matrix is:
$$
K = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

where $(c_x, c_y) = (W/2, H/2)$ is the principal point (image center).

### 3.2 Computing Camera Rays

For each pixel $(u, v)$ in the image, we compute the corresponding 3D ray direction in camera coordinates:

$$
\mathbf{p}_{\text{cam}} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

This gives an un-normalized ray direction. We compute this for all $H \times W$ pixels simultaneously.

---

## 4. World-Frame Ray Transformation

### 4.1 Orientation Data

At each timestamp $t$, we have an orientation represented as either:
- **Rotation matrix** $R_t \in SO(3)$ from VICON
- **Quaternion** $q_t = [w, x, y, z]$ from IMU optimization

If given a quaternion, we convert it to a rotation matrix using the standard formula.

### 4.2 Transform Pipeline

For each camera frame at time $t$:

**Step 1: Camera → Body**
$$
\mathbf{r}_{\text{body}} = R_{bc} \cdot \mathbf{r}_{\text{cam}}
$$

**Step 2: Body → World**
$$
\mathbf{r}_{\text{world}} = R_t \cdot \mathbf{r}_{\text{body}}
$$

Combined:
$$
\mathbf{r}_{\text{world}} = R_t \cdot R_{bc} \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

---

## 5. Spherical Projection (Equirectangular Mapping)

### 5.1 Spherical Coordinates

Given a world-frame ray $\mathbf{r} = [x, y, z]^T$, we convert to spherical coordinates:

$$
\text{Longitude: } \lambda = \arctan2(y, x) \in [-\pi, \pi]
$$

$$
\text{Latitude: } \phi = \arcsin\left(\frac{z}{\|\mathbf{r}\|}\right) \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]
$$

### 5.2 Panorama Pixel Mapping

We map spherical coordinates to panorama pixels $(u_p, v_p)$:

$$
u_p = \left\lfloor \frac{\lambda + \pi}{2\pi} \cdot W_p \right\rfloor
$$

$$
v_p = \left\lfloor \frac{\phi + \pi/2}{\pi} \cdot H_p \right\rfloor
$$

where:
- $W_p, H_p$ are the panorama dimensions (default: 1024 × 512)
- $u_p \in [0, W_p - 1]$ indexes the horizontal axis (longitude)
- $v_p \in [0, H_p - 1]$ indexes the vertical axis (latitude)

---

## 6. Complete Algorithm

### 6.1 Pseudocode

```
ALGORITHM: GeneratePanorama
INPUT: 
    cam_images[N]     - N camera frames, each H×W×3
    cam_ts[N]         - Timestamps for each camera frame
    rotations[M]      - M rotation matrices or quaternions
    rot_ts[M]         - Timestamps for rotations
    step              - Subsampling factor (process every step-th frame)

OUTPUT:
    panorama          - H_p × W_p × 3 panoramic image

1. Initialize panorama canvas to zeros: panorama ← zeros(H_p, W_p, 3)

2. Precompute camera rays for all pixels:
   FOR each pixel (u, v) in [0..W-1] × [0..H-1]:
       cam_rays[:, u*H + v] ← K^{-1} @ [u, v, 1]^T

3. FOR i ← 0 to N-1 with step:
   a. Get current camera timestamp: t ← cam_ts[i]
   
   b. Find closest-past rotation index:
      idx ← max(k : rot_ts[k] ≤ t)
   
   c. Get rotation matrix:
      IF rotations are quaternions:
          R ← quat_to_mat(rotations[idx])
      ELSE:
          R ← rotations[:, :, idx]
   
   d. Transform rays to world frame:
      body_rays ← R_bc @ cam_rays           # (3, H*W)
      world_rays ← R @ body_rays            # (3, H*W)
   
   e. Project to spherical coordinates:
      FOR each ray r = [x, y, z]:
          λ ← atan2(y, x)
          φ ← asin(z / ||r||)
          u_p ← floor((λ + π) / (2π) * W_p)
          v_p ← floor((φ + π/2) / π * H_p)
   
   f. Paint pixels (overwrite mode):
      panorama[v_p, u_p] ← cam_images[i][v, u]

4. Rotate panorama 180° (optional, for orientation alignment)

5. RETURN panorama
```

### 6.2 Timestamp Alignment

We use **closest-past** alignment: for each camera frame at time $t$, we find the most recent orientation measurement:

$$
\text{idx} = \max \{ k : \text{rot\_ts}[k] \leq t \}
$$

This is computed efficiently using binary search (`np.searchsorted`).

---

## 7. Key Implementation Details

### 7.1 Overwrite vs. Alpha Blending

Two pixel update modes are supported:

**Overwrite Mode (default)**: Later frames paint over earlier frames.
```python
panorama[v_pano, u_pano] = flat_colors
```

**Alpha Blending Mode**: Weighted average of overlapping pixels.
```python
# Accumulate colors and counts
np.add.at(color_sum, (v_pano, u_pano), flat_colors)
np.add.at(count, (v_pano, u_pano), 1)

# After processing all frames, compute average
panorama[mask] = color_sum[mask] / count[mask]
```

The alpha blending produces smoother transitions between overlapping regions, reducing visible seams. The formula is:

$$
\text{panorama}(u, v) = \frac{\sum_{i \in \text{frames}} \text{color}_i(u, v)}{|\{i : \text{pixel}(u, v) \in \text{frame}_i\}|}
$$

**Usage**:
```bash
# Overwrite mode (default)
python panorama.py --dataset 1 --mode vicon

# Alpha blending mode  
python panorama_test.py --dataset 10 --blend-mode alpha
```

### 7.2 Vectorization

All operations are vectorized using NumPy for efficiency:
- Camera rays are precomputed once for all pixels: `cam_rays` is shape `(3, H*W)`
- Matrix multiplications apply to all pixels simultaneously
- Spherical coordinate conversion uses element-wise operations

### 7.3 Sign Conventions and 180° Rotation

The final panorama is rotated 180° (`np.rot90(panorama, k=2)`) to align the visual orientation with common conventions. This accounts for the particular camera mounting and coordinate system choices.

---

## 8. Mathematical Summary

### Complete Transformation Pipeline

For pixel $(u, v)$ in camera frame $i$ at time $t_i$:

$$
\boxed{
\begin{bmatrix} u_p \\ v_p \end{bmatrix} = 
\text{Spherical}\left( R_{t_i} \cdot R_{bc} \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right)
}
$$

where:
$$
\text{Spherical}([x, y, z]^T) = 
\begin{bmatrix}
\left\lfloor \frac{\arctan2(y, x) + \pi}{2\pi} \cdot W_p \right\rfloor \\
\left\lfloor \frac{\arcsin(z/\|r\|) + \pi/2}{\pi} \cdot H_p \right\rfloor
\end{bmatrix}
$$

---

## 9. Code Reference

The implementation is in [`panorama.py`](file:///home/dell/ECE276A_PR1/code/part2/panorama.py):

| Function/Method | Description |
|-----------------|-------------|
| `PanoramaGenerator.__init__` | Sets up camera intrinsics and precomputes rays |
| `_precompute_camera_rays` | Computes $K^{-1}[u, v, 1]^T$ for all pixels |
| `_project_to_panorama` | Spherical coordinate conversion |
| `generate_from_rotations` | Main loop over frames |
| `get_closest_past_index` | Binary search for timestamp alignment |

---

## 10. Example Visualization

```
Camera Image (t=t_i)          Orientation (R_t)          Panorama
┌──────────────┐              ┌─────────┐              ┌────────────────────┐
│   ████████   │    ────►     │  World  │    ────►     │        ████        │
│   ████████   │   Transform  │  Frame  │   Project    │      ████████      │
│              │              └─────────┘              │   Previous frames  │
└──────────────┘                                       └────────────────────┘
    60° FOV                      SO(3)                  360° × 180° sphere
```

Each camera frame covers a portion of the sphere. As we process frames from a rotating camera, we gradually fill in the entire panorama.
