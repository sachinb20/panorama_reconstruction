import pickle
import sys
import cv2
import numpy as np
import os

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')
  return d

def main():
    dataset="11"
    cfile = "../../data/testset/cam/cam" + dataset + ".p"
    output_file = "cam" + dataset + ".avi"

    if not os.path.exists(cfile):
        print(f"Error: File {cfile} not found.")
        return

    print(f"Loading {cfile}...")
    try:
        camd = read_data(cfile)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if 'cam' not in camd:
        print("Error: 'cam' key not found in data.")
        return

    imgs = camd['cam'] # Shape: (240, 320, 3, 1259)

    # Check shape
    print(f"Images shape: {imgs.shape}")
    
    # Handle different shapes if necessary, but based on inspection:
    # (H, W, C, N)
    if len(imgs.shape) == 4:
        height, width, channels, num_frames = imgs.shape
    else:
        print(f"Unexpected shape: {imgs.shape}")
        return

    # Define the codec and create VideoWriter object
    # MJPG is usually safe.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

    print(f"Writing {num_frames} frames to {output_file}...")

    for i in range(num_frames):
        # Extract frame i
        # The shape is (H, W, C, N), so we slice the last dimension
        frame = imgs[:, :, :, i]
        
        # Frame is likely RGB, OpenCV expects BGR
        # Let's assume it is RGB and convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame_bgr)
        
        if i % 100 == 0:
            print(f"Processed {i}/{num_frames} frames")

    out.release()
    print(f"Done. Video saved to {output_file}")

if __name__ == "__main__":
    main()
