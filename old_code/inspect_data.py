
import pickle
import sys
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

dataset="2"
# Assuming running from 'code' directory
cfile = "../data/trainset/cam/cam" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

print(f"Checking files: {cfile}, {vfile}")
if not os.path.exists(cfile):
    print(f"File not found: {cfile}")
if not os.path.exists(vfile):
    print(f"File not found: {vfile}")

try:
    print("\n--- Camera Data ---")
    camd = read_data(cfile)
    print("Keys:", camd.keys())
    if 'ts' in camd:
        print("Timestamp shape:", camd['ts'].shape)
        print("First 5 timestamps:", camd['ts'].flatten()[:5])
    if 'cam' in camd:
        print("Camera images shape:", camd['cam'].shape) # Expected (H, W, 3, N) or (N, H, W, 3)

    print("\n--- VICON Data ---")
    vicd = read_data(vfile)
    print("Keys:", vicd.keys())
    if 'ts' in vicd:
        print("Timestamp shape:", vicd['ts'].shape)
        print("First 5 timestamps:", vicd['ts'].flatten()[:5])
    if 'rots' in vicd:
        print("Rotation shape:", vicd['rots'].shape)

except Exception as e:
    print(f"Error: {e}")
