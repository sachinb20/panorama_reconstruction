import pickle
import sys
import numpy as np

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')
  return d

dataset="2"
cfile = "../data/trainset/cam/cam" + dataset + ".p"

try:
    camd = read_data(cfile)
    print("Type of camd:", type(camd))
    if isinstance(camd, dict):
        print("Keys:", camd.keys())
        for k, v in camd.items():
            if hasattr(v, 'shape'):
                print(f"Key: {k}, Shape: {v.shape}, Type: {v.dtype}")
            elif isinstance(v, list):
                print(f"Key: {k}, List length: {len(v)}")
                if len(v) > 0:
                     print(f"First element type: {type(v[0])}")
            else:
                print(f"Key: {k}, Type: {type(v)}")
    else:
        print("camd is not a dict")
except Exception as e:
    print(f"Error: {e}")
