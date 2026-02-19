import os
import pickle
import numpy as np
import shutil

def fill_nan_with_prev(arr):
    """
    Fills NaNs in a numpy array with the previous valid value along the last axis (time).
    Handles 1D, 2D, and 3D arrays by recursively applying forward fill.
    """
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr, 0

    nan_count = np.sum(mask)
    
    # Recursively apply forward fill along the last axis
    _fill_recursive(arr)
        
    return arr, nan_count

def _fill_recursive(arr):
    """Recursively iterate until we reach 1D slices, then forward fill."""
    if arr.ndim == 1:
        _fill_1d(arr)
    else:
        for i in range(arr.shape[0]):
            _fill_recursive(arr[i])

def _fill_1d(arr):
    """In-place forward fill for 1D array."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx, axis=0)
    
    # Use the accumulated indices to fill
    arr[:] = arr[idx]
    
    # Handle leading NaNs (if any remain) by backfilling
    if np.any(np.isnan(arr)):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        arr[:] = arr[idx]

def clean_structure(data):
    """
    Recursively walk through data structure (dict, list, array) and clean NaNs.
    Returns: (cleaned_data, total_nan_count)
    """
    total_nans = 0
    
    if isinstance(data, dict):
        for k, v in data.items():
            cleaned_v, nans = clean_structure(v)
            data[k] = cleaned_v
            total_nans += nans
            
    elif isinstance(data, list):
        for i, v in enumerate(data):
            cleaned_v, nans = clean_structure(v)
            data[i] = cleaned_v
            total_nans += nans
            
    elif isinstance(data, np.ndarray):
        # We only care if it's float type, integers won't have NaNs
        if np.issubdtype(data.dtype, np.floating):
            # Check for NaNs
            if np.isnan(data).any():
                data, nans = fill_nan_with_prev(data)
                total_nans += nans
    
    return data, total_nans

def process_file(filepath):
    """
    Loads file, cleans it, saves backup if needed.
    Returns: Number of NaNs found/fixed.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return 0

    cleaned_data, nan_count = clean_structure(data)

    if nan_count > 0:
        # Rename original to _corrupt.p
        dirname, basename = os.path.split(filepath)
        name, ext = os.path.splitext(basename)
        corrupt_path = os.path.join(dirname, f"{name}_corrupt{ext}")
        
        # Copy original to corrupt path
        shutil.copy2(filepath, corrupt_path)
        print(f"Found {nan_count} NaNs in {basename}. Backed up to {name}_corrupt{ext} and fixed.")
        
        # Save cleaned data
        with open(filepath, 'wb') as f:
            pickle.dump(cleaned_data, f)
            
    return nan_count

def main():
    root_dir = "../data"  # Assuming we run from code/
    total_files_checked = 0
    total_files_corrupt = 0
    total_nans_fixed = 0
    file_stats = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".p") and not filename.endswith("_corrupt.p"):
                filepath = os.path.join(dirpath, filename)
                
                nans = process_file(filepath)
                total_files_checked += 1
                if nans > 0:
                    total_files_corrupt += 1
                    total_nans_fixed += nans
                    file_stats.append((filename, nans))
                    
    print("\n--- Summary ---")
    print(f"Files checked: {total_files_checked}")
    print(f"Corrupt files found: {total_files_corrupt}")
    print(f"Total NaNs fixed: {total_nans_fixed}")
    
    if file_stats:
        print("\n--- Per-file NaN counts ---")
        for fname, nans in file_stats:
            print(f"  {fname}: {nans} NaNs")

if __name__ == "__main__":
    main()
