import os
import h5py
import pandas as pd

def load_cmaps_data(filepath):
    print(f"Loading: {filepath}")
    with h5py.File(filepath, 'r') as f:
        # Assume sensor data is in 'X', operational settings in 'W', RUL in 'Y'
        X = f['X'][:]
        W = f['W'][:]
        Y = f['Y'][:] if 'Y' in f else None

        df = pd.DataFrame(X)
        if Y is not None:
            df['RUL'] = Y
        return df

def load_all_cmaps_files(folder):
    data = {}
    for fname in os.listdir(folder):
        if fname.endswith('.h5'):
            full_path = os.path.join(folder, fname)
            try:
                df = load_cmaps_data(full_path)
                data[fname] = df
            except Exception as e:
                print(f"❌ Failed to load {fname}: {e}")
    return data

if __name__ == "__main__":
    folder = 'analysis2'
    datasets = load_all_cmaps_files(folder)
    print(f"\n✅ Loaded {len(datasets)} files successfully.")
