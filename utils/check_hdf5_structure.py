import h5py

hdf5_file_path = 'data/workspace/hdf5s/waveforms/cough_notcough_train_train.h5'  # 替换为您的 cough_notcough_train.h5 文件路径

try:
    with h5py.File(hdf5_file_path, 'r') as f:
        print(f"HDF5 file structure of: {hdf5_file_path}")
        f.visititems(lambda name, obj: print(f"  {name}: {obj.dtype if isinstance(obj, h5py.Dataset) else 'Group'}"))

except FileNotFoundError:
    print(f"Error: File not found: {hdf5_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
