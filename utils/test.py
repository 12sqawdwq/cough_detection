import h5py
import numpy as np

with h5py.File('/home/featurize/work/audioset_tagging_cnn/data/workspace/hdf5s/waveforms/cough_notcough/cough_notcough_train.h5', 'r') as f:
    print(f['audio_name'][:2])    # 前两个音频名
    print(f['target'][:2])       # 标签应为 [0] 或 [1]
    print(f['waveforms'].shape)   # 应为 (样本数, 32000*10)
