import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import torch
import os
import numpy as np
import pandas as pd

# 设置检查点路径
checkpoint_dir = "E:\\PROJECT\\cough_detection\\data\\Cnn14_DecisionLevelMax_mAP=0.385.pth"  # Make sure this path is correct
at_checkpoint_path = checkpoint_dir # Use the same path for both, or specify different ones if needed
sed_checkpoint_path = checkpoint_dir # Use the same path for both, or specify different ones if needed


# 加载音频文件
audio_path = ("E:\PROJECT\deta_set\\archive\data_audio\data\coughs\\audioset_119.wav")
(audio, sr) = librosa.core.load(audio_path, sr=32000, mono=True)
audio = torch.from_numpy(audio).float().unsqueeze(0)

# 音频标记
at = AudioTagging(checkpoint_path=at_checkpoint_path, device='cuda')
(clipwise_output, embedding) = at.inference(audio)
clipwise_output_tensor = torch.from_numpy(clipwise_output)
top_indices = torch.argsort(clipwise_output_tensor, dim=1, descending=True)[0, :10]
predicted_labels_at = [labels[i] for i in top_indices]

print("音频标记结果 (Audio Tagging):")
for i, label in enumerate(predicted_labels_at):
    print(f"  {i+1}. {label}")

# 声音事件检测
sed = SoundEventDetection(checkpoint_path=sed_checkpoint_path, device='cpu')
framewise_output = sed.inference(audio)
framewise_output_np = framewise_output[0]

# 获取声音事件检测的标签
try:
    from audioset_ontology import ontology
    sed_labels_audioset = [node.display_name for node in ontology.load_ontology().get_descendants()]
except ImportError:
    sed_labels_audioset = [f"SED_Label_{i+1}" for i in range(framewise_output_np.shape[1])]

# 提取咳嗽的标签索引
cough_index = sed_labels_audioset.index("cough") if "cough" in sed_labels_audioset else None


if cough_index is not None:
    cough_probabilities = framewise_output_np[:, cough_index]
    cough_times = np.where(cough_probabilities > 0.2)[0] * (1 / sr)  # 假设概率阈值为 0.2

    valid_indices = cough_probabilities[cough_times.astype(int)] > 0.2
    cough_times = cough_times[valid_indices]
    cough_probabilities_filtered = cough_probabilities[cough_times.astype(int)]

    df_cough = pd.DataFrame({"time": cough_times, "probability": cough_probabilities_filtered})

    output_file = "cough_timestamps.csv"
    df_cough.to_csv(output_file, index=False)

    print(f"\nCough 事件时间戳和概率已保存到 {output_file}")
    print("\nCough 事件时间戳和概率:")
    print(df_cough)

else:
    print("\n未检测到咳嗽事件。")