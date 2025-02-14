import os
import librosa
import torch
import torch.nn as nn
import soundfile as sf  # 用于加载音频文件
import numpy as np

# 定义与训练时相同的 SimpleCNN 模型结构
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_frames):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 计算经过卷积和池化后的特征图尺寸
        self.fc_input_dim = self.calculate_fc_input_dim(input_dim, max_frames)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def calculate_fc_input_dim(self, input_dim, max_frames):
        # 卷积后尺寸: (input_dim - 3 + 2*1) / 1 + 1 = input_dim
        # 池化后尺寸: input_dim / 2
        conv_output_size = input_dim
        pool_output_size = conv_output_size // 2
        return int(32 * pool_output_size * (max_frames // 2))

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(-1, self.fc_input_dim)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_cough(audio_path, model, label_map, n_mfcc=13, max_frames=1292, device='cpu'):
    """
    使用预训练模型预测音频文件是否为咳嗽声。

    Args:
        audio_path (str): 音频文件的绝对路径 (.wav).
        model (nn.Module): 加载的预训练模型.
        label_map (dict): 标签映射字典 (例如 {'cough': 1, 'notcough': 0}).
        n_mfcc (int): MFCC 特征的维度.
        max_frames (int): MFCC 特征的最大帧数.
        device (str): 'cpu' 或 'cuda', 指定设备.

    Returns:
        tuple: (预测类别标签 (str), 咳嗽概率 (float)).
               例如: ('cough', 0.95) 或 ('notcough', 0.10)
    """
    try:
        # 使用 soundfile 加载音频
        audio, sr = sf.read(audio_path)

        # 转为单通道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 确保音频足够长
        min_length = 2048  # 2048 个采样点
        if len(audio) < min_length:
            padding = min_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        # 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=2048, n_mfcc=n_mfcc, hop_length=512)

        # 处理MFCC维度
        if mfcc.shape[1] == 0:
            mfcc = np.zeros((n_mfcc, 1))
        # 标准化
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()

        # 调整时间维度
        if mfcc.shape[1] > max_frames:
            mfcc = mfcc[:, :max_frames]
        else:
            padding = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(device) # 添加 batch 维度和通道维度, 移动到设备

        # 模型预测 (设置为评估模式)
        model.eval()
        with torch.no_grad():
            output = model(mfcc_tensor)
            probabilities = torch.softmax(output, dim=1) # Softmax 获取概率
            predicted_class_id = torch.argmax(probabilities, dim=1).item() # 获取概率最大的类别索引
            cough_probability = probabilities[0, label_map['coughs']].item() # 获取 'cough' 类的概率

        # 反向查找标签
        reverse_label_map = {v: k for k, v in label_map.items()} # 反转 label_map 方便通过 id 找 label
        predicted_label = reverse_label_map[predicted_class_id]

        return predicted_label, cough_probability

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return "error", 0.0 # 发生错误时返回 "error" 标签和 0.0 概率


if __name__ == '__main__':
    # 模型和音频文件路径 (直接在代码中指定)
    model_path = "E:\PROJECT\cough_onserve\coughmodel.pth"
    audio_file_path = "E:\PROJECT\deta_set\\archive\data_audio\data\coughs\\audioset_119.wav"

    label_map = {'coughs': 1, 'notcoughs': 0} # 与训练时相同的标签映射
    input_dim = 13 # MFCC 特征维度
    output_dim = 2 # 输出类别数 (cough/notcough)
    max_frames = 1292 #  最大帧数，与训练时保持一致

    # 选择设备，如果有 CUDA 则使用 CUDA，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:{device}")

    # 初始化模型并加载预训练权重
    model = SimpleCNN(input_dim=input_dim, output_dim=output_dim, max_frames=max_frames).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) # 加载模型到指定设备

    # 进行咳嗽预测
    predicted_label, cough_probability = predict_cough(audio_file_path, model, label_map, device=device)

    if predicted_label != "error":
        print(f"Audio file: {audio_file_path}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Cough Probability: {cough_probability:.4f}")
    else:
        print(f"Error processing audio file: {audio_file_path}")
