import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random
import subprocess

# 获取GPU的利用率
def get_gpu_utilization():
    """通过 nvidia-smi 获取 GPU 利用率"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpu_util = result.stdout.strip()
        return gpu_util
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return "N/A"  # 如果出错，返回 N/A

# 数据增强函数
def augment_time_stretch(y, rate=None):
    """时间拉伸"""
    if rate is None:
        rate = 0.8 + np.random.uniform() * 0.4  # 0.8 ~ 1.2 之间随机
    return librosa.effects.time_stretch(y, rate=rate)

def augment_add_noise(y, noise_factor=None):
    """添加噪声"""
    if noise_factor is None:
        noise_factor = 0.005 + np.random.uniform() * 0.02 # 噪声强度 0.005 ~ 0.02 之间随机
    noise = np.random.randn(len(y))
    augmented_sound = y + noise_factor * noise
    # 归一化，避免音量过大
    return augmented_sound / np.max(np.abs(augmented_sound))

def augment_volume_change(y, volume_factor=None):
    """音量调整"""
    if volume_factor is None:
        volume_factor = 0.8 + np.random.uniform() * 0.4 # 音量调整因子 0.8 ~ 1.2 之间随机
    return y * volume_factor

def augment_time_shift(y, shift_factor=None, sr=22050):
    """时间偏移"""
    if shift_factor is None:
        shift_factor = 0.2 * (np.random.uniform() - 0.5) # 偏移比例 -0.1 ~ 0.1 之间随机
    timeshift_samples = int(y.shape[0] * shift_factor)
    if timeshift_samples > 0:
        padding = (timeshift_samples, 0)
        augmented_sound = np.pad(y, padding, 'constant')[:-timeshift_samples]
    else:
        padding = (0, -timeshift_samples)
        augmented_sound = np.pad(y, padding, 'constant')[-timeshift_samples:]
    return augmented_sound

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_map, transform=None, n_mfcc=13, max_frames=1292, augmentation=False, augmentation_prob=0.5):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.transform = transform
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.audio_files = []
        self.labels = []
        self.augmentation = augmentation # 是否进行数据增强
        self.augmentation_prob = augmentation_prob # 数据增强的概率

        # 遍历音频文件夹，收集音频文件路径和标签
        for label in os.listdir(audio_dir):
            label_path = os.path.join(audio_dir, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith('.wav'):
                        self.audio_files.append(os.path.join(label_path, file))
                        self.labels.append(label)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.label_map.keys()))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            audio_file = self.audio_files[idx]
            label = self.labels[idx]
            audio, sr = sf.read(audio_file)

            # 转为单通道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 确保音频足够长
            min_length = 2048  # 2048 个采样点
            if len(audio) < min_length:
                padding = min_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')

            # 数据增强 (仅对训练集进行增强)
            if self.augmentation and random.random() < self.augmentation_prob:
                augment_type = random.choice(['time_stretch', 'add_noise', 'volume_change', 'time_shift'])
                if augment_type == 'time_stretch':
                    audio = augment_time_stretch(audio)
                elif augment_type == 'add_noise':
                    audio = augment_add_noise(audio)
                elif augment_type == 'volume_change':
                    audio = augment_volume_change(audio)
                elif augment_type == 'time_shift':
                    audio = augment_time_shift(audio, sr=sr)

            # 提取 MFCC 特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=2048, n_mfcc=self.n_mfcc, hop_length=512)

            # MFCC 特征归一化和padding/裁剪
            if mfcc.shape[1] == 0:
                mfcc = np.zeros((self.n_mfcc, 1))
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()
            if mfcc.shape[1] > self.max_frames:
                mfcc = mfcc[:, :self.max_frames]
            else:
                padding = self.max_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

        except Exception as e:
            audio_file = self.audio_files[idx]
            print(f"Warning: Error processing {audio_file}: {str(e)}. 返回默认标签 'notcough' (0).")
            mfcc = torch.zeros((self.n_mfcc, self.max_frames))
            label_tensor = torch.tensor(self.label_map['notcough'])
            return mfcc, label_tensor

        return torch.FloatTensor(mfcc), torch.tensor(self.label_map[label])

# 模型定义 (CNN 模型，加入 Dropout 和 权重初始化)
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_frames):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') # 使用 Kaiming 初始化 convolutional 层
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_input_dim = self.calculate_fc_input_dim(input_dim, max_frames)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu') # 使用 Kaiming 初始化全连接层
        self.dropout = nn.Dropout(p=0.5) # Dropout 正则化
        self.fc2 = nn.Linear(128, output_dim)
        nn.init.xavier_uniform_(self.fc2.weight) # 输出层使用 Xavier 初始化

    def calculate_fc_input_dim(self, input_dim, max_frames):
        conv_output_size = input_dim
        pool_output_size = conv_output_size // 2
        return int(32 * pool_output_size * (max_frames // 2))

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(-1, self.fc_input_dim)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x) # 应用 Dropout
        x = self.fc2(x)
        return x

# 获取GPU的利用率
def get_gpu_utilization():
    """通过 nvidia-smi 获取 GPU 利用率"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpu_util = result.stdout.strip()
        return gpu_util
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return "N/A"  # 如果出错，返回 N/A

if __name__ == '__main__':
    # 超参数定义（不需要命令行传参）
    epochs = 100  # 设置训练的 epochs 数量
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-4
    patience = 10

    # 数据集和数据加载器
    audio_dir = os.path.abspath("data_audio/data")  # 假设数据文件夹名为 "data"
    label_map = {'coughs': 1, 'notcoughs': 0}
    dataset = AudioDataset(audio_dir=audio_dir, label_map=label_map, augmentation=True, augmentation_prob=0.5)

    # 划分训练集、验证集和测试集 (8:1:1 比例)
    train_val_size = int(0.9 * len(dataset))  # 训练集 + 验证集 占 90%
    test_size = len(dataset) - train_val_size  # 测试集 占 10%
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

    train_size = int(0.888 * len(train_val_dataset))  # 训练集占 train_val_dataset 的 88.8% (约为总数据集的 80%)
    val_size = len(train_val_dataset) - train_size  # 验证集占 train_val_dataset 的 11.2% (约为总数据集的 10%)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型、优化器、损失函数、设备
    input_dim = 13  # MFCC 特征维度
    output_dim = 2  # 二分类：cough/notcough
    max_frames = 1292  # MFCC 最大帧数
    model = SimpleCNN(input_dim=input_dim, output_dim=output_dim, max_frames=max_frames)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练过程
    best_val_loss = float('inf')  # 初始化最佳验证损失
    best_model_state = None  # 用于保存最佳模型的参数

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 获取 GPU 利用率并实时更新
                gpu_util = get_gpu_utilization()
                tepoch.set_postfix(loss=running_loss/len(tepoch), accuracy=100 * correct / total, gpu_utilization=gpu_util)

        train_accuracy = 100 * correct / total
        train_loss_epoch = running_loss / len(train_loader)

        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    # 保存最佳模型
    print("Saving the best model...")
    model.load_state_dict(best_model_state)  # 使用最佳验证集模型参数
    model_save_path = "best_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 测试模型
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy with Best Model: {test_accuracy:.2f}%")

