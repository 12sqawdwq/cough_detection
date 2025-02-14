import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf  # 使用 soundfile 来加载音频文件
import numpy as np  # 用于处理填充或裁剪
from tqdm import tqdm  # 导入 tqdm

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_map, transform=None, n_mfcc=13, max_frames=1292):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.transform = transform
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames  # 最大帧数
        self.audio_files = []
        self.labels = []

        # 遍历数据集并加载音频文件路径及标签
        for label in os.listdir(audio_dir):
            label_path = os.path.join(audio_dir, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith('.wav'):
                        self.audio_files.append(os.path.join(label_path, file))
                        self.labels.append(label)

        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.label_map.keys()))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
      try:
        audio_file = self.audio_files[idx]
        label = self.labels[idx]

        # 使用 soundfile 加载音频
        audio, sr = sf.read(audio_file)

        #转为单通道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        #确保音频足够长
        min_length = 2048  # 2048个采样点
        if len(audio) < min_length:
            padding = min_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_fft=2048,n_mfcc=self.n_mfcc,hop_length=512)

        #处理MFCC维度
        if mfcc.shape[1] == 0:
            mfcc = np.zeros((self.n_mfcc, 1))
        #标准化
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()

        #调整时间维度
        if mfcc.shape[1] > self.max_frames:
            mfcc = mfcc[:, :self.max_frames]
        else:
            padding = self.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

      except Exception as e:
       print(f"Error processing {audio_file}: {str(e)}")
       return torch.zeros((self.n_mfcc, self.max_frames)), torch.tensor(-1)

    # 转换为tensor
      return torch.FloatTensor(mfcc), torch.tensor(self.label_map[label])
# 使用绝对路径初始化数据集
audio_dir = os.path.abspath("data_audio/data")  # 使用绝对路径
label_map = {'coughs': 1, 'notcoughs': 0}
dataset = AudioDataset(audio_dir=audio_dir, label_map=label_map)

# 自动划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义简单CNN模型
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

#检查cuda是否能用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

# 初始化模型
model = SimpleCNN(input_dim=13, output_dim=2, max_frames=1292).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
epochs = 10  # 根据需要调整 epoch 数量

# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader_len = len(train_loader)  # 获取 train_loader 的长度

    with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch", bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as tepoch:
        for i, (inputs, labels) in enumerate(tepoch): # 使用 enumerate 获取批次索引 i
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1).float())  # 显式增加通道维度并移动到 device
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算批次 accuracy
            batch_accuracy = 100 * correct / total

            # 更新 tqdm 后缀，显示动态指标
            tepoch.set_postfix({
                'loss': f'{loss.item():.4f}', # 显示当前 batch loss
                'accuracy': f'{batch_accuracy:.2f}%', # 显示当前 batch accuracy
                'avg_loss': f'{running_loss/(i+1):.4f}', # 显示平均 loss
                'avg_acc': f'{100 * correct / total:.2f}%' # 显示平均 accuracy
            })

    # 计算并打印 epoch 级别的平均 loss 和 accuracy
    train_epoch_loss = running_loss / train_loader_len
    train_epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.2f}%")

    # 测试模型
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_loader_len = len(test_loader)

    with tqdm(test_loader, desc=f"Epoch [{epoch+1}/{epochs}] - Testing", unit="batch",  bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as tepoch:
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1).float())  # 显式增加通道维度并移动到 device
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 计算批次 accuracy
            batch_accuracy = 100 * test_correct / test_total

            # 更新 tqdm 后缀，显示动态指标
            tepoch.set_postfix({
                'loss': f'{loss.item():.4f}', # 显示当前 batch loss
                'accuracy': f'{batch_accuracy:.2f}%',  # 显示当前 batch accuracy
                'avg_loss': f'{test_loss/(len(tepoch)):.4f}', # 显示平均 loss
                'avg_acc': f'{100 * test_correct / test_total:.2f}%' # 显示平均 accuracy
            })

    # 计算并打印 test 级别的平均 loss 和 accuracy
    test_epoch_loss = test_loss / test_loader_len
    test_epoch_accuracy = 100 * test_correct / test_total
    print(f"Epoch [{epoch+1}/{epochs}] - Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.2f}%")

# 保存模型
model_save_path = os.path.abspath("cough_model.pth")  # 使用绝对路径保存模型
torch.save(model.state_dict(), model_save_path)
print("The model has been saved to", model_save_path)
