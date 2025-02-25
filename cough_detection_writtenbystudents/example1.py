import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels

def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
            clipwise_output[sorted_indexes[k]]))

def plot_sound_event_detection_result(framewise_output, sr, hop_length):
    """Visualization of sound event detection result with dynamic cough highlighting.
    Args:
      framewise_output: (time_steps, classes_num)
      sr: Sample rate
      hop_length: Number of samples between successive frames
    """
    out_fig_path = 'results/sed_result_adaptive_highlight.png' # 修改输出文件名
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    top_n_idxes = idxes[0:5] # 保留 Top N，但排除 cough

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    time = np.arange(0, framewise_output.shape[0]) * hop_length / sr

    # 创建 Figure 和 Subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 6)) # 创建两个子图，共享 x 轴
    fig.suptitle('Sound Event Detection Framewise Output') # 总标题

    # 子图 1: Cough 事件
    cough_idx = labels.index('Cough')
    cough_probabilities = framewise_output[:, cough_idx]
    cough_line, = ax1.plot(time, cough_probabilities, label='cough', linewidth=2, color='red')
    ax1.set_ylabel('Cough Probability')
    ax1.set_ylim(0, 1.)
    ax1.grid(True) # 添加网格

    # 动态高亮显示 Cough 概率超过阈值的时间区间
    cough_highlight_threshold = 0.1  # 设置咳嗽概率高亮阈值 (可以调整)
    cough_segments = []
    segment_start_time = None
    for i, prob in enumerate(cough_probabilities):
        if prob > cough_highlight_threshold:
            current_time = time[i]
            if segment_start_time is None:
                segment_start_time = current_time # 记录段开始时间
        else:
            if segment_start_time is not None:
                segment_end_time = time[i-1] # 记录段结束时间
                cough_segments.append((segment_start_time, segment_end_time))
                segment_start_time = None # 重置段开始时间
    # 处理音频结尾的段
    if segment_start_time is not None:
        cough_segments.append((segment_start_time, time[-1]))


    for start_time, end_time in cough_segments:
        ax1.axvspan(start_time, end_time, facecolor='yellow', alpha=0.3, label='Cough event (Prob > {:.2f})'.format(cough_highlight_threshold) if start_time == cough_segments[0][0] else None) # 只在第一个阴影区域添加标签

    ax1.legend(loc='upper right') # 图例位置调整


    # 子图 2: Top N (非 Cough) 事件 (代码与之前版本相同)
    lines_top_n = []
    for idx in top_n_idxes:
        if idx != cough_idx: # 排除 cough 事件，避免重复绘制
            line, = ax2.plot(time, framewise_output[:, idx], label=ix_to_lb[idx])
            lines_top_n.append(line)

    ax2.set_ylabel('Probability (Top Events)')
    ax2.set_xlabel('Time (s)') # x 轴标签只在底部子图显示
    ax2.set_ylim(0, 1.)
    ax2.grid(True) # 添加网格
    ax2.legend(handles=lines_top_n, loc='upper right') # 图例位置调整


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整子图布局
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))
if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound event detection."""
    device = 'cuda' # 'cuda' | 'cpu'
    audio_path = "E:\PROJECT\deta_set\\archive\data_audio\data\coughs\\audioset_2.wav"
    sr = 32000
    hop_length = 320  # This should match the hop length used in the model
    (audio, _) = librosa.core.load(audio_path, sr=sr, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

    print_audio_tagging_result(clipwise_output[0])

    print('------ Sound event detection ------')
    sed = SoundEventDetection(
        checkpoint_path="E:\\PROJECT\\cough_detection\\data\\Cnn14_mAP=0.431.pth",
        device=device,
        interpolate_mode='nearest', # 'nearest'
    )
    framewise_output = sed.inference(audio)
    """(batch_size, time_steps, classes_num)"""

    plot_sound_event_detection_result(framewise_output[0], sr, hop_length)