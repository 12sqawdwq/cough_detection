import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import matplotlib.pyplot as plt
import os
# import cv2
# import IPython.display as ipd

EPS = 1e-8

# 获取频谱函数
def get_spectrogram(wav):
    D = librosa.stft(wav, n_fft=480, hop_length=160, win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect

# 加载wav
def load_wav_files(input_path):
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        wav, sr = librosa.load(file_path, sr=None)
        print(wav.shape, wav.max(), wav.min())

        # 绘制波形
        plt.figure()
        plt.plot(wav)
        plt.show()

        # 绘制频谱
        spect = get_spectrogram(wav)
        plt.figure()
        plt.imshow(spect, aspect='auto', origin='lower')
        plt.show()

        # 绘制MFCC
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, hop_length=160)
        plt.figure()
        plt.imshow(mfcc, aspect='auto', origin='lower')
        plt.show()

if __name__ == '__main__':
    input_path = 'E:\\PROJECT\\deta_set\\cough_test'  # 实际输入文件夹路径
    output_path = 'E:\\PROJECT\\deta_set\\cough_test\\output2'  # 实际输出文件夹路径
    load_wav_files(input_path)  # 调用函数