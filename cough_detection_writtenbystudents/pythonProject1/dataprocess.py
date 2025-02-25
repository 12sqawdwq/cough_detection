import os
from pydub import AudioSegment


def convert_to_wav(input_folder, output_folder):
    """
    将 input_folder 中所有音频文件转换为 WAV 格式并保存到 output_folder。
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 如果是文件夹则跳过
        if os.path.isdir(file_path):
            continue

        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 如果已经是 wav，通常就不需要重复转换
        if ext == '.wav':
            continue

        # 尝试转换
        try:
            # 读入音频（pydub 自动调用 ffmpeg 解码）
            sound = AudioSegment.from_file(file_path, format=ext.replace('.', ''))

            # 构造目标路径：output_folder + name + '.wav'
            output_file = os.path.join(output_folder, name + '.wav')
            sound = sound.set_frame_rate(16000).set_channels(1)

            # 导出为 wav 格式
            sound.export(output_file, format='wav')
            print(f"转换成功: {file_path} -> {output_file}")

        except Exception as e:
            print(f"转换失败: {file_path}, 错误信息: {e}")


if __name__ == '__main__':
    input_path = "E:\PROJECT\deta_set\public_dataset" #实际输入文件夹路径
    output_path = "E:\PROJECT\deta_set\public_dataset_output" #实际输出文件夹路径
    convert_to_wav(input_path, output_path) #调用函数