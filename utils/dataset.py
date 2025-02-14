import numpy as np
import argparse
import os
import glob
import time
import logging
import h5py
import librosa
import random

from utilities import (
    create_folder,
    pad_or_truncate,
    float32_to_int16,
    create_logging  # 确保从utilities导入
)
import config

def pack_waveforms_to_hdf5(args):
    """优化后的HDF5打包函数，支持二分类数据集"""
    
    # 参数解析
    audios_root = args.audios_dir
    workspace_dir = args.workspace_dir
    dataset_name = args.dataset_name
    
    # 自动生成标准化路径
    waveforms_hdf5_dir = os.path.join(
        workspace_dir, 
        'hdf5s', 
        'waveforms', 
        dataset_name
    )
    create_folder(waveforms_hdf5_dir)
    
    train_h5_path = os.path.join(
        waveforms_hdf5_dir, 
        f'{dataset_name}_train.h5'
    )
    test_h5_path = os.path.join(
        waveforms_hdf5_dir, 
        f'{dataset_name}_test.h5'
    )

    # 日志配置
    logs_dir = os.path.join(workspace_dir, 'logs', 'pack_waveforms')
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')  # 初始化日志
    logging.info(f'{"="*30} 开始打包数据集: {dataset_name} {"="*30}')

    # 类别配置
    categories = ['notcough', 'cough']
    category_to_label = {cat: idx for idx, cat in enumerate(categories)}
    
    # ===================== 数据收集阶段 =====================
    audio_paths, targets, audio_names = [], [], []
    
    for category in categories:
        category_dir = os.path.join(audios_root, category)
        if not os.path.exists(category_dir):
            logging.error(f'严重错误: 缺失类别目录 {category_dir}')
            raise FileNotFoundError(f'目录不存在: {category_dir}')

        # 收集所有.wav文件路径
        for audio_path in glob.glob(os.path.join(category_dir, '*.wav')):
            audio_paths.append(audio_path)
            targets.append(category_to_label[category])
            audio_names.append(os.path.basename(audio_path))
            logging.debug(f'找到文件: {audio_path}')

    total_samples = len(audio_paths)
    if total_samples == 0:
        raise ValueError("错误: 未找到任何.wav文件！请检查数据路径")
    
    logging.info(f'共找到 {total_samples} 个样本')
    logging.info(f'类别分布: notcough={targets.count(0)} | cough={targets.count(1)}')

    # ===================== 数据分割阶段 =====================
    indices = list(range(total_samples))
    random.shuffle(indices)
    split_idx = int(total_samples * 0.8)
    
    # ===================== 训练集打包阶段 =====================
    def process_and_save(audio_paths, indices, output_path):
        """处理并保存数据到HDF5"""
        waveforms = []
        valid_indices = []
        
        # 阶段1: 加载和预处理音频
        for idx in indices:
            audio_path = audio_paths[idx]
            try:
                # 加载音频
                audio, _ = librosa.load(
                    audio_path, 
                    sr=config.sample_rate, 
                    mono=True
                )
                # 裁剪/填充
                audio = pad_or_truncate(audio, config.clip_samples)
                # 数值裁剪
                audio = np.clip(audio, -1.0, 1.0)
                # 转换数据类型
                waveform = float32_to_int16(audio)
                waveforms.append(waveform)
                valid_indices.append(idx)
            except Exception as e:
                logging.error(f"无法处理 {audio_path}: {str(e)}")
                continue

        if len(waveforms) == 0:
            raise ValueError("错误: 无有效波形数据！请检查音频文件")

        # 阶段2: 写入HDF5
        with h5py.File(output_path, 'w') as hf:
            # 写入音频名称
            hf.create_dataset(
                'audio_name',
                data=np.array([audio_names[i].encode() for i in valid_indices]),
                dtype='S100'
            )
            # 写入标签
            hf.create_dataset(
                'target',
                data=np.array([targets[i] for i in valid_indices]),
                dtype=np.int32
            )
            # 写入波形数据（关键修复点）
            hf.create_dataset(
                'waveform',  # 注意是单数形式
                data=np.stack(waveforms),
                dtype=np.int16
            )
            # 写入元数据
            hf.attrs['sample_rate'] = config.sample_rate
            hf.attrs['clip_samples'] = config.clip_samples
            hf.attrs['classes'] = np.array(categories, dtype='S10')

        logging.info(f'成功写入 {len(waveforms)} 样本到 {output_path}')
        logging.info(f'波形数据维度: {np.stack(waveforms).shape}')

    # 处理训练集
    logging.info('开始处理训练集...')
    process_and_save(
        audio_paths, 
        indices[:split_idx], 
        train_h5_path
    )

    # 处理测试集
    logging.info('开始处理测试集...')
    process_and_save(
        audio_paths, 
        indices[split_idx:], 
        test_h5_path
    )

    logging.info(f'{"="*30} 数据集打包完成 {"="*30}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # 打包参数配置
    parser_pack = subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack.add_argument(
        '--audios_dir', 
        required=True,
        help='包含咳嗽/非咳嗽子目录的数据集根目录'
    )
    parser_pack.add_argument(
        '--workspace_dir', 
        default='data/workspace',
        help='工作区根目录'
    )
    parser_pack.add_argument(
        '--dataset_name', 
        required=True,
        help='数据集标识名称 (例: cough_notcough)'
    )

    args = parser.parse_args()

    if args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)
