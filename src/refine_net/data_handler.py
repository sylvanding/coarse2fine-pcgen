import torch
import random
import src.refine_net.util as util
from torch.utils.data import Dataset
from pathlib import Path
import os
import h5py
import numpy as np
from typing import List, Optional, Tuple
NUM_CLUSTERS = 5


def get_dataset(mode='h5_paired'):
    mode = mode.lower().strip()

    if mode == 'h5_paired':
        return H5PairedDataset
    elif mode == 'h5_inference':
        return H5InferenceDataset


class H5PairedDataset(Dataset):
    """
    H5配对点云数据集，用于加载GT和噪声点云数据
    
    从两个H5文件分别加载GT点云和噪声点云，对它们进行点采样得到相同点数的数据对。
    网络将在噪声点云上预测offset来修正为GT点云。
    支持训练/验证数据集分割和点云归一化。
    """
    
    def __init__(self, dummy_pc, real_device, args, split='train'):
        """
        初始化H5配对数据集
        
        Args:
            dummy_pc: 占位参数，为了兼容原有接口
            real_device: 计算设备
            args: 参数配置对象
            split: 数据集分割类型 ('train', 'val', 'inference')
        """
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.split = split
        
        # 归一化参数
        self.volume_dims = getattr(args, 'volume_dims', [20000, 20000, 2500])
        self.padding = getattr(args, 'padding', [0, 0, 100])
        
        # 加载GT和噪声点云数据
        self.gt_data = self._load_h5_data(args.gt_h5_path, args.gt_data_key)
        self.noisy_data = self._load_h5_data(args.noisy_h5_path, args.noisy_data_key)
        
        # 验证数据匹配性
        assert self.gt_data.shape[0] == self.noisy_data.shape[0], \
            f"GT和噪声数据的样本数不匹配: {self.gt_data.shape[0]} vs {self.noisy_data.shape[0]}"
        
        self.num_samples = self.gt_data.shape[0]
        
        # 数据集分割
        self._split_dataset()
        
        print(f"加载H5配对数据集 ({split}): {len(self.sample_indices)} 个样本")
        print(f"GT点云: {self.gt_data.shape}")
        print(f"噪声点云: {self.noisy_data.shape}")
        
    def _split_dataset(self):
        """根据训练数据占比分割数据集"""
        train_ratio = getattr(self.args, 'train_ratio', 0.8)
        num_train = int(self.num_samples * train_ratio)
        
        # 固定随机种子确保分割一致性
        np.random.seed(42)
        all_indices = np.random.permutation(self.num_samples)
        
        if self.split == 'train':
            self.sample_indices = all_indices[:num_train]
        elif self.split == 'val':
            self.sample_indices = all_indices[num_train:]
        elif self.split == 'inference':
            # 推理使用验证集前4个样本
            val_indices = all_indices[num_train:]
            self.sample_indices = val_indices[:min(4, len(val_indices))]
        else:
            raise ValueError(f"未知的split类型: {self.split}")
            
        print(f"数据集分割: 训练={num_train}, 验证={self.num_samples-num_train}, 当前split={self.split}")
        
    def _load_h5_data(self, file_path: str, data_key: str = 'data') -> np.ndarray:
        """
        从H5文件加载点云数据
        
        Args:
            file_path: H5文件路径
            data_key: 数据键名
            
        Returns:
            numpy.ndarray: 形状为 (samples, points, 3) 的点云数据
        """
        with h5py.File(file_path, 'r') as f:
            if data_key not in f:
                available_keys = list(f.keys())
                raise ValueError(f"数据键 '{data_key}' 不存在。可用键: {available_keys}")
            
            data = f[data_key][:]
            
            if len(data.shape) != 3 or data.shape[2] != 3:
                raise ValueError(f"数据格式错误，期望 (样本数, 点数, 3)，得到 {data.shape}")
                
        return data.astype(np.float32)
    
    def normalize_pointcloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        归一化点云到[-1, 1]范围
        
        Args:
            point_cloud: 形状为 (N, 3) 的点云
            
        Returns:
            numpy.ndarray: 归一化后的点云
        """
        # 计算实际空间范围: volume_dims ± padding
        # x, y, z 坐标范围分别为 [-padding[i], volume_dims[i] + padding[i]]
        x_min, x_max = -self.padding[0], self.volume_dims[0] + self.padding[0]  # [-0, 20000+0] = [0, 20000]
        y_min, y_max = -self.padding[1], self.volume_dims[1] + self.padding[1]  # [-0, 20000+0] = [0, 20000]
        z_min, z_max = -self.padding[2], self.volume_dims[2] + self.padding[2]  # [-100, 2500+100] = [-100, 2600]
        
        # 将点云坐标从原始范围归一化到 [-1, 1]
        normalized = point_cloud.copy()
        normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min) * 2 - 1  # [x_min, x_max] -> [-1, 1]
        normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min) * 2 - 1  # [y_min, y_max] -> [-1, 1] 
        # normalized[:, 2] = (normalized[:, 2] - z_min) / (z_max - z_min) * 2 - 1  # [z_min, z_max] -> [-1, 1]
        normalized[:, 2] = (normalized[:, 2] - z_min) / (y_max - y_min) * 2 - 1  # [z_min, z_max] -> [-1, 1] # 无缩放, 统一系
        
        return normalized
    
    def denormalize_pointcloud(self, normalized_pc: np.ndarray) -> np.ndarray:
        """
        反归一化点云从[-1, 1]范围
        
        Args:
            normalized_pc: 形状为 (N, 3) 的归一化点云
            
        Returns:
            numpy.ndarray: 反归一化后的点云
        """
        # 计算实际空间范围: volume_dims ± padding
        # x, y, z 坐标范围分别为 [-padding[i], volume_dims[i] + padding[i]]
        x_min, x_max = -self.padding[0], self.volume_dims[0] + self.padding[0]  # [-0, 20000+0] = [0, 20000]
        y_min, y_max = -self.padding[1], self.volume_dims[1] + self.padding[1]  # [-0, 20000+0] = [0, 20000]
        z_min, z_max = -self.padding[2], self.volume_dims[2] + self.padding[2]  # [-100, 2500+100] = [-100, 2600]
        
        # 从 [-1, 1] 反归一化到原始坐标范围
        denormalized = normalized_pc.copy()
        denormalized[:, 0] = (denormalized[:, 0] + 1) / 2 * (x_max - x_min) + x_min  # [-1, 1] -> [x_min, x_max]
        denormalized[:, 1] = (denormalized[:, 1] + 1) / 2 * (y_max - y_min) + y_min  # [-1, 1] -> [y_min, y_max]
        # denormalized[:, 2] = (denormalized[:, 2] + 1) / 2 * (z_max - z_min) + z_min  # [-1, 1] -> [z_min, z_max]
        denormalized[:, 2] = (denormalized[:, 2] + 1) / 2 * (y_max - y_min) + z_min  # [-1, 1] -> [z_min, z_max] # 无缩放,统一系
        
        return denormalized

    def _sample_points(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        """
        对点云进行采样，获得指定数量的点
        
        Args:
            point_cloud: 形状为 (N, 3) 的点云
            num_points: 目标点数
            
        Returns:
            numpy.ndarray: 采样后的点云，形状为 (num_points, 3)
        """
        N = point_cloud.shape[0]
        
        if N >= num_points:
            # 随机采样
            indices = np.random.choice(N, num_points, replace=False)
            return point_cloud[indices]
        else:
            # 需要重复采样
            indices = np.random.choice(N, num_points, replace=True)
            return point_cloud[indices]
    
    def __getitem__(self, item):
        """
        获取一对训练数据
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (GT点云, 噪声点云)
            - GT点云: 形状为 (3, num_points)，作为监督目标
            - 噪声点云: 形状为 (3, num_points)，作为网络输入
        """
        # 从分割后的索引中选择样本
        if self.split == 'inference':
            # 推理模式：按顺序选择样本
            sample_idx = self.sample_indices[item % len(self.sample_indices)]
        else:
            # 训练/验证模式：随机选择样本
            sample_idx = np.random.choice(self.sample_indices)
        
        gt_pc = self.gt_data[sample_idx]  # (points, 3)
        noisy_pc = self.noisy_data[sample_idx]  # (points, 3)
        
        # 归一化点云
        gt_pc = self.normalize_pointcloud(gt_pc)
        noisy_pc = self.normalize_pointcloud(noisy_pc)
        
        if self.split == 'inference':
            # 推理模式：保持原始点数，不进行采样
            gt_sampled = gt_pc
            noisy_sampled = noisy_pc
        else:
            # 训练/验证模式：对两个点云分别采样到相同点数
            num_points = getattr(self.args, 'sample_points', 2048)
            gt_sampled = self._sample_points(gt_pc, num_points)
            noisy_sampled = self._sample_points(noisy_pc, num_points)
        
        # 转换为torch tensor并调整维度: (points, 3) -> (3, points)
        gt_tensor = torch.from_numpy(gt_sampled).transpose(0, 1).to(self.device)
        noisy_tensor = torch.from_numpy(noisy_sampled).transpose(0, 1).to(self.device)
        
        return gt_tensor, noisy_tensor
    
    def __len__(self):
        """返回数据集长度"""
        if self.split == 'inference':
            # 推理模式：返回实际样本数
            return len(self.sample_indices)
        else:
            # 训练/验证模式：返回迭代次数
            return self.args.iterations
    
    def single(self, size):
        """兼容原有接口，返回单个随机采样的噪声点云"""
        sample_idx = np.random.choice(self.sample_indices)
        noisy_pc = self.noisy_data[sample_idx]
        # 归一化
        noisy_pc = self.normalize_pointcloud(noisy_pc)
        sampled = self._sample_points(noisy_pc, size)
        return torch.from_numpy(sampled).transpose(0, 1).unsqueeze(0).to(self.real_device)


class H5InferenceDataset(H5PairedDataset):
    """
    推理专用数据集，使用验证集前4个样本的原始点数
    """
    
    def __init__(self, dummy_pc, real_device, args):
        super().__init__(dummy_pc, real_device, args, split='inference')
        
    def __getitem__(self, item):
        """
        获取推理数据
        
        Returns:
            dict: 包含归一化前后点云和样本信息的字典
        """
        sample_idx = self.sample_indices[item]
        
        gt_pc = self.gt_data[sample_idx]  # (points, 3)
        noisy_pc = self.noisy_data[sample_idx]  # (points, 3)
        
        # 保存原始点云（反归一化时需要）
        gt_original = gt_pc.copy()
        noisy_original = noisy_pc.copy()
        
        # 归一化点云
        gt_normalized = self.normalize_pointcloud(gt_pc)
        noisy_normalized = self.normalize_pointcloud(noisy_pc)
        
        # 转换为torch tensor: (points, 3) -> (3, points)
        gt_tensor = torch.from_numpy(gt_normalized).transpose(0, 1).to(self.device)
        noisy_tensor = torch.from_numpy(noisy_normalized).transpose(0, 1).to(self.device)
        
        return {
            'gt_normalized': gt_tensor,
            'noisy_normalized': noisy_tensor,
            'gt_original': gt_original,
            'noisy_original': noisy_original,
            'sample_idx': sample_idx,
            'num_points': gt_pc.shape[0]
        }

