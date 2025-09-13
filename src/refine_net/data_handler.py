import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import logging
from typing import Optional, Tuple
from scipy.ndimage import zoom

NUM_CLUSTERS = 5

# 设置模块级别日志
logger = logging.getLogger(__name__)


def get_dataset(mode='h5_paired'):
    """
    获取数据集类
    
    Args:
        mode (str): 数据集模式
            - 'h5_paired': 标准配对数据集（惰性加载）
            - 'h5_inference': 推理数据集（惰性加载）
            - 'h5_lazy_cached': 带缓存的惰性加载数据集（推荐）
    
    Returns:
        Dataset class: 对应的数据集类
    """
    mode = mode.lower().strip()

    if mode == 'h5_paired':
        return H5PairedDataset
    elif mode == 'h5_inference':
        return H5InferenceDataset
    elif mode == 'h5_lazy_cached':
        return LazyH5Dataset
    else:
        raise ValueError(f"未知的数据集模式: {mode}。支持的模式: 'h5_paired', 'h5_inference', 'h5_lazy_cached'")


class H5PairedDataset(Dataset):
    """
    H5配对点云数据集，用于加载GT和噪声点云数据（惰性加载版本）
    
    从两个H5文件分别加载GT点云和噪声点云，对它们进行点采样得到相同点数的数据对。
    网络将在噪声点云上预测offset来修正为GT点云。
    支持训练/验证数据集分割和点云归一化。
    
    使用惰性加载策略：
    - 初始化时只获取数据集元信息，不加载实际数据
    - 在__getitem__中按需读取单个样本
    - 支持多进程数据加载（每个worker独立打开文件）
    """
    
    def __init__(self, dummy_pc, real_device, args, split='train'):
        """
        初始化H5配对数据集（惰性加载）
        
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
        
        # 存储文件路径和数据键，不立即加载数据
        self.gt_h5_path = args.gt_h5_path
        self.gt_data_key = args.gt_data_key
        self.noisy_h5_path = args.noisy_h5_path
        self.noisy_data_key = args.noisy_data_key
        
        # 体素指导参数
        self.use_voxel_guidance = getattr(args, 'use_voxel_guidance', False)
        self.voxel_h5_path = getattr(args, 'voxel_h5_path', None)
        self.voxel_data_key = getattr(args, 'voxel_data_key', 'voxel_grids')
        self.voxel_grid_size = getattr(args, 'voxel_grid_size', 64)
        
        # 文件句柄（在worker进程中延迟初始化）
        self.gt_h5_file = None
        self.noisy_h5_file = None
        self.voxel_h5_file = None
        
        # 验证文件存在性并获取数据集元信息
        self._validate_files_and_get_metadata()
        
        # 数据集分割
        self._split_dataset()
        
        print(f"初始化H5配对数据集 ({split}): {len(self.sample_indices)} 个样本")
        print(f"GT文件: {self.gt_h5_path}")
        print(f"噪声文件: {self.noisy_h5_path}")
        if self.use_voxel_guidance:
            print(f"体素文件: {self.voxel_h5_path}")
    
    def _validate_files_and_get_metadata(self):
        """验证文件存在性并获取数据集元信息"""
        if not os.path.exists(self.gt_h5_path):
            raise FileNotFoundError(f"GT文件不存在: {self.gt_h5_path}")
        if not os.path.exists(self.noisy_h5_path):
            raise FileNotFoundError(f"噪声文件不存在: {self.noisy_h5_path}")
        
        # 临时打开文件获取元信息
        with h5py.File(self.gt_h5_path, 'r') as gt_file:
            if self.gt_data_key not in gt_file:
                available_keys = list(gt_file.keys())
                raise ValueError(f"GT数据键 '{self.gt_data_key}' 不存在。可用键: {available_keys}")
            self.gt_shape = gt_file[self.gt_data_key].shape
            self.gt_dtype = gt_file[self.gt_data_key].dtype
        
        with h5py.File(self.noisy_h5_path, 'r') as noisy_file:
            if self.noisy_data_key not in noisy_file:
                available_keys = list(noisy_file.keys())
                raise ValueError(f"噪声数据键 '{self.noisy_data_key}' 不存在。可用键: {available_keys}")
            self.noisy_shape = noisy_file[self.noisy_data_key].shape
            self.noisy_dtype = noisy_file[self.noisy_data_key].dtype
        
        # 验证数据形状匹配
        assert self.gt_shape[0] == self.noisy_shape[0], \
            f"GT和噪声数据的样本数不匹配: {self.gt_shape[0]} vs {self.noisy_shape[0]}"
        
        self.num_samples = self.gt_shape[0]
        
        # 验证体素文件（如果使用）
        if self.use_voxel_guidance:
            if self.voxel_h5_path and os.path.exists(self.voxel_h5_path):
                with h5py.File(self.voxel_h5_path, 'r') as voxel_file:
                    if self.voxel_data_key not in voxel_file:
                        available_keys = list(voxel_file.keys())
                        raise ValueError(f"体素数据键 '{self.voxel_data_key}' 不存在。可用键: {available_keys}")
                    self.voxel_shape = voxel_file[self.voxel_data_key].shape
                    self.voxel_dtype = voxel_file[self.voxel_data_key].dtype
                    
                    assert self.voxel_shape[0] == self.num_samples, \
                        f"体素数据和点云数据的样本数不匹配: {self.voxel_shape[0]} vs {self.num_samples}"
                print(f"体素数据形状: {self.voxel_shape}")
            else:
                print(f"警告: 启用了体素指导但体素文件不存在或路径无效: {self.voxel_h5_path}")
                self.use_voxel_guidance = False
    
    def _ensure_file_handles(self):
        """确保文件句柄已打开（在worker进程中调用）"""
        if self.gt_h5_file is None:
            try:
                self.gt_h5_file = h5py.File(self.gt_h5_path, 'r')
            except Exception as e:
                raise RuntimeError(f"无法打开GT文件 {self.gt_h5_path}: {e}")
        
        if self.noisy_h5_file is None:
            try:
                self.noisy_h5_file = h5py.File(self.noisy_h5_path, 'r')
            except Exception as e:
                raise RuntimeError(f"无法打开噪声文件 {self.noisy_h5_path}: {e}")
        
        if self.use_voxel_guidance and self.voxel_h5_file is None:
            try:
                self.voxel_h5_file = h5py.File(self.voxel_h5_path, 'r')
            except Exception as e:
                raise RuntimeError(f"无法打开体素文件 {self.voxel_h5_path}: {e}")
    
    def _load_single_sample(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        从HDF5文件加载单个样本
        
        Args:
            sample_idx: 样本索引
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
            (GT点云, 噪声点云, 体素数据或None)
        """
        # 确保文件句柄已打开
        self._ensure_file_handles()
        
        try:
            # 读取单个样本
            gt_data = self.gt_h5_file[self.gt_data_key][sample_idx].astype(np.float32)
            noisy_data = self.noisy_h5_file[self.noisy_data_key][sample_idx].astype(np.float32)
            
            voxel_data = None
            if self.use_voxel_guidance and self.voxel_h5_file is not None:
                voxel_data = self.voxel_h5_file[self.voxel_data_key][sample_idx].astype(np.float32)
                zoom_factor = self.voxel_grid_size / voxel_data.shape[1]
                voxel_data = zoom(voxel_data, zoom_factor, order=1)
            
            return gt_data, noisy_data, voxel_data
            
        except Exception as e:
            raise RuntimeError(f"读取样本 {sample_idx} 时出错: {e}")
        
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
        
    def __del__(self):
        """析构函数，确保HDF5文件正确关闭"""
        self._close_file_handles()
    
    def _close_file_handles(self):
        """关闭所有HDF5文件句柄"""
        try:
            if hasattr(self, 'gt_h5_file') and self.gt_h5_file is not None:
                self.gt_h5_file.close()
                self.gt_h5_file = None
        except Exception:
            pass
        
        try:
            if hasattr(self, 'noisy_h5_file') and self.noisy_h5_file is not None:
                self.noisy_h5_file.close()
                self.noisy_h5_file = None
        except Exception:
            pass
        
        try:
            if hasattr(self, 'voxel_h5_file') and self.voxel_h5_file is not None:
                self.voxel_h5_file.close()
                self.voxel_h5_file = None
        except Exception:
            pass
    
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
        获取一对训练数据（惰性加载版本）
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            根据是否使用体素指导返回不同的数据格式
            - 不使用体素指导: (GT点云, 噪声点云)
            - 使用体素指导: (GT点云, 噪声点云, 体素数据)
            - GT点云: 形状为 (3, num_points)，作为监督目标
            - 噪声点云: 形状为 (3, num_points)，作为网络输入
            - 体素数据: 形状为 (C, D, H, W)，用于体素指导
        """
        # 从分割后的索引中选择样本
        if self.split == 'inference':
            # 推理模式：按顺序选择样本
            sample_idx = self.sample_indices[item % len(self.sample_indices)]
        else:
            # 训练/验证模式：随机选择样本
            sample_idx = np.random.choice(self.sample_indices)
        
        # 惰性加载：只读取需要的单个样本
        gt_pc, noisy_pc, voxel_data = self._load_single_sample(sample_idx)
        
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
        
        # 如果使用体素指导，返回体素数据
        if self.use_voxel_guidance and voxel_data is not None:
            # 确保体素数据有正确的维度
            if len(voxel_data.shape) == 3:
                # 添加通道维度: (D, H, W) -> (1, D, H, W)
                voxel_data = voxel_data[np.newaxis, ...]
            
            # 转换为torch tensor
            voxel_tensor = torch.from_numpy(voxel_data.astype(np.float32)).to(self.device)
            
            return gt_tensor, noisy_tensor, voxel_tensor
        else:
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
        """兼容原有接口，返回单个随机采样的噪声点云（惰性加载版本）"""
        sample_idx = np.random.choice(self.sample_indices)
        
        # 惰性加载：只读取噪声点云
        _, noisy_pc, _ = self._load_single_sample(sample_idx)
        
        # 归一化
        noisy_pc = self.normalize_pointcloud(noisy_pc)
        sampled = self._sample_points(noisy_pc, size)
        return torch.from_numpy(sampled).transpose(0, 1).unsqueeze(0).to(self.real_device)


class H5InferenceDataset(H5PairedDataset):
    """
    推理专用数据集，使用验证集前4个样本的原始点数（惰性加载版本）
    """
    
    def __init__(self, dummy_pc, real_device, args):
        super().__init__(dummy_pc, real_device, args, split='inference')
        
    def __getitem__(self, item):
        """
        获取推理数据（惰性加载版本）
        
        Returns:
            dict: 包含归一化前后点云、体素数据和样本信息的字典
        """
        sample_idx = self.sample_indices[item]
        
        # 惰性加载：只读取需要的单个样本
        gt_pc, noisy_pc, voxel_data = self._load_single_sample(sample_idx)
        
        # 保存原始点云（反归一化时需要）
        gt_original = gt_pc.copy()
        noisy_original = noisy_pc.copy()
        
        # 归一化点云
        gt_normalized = self.normalize_pointcloud(gt_pc)
        noisy_normalized = self.normalize_pointcloud(noisy_pc)
        
        # 转换为torch tensor: (points, 3) -> (3, points)
        gt_tensor = torch.from_numpy(gt_normalized).transpose(0, 1).to(self.device)
        noisy_tensor = torch.from_numpy(noisy_normalized).transpose(0, 1).to(self.device)
        
        result = {
            'gt_normalized': gt_tensor,
            'noisy_normalized': noisy_tensor,
            'gt_original': gt_original,
            'noisy_original': noisy_original,
            'sample_idx': sample_idx,
            'num_points': gt_pc.shape[0]
        }
        
        # 如果使用体素指导，添加体素数据
        if self.use_voxel_guidance and voxel_data is not None:
            # 确保体素数据有正确的维度
            if len(voxel_data.shape) == 3:
                # 添加通道维度: (D, H, W) -> (1, D, H, W)
                voxel_data = voxel_data[np.newaxis, ...]
            
            # 转换为torch tensor
            voxel_tensor = torch.from_numpy(voxel_data.astype(np.float32)).to(self.device)
            result['voxel_grid'] = voxel_tensor
        
        return result


class LazyH5Dataset(H5PairedDataset):
    """
    带性能优化的惰性加载H5数据集
    
    额外功能：
    - LRU缓存：缓存最近访问的样本以减少磁盘I/O
    - 性能监控：记录数据加载时间和缓存命中率
    - 批量预加载：可选的后台批量预加载机制
    """
    
    def __init__(self, dummy_pc, real_device, args, split='train'):
        super().__init__(dummy_pc, real_device, args, split)
        
        # 缓存参数
        self.cache_size = getattr(args, 'sample_cache_size', 100)  # 缓存样本数
        self.enable_cache = getattr(args, 'enable_sample_cache', True)
        
        # 简单的LRU缓存实现
        if self.enable_cache:
            self.sample_cache = {}  # {sample_idx: (gt_pc, noisy_pc, voxel_data)}
            self.cache_access_order = []  # 访问顺序，用于LRU
            
        # 性能统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_load_time = 0.0
        
        logger.info(f"初始化惰性H5数据集，缓存大小: {self.cache_size}")
    
    def _load_single_sample_with_cache(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        带缓存的单样本加载
        
        Args:
            sample_idx: 样本索引
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
            (GT点云, 噪声点云, 体素数据或None)
        """
        import time
        start_time = time.time()
        
        if self.enable_cache and sample_idx in self.sample_cache:
            # 缓存命中
            self.cache_hits += 1
            
            # 更新访问顺序（移到最后）
            if sample_idx in self.cache_access_order:
                self.cache_access_order.remove(sample_idx)
            self.cache_access_order.append(sample_idx)
            
            data = self.sample_cache[sample_idx]
            
            self.total_load_time += time.time() - start_time
            return data
        
        # 缓存未命中，从磁盘加载
        self.cache_misses += 1
        data = self._load_single_sample(sample_idx)
        
        # 添加到缓存
        if self.enable_cache:
            self._add_to_cache(sample_idx, data)
        
        self.total_load_time += time.time() - start_time
        return data
    
    def _add_to_cache(self, sample_idx: int, data: Tuple):
        """将样本添加到缓存"""
        # 如果缓存已满，移除最久未使用的样本
        if len(self.sample_cache) >= self.cache_size:
            oldest_idx = self.cache_access_order.pop(0)
            del self.sample_cache[oldest_idx]
        
        # 添加新样本
        self.sample_cache[sample_idx] = data
        self.cache_access_order.append(sample_idx)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.sample_cache) if self.enable_cache else 0,
            'avg_load_time': self.total_load_time / total_accesses if total_accesses > 0 else 0.0
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self.sample_cache.clear()
            self.cache_access_order.clear()
        
        logger.info("缓存已清空")
    
    def __getitem__(self, item):
        """重写以使用缓存版本的数据加载"""
        # 从分割后的索引中选择样本
        if self.split == 'inference':
            # 推理模式：按顺序选择样本
            sample_idx = self.sample_indices[item % len(self.sample_indices)]
        else:
            # 训练/验证模式：随机选择样本
            sample_idx = np.random.choice(self.sample_indices)
        
        # 使用缓存的惰性加载
        gt_pc, noisy_pc, voxel_data = self._load_single_sample_with_cache(sample_idx)
        
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
        
        # 如果使用体素指导，返回体素数据
        if self.use_voxel_guidance and voxel_data is not None:
            # 确保体素数据有正确的维度
            if len(voxel_data.shape) == 3:
                # 添加通道维度: (D, H, W) -> (1, D, H, W)
                voxel_data = voxel_data[np.newaxis, ...]
            
            # 转换为torch tensor
            voxel_tensor = torch.from_numpy(voxel_data.astype(np.float32)).to(self.device)
            
            return gt_tensor, noisy_tensor, voxel_tensor
        else:
            return gt_tensor, noisy_tensor
    
    def __del__(self):
        """析构时打印性能统计"""
        try:
            stats = self.get_cache_stats()
            if stats['cache_hits'] + stats['cache_misses'] > 0:
                logger.info(f"数据集性能统计: 缓存命中率={stats['hit_rate']:.2%}, "
                          f"平均加载时间={stats['avg_load_time']:.4f}秒")
        except Exception:
            pass
        
        # 调用父类析构
        super().__del__()

