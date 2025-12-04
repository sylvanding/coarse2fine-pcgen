"""
体素数据集

用于训练3D Diffusion模型的体素数据集，支持动态点云到体素转换和预渲染体素加载。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path
import h5py
import yaml
import glob

from .h5_loader import PointCloudH5Loader
from ..voxel.converter import PointCloudToVoxel

logger = logging.getLogger(__name__)


class VoxelDataset(Dataset):
    """
    体素数据集
    
    从H5点云文件动态生成体素数据用于训练3D Diffusion模型。
    支持数据预处理、缓存和多种体素化参数。
    """
    
    def __init__(
        self,
        h5_file_path: str,
        voxel_size: int = 64,
        voxelization_method: str = 'gaussian',
        data_key: str = 'point_clouds',
        sigma: float = 1.0,
        volume_dims: Optional[List[float]] = None,
        padding: Optional[List[float]] = None,
        normalize: bool = True,
        augment: bool = False,
        cache_voxels: bool = False,
        max_cache_size: int = 1000,
        subset_indices: Optional[List[int]] = None
    ):
        """
        初始化体素数据集
        
        Args:
            h5_file_path (str): H5点云文件路径
            voxel_size (int): 体素网格分辨率
            voxelization_method (str): 体素化方法 ('occupancy', 'density', 'gaussian')
            data_key (str): H5文件中的数据键名
            sigma (float): 高斯体素化的标准差
            volume_dims (Optional[List[float]]): 体积尺寸 [x, y, z] (nm)
            padding (Optional[List[float]]): 体积边界填充 [x, y, z] (nm)
            normalize (bool): 是否将体素值归一化到[-1, 1]
            augment (bool): 是否启用数据增强
            cache_voxels (bool): 是否缓存转换后的体素
            max_cache_size (int): 最大缓存大小
            subset_indices (Optional[List[int]]): 使用数据子集的索引
        """
        self.h5_file_path = h5_file_path
        self.voxel_size = voxel_size
        self.voxelization_method = voxelization_method
        self.data_key = data_key
        self.sigma = sigma
        self.normalize = normalize
        self.augment = augment
        self.cache_voxels = cache_voxels
        self.max_cache_size = max_cache_size
        
        # 初始化点云加载器
        self.loader = PointCloudH5Loader(h5_file_path, data_key)
        
        # 确定数据集大小
        if subset_indices is not None:
            self.indices = subset_indices
            self.num_samples = len(subset_indices)
        else:
            self.indices = list(range(self.loader.num_samples))
            self.num_samples = self.loader.num_samples
        
        # 初始化体素转换器
        self.converter = PointCloudToVoxel(
            voxel_size=voxel_size,
            method=voxelization_method,
            volume_dims=volume_dims or [20000, 20000, 2500],
            padding=padding or [0, 0, 100]
        )
        
        # 初始化缓存
        self.voxel_cache = {} if cache_voxels else None
        
        logger.info("初始化体素数据集:")
        logger.info(f"  H5文件: {h5_file_path}")
        logger.info(f"  数据集大小: {self.num_samples}")
        logger.info(f"  体素大小: {voxel_size}^3")
        logger.info(f"  体素化方法: {voxelization_method}")
        logger.info(f"  缓存启用: {cache_voxels}")
        
        # 预计算一些统计信息
        self._compute_dataset_stats()
    
    def _compute_dataset_stats(self):
        """计算数据集统计信息"""
        logger.info("计算数据集统计信息...")
        
        # 采样少量数据来估算统计信息
        sample_size = min(10, self.num_samples)
        sample_indices = np.random.choice(len(self.indices), sample_size, replace=False)
        
        voxel_values = []
        for i in sample_indices:
            try:
                point_cloud = self.loader.load_single_cloud(self.indices[i])
                if self.voxelization_method == 'gaussian':
                    voxel_grid = self.converter.convert(point_cloud, sigma=self.sigma)
                else:
                    voxel_grid = self.converter.convert(point_cloud)
                voxel_values.extend(voxel_grid.flatten())
            except Exception as e:
                logger.warning(f"计算统计信息时跳过样本 {i}: {e}")
        
        if voxel_values:
            voxel_values = np.array(voxel_values)
            self.data_mean = np.mean(voxel_values)
            self.data_std = np.std(voxel_values)
            self.data_min = np.min(voxel_values)
            self.data_max = np.max(voxel_values)
            
            logger.info(f"数据统计: 均值={self.data_mean:.4f}, 标准差={self.data_std:.4f}")
            logger.info(f"数据范围: [{self.data_min:.4f}, {self.data_max:.4f}]")
        else:
            logger.warning("无法计算数据统计信息，使用默认值")
            self.data_mean = 0.0
            self.data_std = 1.0
            self.data_min = 0.0
            self.data_max = 1.0
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含体素数据和元信息的字典
        """
        # 获取真实的数据索引
        real_idx = self.indices[idx]
        
        # 检查缓存
        if self.cache_voxels and real_idx in self.voxel_cache:
            voxel_grid = self.voxel_cache[real_idx].copy()
        else:
            # 从H5文件加载点云
            point_cloud = self.loader.load_single_cloud(real_idx)
            
            # 转换为体素
            if self.voxelization_method == 'gaussian':
                voxel_grid = self.converter.convert(point_cloud, sigma=self.sigma)
            else:
                voxel_grid = self.converter.convert(point_cloud)
            
            # 添加到缓存
            if self.cache_voxels and len(self.voxel_cache) < self.max_cache_size:
                self.voxel_cache[real_idx] = voxel_grid.copy()
        
        # 数据增强
        if self.augment:
            voxel_grid = self._apply_augmentation(voxel_grid)
        
        # 归一化
        if self.normalize:
            voxel_grid = self._normalize_voxel(voxel_grid)
        
        # 转换为tensor并添加通道维度
        voxel_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0)  # (1, D, H, W)
        
        return {
            'voxel': voxel_tensor,
            'index': idx,
            'real_index': real_idx
        }
    
    def _apply_augmentation(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            
        Returns:
            np.ndarray: 增强后的体素网格
        """
        # 随机旋转（90度的倍数）
        if np.random.random() < 0.5:
            # 围绕z轴旋转
            k = np.random.randint(0, 4)
            voxel_grid = np.rot90(voxel_grid, k, axes=(0, 1))
        
        # 随机翻转
        if np.random.random() < 0.5:
            # x轴翻转
            voxel_grid = np.flip(voxel_grid, axis=0)
        
        if np.random.random() < 0.5:
            # y轴翻转
            voxel_grid = np.flip(voxel_grid, axis=1)
        
        # 添加少量噪声
        if np.random.random() < 0.3:
            noise_scale = 0.01 * (self.data_max - self.data_min)
            noise = np.random.normal(0, noise_scale, voxel_grid.shape)
            voxel_grid = voxel_grid + noise
            voxel_grid = np.clip(voxel_grid, self.data_min, self.data_max)
        
        return voxel_grid
    
    def _normalize_voxel(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        归一化体素网格到[-1, 1]范围
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            
        Returns:
            np.ndarray: 归一化后的体素网格
        """
        # 将[data_min, data_max]映射到[-1, 1]
        if self.data_max > self.data_min:
            normalized = 2.0 * (voxel_grid - self.data_min) / (self.data_max - self.data_min) - 1.0
        else:
            normalized = voxel_grid - self.data_min
        
        return np.clip(normalized, -1.0, 1.0)
    
    def denormalize_voxel(self, normalized_voxel: torch.Tensor) -> torch.Tensor:
        """
        反归一化体素网格
        
        Args:
            normalized_voxel (torch.Tensor): 归一化的体素网格
            
        Returns:
            torch.Tensor: 反归一化后的体素网格
        """
        if self.normalize:
            # 将[-1, 1]映射回[data_min, data_max]
            denormalized = (normalized_voxel + 1.0) * (self.data_max - self.data_min) / 2.0 + self.data_min
            return torch.clamp(denormalized, self.data_min, self.data_max)
        else:
            return normalized_voxel
    
    def get_sample_voxel(self, idx: int, denormalize: bool = True) -> np.ndarray:
        """
        获取单个样本的体素网格（用于可视化）
        
        Args:
            idx (int): 样本索引
            denormalize (bool): 是否反归一化
            
        Returns:
            np.ndarray: 体素网格
        """
        sample = self[idx]
        voxel = sample['voxel'].squeeze(0).numpy()  # 移除通道维度
        
        if denormalize and self.normalize:
            # 反归一化
            voxel = (voxel + 1.0) * (self.data_max - self.data_min) / 2.0 + self.data_min
            voxel = np.clip(voxel, self.data_min, self.data_max)
        
        return voxel
    
    def save_sample_as_tiff(self, idx: int, output_path: str, denormalize: bool = True):
        """
        将样本保存为TIFF文件
        
        Args:
            idx (int): 样本索引
            output_path (str): 输出文件路径
            denormalize (bool): 是否反归一化
        """
        voxel = self.get_sample_voxel(idx, denormalize)
        self.converter.save_as_tiff(voxel, output_path)
        logger.info(f"已保存样本 {idx} 为 TIFF: {output_path}")
    
    def clear_cache(self):
        """清空缓存"""
        if self.voxel_cache is not None:
            self.voxel_cache.clear()
            logger.info("已清空体素缓存")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        if self.voxel_cache is not None:
            return {
                'cache_enabled': True,
                'cache_size': len(self.voxel_cache),
                'max_cache_size': self.max_cache_size,
                'cache_usage': len(self.voxel_cache) / self.max_cache_size
            }
        else:
            return {'cache_enabled': False}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集详细信息"""
        return {
            'h5_file_path': self.h5_file_path,
            'num_samples': self.num_samples,
            'voxel_size': self.voxel_size,
            'voxelization_method': self.voxelization_method,
            'data_key': self.data_key,
            'sigma': self.sigma,
            'normalize': self.normalize,
            'augment': self.augment,
            'data_stats': {
                'mean': self.data_mean,
                'std': self.data_std,
                'min': self.data_min,
                'max': self.data_max
            },
            'cache_info': self.get_cache_info()
        }


def create_train_val_datasets(
    h5_file_path: str,
    train_ratio: float = 0.8,
    voxel_size: int = 64,
    voxelization_method: str = 'gaussian',
    **kwargs
) -> Tuple[VoxelDataset, VoxelDataset]:
    """
    创建训练和验证数据集
    
    Args:
        h5_file_path (str): H5文件路径
        train_ratio (float): 训练集比例
        voxel_size (int): 体素大小
        voxelization_method (str): 体素化方法
        **kwargs: 其他VoxelDataset参数
        
    Returns:
        Tuple[VoxelDataset, VoxelDataset]: (训练集, 验证集)
    """
    # 获取数据集总大小
    loader = PointCloudH5Loader(h5_file_path, kwargs.get('data_key', 'point_clouds'))
    total_samples = loader.num_samples
    
    # 生成随机索引
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # 分割训练和验证集
    train_size = int(total_samples * train_ratio)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    
    # 创建数据集
    train_kwargs = kwargs.copy()
    train_kwargs['augment'] = kwargs.get('augment', True)
    train_kwargs['cache_voxels'] = kwargs.get('cache_voxels', True)
    
    val_kwargs = kwargs.copy()
    val_kwargs['augment'] = False  # 验证集不使用增强
    val_kwargs['cache_voxels'] = kwargs.get('cache_voxels', True)
    
    train_dataset = VoxelDataset(
        h5_file_path=h5_file_path,
        voxel_size=voxel_size,
        voxelization_method=voxelization_method,
        subset_indices=train_indices,
        **train_kwargs
    )
    
    val_dataset = VoxelDataset(
        h5_file_path=h5_file_path,
        voxel_size=voxel_size,
        voxelization_method=voxelization_method,
        subset_indices=val_indices,
        **val_kwargs
    )
    
    logger.info(f"创建数据集: 训练={len(train_dataset)}, 验证={len(val_dataset)}")
    
    return train_dataset, val_dataset


class PrerenderedVoxelDataset(Dataset):
    """
    预渲染体素数据集
    
    直接从预渲染的H5文件加载体素数据，大幅加速训练速度。
    """
    
    def __init__(
        self,
        prerendered_dir: str,
        normalize: bool = True,
        augment: bool = False,
        subset_indices: Optional[List[int]] = None
    ):
        """
        初始化预渲染体素数据集
        
        Args:
            prerendered_dir (str): 预渲染体素文件目录
            normalize (bool): 是否归一化到[-1, 1]
            augment (bool): 是否启用数据增强
            subset_indices (Optional[List[int]]): 使用数据子集的索引
        """
        self.prerendered_dir = Path(prerendered_dir)
        self.normalize = normalize
        self.augment = augment
        
        # 加载元数据
        metadata_path = self.prerendered_dir / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = yaml.safe_load(f)
        
        logger.info(f"加载预渲染体素数据集: {prerendered_dir}")
        logger.info(f"  体素大小: {self.metadata['voxel_size']}^3")
        logger.info(f"  体素化方法: {self.metadata['voxelization_method']}")
        logger.info(f"  总样本数: {self.metadata['total_samples']}")
        
        # 查找所有批次文件
        batch_files = sorted(glob.glob(str(self.prerendered_dir / "voxels_batch_*.h5")))
        if not batch_files:
            raise FileNotFoundError(f"未找到批次文件: {self.prerendered_dir}")
        
        logger.info(f"  批次文件数: {len(batch_files)}")
        
        # 构建索引映射
        self.batch_files = []
        self.index_map = {}  # global_idx -> (batch_file_idx, local_idx)
        
        global_idx = 0
        for batch_file in batch_files:
            with h5py.File(batch_file, 'r') as f:
                batch_size = f['voxels'].shape[0]
                self.batch_files.append(batch_file)
                
                for local_idx in range(batch_size):
                    self.index_map[global_idx] = (len(self.batch_files) - 1, local_idx)
                    global_idx += 1
        
        self.total_samples = global_idx
        
        # 处理子集索引
        if subset_indices is not None:
            self.indices = [idx for idx in subset_indices if idx < self.total_samples]
            self.num_samples = len(self.indices)
            logger.info(f"  使用子集: {self.num_samples} 样本")
        else:
            self.indices = list(range(self.total_samples))
            self.num_samples = self.total_samples
        
        # 计算数据统计信息（从第一个批次采样）
        self._compute_dataset_stats()
        
        # H5文件句柄缓存（用于加速访问）
        self.h5_handles = {}
    
    def _compute_dataset_stats(self):
        """计算数据集统计信息"""
        logger.info("计算数据集统计信息...")
        
        # 从第一个批次文件采样
        sample_size = min(10, self.num_samples)
        sample_indices = np.random.choice(len(self.indices), sample_size, replace=False)
        
        voxel_values = []
        with h5py.File(self.batch_files[0], 'r') as f:
            for i in sample_indices:
                real_idx = self.indices[i]
                if real_idx in self.index_map:
                    batch_idx, local_idx = self.index_map[real_idx]
                    if batch_idx == 0:  # 只从第一个批次采样
                        voxel = f['voxels'][local_idx]
                        voxel_values.extend(voxel.flatten())
        
        if voxel_values:
            voxel_values = np.array(voxel_values)
            self.data_mean = np.mean(voxel_values)
            self.data_std = np.std(voxel_values)
            self.data_min = np.min(voxel_values)
            self.data_max = np.max(voxel_values)
            
            logger.info(f"数据统计: 均值={self.data_mean:.4f}, 标准差={self.data_std:.4f}")
            logger.info(f"数据范围: [{self.data_min:.4f}, {self.data_max:.4f}]")
        else:
            logger.warning("无法计算数据统计信息，使用默认值")
            self.data_mean = 0.0
            self.data_std = 1.0
            self.data_min = 0.0
            self.data_max = 1.0
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含体素数据和元信息的字典
        """
        # 获取真实的数据索引
        real_idx = self.indices[idx]
        
        if real_idx not in self.index_map:
            raise IndexError(f"索引 {real_idx} 不在数据集范围内")
        
        batch_idx, local_idx = self.index_map[real_idx]
        batch_file = self.batch_files[batch_idx]
        
        # 从H5文件加载体素
        with h5py.File(batch_file, 'r') as f:
            voxel_grid = f['voxels'][local_idx].astype(np.float32)
        
        # 数据增强
        if self.augment:
            voxel_grid = self._apply_augmentation(voxel_grid)
        
        # 归一化
        if self.normalize:
            voxel_grid = self._normalize_voxel(voxel_grid)
        
        # 转换为tensor并添加通道维度
        voxel_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0)  # (1, D, H, W)
        
        return {
            'voxel': voxel_tensor,
            'index': idx,
            'real_index': real_idx
        }
    
    def _apply_augmentation(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            
        Returns:
            np.ndarray: 增强后的体素网格
        """
        # 随机旋转（90度的倍数）
        if np.random.random() < 0.5:
            # 围绕z轴旋转
            k = np.random.randint(0, 4)
            voxel_grid = np.rot90(voxel_grid, k, axes=(0, 1))
        
        # 随机翻转
        if np.random.random() < 0.5:
            # x轴翻转
            voxel_grid = np.flip(voxel_grid, axis=0).copy()
        
        if np.random.random() < 0.5:
            # y轴翻转
            voxel_grid = np.flip(voxel_grid, axis=1).copy()
        
        # 添加少量噪声
        if np.random.random() < 0.3:
            noise_scale = 0.01 * (self.data_max - self.data_min)
            noise = np.random.normal(0, noise_scale, voxel_grid.shape)
            voxel_grid = voxel_grid + noise
            voxel_grid = np.clip(voxel_grid, self.data_min, self.data_max)
        
        return voxel_grid
    
    def _normalize_voxel(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        归一化体素网格到[-1, 1]范围
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            
        Returns:
            np.ndarray: 归一化后的体素网格
        """
        # 将[data_min, data_max]映射到[-1, 1]
        if self.data_max > self.data_min:
            normalized = 2.0 * (voxel_grid - self.data_min) / (self.data_max - self.data_min) - 1.0
        else:
            normalized = voxel_grid - self.data_min
        
        return np.clip(normalized, -1.0, 1.0)
    
    def denormalize_voxel(self, normalized_voxel: torch.Tensor) -> torch.Tensor:
        """
        反归一化体素网格
        
        Args:
            normalized_voxel (torch.Tensor): 归一化的体素网格
            
        Returns:
            torch.Tensor: 反归一化后的体素网格
        """
        if self.normalize:
            # 将[-1, 1]映射回[data_min, data_max]
            denormalized = (normalized_voxel + 1.0) * (self.data_max - self.data_min) / 2.0 + self.data_min
            return torch.clamp(denormalized, self.data_min, self.data_max)
        else:
            return normalized_voxel
    
    def clear_cache(self):
        """清空H5文件句柄缓存"""
        for handle in self.h5_handles.values():
            handle.close()
        self.h5_handles.clear()
        logger.info("已清空H5文件句柄缓存")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集详细信息"""
        return {
            'prerendered_dir': str(self.prerendered_dir),
            'num_samples': self.num_samples,
            'voxel_size': self.metadata['voxel_size'],
            'voxelization_method': self.metadata['voxelization_method'],
            'normalize': self.normalize,
            'augment': self.augment,
            'data_stats': {
                'mean': self.data_mean,
                'std': self.data_std,
                'min': self.data_min,
                'max': self.data_max
            },
            'metadata': self.metadata
        }
    
    def __del__(self):
        """析构函数，关闭所有H5文件句柄"""
        self.clear_cache()


def create_prerendered_train_val_datasets(
    prerendered_dir: str,
    train_ratio: float = 0.8,
    **kwargs
) -> Tuple[PrerenderedVoxelDataset, PrerenderedVoxelDataset]:
    """
    创建预渲染数据的训练和验证数据集
    
    Args:
        prerendered_dir (str): 预渲染数据目录
        train_ratio (float): 训练集比例
        **kwargs: 其他PrerenderedVoxelDataset参数
        
    Returns:
        Tuple[PrerenderedVoxelDataset, PrerenderedVoxelDataset]: (训练集, 验证集)
    """
    # 首先创建一个临时数据集来获取总样本数
    temp_dataset = PrerenderedVoxelDataset(prerendered_dir, normalize=False, augment=False)
    total_samples = temp_dataset.total_samples
    
    # 生成随机索引
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # 分割训练和验证集
    train_size = int(total_samples * train_ratio)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    
    # 创建数据集
    train_kwargs = kwargs.copy()
    train_kwargs['augment'] = kwargs.get('augment', True)
    
    val_kwargs = kwargs.copy()
    val_kwargs['augment'] = False  # 验证集不使用增强
    
    train_dataset = PrerenderedVoxelDataset(
        prerendered_dir=prerendered_dir,
        subset_indices=train_indices,
        **train_kwargs
    )
    
    val_dataset = PrerenderedVoxelDataset(
        prerendered_dir=prerendered_dir,
        subset_indices=val_indices,
        **val_kwargs
    )
    
    logger.info(f"创建预渲染数据集: 训练={len(train_dataset)}, 验证={len(val_dataset)}")
    
    return train_dataset, val_dataset


class VoxelCollator:
    """
    体素数据的批次整理器
    
    用于DataLoader中整理批次数据。
    """
    
    def __init__(self, pad_to_size: Optional[int] = None):
        """
        Args:
            pad_to_size (Optional[int]): 如果指定，将体素填充到该大小
        """
        self.pad_to_size = pad_to_size
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch (List[Dict]): 批次中的样本列表
            
        Returns:
            Dict[str, torch.Tensor]: 整理后的批次数据
        """
        # 提取所有体素
        voxels = [item['voxel'] for item in batch]
        indices = [item['index'] for item in batch]
        real_indices = [item['real_index'] for item in batch]
        
        # 堆叠体素
        voxel_batch = torch.stack(voxels, dim=0)
        
        # 如果需要填充
        if self.pad_to_size is not None:
            current_size = voxel_batch.shape[-1]  # 假设是立方体
            if current_size < self.pad_to_size:
                pad_size = self.pad_to_size - current_size
                pad = (0, pad_size, 0, pad_size, 0, pad_size)  # (left, right, top, bottom, front, back)
                voxel_batch = F.pad(voxel_batch, pad, mode='constant', value=0)
        
        return {
            'voxel': voxel_batch,
            'index': torch.tensor(indices),
            'real_index': torch.tensor(real_indices)
        }