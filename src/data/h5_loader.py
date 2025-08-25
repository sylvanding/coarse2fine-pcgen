"""
H5点云数据加载器

用于读取HDF5格式的点云数据文件，支持shape为(样本数, 点数, 3)的数据格式。
"""

import h5py
import numpy as np
from typing import Optional, Tuple, Union, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudH5Loader:
    """
    点云H5文件加载器
    
    支持读取shape为(样本数, 点数, 3)的点云数据，其中3代表xyz坐标。
    
    Attributes:
        file_path (str): H5文件路径
        data_key (str): 数据在H5文件中的键名，默认为'data'
    """
    
    def __init__(self, file_path: str, data_key: str = 'data'):
        """
        初始化H5加载器
        
        Args:
            file_path (str): H5文件的路径
            data_key (str): 数据在H5文件中的键名，默认为'data'
        
        Raises:
            FileNotFoundError: 当文件不存在时抛出
            ValueError: 当数据格式不正确时抛出
        """
        self.file_path = file_path
        self.data_key = data_key
        
        # 验证文件并获取数据信息
        self._validate_file()
        
    def _validate_file(self) -> None:
        """
        验证H5文件的格式和数据完整性
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                if self.data_key not in f:
                    available_keys = list(f.keys())
                    raise ValueError(
                        f"数据键 '{self.data_key}' 不存在于H5文件中。"
                        f"可用的键: {available_keys}"
                    )
                
                data = f[self.data_key]
                shape = data.shape
                
                # 验证数据维度
                if len(shape) != 3:
                    raise ValueError(
                        f"数据应该是3维的 (样本数, 点数, 3)，但得到了 {len(shape)} 维"
                    )
                
                # 验证最后一维是否为3（xyz坐标）
                if shape[2] != 3:
                    raise ValueError(
                        f"最后一维应该是3（xyz坐标），但得到了 {shape[2]}"
                    )
                
                self._num_samples = shape[0]
                self._num_points = shape[1]
                
                logger.info(f"成功加载H5文件: {self.file_path}")
                logger.info(f"数据shape: {shape}")
                logger.info(f"样本数: {self._num_samples}, 每样本点数: {self._num_points}")
                
        except OSError as e:
            raise FileNotFoundError(f"无法打开文件 {self.file_path}: {e}")
    
    @property
    def num_samples(self) -> int:
        """获取样本数量"""
        return self._num_samples
    
    @property
    def num_points(self) -> int:
        """获取每个样本的点数"""
        return self._num_points
    
    @property
    def data_shape(self) -> Tuple[int, int, int]:
        """获取数据的完整shape"""
        return (self._num_samples, self._num_points, 3)
    
    def load_single_cloud(self, index: int) -> np.ndarray:
        """
        加载单个点云样本
        
        Args:
            index (int): 样本索引，范围[0, num_samples-1]
        
        Returns:
            np.ndarray: shape为(点数, 3)的点云数据，dtype为float32
        
        Raises:
            IndexError: 当索引超出范围时抛出
        """
        if index < 0 or index >= self._num_samples:
            raise IndexError(
                f"样本索引 {index} 超出范围 [0, {self._num_samples-1}]"
            )
        
        with h5py.File(self.file_path, 'r') as f:
            point_cloud = f[self.data_key][index].astype(np.float32)
            
        # logger.info(f"加载样本 {index}，点云shape: {point_cloud.shape}")
        return point_cloud
    
    def load_multiple_clouds(self, indices: List[int]) -> np.ndarray:
        """
        加载多个点云样本
        
        Args:
            indices (List[int]): 样本索引列表
        
        Returns:
            np.ndarray: shape为(len(indices), 点数, 3)的点云数据
        
        Raises:
            IndexError: 当任何索引超出范围时抛出
        """
        # 验证所有索引
        for idx in indices:
            if idx < 0 or idx >= self._num_samples:
                raise IndexError(
                    f"样本索引 {idx} 超出范围 [0, {self._num_samples-1}]"
                )
        
        with h5py.File(self.file_path, 'r') as f:
            point_clouds = f[self.data_key][indices].astype(np.float32)
            
        logger.info(f"加载 {len(indices)} 个样本，数据shape: {point_clouds.shape}")
        return point_clouds
    
    def load_all_clouds(self) -> np.ndarray:
        """
        加载所有点云样本
        
        Returns:
            np.ndarray: shape为(样本数, 点数, 3)的完整数据集
        
        Note:
            对于大型数据集，请谨慎使用此方法，可能会消耗大量内存
        """
        with h5py.File(self.file_path, 'r') as f:
            all_clouds = f[self.data_key][:].astype(np.float32)
        
        logger.info(f"加载全部数据，shape: {all_clouds.shape}")
        logger.warning("已加载全部数据到内存，请确保有足够的内存空间")
        return all_clouds
    
    def get_batch_iterator(self, batch_size: int, shuffle: bool = False) -> 'BatchIterator':
        """
        创建批次迭代器
        
        Args:
            batch_size (int): 批次大小
            shuffle (bool): 是否打乱数据顺序
        
        Returns:
            BatchIterator: 批次迭代器对象
        """
        return BatchIterator(self, batch_size, shuffle)
    
    def get_point_cloud_bounds(self, sample_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算点云的边界框
        
        Args:
            sample_indices (Optional[List[int]]): 要计算边界的样本索引，None表示使用所有样本
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (min_bounds, max_bounds)，每个都是shape为(3,)的数组
        """
        if sample_indices is None:
            # 使用所有样本
            with h5py.File(self.file_path, 'r') as f:
                data = f[self.data_key]
                min_bounds = np.min(data, axis=(0, 1))
                max_bounds = np.max(data, axis=(0, 1))
        else:
            # 使用指定样本
            clouds = self.load_multiple_clouds(sample_indices)
            min_bounds = np.min(clouds, axis=(0, 1))
            max_bounds = np.max(clouds, axis=(0, 1))
        
        logger.info(f"点云边界: min={min_bounds}, max={max_bounds}")
        return min_bounds, max_bounds


class BatchIterator:
    """
    批次迭代器
    
    用于按批次迭代点云数据，支持数据打乱。
    """
    
    def __init__(self, loader: PointCloudH5Loader, batch_size: int, shuffle: bool = False):
        """
        初始化批次迭代器
        
        Args:
            loader (PointCloudH5Loader): 数据加载器
            batch_size (int): 批次大小
            shuffle (bool): 是否打乱数据
        """
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indices = np.arange(loader.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        """返回迭代器自身"""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> np.ndarray:
        """
        获取下一个批次
        
        Returns:
            np.ndarray: shape为(batch_size, 点数, 3)的批次数据
        
        Raises:
            StopIteration: 当没有更多批次时抛出
        """
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        batch_data = self.loader.load_multiple_clouds(batch_indices.tolist())
        
        self.current_idx = end_idx
        return batch_data
    
    def __len__(self) -> int:
        """返回批次总数"""
        return (self.loader.num_samples + self.batch_size - 1) // self.batch_size
