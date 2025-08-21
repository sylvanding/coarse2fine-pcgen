"""
点云到体素转换器

实现点云数据到3D体素网格的转换，支持多种体素化策略。
"""

import numpy as np
import tifffile
from typing import Tuple, Optional
import logging
from scipy.ndimage import gaussian_filter, zoom

# 配置日志
logger = logging.getLogger(__name__)


class PointCloudToVoxel:
    """
    点云到体素转换器
    
    将3D点云数据转换为规则的3D体素网格表示。支持多种体素化方法
    和后处理选项。
    
    Attributes:
        voxel_size (int): 体素网格的分辨率（每个维度的体素数）
        method (str): 体素化方法，'occupancy' 或 'density'
        bounds (Optional[Tuple]): 点云的边界框 ((min_x, min_y, min_z), (max_x, max_y, max_z))
        volume_dims (List[float]): 体积尺寸 [x, y, z] (单位: nm)
        padding (List[float]): 体积边界填充 [x, y, z] (单位: nm)
    """
    
    def __init__(
        self, 
        voxel_size: int = 64, 
        method: str = 'occupancy',
        bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        padding_ratio: float = 0.1,
        volume_dims: Optional[list] = None,
        padding: Optional[list] = None
    ):
        """
        初始化体素转换器
        
        Args:
            voxel_size (int): 体素网格分辨率，默认64x64x64
            method (str): 体素化方法
                - 'occupancy': 二值占有网格，有点为1，无点为0
                - 'density': 密度网格，值代表该体素内的点数
                - 'gaussian': 高斯密度网格，点周围有连续密度分布
            bounds (Optional[Tuple]): 自定义边界框，None时自动计算
            padding_ratio (float): 边界扩展比例，用于避免边界点被裁剪
            volume_dims (Optional[list]): 体积尺寸 [x, y, z] (单位: nm)，默认[20000, 20000, 2500]
            padding (Optional[list]): 体积边界填充 [x, y, z] (单位: nm)，默认[0, 0, 100]
        
        Raises:
            ValueError: 当参数无效时抛出
        """
        if voxel_size <= 0:
            raise ValueError("体素大小必须大于0")
        
        if method not in ['occupancy', 'density', 'gaussian']:
            raise ValueError(
                f"不支持的体素化方法: {method}。"
                f"支持的方法: 'occupancy', 'density', 'gaussian'"
            )
        
        if not 0 <= padding_ratio <= 1:
            raise ValueError("padding_ratio必须在[0, 1]范围内")
        
        # 设置默认的体积参数
        if volume_dims is None:
            volume_dims = [20000, 20000, 2500]  # (x, y, z) dimensions in nm
        if padding is None:
            padding = [0, 0, 100]  # Padding around the volume in nm
        
        self.voxel_size = voxel_size
        self.method = method
        self.bounds = bounds
        self.padding_ratio = padding_ratio
        self.volume_dims = np.array(volume_dims, dtype=np.float32)
        self.padding = np.array(padding, dtype=np.float32)
        
        # 计算固定的边界
        self.fixed_min_bounds = -self.padding
        self.fixed_max_bounds = self.volume_dims + self.padding
        
        logger.info(f"初始化体素转换器: size={voxel_size}, method={method}")
        logger.info(f"体积尺寸: {self.volume_dims} nm")
        logger.info(f"填充: {self.padding} nm")
        logger.info(f"固定边界: [{self.fixed_min_bounds}] - [{self.fixed_max_bounds}]")
    
    def _compute_bounds(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算点云的边界框
        
        Args:
            point_cloud (np.ndarray): shape为(N, 3)的点云
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (min_bounds, max_bounds)
        """
        if self.bounds is not None:
            min_bounds = np.array(self.bounds[0])
            max_bounds = np.array(self.bounds[1])
        else:
            # 使用固定的体积边界而不是基于点云自动计算
            min_bounds = self.fixed_min_bounds.copy()
            max_bounds = self.fixed_max_bounds.copy()
        
        return min_bounds, max_bounds
    
    def _normalize_points(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将点云坐标归一化到[0, voxel_size-1]范围
        
        Args:
            point_cloud (np.ndarray): 原始点云，shape为(N, 3)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (归一化后的点云, min_bounds, max_bounds)
        """
        min_bounds, max_bounds = self._compute_bounds(point_cloud)
        
        # 归一化到[0, voxel_size-1]
        normalized_points = (point_cloud - min_bounds) / (max_bounds - min_bounds)
        normalized_points = normalized_points * (self.voxel_size - 1)
        
        # 确保点在有效范围内
        normalized_points = np.clip(normalized_points, 0, self.voxel_size - 1)
        
        return normalized_points, min_bounds, max_bounds
    
    def _occupancy_voxelization(self, normalized_points: np.ndarray) -> np.ndarray:
        """
        占有网格体素化
        
        Args:
            normalized_points (np.ndarray): 归一化后的点云
        
        Returns:
            np.ndarray: 二值体素网格，shape为(voxel_size, voxel_size, voxel_size)
        """
        voxel_grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.uint8)
        
        # 将点坐标转换为整数索引
        voxel_indices = np.floor(normalized_points).astype(np.int32)
        
        # 设置占有体素为1
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
        
        return voxel_grid
    
    def _density_voxelization(self, normalized_points: np.ndarray) -> np.ndarray:
        """
        密度网格体素化
        
        Args:
            normalized_points (np.ndarray): 归一化后的点云
        
        Returns:
            np.ndarray: 密度体素网格，值为该体素内的点数
        """
        voxel_grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
        
        # 将点坐标转换为整数索引
        voxel_indices = np.floor(normalized_points).astype(np.int32)
        
        # 统计每个体素内的点数
        unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        voxel_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = counts
        
        return voxel_grid
    
    def _gaussian_voxelization(self, normalized_points: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        高斯密度体素化
        
        Args:
            normalized_points (np.ndarray): 归一化后的点云
            sigma (float): 高斯核的标准差
        
        Returns:
            np.ndarray: 高斯密度体素网格
        """
        # 先创建密度网格
        density_grid = self._density_voxelization(normalized_points)
        
        # 应用高斯滤波
        gaussian_grid = gaussian_filter(density_grid.astype(np.float32), sigma=sigma)
        
        return gaussian_grid
    
    def convert(self, point_cloud: np.ndarray, **kwargs) -> np.ndarray:
        """
        将点云转换为体素网格
        
        Args:
            point_cloud (np.ndarray): 输入点云，shape为(N, 3)
            **kwargs: 方法特定的参数
                - sigma (float): 高斯方法的标准差，默认1.0
        
        Returns:
            np.ndarray: 体素网格，shape为(voxel_size, voxel_size, voxel_size)
        
        Raises:
            ValueError: 当点云格式不正确时抛出
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(
                f"点云应该是shape为(N, 3)的数组，但得到了shape: {point_cloud.shape}"
            )
        
        if len(point_cloud) == 0:
            logger.warning("输入点云为空，返回零体素网格")
            return np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        
        logger.info(f"开始转换点云，点数: {len(point_cloud)}")
        
        # 归一化点云坐标
        normalized_points, min_bounds, max_bounds = self._normalize_points(point_cloud)
        
        # 根据方法进行体素化
        if self.method == 'occupancy':
            voxel_grid = self._occupancy_voxelization(normalized_points)
        elif self.method == 'density':
            voxel_grid = self._density_voxelization(normalized_points)
        elif self.method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            voxel_grid = self._gaussian_voxelization(normalized_points, sigma)
        
        # 归一化体素网格
        if voxel_grid.max() > voxel_grid.min():
            voxel_grid = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
        else:
            voxel_grid = voxel_grid - voxel_grid.min()
        
        # 保存转换信息供后续使用
        self._last_min_bounds = min_bounds
        self._last_max_bounds = max_bounds
        
        logger.info(f"体素转换完成，网格shape: {voxel_grid.shape}")
        logger.info(f"体素值范围: [{np.min(voxel_grid):.3f}, {np.max(voxel_grid):.3f}]")
        
        return voxel_grid
    
    def convert_batch(self, point_clouds: np.ndarray, **kwargs) -> np.ndarray:
        """
        批量转换点云
        
        Args:
            point_clouds (np.ndarray): 批次点云，shape为(B, N, 3)
            **kwargs: 方法特定的参数
        
        Returns:
            np.ndarray: 批次体素网格，shape为(B, voxel_size, voxel_size, voxel_size)
        """
        batch_size = point_clouds.shape[0]
        voxel_grids = []
        
        logger.info(f"开始批量转换，批次大小: {batch_size}")
        
        for i in range(batch_size):
            voxel_grid = self.convert(point_clouds[i], **kwargs)
            voxel_grids.append(voxel_grid)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i + 1}/{batch_size} 个样本的转换")
        
        batch_voxels = np.stack(voxel_grids, axis=0)
        logger.info(f"批量转换完成，输出shape: {batch_voxels.shape}")
        
        return batch_voxels
    
    def save_as_tiff(self, voxel_grid: np.ndarray, output_path: str) -> None:
        """
        将体素网格保存为TIFF文件
        
        Args:
            voxel_grid (np.ndarray): 体素网格
            output_path (str): 输出文件路径
        """
        # 确保数据类型适合TIFF格式
        if voxel_grid.dtype == np.bool_ or voxel_grid.dtype == np.uint8:
            # 布尔或uint8类型直接保存
            tiff_data = voxel_grid.astype(np.uint8)
        else:
            # 浮点类型归一化到[0, 255]
            min_val = np.min(voxel_grid)
            max_val = np.max(voxel_grid)
            
            if max_val > min_val:
                normalized = (voxel_grid - min_val) / (max_val - min_val)
            else:
                normalized = voxel_grid - min_val
            
            tiff_data = (normalized * 255).astype(np.uint8)
        
        # 保存为3D TIFF
        tifffile.imwrite(output_path, tiff_data)
        logger.info(f"体素网格已保存为TIFF: {output_path}")
        logger.info(f"TIFF文件shape: {tiff_data.shape}, dtype: {tiff_data.dtype}")
    
    def voxel_to_points(self, voxel_grid: np.ndarray, threshold: float = 0.0, 
                       num_points: Optional[int] = None, method: str = 'probabilistic') -> np.ndarray:
        """
        将体素网格转换回点云（采样）- 改进版本避免网格化效应
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            threshold (float): 体素值阈值，低于此值的体素不参与采样 (默认: 0.0)
            num_points (Optional[int]): 目标点数，None时自动根据体素值总和确定
            method (str): 采样方法
                - 'probabilistic': 基于概率分布的连续采样（推荐，避免网格化）
                - 'center': 体素中心采样（传统方法）
                - 'random': 体素内随机采样（传统方法）
                - 'weighted': 基于体素值概率的加权采样（传统方法）
        
        Returns:
            np.ndarray: 采样得到的点云，shape为(M, 3)
        """
        if voxel_grid.shape[0] != voxel_grid.shape[1] or voxel_grid.shape[0] != voxel_grid.shape[2]:
            raise ValueError("体素网格必须是立方体")
        
        scale = self.voxel_size / voxel_grid.shape[0]
        
        if method == 'probabilistic':
            return self._probabilistic_sampling(voxel_grid, threshold, num_points) * scale
        else:
            return self._traditional_sampling(voxel_grid, threshold, num_points, method) * scale
    
    def _probabilistic_sampling(self, voxel_grid: np.ndarray, threshold: float = 0.0, 
                               num_points: Optional[int] = None) -> np.ndarray:
        """
        基于概率分布的连续采样方法，避免网格化效应
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            threshold (float): 体素值阈值
            num_points (Optional[int]): 目标点数
            
        Returns:
            np.ndarray: 采样得到的点云，shape为(M, 3)
        """
        logger.info("使用概率分布采样方法（避免网格化效应）")
        
        # 过滤低于阈值的体素
        valid_mask = voxel_grid > threshold
        
        if not np.any(valid_mask):
            logger.warning("没有体素超过阈值，返回空点云")
            return np.empty((0, 3))
        
        # 计算总体素值
        total_value = np.sum(voxel_grid[valid_mask])
        
        if total_value == 0:
            logger.warning("有效体素的总值为0，返回空点云")
            return np.empty((0, 3))
        
        # 确定目标点数
        if num_points is None:
            # 自动确定点数：基于总体素值的某个倍数
            # 这里使用启发式规则：总体素值 * 1000（可调整）
            num_points = max(1000, int(total_value * 1000))
            logger.info(f"自动确定目标点数: {num_points}")
        
        logger.info(f"开始概率采样：目标点数={num_points:,}, 总体素值={total_value:.3f}")
        
        # 选择最佳的采样算法
        num_valid_voxels = np.sum(valid_mask)
        
        if num_points > num_valid_voxels * 10:
            # 大量采样时使用别名采样算法（O(1)时间复杂度）
            logger.info("使用别名采样算法（高效处理大量采样）")
            return self._alias_sampling(voxel_grid, valid_mask, num_points, total_value)
        else:
            # 中等采样量使用CDF采样（内存友好）
            logger.info("使用CDF采样算法（内存友好）")
            return self._cdf_sampling(voxel_grid, valid_mask, num_points, total_value)
    
    def _cdf_sampling(self, voxel_grid: np.ndarray, valid_mask: np.ndarray, 
                     num_points: int, total_value: float) -> np.ndarray:
        """
        使用累积分布函数进行高效采样
        
        Args:
            voxel_grid (np.ndarray): 体素网格
            valid_mask (np.ndarray): 有效体素掩码
            num_points (int): 目标点数
            total_value (float): 总体素值
            
        Returns:
            np.ndarray: 采样点云
        """
        # 获取有效体素的坐标和值
        valid_coords = np.argwhere(valid_mask)  # shape: (N_valid, 3)
        valid_values = voxel_grid[valid_mask]   # shape: (N_valid,)
        
        # 构建累积概率分布
        probabilities = valid_values / total_value
        cumulative_probs = np.cumsum(probabilities)
        
        # 确保最后一个元素为1.0（避免浮点误差）
        cumulative_probs[-1] = 1.0
        
        logger.info(f"构建CDF完成，有效体素数: {len(valid_coords)}")
        
        # 生成随机数进行采样
        random_values = np.random.random(num_points)
        
        # 使用二分查找找到对应的体素索引
        voxel_indices = np.searchsorted(cumulative_probs, random_values)
        
        # 获取采样的体素坐标
        sampled_voxel_coords = valid_coords[voxel_indices]  # shape: (num_points, 3)
        
        # 在每个体素内生成连续的随机位置（避免离散化）
        # 使用多种分布来增加自然性
        subvoxel_offsets = self._generate_subvoxel_offsets(num_points, method='mixed')
        
        # 计算最终的连续坐标
        final_coords = sampled_voxel_coords.astype(np.float32) + subvoxel_offsets
        
        # 反归一化到原始坐标空间
        point_cloud = self._denormalize_coordinates(final_coords)
        
        logger.info(f"概率采样完成，生成 {len(point_cloud)} 个点")
        return point_cloud
    
    def _alias_sampling(self, voxel_grid: np.ndarray, valid_mask: np.ndarray, 
                       num_points: int, total_value: float) -> np.ndarray:
        """
        使用别名采样算法进行高效大量采样
        
        别名采样 (Alias Method) 是一种可以在O(1)时间内完成单次采样的算法，
        预处理时间复杂度为O(n)，适合大量重复采样的场景。
        
        Args:
            voxel_grid (np.ndarray): 体素网格
            valid_mask (np.ndarray): 有效体素掩码
            num_points (int): 目标点数
            total_value (float): 总体素值
            
        Returns:
            np.ndarray: 采样点云
        """
        # 获取有效体素的坐标和值
        valid_coords = np.argwhere(valid_mask)  # shape: (N_valid, 3)
        valid_values = voxel_grid[valid_mask]   # shape: (N_valid,)
        n_voxels = len(valid_coords)
        
        # 构建别名表
        logger.info(f"构建别名表，有效体素数: {n_voxels}")
        alias_table, prob_table = self._build_alias_table(valid_values / total_value)
        
        # 批量采样
        logger.info("开始批量别名采样")
        batch_size = min(num_points, 100000)  # 控制内存使用
        sampled_coords_list = []
        
        remaining_points = num_points
        while remaining_points > 0:
            current_batch = min(batch_size, remaining_points)
            
            # 生成随机数
            random_indices = np.random.randint(0, n_voxels, current_batch)
            random_probs = np.random.random(current_batch)
            
            # 别名采样决策
            use_primary = random_probs < prob_table[random_indices]
            final_indices = np.where(use_primary, random_indices, alias_table[random_indices])
            
            # 获取采样的体素坐标
            batch_coords = valid_coords[final_indices]
            sampled_coords_list.append(batch_coords)
            
            remaining_points -= current_batch
            
            if len(sampled_coords_list) % 10 == 0:
                logger.info(f"已完成 {num_points - remaining_points:,}/{num_points:,} 点的采样")
        
        # 合并所有批次
        sampled_voxel_coords = np.vstack(sampled_coords_list)
        
        # 在每个体素内生成连续的随机位置
        subvoxel_offsets = self._generate_subvoxel_offsets(num_points, method='mixed')
        
        # 计算最终的连续坐标
        final_coords = sampled_voxel_coords.astype(np.float32) + subvoxel_offsets
        
        # 反归一化到原始坐标空间
        point_cloud = self._denormalize_coordinates(final_coords)
        
        logger.info(f"别名采样完成，生成 {len(point_cloud)} 个点")
        return point_cloud
    
    def _build_alias_table(self, probabilities: np.ndarray) -> tuple:
        """
        构建别名采样表
        
        Args:
            probabilities (np.ndarray): 归一化的概率分布
            
        Returns:
            tuple: (alias_table, prob_table)
        """
        n = len(probabilities)
        
        # 初始化表格
        alias_table = np.zeros(n, dtype=np.int32)
        prob_table = np.zeros(n, dtype=np.float32)
        
        # 缩放概率
        scaled_probs = probabilities * n
        
        # 分离小于1和大于等于1的概率
        small_indices = []
        large_indices = []
        
        for i in range(n):
            if scaled_probs[i] < 1.0:
                small_indices.append(i)
            else:
                large_indices.append(i)
        
        # 构建别名表
        while small_indices and large_indices:
            small_idx = small_indices.pop()
            large_idx = large_indices.pop()
            
            prob_table[small_idx] = scaled_probs[small_idx]
            alias_table[small_idx] = large_idx
            
            # 更新大概率值
            scaled_probs[large_idx] -= (1.0 - scaled_probs[small_idx])
            
            if scaled_probs[large_idx] < 1.0:
                small_indices.append(large_idx)
            else:
                large_indices.append(large_idx)
        
        # 处理剩余的项
        while large_indices:
            large_idx = large_indices.pop()
            prob_table[large_idx] = 1.0
            
        while small_indices:
            small_idx = small_indices.pop()
            prob_table[small_idx] = 1.0
        
        return alias_table, prob_table
    
    def _generate_subvoxel_offsets(self, num_points: int, method: str = 'mixed') -> np.ndarray:
        """
        生成体素内的连续偏移，增加自然性
        
        Args:
            num_points (int): 点数
            method (str): 偏移生成方法
                - 'uniform': 均匀分布
                - 'gaussian': 高斯分布（中心偏向）
                - 'mixed': 混合分布（推荐）
                
        Returns:
            np.ndarray: 偏移量，shape为(num_points, 3)
        """
        if method == 'uniform':
            # 简单的均匀分布
            return np.random.random((num_points, 3))
        
        elif method == 'gaussian':
            # 高斯分布，中心化在体素中心
            offsets = np.random.normal(0.5, 0.15, (num_points, 3))
            return np.clip(offsets, 0.0, 1.0)
        
        elif method == 'mixed':
            # 混合分布：70%均匀分布 + 30%中心偏向高斯分布
            uniform_count = int(num_points * 0.7)
            gaussian_count = num_points - uniform_count
            
            uniform_offsets = np.random.random((uniform_count, 3))
            gaussian_offsets = np.clip(
                np.random.normal(0.5, 0.2, (gaussian_count, 3)), 
                0.0, 1.0
            )
            
            # 合并并随机打乱
            all_offsets = np.vstack([uniform_offsets, gaussian_offsets])
            np.random.shuffle(all_offsets)
            return all_offsets
        
        else:
            # 默认使用均匀分布
            return np.random.random((num_points, 3))
    
    def _traditional_sampling(self, voxel_grid: np.ndarray, threshold: float, 
                             num_points: Optional[int], method: str) -> np.ndarray:
        """
        传统的体素采样方法（保持向后兼容）
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            threshold (float): 体素值阈值
            num_points (Optional[int]): 目标点数
            method (str): 采样方法
            
        Returns:
            np.ndarray: 采样得到的点云
        """
        # 找到超过阈值的体素
        occupied_voxels = np.where(voxel_grid > threshold)
        
        if len(occupied_voxels[0]) == 0:
            logger.warning("没有体素超过阈值，返回空点云")
            return np.empty((0, 3))
        
        # 获取体素坐标和对应的值
        voxel_coords = np.stack(occupied_voxels, axis=1).astype(np.float32)
        voxel_values = voxel_grid[occupied_voxels]
        
        # 根据采样方法和目标点数进行采样
        if num_points is not None and num_points < len(voxel_coords):
            if method == 'weighted':
                # 基于体素值进行概率采样
                weights = voxel_values / np.sum(voxel_values)
                indices = np.random.choice(len(voxel_coords), num_points, replace=False, p=weights)
            else:
                # 随机采样
                indices = np.random.choice(len(voxel_coords), num_points, replace=False)
            voxel_coords = voxel_coords[indices]
            voxel_values = voxel_values[indices]
        elif num_points is not None and num_points > len(voxel_coords):
            # 需要重复采样
            logger.warning(f"目标点数({num_points})大于可用体素数({len(voxel_coords)})，将进行重复采样")
            if method == 'weighted':
                weights = voxel_values / np.sum(voxel_values)
                indices = np.random.choice(len(voxel_coords), num_points, replace=True, p=weights)
            else:
                indices = np.random.choice(len(voxel_coords), num_points, replace=True)
            voxel_coords = voxel_coords[indices]
            voxel_values = voxel_values[indices]
        
        # 根据采样方法生成最终点坐标
        if method == 'center':
            # 体素中心采样（添加0.5偏移使其位于体素中心）
            final_coords = voxel_coords + 0.5
        elif method == 'random':
            # 体素内随机采样
            random_offset = np.random.random((len(voxel_coords), 3))
            final_coords = voxel_coords + random_offset
        else:  # weighted or any other method defaults to center
            final_coords = voxel_coords + 0.5
        
        # 反归一化到原始坐标空间
        point_cloud = self._denormalize_coordinates(final_coords)
        
        logger.info(f"从体素网格采样得到 {len(point_cloud)} 个点 (方法: {method})")
        return point_cloud
    
    def _denormalize_coordinates(self, normalized_coords: np.ndarray) -> np.ndarray:
        """
        将归一化坐标转换回原始坐标空间
        
        Args:
            normalized_coords (np.ndarray): 归一化坐标
            
        Returns:
            np.ndarray: 原始坐标空间的点云
        """
        if hasattr(self, '_last_min_bounds') and hasattr(self, '_last_max_bounds'):
            point_cloud = normalized_coords / (self.voxel_size - 1)
            point_cloud = point_cloud * (self._last_max_bounds - self._last_min_bounds)
            point_cloud += self._last_min_bounds
        else:
            logger.warning("没有保存的边界信息，返回归一化坐标")
            point_cloud = normalized_coords / self.voxel_size
        
        return point_cloud
    
    def upsample_voxel_grid(self, voxel_grid: np.ndarray, scale_factor: float = 2.0, 
                           method: str = 'linear') -> np.ndarray:
        """
        对体素网格进行上采样以提升分辨率
        
        Args:
            voxel_grid (np.ndarray): 输入体素网格
            scale_factor (float): 放大倍数，默认2.0
            method (str): 插值方法
                - 'linear': 线性插值（默认）
                - 'nearest': 最近邻插值
                - 'cubic': 三次插值（需要scipy > 1.6.0）
        
        Returns:
            np.ndarray: 上采样后的体素网格
        """
        if scale_factor <= 1.0:
            logger.warning(f"缩放因子 {scale_factor} <= 1.0，返回原网格")
            return voxel_grid.copy()
        
        logger.info(f"开始上采样体素网格，缩放因子: {scale_factor}, 方法: {method}")
        logger.info(f"原始网格shape: {voxel_grid.shape}")
        
        # 根据方法选择插值阶数
        if method == 'nearest':
            order = 0
        elif method == 'linear':
            order = 1
        elif method == 'cubic':
            order = 3
        else:
            logger.warning(f"未知的插值方法 {method}，使用线性插值")
            order = 1
        
        try:
            # 使用scipy.ndimage.zoom进行上采样
            upsampled_grid = zoom(voxel_grid, scale_factor, order=order, mode='nearest')
            
            logger.info(f"上采样完成，新网格shape: {upsampled_grid.shape}")
            logger.info(f"数据范围: [{np.min(upsampled_grid):.3f}, {np.max(upsampled_grid):.3f}]")
            
            return upsampled_grid
            
        except Exception as e:
            logger.error(f"上采样过程中发生错误: {e}")
            raise
    
    def save_point_cloud(self, point_cloud: np.ndarray, output_path: str) -> None:
        """
        保存点云数据到CSV文件
        
        Args:
            point_cloud (np.ndarray): 点云数据，shape为(N, 3)
            output_path (str): 输出文件路径（CSV格式）
        """
        # 确保输出路径是CSV格式
        if not output_path.lower().endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
        
        # 使用指定的表头格式保存CSV文件
        np.savetxt(output_path, point_cloud, fmt='%.6f', delimiter=',', 
                  header='x [nm],y [nm],z [nm]', comments='')
        
        logger.info(f"点云已保存为CSV文件: {output_path}")
        logger.info(f"保存的点云包含 {len(point_cloud)} 个点")
    
    def get_conversion_info(self) -> dict:
        """
        获取最近一次转换的详细信息
        
        Returns:
            dict: 包含转换参数的字典
        """
        info = {
            'voxel_size': self.voxel_size,
            'method': self.method,
            'padding_ratio': self.padding_ratio,
            'volume_dims': self.volume_dims.tolist(),
            'padding': self.padding.tolist(),
            'fixed_min_bounds': self.fixed_min_bounds.tolist(),
            'fixed_max_bounds': self.fixed_max_bounds.tolist()
        }
        
        if hasattr(self, '_last_min_bounds'):
            info['last_min_bounds'] = self._last_min_bounds.tolist()
            info['last_max_bounds'] = self._last_max_bounds.tolist()
        
        return info
