"""
点云数据处理脚本

从CSV文件中随机提取点云区域，进行数据增强（旋转、裁剪），
并保存为HDF5格式用于深度学习训练。

功能：
    - 递归读取目录下所有CSV点云文件
    - 随机旋转点云（绕Z轴）
    - 随机提取指定大小的点云区域
    - 过滤点数不足的区域
    - 坐标归一化和精度控制
    - 批量生成样本并保存为H5文件

Usage:
    python scripts/exp-data-process/process_pointcloud_data.py

Author: AI Assistant
Date: 2025-10-05
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_csv_files(root_dir: str) -> List[Path]:
    """
    递归查找目录下所有必要列存在的CSV文件
    
    Args:
        root_dir (str): 根目录路径
        
    Returns:
        List[Path]: 必需列存在的CSV文件路径列表
        
    Raises:
        ValueError: 当目录不存在或没有找到必需列存在的CSV文件时
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise ValueError(f"目录不存在: {root_dir}")
    
    if not root_path.is_dir():
        raise ValueError(f"路径不是目录: {root_dir}")
    
    # 递归查找所有CSV文件
    csv_files = list(root_path.rglob("*.csv"))
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {root_dir} 及其子目录中未找到必需列存在的CSV文件")
    
    # 只保留必需的列存在的csv文件
    required_columns = ["x [nm]", "y [nm]", "z [nm]"]
    csv_files = [file for file in csv_files if all(col in pd.read_csv(file).columns for col in required_columns)]
    if len(csv_files) == 0:
        raise ValueError(f"在 {root_dir} 及其子目录中未找到必需的列: {required_columns}")
    
    logger.info(f"找到 {len(csv_files)} 个符合要求的CSV文件(必需的列存在: {required_columns})")
    return csv_files


def load_csv_pointcloud(csv_path: Path) -> Optional[np.ndarray]:
    """
    从CSV文件加载点云数据
    
    CSV文件格式要求：
        - 表头必须包含 "x [nm]", "y [nm]", "z [nm]"
        - 坐标单位为纳米(nm)
    
    Args:
        csv_path (Path): CSV文件路径
        
    Returns:
        Optional[np.ndarray]: 形状为(N, 3)的点云数组，如果加载失败返回None
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查必需的列是否存在
        required_columns = ["x [nm]", "y [nm]", "z [nm]"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(
                f"文件 {csv_path.name} 缺少必需的列: {missing_columns}，跳过该文件"
            )
            return None
        
        # 提取xyz坐标
        points = df[required_columns].values.astype(np.float64)
        
        # 检查是否有有效数据
        if len(points) == 0:
            logger.warning(f"文件 {csv_path.name} 为空，跳过该文件")
            return None
        
        # 检查是否有NaN值
        if np.any(np.isnan(points)):
            logger.warning(f"文件 {csv_path.name} 包含NaN值，移除这些点")
            points = points[~np.isnan(points).any(axis=1)]
            
            if len(points) == 0:
                logger.warning(f"文件 {csv_path.name} 移除NaN后无有效数据，跳过该文件")
                return None
        
        logger.debug(f"成功加载 {csv_path.name}，包含 {len(points)} 个点")
        return points
        
    except Exception as e:
        logger.error(f"加载文件 {csv_path.name} 时出错: {e}")
        return None


def rotate_pointcloud_z(points: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    绕Z轴旋转点云
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        angle (Optional[float]): 旋转角度（弧度），如果为None则随机生成
        
    Returns:
        np.ndarray: 旋转后的点云
    """
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    
    # 构建绕Z轴的旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    
    # 应用旋转
    rotated_points = points @ rotation_matrix.T
    
    return rotated_points


def extract_random_region(
    points: np.ndarray,
    region_size: Tuple[float, float, float],
    min_points: int = 100,
    max_attempts: int = 10
) -> Optional[np.ndarray]:
    """
    从点云中随机提取一个指定大小的区域
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        region_size (Tuple[float, float, float]): 区域大小(x_size, y_size, z_size)，单位nm
        min_points (int): 区域内最少点数要求
        max_attempts (int): 最大尝试次数
        
    Returns:
        Optional[np.ndarray]: 提取的点云区域，如果失败返回None
    """
    if len(points) < min_points:
        logger.debug(f"点云总数 {len(points)} 少于最小要求 {min_points}")
        return None
    
    # 计算点云边界
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    x_size, y_size, z_size = region_size
    
    # 检查点云是否足够大
    point_cloud_size = max_coords - min_coords
    if (point_cloud_size[0] < x_size or 
        point_cloud_size[1] < y_size or 
        point_cloud_size[2] < z_size):
        logger.debug(
            f"点云尺寸 {point_cloud_size} 小于所需区域 {region_size}"
        )
        return None
    
    # 尝试随机提取区域
    for attempt in range(max_attempts):
        # 随机选择区域的起始点
        x_start = np.random.uniform(min_coords[0], max_coords[0] - x_size)
        y_start = np.random.uniform(min_coords[1], max_coords[1] - y_size)
        z_start = np.random.uniform(min_coords[2], max_coords[2] - z_size)
        
        # 定义区域边界
        x_end = x_start + x_size
        y_end = y_start + y_size
        z_end = z_start + z_size
        
        # 筛选在区域内的点
        mask = (
            (points[:, 0] >= x_start) & (points[:, 0] <= x_end) &
            (points[:, 1] >= y_start) & (points[:, 1] <= y_end) &
            (points[:, 2] >= z_start) & (points[:, 2] <= z_end)
        )
        
        region_points = points[mask]
        
        # 检查点数是否满足要求
        if len(region_points) >= min_points:
            logger.debug(
                f"成功提取区域，包含 {len(region_points)} 个点 "
                f"(尝试 {attempt + 1}/{max_attempts})"
            )
            return region_points
    
    logger.debug(f"经过 {max_attempts} 次尝试后仍未找到满足条件的区域")
    return None


def translate_to_origin(points: np.ndarray) -> np.ndarray:
    """
    将点云的最小角点平移到原点(0, 0, 0)
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        
    Returns:
        np.ndarray: 平移后的点云
    """
    min_coords = np.min(points, axis=0)
    translated_points = points - min_coords
    return translated_points


def round_coordinates(points: np.ndarray, decimals: int = 2) -> np.ndarray:
    """
    将坐标值精确到指定小数位数
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        decimals (int): 保留的小数位数
        
    Returns:
        np.ndarray: 精确后的点云
    """
    return np.round(points, decimals=decimals)


def adjust_point_count(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    调整点云的点数到目标数量
    
    如果点数少于目标数量，随机复制一些点；
    如果点数多于目标数量，随机删除一些点。
    
    Args:
        points (np.ndarray): 输入点云，形状为(N, 3)
        target_count (int): 目标点数
        
    Returns:
        np.ndarray: 调整后的点云，形状为(target_count, 3)
    """
    current_count = len(points)
    
    if current_count == target_count:
        return points.copy()
    elif current_count < target_count:
        # 点数不足，需要复制一些点
        # 计算需要复制的点数
        points_to_add = target_count - current_count
        
        # 随机选择要复制的点的索引
        indices_to_copy = np.random.choice(current_count, size=points_to_add, replace=True)
        
        # 复制选中的点
        additional_points = points[indices_to_copy]
        
        # 合并原始点和复制的点
        adjusted_points = np.vstack([points, additional_points])
        
        return adjusted_points
    else:
        # 点数过多，需要删除一些点
        # 随机选择要保留的点的索引
        indices_to_keep = np.random.choice(current_count, size=target_count, replace=False)
        
        # 返回选中的点
        return points[indices_to_keep]


def generate_samples(
    csv_files: List[Path],
    num_samples: int,
    region_size: Tuple[float, float, float],
    target_points: int,
    min_points: int = 100,
    max_attempts: int = 10,
    decimals: int = 2,
    random_seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    从CSV文件列表中生成指定数量的点云样本
    
    处理流程：
        1. 随机选择CSV文件
        2. 加载点云
        3. 随机旋转（绕Z轴）
        4. 随机提取区域
        5. 平移到原点
        6. 精确坐标
        7. 调整点数到目标数量
    
    Args:
        csv_files (List[Path]): CSV文件路径列表
        num_samples (int): 需要生成的样本数量
        region_size (Tuple[float, float, float]): 提取区域大小(x, y, z)，单位nm
        target_points (int): 期望的点数，必须大于等于min_points
        min_points (int): 每个样本的最少点数
        max_attempts (int): 每个CSV文件的最大尝试次数
        decimals (int): 坐标保留的小数位数
        random_seed (Optional[int]): 随机种子，用于可重复性
        
    Returns:
        List[np.ndarray]: 生成的点云样本列表，每个样本都有target_points个点
        
    Raises:
        ValueError: 当target_points小于min_points时
        RuntimeError: 当无法生成足够的样本时
    """
    # 参数验证
    if target_points < min_points:
        raise ValueError(f"target_points ({target_points}) 必须大于等于 min_points ({min_points})")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    samples = []
    failed_attempts = 0
    max_failed_attempts = num_samples * 100  # 设置一个上限避免无限循环
    
    logger.info(f"开始生成 {num_samples} 个样本...")
    
    with tqdm(total=num_samples, desc="生成样本") as pbar:
        while len(samples) < num_samples:
            # 检查是否失败次数过多
            if failed_attempts >= max_failed_attempts:
                raise RuntimeError(
                    f"无法生成足够的样本。已生成 {len(samples)}/{num_samples} 个样本，"
                    f"失败次数: {failed_attempts}"
                )
            
            # 随机选择一个CSV文件
            csv_file = np.random.choice(csv_files)
            
            # 加载点云
            points = load_csv_pointcloud(csv_file)
            if points is None:
                failed_attempts += 1
                continue
            
            # 随机旋转
            rotated_points = rotate_pointcloud_z(points)
            
            # 提取区域
            region = extract_random_region(
                rotated_points,
                region_size,
                min_points,
                max_attempts
            )
            
            if region is None:
                failed_attempts += 1
                logger.info(f"由于选取的区域不符合要求(区域点数少于最小要求/区域尺寸小于指定大小)，跳过文件: {csv_file}")
                continue
            
            # 平移到原点
            translated_region = translate_to_origin(region)
            
            # 精确坐标
            final_points = round_coordinates(translated_region, decimals)
            
            # 调整点数到目标数量
            adjusted_points = adjust_point_count(final_points, target_points)
            
            # 添加到样本列表
            samples.append(adjusted_points)
            pbar.update(1)
    
    logger.info(
        f"成功生成 {len(samples)} 个样本，失败尝试次数: {failed_attempts}"
    )
    
    return samples


def generate_samples_with_source(
    csv_files: List[Path],
    num_samples: int,
    region_size: Tuple[float, float, float],
    target_points: int,
    min_points: int = 100,
    max_attempts: int = 10,
    decimals: int = 2,
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[Path]]:
    """
    从CSV文件列表中生成指定数量的点云样本，并记录每个样本的来源文件
    
    处理流程：
        1. 随机选择CSV文件
        2. 加载点云
        3. 随机旋转（绕Z轴）
        4. 随机提取区域
        5. 平移到原点
        6. 精确坐标
        7. 调整点数到目标数量
    
    Args:
        csv_files (List[Path]): CSV文件路径列表
        num_samples (int): 需要生成的样本数量
        region_size (Tuple[float, float, float]): 提取区域大小(x, y, z)，单位nm
        target_points (int): 期望的点数，必须大于等于min_points
        min_points (int): 每个样本的最少点数
        max_attempts (int): 每个CSV文件的最大尝试次数
        decimals (int): 坐标保留的小数位数
        random_seed (Optional[int]): 随机种子，用于可重复性
        
    Returns:
        Tuple[List[np.ndarray], List[Path]]: 
            - 生成的点云样本列表，每个样本都有target_points个点
            - 每个样本对应的源CSV文件路径列表
        
    Raises:
        ValueError: 当target_points小于min_points时
        RuntimeError: 当无法生成足够的样本时
    """
    # 参数验证
    if target_points < min_points:
        raise ValueError(f"target_points ({target_points}) 必须大于等于 min_points ({min_points})")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    samples = []
    source_files = []
    failed_attempts = 0
    max_failed_attempts = num_samples * 100  # 设置一个上限避免无限循环
    
    logger.info(f"开始生成 {num_samples} 个样本...")
    
    with tqdm(total=num_samples, desc="生成样本") as pbar:
        while len(samples) < num_samples:
            # 检查是否失败次数过多
            if failed_attempts >= max_failed_attempts:
                raise RuntimeError(
                    f"无法生成足够的样本。已生成 {len(samples)}/{num_samples} 个样本，"
                    f"失败次数: {failed_attempts}"
                )
            
            # 随机选择一个CSV文件
            csv_file = np.random.choice(csv_files)
            
            # 加载点云
            points = load_csv_pointcloud(csv_file)
            if points is None:
                failed_attempts += 1
                continue
            
            # 随机旋转
            rotated_points = rotate_pointcloud_z(points)
            
            # 提取区域
            region = extract_random_region(
                rotated_points,
                region_size,
                min_points,
                max_attempts
            )
            
            if region is None:
                failed_attempts += 1
                logger.info(f"由于选取的区域不符合要求(区域点数少于最小要求/区域尺寸小于指定大小)，跳过文件: {csv_file}")
                continue
            
            # 平移到原点
            translated_region = translate_to_origin(region)
            
            # 精确坐标
            final_points = round_coordinates(translated_region, decimals)
            
            # 调整点数到目标数量
            adjusted_points = adjust_point_count(final_points, target_points)
            
            # 添加到样本列表和来源文件列表
            samples.append(adjusted_points)
            source_files.append(csv_file)
            pbar.update(1)
    
    logger.info(
        f"成功生成 {len(samples)} 个样本，失败尝试次数: {failed_attempts}"
    )
    
    return samples, source_files


def save_samples_to_h5(
    samples: List[np.ndarray],
    output_path: str,
    dataset_name: str = "point_clouds",
    compression: str = "gzip"
) -> None:
    """
    将点云样本保存为HDF5文件
    
    HDF5文件结构：
        - dataset_name: 形状为(num_samples, points_per_sample, 3)的数组
        - 属性: num_samples, points_per_sample
    
    Args:
        samples (List[np.ndarray]): 点云样本列表，每个样本应有相同的点数
        output_path (str): 输出H5文件路径
        dataset_name (str): 数据集名称
        compression (str): 压缩方法
        
    Raises:
        ValueError: 当样本列表为空或样本点数不一致时
    """
    if len(samples) == 0:
        raise ValueError("样本列表为空，无法保存")
    
    # 验证所有样本的点数是否一致
    num_samples = len(samples)
    point_counts = [len(sample) for sample in samples]
    points_per_sample = point_counts[0]
    
    if not all(count == points_per_sample for count in point_counts):
        raise ValueError(f"所有样本的点数必须一致，但发现不同的点数: {set(point_counts)}")
    
    logger.info(f"样本统计:")
    logger.info(f"  - 样本数量: {num_samples}")
    logger.info(f"  - 每个样本点数: {points_per_sample}")
    
    # 创建数组，所有样本点数相同，无需填充
    samples_array = np.stack(samples, axis=0)
    
    # 保存到H5文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 创建数据集
        dataset = f.create_dataset(
            dataset_name,
            data=samples_array,
            compression=compression
        )
        
        # 添加元数据
        dataset.attrs['num_samples'] = num_samples
        dataset.attrs['points_per_sample'] = points_per_sample
        dataset.attrs['description'] = "Point cloud samples extracted from EXP CSV files"
    
    logger.info(f"样本已保存到: {output_path}")
    logger.info(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """主函数：演示完整的数据处理流程"""
    
    # =========================
    # 配置参数
    # =========================
    
    # 输入目录（包含CSV文件）
    INPUT_DIR = "/path/to/your/csv/files"
    
    # 输出H5文件路径
    OUTPUT_H5 = "outputs/processed_data/pointcloud_samples.h5"
    
    # 生成样本数量
    NUM_SAMPLES = 100
    
    # 提取区域大小（单位：nm）
    REGION_SIZE = (1000.0, 1000.0, 500.0)  # x, y, z
    
    # 期望点数
    TARGET_POINTS = 1024
    
    # 最小点数要求
    MIN_POINTS = 100
    
    # 每个CSV的最大尝试次数
    MAX_ATTEMPTS = 10
    
    # 坐标精度（小数位数）
    DECIMALS = 2
    
    # 随机种子（可选，用于可重复性）
    RANDOM_SEED = 42
    
    # =========================
    # 执行处理流程
    # =========================
    
    try:
        # 1. 查找所有CSV文件
        logger.info("=" * 60)
        logger.info("步骤 1: 查找CSV文件")
        logger.info("=" * 60)
        csv_files = find_all_csv_files(INPUT_DIR)
        
        # 2. 生成样本
        logger.info("\n" + "=" * 60)
        logger.info("步骤 2: 生成点云样本")
        logger.info("=" * 60)
        samples = generate_samples(
            csv_files=csv_files,
            num_samples=NUM_SAMPLES,
            region_size=REGION_SIZE,
            target_points=TARGET_POINTS,
            min_points=MIN_POINTS,
            max_attempts=MAX_ATTEMPTS,
            decimals=DECIMALS,
            random_seed=RANDOM_SEED
        )
        
        # 3. 保存到H5文件
        logger.info("\n" + "=" * 60)
        logger.info("步骤 3: 保存到H5文件")
        logger.info("=" * 60)
        save_samples_to_h5(samples, OUTPUT_H5)
        
        logger.info("\n" + "=" * 60)
        logger.info("处理完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
