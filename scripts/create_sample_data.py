#!/usr/bin/env python3
"""
示例数据生成脚本

生成用于测试的H5格式点云数据文件。
"""

import argparse
import numpy as np
import h5py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sphere_points(center, radius, num_points):
    """
    生成球形点云
    
    Args:
        center (tuple): 球心坐标 (x, y, z)
        radius (float): 球半径
        num_points (int): 点数
    
    Returns:
        np.ndarray: 球形点云
    """
    # 生成球面上的随机点
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    
    # 为了生成实心球，添加径向随机性
    r = radius * np.cbrt(np.random.uniform(0, 1, num_points))
    
    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)
    
    return np.column_stack([x, y, z]).astype(np.float32)


def generate_cube_points(center, size, num_points):
    """
    生成立方体点云
    
    Args:
        center (tuple): 立方体中心
        size (float): 立方体边长
        num_points (int): 点数
    
    Returns:
        np.ndarray: 立方体点云
    """
    half_size = size / 2
    
    x = np.random.uniform(center[0] - half_size, center[0] + half_size, num_points)
    y = np.random.uniform(center[1] - half_size, center[1] + half_size, num_points)
    z = np.random.uniform(center[2] - half_size, center[2] + half_size, num_points)
    
    return np.column_stack([x, y, z]).astype(np.float32)


def generate_bunny_like_shape(num_points):
    """
    生成类似兔子的复杂形状点云
    
    Args:
        num_points (int): 点数
    
    Returns:
        np.ndarray: 复杂形状点云
    """
    points = []
    
    # 身体（椭球）
    body_points = int(num_points * 0.5)
    phi = np.random.uniform(0, 2 * np.pi, body_points)
    theta = np.random.uniform(0, np.pi, body_points)
    r = np.cbrt(np.random.uniform(0, 1, body_points))
    
    # 椭球参数
    a, b, c = 2.0, 1.5, 1.0
    x = a * r * np.sin(theta) * np.cos(phi)
    y = b * r * np.sin(theta) * np.sin(phi)
    z = c * r * np.cos(theta)
    body = np.column_stack([x, y, z])
    points.append(body)
    
    # 头部（球）
    head_points = int(num_points * 0.3)
    head = generate_sphere_points((0, 0, 1.5), 0.8, head_points)
    points.append(head)
    
    # 耳朵（两个小椭球）
    ear_points = int(num_points * 0.1)
    ear1 = generate_sphere_points((-0.4, 0.3, 2.2), 0.3, ear_points // 2)
    ear2 = generate_sphere_points((0.4, 0.3, 2.2), 0.3, ear_points // 2)
    points.append(ear1)
    points.append(ear2)
    
    # 尾巴（小球）
    tail_points = int(num_points * 0.1)
    tail = generate_sphere_points((0, -2.2, 0.2), 0.4, tail_points)
    points.append(tail)
    
    all_points = np.vstack(points)
    
    # 随机选择指定数量的点
    if len(all_points) > num_points:
        indices = np.random.choice(len(all_points), num_points, replace=False)
        all_points = all_points[indices]
    
    return all_points.astype(np.float32)


def create_sample_dataset(output_path, num_samples=10, points_per_sample=5000):
    """
    创建示例数据集
    
    Args:
        output_path (str): 输出文件路径
        num_samples (int): 样本数量
        points_per_sample (int): 每个样本的点数
    """
    logger.info(f"开始生成示例数据集: {num_samples} 个样本，每个 {points_per_sample} 个点")
    
    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据存储
    all_samples = []
    
    for i in range(num_samples):
        logger.info(f"生成第 {i+1}/{num_samples} 个样本")
        
        # 随机选择形状类型
        shape_type = np.random.choice(['sphere', 'cube', 'bunny'])
        
        if shape_type == 'sphere':
            # 随机参数的球
            center = np.random.uniform(-5, 5, 3)
            radius = np.random.uniform(1, 3)
            sample = generate_sphere_points(center, radius, points_per_sample)
            
        elif shape_type == 'cube':
            # 随机参数的立方体
            center = np.random.uniform(-5, 5, 3)
            size = np.random.uniform(2, 6)
            sample = generate_cube_points(center, size, points_per_sample)
            
        else:  # bunny
            # 复杂形状
            sample = generate_bunny_like_shape(points_per_sample)
            # 添加随机变换
            scale = np.random.uniform(0.5, 2.0)
            rotation = np.random.uniform(0, 2*np.pi)
            translation = np.random.uniform(-3, 3, 3)
            
            # 简单旋转（绕z轴）
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rotation_matrix = np.array([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ])
            
            sample = sample * scale
            sample = sample @ rotation_matrix.T
            sample += translation
        
        # 添加轻微噪声
        noise = np.random.normal(0, 0.05, sample.shape)
        sample += noise
        
        all_samples.append(sample)
    
    # 转换为numpy数组
    dataset = np.stack(all_samples, axis=0)
    
    logger.info(f"数据集shape: {dataset.shape}")
    logger.info(f"数据范围: [{np.min(dataset):.3f}, {np.max(dataset):.3f}]")
    
    # 保存到H5文件
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=dataset, compression='gzip')
        
        # 添加元数据
        f.attrs['num_samples'] = num_samples
        f.attrs['points_per_sample'] = points_per_sample
        f.attrs['description'] = 'Synthetic point cloud dataset for testing'
        f.attrs['coordinate_system'] = 'xyz'
    
    logger.info(f"示例数据集已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成示例点云数据')
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/sample_pointclouds.h5',
        help='输出文件路径'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='样本数量'
    )
    
    parser.add_argument(
        '--points-per-sample',
        type=int,
        default=5000,
        help='每个样本的点数'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    try:
        create_sample_dataset(
            args.output,
            args.num_samples,
            args.points_per_sample
        )
        print(f"✅ 成功生成示例数据: {args.output}")
        
    except Exception as e:
        logger.error(f"生成数据时发生错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
