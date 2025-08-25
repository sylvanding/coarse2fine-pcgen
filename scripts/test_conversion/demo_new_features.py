#!/usr/bin/env python3
"""
新功能演示脚本

演示体素采样回点云和体素上采样功能的使用
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.voxel.converter import PointCloudToVoxel


def create_demo_point_cloud(num_points=1000):
    """创建一个简单的演示点云数据"""
    # 创建一个球形点云
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)
    
    theta = np.arccos(costheta)
    r = 1000 * np.cbrt(u)  # 半径1000nm
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack([x, y, z])


def demo_new_features():
    """演示新功能"""
    print("🚀 开始演示体素转换新功能")
    print("="*60)
    
    # 1. 创建演示数据
    print("1. 创建演示点云数据...")
    original_points = create_demo_point_cloud(5000)
    print(f"   原始点云: {len(original_points)} 个点")
    
    # 2. 创建转换器
    print("\n2. 创建体素转换器...")
    converter = PointCloudToVoxel(
        voxel_size=64,
        method='gaussian',
        volume_dims=[2500, 2500, 2500],
        padding=[100, 100, 100]
    )
    
    # 3. 点云转体素
    print("\n3. 点云转换为体素网格...")
    voxel_grid = converter.convert(original_points, sigma=1.5)
    print(f"   体素网格shape: {voxel_grid.shape}")
    print(f"   占有体素数: {np.sum(voxel_grid > 0.1)}")
    
    # 4. 体素上采样
    print("\n4. 体素网格上采样...")
    upsampled_grid = converter.upsample_voxel_grid(voxel_grid, scale_factor=2.0, method='linear')
    print(f"   上采样后shape: {upsampled_grid.shape}")
    print(f"   上采样占有体素数: {np.sum(upsampled_grid > 0.1)}")
    
    # 5. 体素采样回点云 - 不同方法演示
    print("\n5. 体素采样回点云 (不同方法对比)...")
    
    # 5.1 中心采样
    sampled_center = converter.voxel_to_points(
        upsampled_grid, threshold=0.1, num_points=3000, method='center'
    )
    print(f"   中心采样: {len(sampled_center)} 个点")
    
    # 5.2 随机采样
    sampled_random = converter.voxel_to_points(
        upsampled_grid, threshold=0.1, num_points=3000, method='random'
    )
    print(f"   随机采样: {len(sampled_random)} 个点")
    
    # 5.3 加权采样
    sampled_weighted = converter.voxel_to_points(
        upsampled_grid, threshold=0.1, num_points=3000, method='weighted'
    )
    print(f"   加权采样: {len(sampled_weighted)} 个点")
    
    # 6. 保存结果
    print("\n6. 保存演示结果...")
    os.makedirs('output/demo', exist_ok=True)
    
    # 保存体素网格
    converter.save_as_tiff(voxel_grid, 'output/demo/original_voxel.tiff')
    converter.save_as_tiff(upsampled_grid, 'output/demo/upsampled_voxel.tiff')
    
    # 保存点云为CSV格式
    converter.save_point_cloud(original_points, 'output/demo/original_points.csv')
    converter.save_point_cloud(sampled_center, 'output/demo/sampled_center.csv')
    converter.save_point_cloud(sampled_random, 'output/demo/sampled_random.csv')
    converter.save_point_cloud(sampled_weighted, 'output/demo/sampled_weighted.csv')
    
    print("   ✅ 结果已保存到 output/demo/ 目录")
    
    # 7. 统计对比
    print("\n7. 统计对比:")
    print(f"   原始点云范围: {np.ptp(original_points, axis=0)}")
    print(f"   中心采样范围: {np.ptp(sampled_center, axis=0)}")
    print(f"   随机采样范围: {np.ptp(sampled_random, axis=0)}")
    print(f"   加权采样范围: {np.ptp(sampled_weighted, axis=0)}")
    
    print("\n🎉 演示完成!")
    print("="*60)


if __name__ == '__main__':
    demo_new_features()
