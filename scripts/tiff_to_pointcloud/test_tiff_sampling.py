#!/usr/bin/env python3
"""
TIFF到点云采样功能测试脚本

创建测试用的3D TIFF文件并验证采样功能
"""

import numpy as np
import tifffile
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

def create_test_tiff(output_path: str, shape: tuple = (64, 64, 32)):
    """
    创建测试用的3D TIFF文件
    
    Args:
        output_path (str): 输出文件路径
        shape (tuple): 3D网格尺寸
    """
    print(f"创建测试TIFF文件: {output_path}")
    print(f"网格尺寸: {shape}")
    
    # 创建一个包含几何形状的测试体素网格
    voxel_grid = np.zeros(shape, dtype=np.float32)
    
    # 添加一个球体
    center = np.array(shape) // 2
    radius = min(shape) // 4
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                distance = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if distance <= radius:
                    # 创建渐变密度，中心密度高，边缘密度低
                    density = 1.0 - (distance / radius)
                    voxel_grid[i, j, k] = density
    
    # 添加一些随机噪声
    noise = np.random.random(shape) * 0.1
    voxel_grid = voxel_grid + noise
    voxel_grid = np.clip(voxel_grid, 0, 1)
    
    # 转换为uint8格式保存
    tiff_data = (voxel_grid * 255).astype(np.uint8)
    
    # 保存TIFF文件
    tifffile.imwrite(output_path, tiff_data)
    
    print(f"测试TIFF文件已创建")
    print(f"数据范围: [0, 255]")
    print(f"非零体素数: {np.sum(tiff_data > 0):,}")
    print(f"占有率: {np.sum(tiff_data > 0) / tiff_data.size:.4f}")
    
    return output_path

def run_sampling_test(tiff_path: str, output_dir: str):
    """
    运行采样测试
    
    Args:
        tiff_path (str): 输入TIFF文件路径
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的采样方法
    methods = ['probabilistic', 'center', 'random', 'weighted']
    num_points = 10000
    
    print(f"\n开始采样测试...")
    print(f"目标点数: {num_points}")
    
    for method in methods:
        print(f"\n测试采样方法: {method}")
        
        # 构建输出文件路径
        output_csv = os.path.join(output_dir, f"test_{method}.csv")
        
        # 构建命令
        cmd = [
            sys.executable,
            str(script_dir / "tiff_to_pointcloud.py"),
            "--input", tiff_path,
            "--output", output_csv,
            "--method", method,
            "--num-points", str(num_points),
            "--threshold", "0.1",
            "--verbose"
        ]
        
        # 执行命令
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {method} 采样成功")
            print(f"输出文件: {output_csv}")
            
            # 检查输出文件
            if os.path.exists(output_csv):
                # 读取点云数据验证（不使用pandas）
                with open(output_csv, 'r') as f:
                    lines = f.readlines()
                    # 减去表头行
                    point_count = len(lines) - 1 if lines and lines[0].startswith('x') else len(lines)
                print(f"实际采样点数: {point_count}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {method} 采样失败:")
            print(f"错误信息: {e.stderr}")

def main():
    """主函数"""
    print("TIFF到点云采样功能测试")
    print("=" * 50)
    
    # 创建测试目录
    test_dir = script_dir / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试TIFF文件
    tiff_path = test_dir / "test_sphere.tiff"
    create_test_tiff(str(tiff_path))
    
    # 运行采样测试
    output_dir = test_dir / "results"
    run_sampling_test(str(tiff_path), str(output_dir))
    
    print(f"\n测试完成!")
    print(f"测试文件位于: {test_dir}")
    print(f"结果文件位于: {output_dir}")

if __name__ == "__main__":
    main()
