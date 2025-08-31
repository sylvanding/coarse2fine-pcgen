#!/usr/bin/env python3
"""
快速功能验证脚本
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

from tiff_to_pointcloud import TiffToPointCloud

def main():
    """快速测试功能"""
    print("快速功能验证测试")
    print("=" * 30)
    
    # 1. 创建简单的测试数据
    print("1. 创建测试体素数据...")
    test_shape = (32, 32, 16)
    voxel_data = np.zeros(test_shape, dtype=np.float32)
    
    # 在中心创建一个小球体
    center = np.array(test_shape) // 2
    for i in range(test_shape[0]):
        for j in range(test_shape[1]):
            for k in range(test_shape[2]):
                distance = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if distance <= 8:
                    voxel_data[i, j, k] = 1.0 - (distance / 8)
    
    print(f"测试数据shape: {voxel_data.shape}")
    print(f"非零体素数: {np.sum(voxel_data > 0)}")
    
    # 2. 测试转换器
    print("\n2. 测试TiffToPointCloud类...")
    converter = TiffToPointCloud()
    
    # 3. 测试采样
    print("\n3. 测试点云采样...")
    try:
        point_cloud = converter.sample_to_pointcloud(
            voxel_data,
            threshold=0.1,
            num_points=1000,
            method='probabilistic'
        )
        
        print(f"采样成功!")
        print(f"生成点数: {len(point_cloud)}")
        print(f"点云范围:")
        if len(point_cloud) > 0:
            print(f"  X: [{np.min(point_cloud[:, 0]):.1f}, {np.max(point_cloud[:, 0]):.1f}]")
            print(f"  Y: [{np.min(point_cloud[:, 1]):.1f}, {np.max(point_cloud[:, 1]):.1f}]")
            print(f"  Z: [{np.min(point_cloud[:, 2]):.1f}, {np.max(point_cloud[:, 2]):.1f}]")
        
        # 4. 测试保存
        print("\n4. 测试CSV保存...")
        test_dir = script_dir / "test_data"
        test_dir.mkdir(exist_ok=True)
        
        output_file = test_dir / "test_points.csv"
        converter.save_pointcloud(point_cloud, str(output_file))
        
        if output_file.exists():
            print(f"CSV保存成功: {output_file}")
            # 验证文件内容
            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"文件包含 {len(lines)} 行 (包含表头)")
        
        print("\n✅ 所有功能测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
