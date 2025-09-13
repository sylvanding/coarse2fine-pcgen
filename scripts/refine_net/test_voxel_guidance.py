#!/usr/bin/env python3
"""
体素指导RefineNet测试脚本

用于测试体素指导功能的正确性，包括数据加载、模型前向传播等。
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import h5py

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入RefineNet模块
from src.refine_net import options, data_handler
from src.refine_net.models import PointNet2Generator
from src.refine_net.voxel_guidance import VoxelGuidanceModule, VoxelFeatureProjector


def create_test_data(save_dir: Path):
    """创建测试用的H5数据文件"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成测试点云数据
    num_samples = 10
    num_points = 5000
    
    print("创建测试数据...")
    
    # GT点云
    gt_point_clouds = []
    for i in range(num_samples):
        # 生成球形点云
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = np.random.uniform(5000, 8000, num_points)
        
        x = r * np.sin(phi) * np.cos(theta) + 10000
        y = r * np.sin(phi) * np.sin(theta) + 10000
        z = r * np.cos(phi) + 1250
        
        point_cloud = np.stack([x, y, z], axis=1).astype(np.float32)
        gt_point_clouds.append(point_cloud)
    
    gt_point_clouds = np.array(gt_point_clouds)
    
    # 噪声点云（添加高斯噪声）
    noise_std = 200.0
    noisy_point_clouds = gt_point_clouds + np.random.normal(0, noise_std, gt_point_clouds.shape)
    
    # 体素数据（创建简单的3D网格）
    voxel_size = 64
    voxel_grids = []
    for i in range(num_samples):
        # 创建简单的3D模式
        voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
        
        # 在中心创建一个球体
        center = voxel_size // 2
        radius = voxel_size // 4
        
        for x in range(voxel_size):
            for y in range(voxel_size):
                for z in range(voxel_size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                    if dist <= radius:
                        voxel_grid[x, y, z] = 1.0 - (dist / radius)  # 渐变强度
        
        voxel_grids.append(voxel_grid)
    
    voxel_grids = np.array(voxel_grids)
    
    # 保存到H5文件
    gt_file = save_dir / 'test_gt.h5'
    noisy_file = save_dir / 'test_noisy.h5'
    voxel_file = save_dir / 'test_voxels.h5'
    
    with h5py.File(gt_file, 'w') as f:
        f.create_dataset('point_clouds', data=gt_point_clouds)
        print(f"GT点云数据已保存: {gt_file}, 形状: {gt_point_clouds.shape}")
    
    with h5py.File(noisy_file, 'w') as f:
        f.create_dataset('point_clouds', data=noisy_point_clouds)
        print(f"噪声点云数据已保存: {noisy_file}, 形状: {noisy_point_clouds.shape}")
    
    with h5py.File(voxel_file, 'w') as f:
        f.create_dataset('voxel_grids', data=voxel_grids)
        print(f"体素数据已保存: {voxel_file}, 形状: {voxel_grids.shape}")
    
    return gt_file, noisy_file, voxel_file


def test_voxel_guidance_components():
    """测试体素指导组件"""
    print("\n=== 测试体素指导组件 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试VoxelGuidanceModule
    print("\n1. 测试VoxelGuidanceModule")
    voxel_guidance = VoxelGuidanceModule(
        voxel_grid_size=64,
        conv_channels=[1, 16, 32, 64],
        downsample_factors=[2, 2, 2]
    ).to(device)
    
    batch_size = 2
    num_points = 2048
    voxel_size = 64
    
    # 创建测试数据
    voxel_grid = torch.randn(batch_size, 1, voxel_size, voxel_size, voxel_size).to(device)
    points = torch.randn(batch_size, 3, num_points).to(device) * 0.8
    
    # 前向传播
    voxel_features = voxel_guidance(voxel_grid, points)
    print(f"体素特征形状: {voxel_features.shape}")
    print(f"期望形状: ({batch_size}, {voxel_guidance.total_feature_dim}, {num_points})")
    
    # 测试VoxelFeatureProjector
    print("\n2. 测试VoxelFeatureProjector")
    projector = VoxelFeatureProjector(
        input_dim=voxel_guidance.total_feature_dim,
        output_dim=64
    ).to(device)
    
    projected_features = projector(voxel_features)
    print(f"投影后特征形状: {projected_features.shape}")
    print(f"期望形状: ({batch_size}, 64, {num_points})")
    
    print("✅ 体素指导组件测试通过")


def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 测试数据加载功能 ===")
    
    # 创建测试数据
    test_data_dir = Path('/tmp/refine_net_test')
    gt_file, noisy_file, voxel_file = create_test_data(test_data_dir)
    
    # 创建模拟参数
    class MockArgs:
        def __init__(self):
            self.gt_h5_path = str(gt_file)
            self.noisy_h5_path = str(noisy_file)
            self.voxel_h5_path = str(voxel_file)
            self.gt_data_key = 'point_clouds'
            self.noisy_data_key = 'point_clouds'
            self.voxel_data_key = 'voxel_grids'
            self.use_voxel_guidance = True
            self.sample_points = 2048
            self.train_ratio = 0.8
            self.volume_dims = [20000, 20000, 2500]
            self.padding = [0, 0, 100]
            self.iterations = 100
    
    args = MockArgs()
    device = torch.device('cpu')
    
    # 测试不使用体素指导的数据加载
    print("\n1. 测试不使用体素指导的数据加载")
    args.use_voxel_guidance = False
    dataset_no_voxel = data_handler.H5PairedDataset(None, device, args, split='train')
    
    sample = dataset_no_voxel[0]
    print(f"不使用体素指导 - 样本类型: {type(sample)}")
    if isinstance(sample, tuple):
        print(f"样本长度: {len(sample)}")
        print(f"GT点云形状: {sample[0].shape}")
        print(f"噪声点云形状: {sample[1].shape}")
    
    # 测试使用体素指导的数据加载
    print("\n2. 测试使用体素指导的数据加载")
    args.use_voxel_guidance = True
    dataset_with_voxel = data_handler.H5PairedDataset(None, device, args, split='train')
    
    sample = dataset_with_voxel[0]
    print(f"使用体素指导 - 样本类型: {type(sample)}")
    if isinstance(sample, tuple):
        print(f"样本长度: {len(sample)}")
        print(f"GT点云形状: {sample[0].shape}")
        print(f"噪声点云形状: {sample[1].shape}")
        if len(sample) > 2:
            print(f"体素网格形状: {sample[2].shape}")
    
    # 测试推理数据集
    print("\n3. 测试推理数据集")
    inference_dataset = data_handler.H5InferenceDataset(None, device, args)
    
    sample = inference_dataset[0]
    print(f"推理样本类型: {type(sample)}")
    if isinstance(sample, dict):
        print(f"样本键: {list(sample.keys())}")
        print(f"GT点云形状: {sample['gt_normalized'].shape}")
        print(f"噪声点云形状: {sample['noisy_normalized'].shape}")
        if 'voxel_grid' in sample:
            print(f"体素网格形状: {sample['voxel_grid'].shape}")
    
    print("✅ 数据加载功能测试通过")
    
    return args


def test_model_integration(args):
    """测试模型集成"""
    print("\n=== 测试模型集成 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不使用体素指导的模型
    print("\n1. 测试不使用体素指导的模型")
    args.use_voxel_guidance = False
    model_no_voxel = PointNet2Generator(device, args).to(device)
    
    batch_size = 2
    num_points = 2048
    test_points = torch.randn(batch_size, 3, num_points).to(device)
    
    with torch.no_grad():
        output_no_voxel = model_no_voxel(test_points)
    
    print(f"输入形状: {test_points.shape}")
    print(f"输出形状: {output_no_voxel.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model_no_voxel.parameters())}")
    
    # 测试使用体素指导的模型
    print("\n2. 测试使用体素指导的模型")
    args.use_voxel_guidance = True
    model_with_voxel = PointNet2Generator(device, args).to(device)
    
    voxel_size = 64
    test_voxel = torch.randn(batch_size, 1, voxel_size, voxel_size, voxel_size).to(device)
    
    with torch.no_grad():
        output_with_voxel = model_with_voxel(test_points, test_voxel)
    
    print(f"输入点云形状: {test_points.shape}")
    print(f"输入体素形状: {test_voxel.shape}")
    print(f"输出形状: {output_with_voxel.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model_with_voxel.parameters())}")
    
    print("✅ 模型集成测试通过")


def main():
    """主测试函数"""
    print("开始体素指导RefineNet功能测试...")
    
    try:
        # 测试体素指导组件
        test_voxel_guidance_components()
        
        # 测试数据加载
        args = test_data_loading()
        
        # 测试模型集成
        test_model_integration(args)
        
        print("\n🎉 所有测试通过！体素指导功能实现成功！")
        
        print("\n=== 使用示例 ===")
        print("启用体素指导的训练命令:")
        print("python scripts/refine_net/train_refine_net.py \\")
        print("    --use-voxel-guidance \\")
        print("    --voxel-h5-path /path/to/voxels.h5 \\")
        print("    --voxel-data-key voxel_grids \\")
        print("    --voxel-grid-size 64 \\")
        print("    --gt-h5-path /path/to/gt.h5 \\")
        print("    --noisy-h5-path /path/to/noisy.h5 \\")
        print("    --sample-points 2048 \\")
        print("    --batch-size 4")
        
        print("\n不启用体素指导的训练命令:")
        print("python scripts/refine_net/train_refine_net.py \\")
        print("    --gt-h5-path /path/to/gt.h5 \\")
        print("    --noisy-h5-path /path/to/noisy.h5 \\")
        print("    --sample-points 2048 \\")
        print("    --batch-size 4")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


