#!/usr/bin/env python3
"""
RefineNet推理脚本

用于加载训练好的RefineNet模型进行点云修正推理。
支持大规模点云分批处理和结果保存为CSV格式。
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入RefineNet模块
from src.refine_net import options, models, util
from src.refine_net.data_handler import H5InferenceDataset

def load_model(model_path, device, args):
    """加载训练好的模型"""
    model = models.PointNet2Generator(device, args)
    
    if model_path.suffix == '.pt':
        # 直接加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # 加载完整检查点
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def process_large_pointcloud(model, noisy_pc, batch_size, device):
    """
    分批处理大规模点云
    
    Args:
        model: 模型
        noisy_pc: 噪声点云 (3, N)
        batch_size: 批处理大小
        device: 设备
        
    Returns:
        torch.Tensor: 修正后的点云 (3, N)
    """
    num_points = noisy_pc.shape[1]
    refined_batches = []
    
    for start_idx in tqdm(range(0, num_points, batch_size), desc="处理批次"):
        end_idx = min(start_idx + batch_size, num_points)
        
        # 获取批次
        batch = noisy_pc[:, start_idx:end_idx]  # (3, batch_size)
        
        # 如果批次太小，进行padding
        if batch.shape[1] < batch_size:
            padding_size = batch_size - batch.shape[1]
            # 重复最后几个点进行padding
            repeat_indices = torch.randint(0, batch.shape[1], (padding_size,), device=device)
            padding = batch[:, repeat_indices]
            batch = torch.cat([batch, padding], dim=1)
        
        # 模型推理
        with torch.no_grad():
            refined_batch = model(batch.unsqueeze(0))[0]  # (3, batch_size)
        
        # 移除padding
        if end_idx - start_idx < batch_size:
            refined_batch = refined_batch[:, :end_idx - start_idx]
        
        refined_batches.append(refined_batch)
    
    # 拼接所有批次
    refined_pc = torch.cat(refined_batches, dim=1)
    return refined_pc

def inference_single_sample(model, inference_dataset, sample_idx, device, args, output_dir):
    """对单个样本进行推理"""
    
    data = inference_dataset[sample_idx]
    
    # 获取数据
    gt_normalized = data['gt_normalized'].to(device)  # (3, N)
    noisy_normalized = data['noisy_normalized'].to(device)  # (3, N)
    gt_original = data['gt_original']  # (N, 3)
    noisy_original = data['noisy_original']  # (N, 3)
    sample_id = data['sample_idx']
    num_points = data['num_points']
    
    print(f"\n处理样本 {sample_id}, 点数: {num_points}")
    
    # 推理
    with torch.no_grad():
        if num_points > args.sample_points:
            print(f"大规模点云，分批处理（批大小: {args.sample_points}）")
            refined_normalized = process_large_pointcloud(
                model, noisy_normalized, args.sample_points, device
            )
        else:
            # 直接处理
            refined_normalized = model(noisy_normalized.unsqueeze(0))[0]  # (3, N)
    
    # 反归一化: (3, N) -> (N, 3)
    refined_normalized_np = refined_normalized.transpose(0, 1).cpu().numpy()
    refined_original = inference_dataset.denormalize_pointcloud(refined_normalized_np)
    
    # 保存结果到CSV
    sample_dir = output_dir / f'sample_{sample_id:03d}'
    sample_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(gt_original, columns=['x', 'y', 'z']).to_csv(
        sample_dir / 'gt.csv', index=False)
    pd.DataFrame(noisy_original, columns=['x', 'y', 'z']).to_csv(
        sample_dir / 'noisy.csv', index=False)
    pd.DataFrame(refined_original, columns=['x', 'y', 'z']).to_csv(
        sample_dir / 'refined.csv', index=False)
    
    print(f"样本 {sample_id} 推理完成，结果保存到: {sample_dir}")
    
    return {
        'sample_id': sample_id,
        'num_points': num_points,
        'output_dir': sample_dir
    }

def main_inference():
    """主推理函数"""
    parser = argparse.ArgumentParser(description='RefineNet点云修正推理')
    
    # 模型和数据参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--gt-h5-path', type=str, required=True,
                        help='GT点云H5文件路径')
    parser.add_argument('--noisy-h5-path', type=str, required=True,
                        help='噪声点云H5文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='推理结果输出目录')
    
    # 数据格式参数
    parser.add_argument('--gt-data-key', type=str, default='point_clouds',
                        help='GT数据在H5文件中的键名')
    parser.add_argument('--noisy-data-key', type=str, default='point_clouds',
                        help='噪声数据在H5文件中的键名')
    
    # 处理参数
    parser.add_argument('--sample-points', type=int, default=8192,
                        help='批处理采样点数')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练数据占比（用于数据集分割）')
    parser.add_argument('--sample-idx', type=int, default=-1,
                        help='要处理的样本索引，-1表示处理所有推理样本')
    
    # 归一化参数
    parser.add_argument('--volume-dims', type=int, nargs=3, default=[20000, 20000, 2500],
                        help='体积维度 [x, y, z]')
    parser.add_argument('--padding', type=int, nargs=3, default=[0, 0, 100],
                        help='边界填充 [x, y, z]')
    
    # 其他参数
    parser.add_argument('--init-var', type=float, default=0.2,
                        help='模型参数初始化方差')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    if not os.path.exists(args.gt_h5_path):
        raise FileNotFoundError(f"GT H5文件不存在: {args.gt_h5_path}")
    
    if not os.path.exists(args.noisy_h5_path):
        raise FileNotFoundError(f"噪声H5文件不存在: {args.noisy_h5_path}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    print("=== RefineNet推理配置 ===")
    print(f"模型路径: {args.model_path}")
    print(f"GT H5文件: {args.gt_h5_path}")
    print(f"噪声H5文件: {args.noisy_h5_path}")
    print(f"批处理点数: {args.sample_points}")
    print(f"输出目录: {output_dir}")
    print(f"体积维度: {args.volume_dims}")
    print(f"边界填充: {args.padding}")
    print("=" * 30)
    
    try:
        # 加载模型
        print("加载模型...")
        model = load_model(Path(args.model_path), device, args)
        print(f'模型参数数量: {util.n_params(model)}')
        
        # 创建推理数据集
        print("创建推理数据集...")
        dummy_pc = torch.zeros((100, 3))
        inference_dataset = H5InferenceDataset(dummy_pc, device, args)
        
        print(f"推理样本数: {len(inference_dataset)}")
        
        # 执行推理
        results = []
        if args.sample_idx >= 0:
            # 处理单个样本
            if args.sample_idx >= len(inference_dataset):
                raise IndexError(f"样本索引超出范围: {args.sample_idx} >= {len(inference_dataset)}")
            
            result = inference_single_sample(
                model, inference_dataset, args.sample_idx, device, args, output_dir
            )
            results.append(result)
        else:
            # 处理所有样本
            for i in range(len(inference_dataset)):
                result = inference_single_sample(
                    model, inference_dataset, i, device, args, output_dir
                )
                results.append(result)
        
        # 保存推理汇总
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / 'inference_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n🎉 推理完成！")
        print(f"处理样本数: {len(results)}")
        print(f"结果保存到: {output_dir}")
        print(f"推理汇总: {summary_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main_inference()
    sys.exit(exit_code)
