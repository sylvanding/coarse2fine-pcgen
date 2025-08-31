"""
H5点云修正推理脚本

支持对大规模噪声点云进行分批修正，并将结果保存为CSV文件。
对于点数超过采样点数的大规模点云，会自动分批处理。
"""

try:
    import open3d
except:
    pass
import torch
import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path
from tqdm import tqdm
import src.refine_net.options as options
import src.refine_net.util as util
from src.refine_net.models import PointNet2Generator
from src.refine_net.data_handler import H5PairedDataset


def load_large_pointcloud_from_h5(h5_path: str, data_key: str = 'data', sample_idx: int = 0) -> np.ndarray:
    """
    从H5文件加载大规模点云
    
    Args:
        h5_path: H5文件路径
        data_key: 数据键名
        sample_idx: 样本索引
        
    Returns:
        np.ndarray: 点云数据，形状为 (N, 3)
    """
    with h5py.File(h5_path, 'r') as f:
        if data_key not in f:
            available_keys = list(f.keys())
            raise ValueError(f"数据键 '{data_key}' 不存在。可用键: {available_keys}")
        
        data = f[data_key]
        if sample_idx >= data.shape[0]:
            raise IndexError(f"样本索引 {sample_idx} 超出范围，最大索引: {data.shape[0] - 1}")
        
        point_cloud = data[sample_idx]  # (N, 3)
        
    return point_cloud.astype(np.float32)


def batch_sample_pointcloud(point_cloud: np.ndarray, sample_size: int, overlap_ratio: float = 0.1) -> list:
    """
    将大规模点云分批采样
    
    Args:
        point_cloud: 原始点云 (N, 3)
        sample_size: 每批采样点数
        overlap_ratio: 批次间重叠比例
        
    Returns:
        list: 分批采样结果列表，每个元素为 (sample_size, 3)
    """
    N = point_cloud.shape[0]
    
    if N <= sample_size:
        # 点数不足，直接返回
        return [point_cloud]
    
    batches = []
    step_size = int(sample_size * (1 - overlap_ratio))
    
    for start_idx in range(0, N, step_size):
        end_idx = min(start_idx + sample_size, N)
        
        if end_idx - start_idx < sample_size:
            # 最后一批点数不足，从末尾采样
            indices = np.random.choice(N, sample_size, replace=False)
            batch = point_cloud[indices]
        else:
            # 连续采样
            batch = point_cloud[start_idx:end_idx]
        
        batches.append(batch)
        
        if end_idx >= N:
            break
    
    return batches


def refine_pointcloud_batches(model, point_cloud_batches: list, device, sample_size: int) -> list:
    """
    对分批点云进行修正
    
    Args:
        model: 训练好的模型
        point_cloud_batches: 分批点云列表
        device: 计算设备
        sample_size: 采样点数
        
    Returns:
        list: 修正后的点云批次
    """
    model.eval()
    refined_batches = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(point_cloud_batches, desc="处理点云批次")):
            # 确保批次大小正确
            if batch.shape[0] != sample_size:
                # 重新采样到指定大小
                indices = np.random.choice(batch.shape[0], sample_size, replace=True)
                batch = batch[indices]
            
            # 转换为tensor: (N, 3) -> (3, N) -> (1, 3, N)
            batch_tensor = torch.from_numpy(batch).transpose(0, 1).unsqueeze(0).to(device)
            
            # 模型推理
            refined_batch = model(batch_tensor)  # (1, 3, N)
            
            # 转换回numpy: (1, 3, N) -> (3, N) -> (N, 3)
            refined_np = refined_batch[0].transpose(0, 1).cpu().numpy()
            
            refined_batches.append(refined_np)
    
    return refined_batches


def merge_refined_batches(refined_batches: list, original_size: int, merge_strategy: str = 'concat') -> np.ndarray:
    """
    合并修正后的点云批次
    
    Args:
        refined_batches: 修正后的点云批次列表
        original_size: 原始点云大小
        merge_strategy: 合并策略 ('concat' 或 'average')
        
    Returns:
        np.ndarray: 合并后的点云 (M, 3)
    """
    if merge_strategy == 'concat':
        # 简单拼接所有批次
        merged = np.concatenate(refined_batches, axis=0)
        
        # 如果点数过多，随机采样到目标大小
        if merged.shape[0] > original_size:
            indices = np.random.choice(merged.shape[0], original_size, replace=False)
            merged = merged[indices]
        
        return merged
    
    elif merge_strategy == 'average':
        # TODO: 实现基于距离的加权平均合并
        # 目前使用简单拼接
        return merge_refined_batches(refined_batches, original_size, 'concat')
    
    else:
        raise ValueError(f"未知的合并策略: {merge_strategy}")


def save_pointcloud_to_csv(point_cloud: np.ndarray, output_path: str):
    """
    保存点云到CSV文件
    
    Args:
        point_cloud: 点云数据 (N, 3)
        output_path: 输出文件路径
    """
    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
    df.to_csv(output_path, index=False)
    print(f"点云已保存到: {output_path}")


def inference_single_sample(args, model, device, sample_idx: int = 0):
    """
    对单个样本进行推理
    
    Args:
        args: 参数配置
        model: 训练好的模型
        device: 计算设备
        sample_idx: 样本索引
    """
    print(f"\n=== 开始处理样本 {sample_idx} ===")
    
    # 加载噪声点云
    noisy_pc = load_large_pointcloud_from_h5(args.noisy_h5_path, args.noisy_data_key, sample_idx)
    print(f"原始噪声点云大小: {noisy_pc.shape}")
    
    # 分批处理
    if noisy_pc.shape[0] > args.sample_points:
        print(f"点云过大，将分批处理（批大小: {args.sample_points}）")
        
        # 分批采样
        batches = batch_sample_pointcloud(noisy_pc, args.sample_points)
        print(f"分为 {len(batches)} 个批次")
        
        # 批量修正
        refined_batches = refine_pointcloud_batches(model, batches, device, args.sample_points)
        
        # 合并结果
        refined_pc = merge_refined_batches(refined_batches, noisy_pc.shape[0])
        
    else:
        print("直接处理整个点云")
        # 直接处理
        refined_batches = refine_pointcloud_batches(model, [noisy_pc], device, args.sample_points)
        refined_pc = refined_batches[0]
    
    print(f"修正后点云大小: {refined_pc.shape}")
    
    # 保存结果
    output_dir = Path(args.save_path) / 'inference_results'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存原始噪声点云
    save_pointcloud_to_csv(noisy_pc, output_dir / f'noisy_sample_{sample_idx:03d}.csv')
    
    # 保存修正后点云
    save_pointcloud_to_csv(refined_pc, output_dir / f'refined_sample_{sample_idx:03d}.csv')
    
    # 如果有GT数据，也加载并保存
    if os.path.exists(args.gt_h5_path):
        try:
            gt_pc = load_large_pointcloud_from_h5(args.gt_h5_path, args.gt_data_key, sample_idx)
            save_pointcloud_to_csv(gt_pc, output_dir / f'gt_sample_{sample_idx:03d}.csv')
            print(f"GT点云大小: {gt_pc.shape}")
        except Exception as e:
            print(f"无法加载GT点云: {e}")
    
    print(f"=== 样本 {sample_idx} 处理完成 ===\n")


def inference_all_samples(args, model, device):
    """
    对所有样本进行推理
    
    Args:
        args: 参数配置
        model: 训练好的模型
        device: 计算设备
    """
    # 获取样本总数
    with h5py.File(args.noisy_h5_path, 'r') as f:
        num_samples = f[args.noisy_data_key].shape[0]
    
    print(f"总共 {num_samples} 个样本")
    
    # 处理每个样本
    for sample_idx in range(num_samples):
        inference_single_sample(args, model, device, sample_idx)


def main():
    """主函数"""
    parser = options.get_parser('H5点云修正推理')
    
    # 添加推理特定参数
    parser.add_argument('--model-path', type=Path, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--sample-idx', type=int, default=-1,
                        help='要处理的样本索引，-1表示处理所有样本')
    parser.add_argument('--merge-strategy', type=str, default='concat',
                        choices=['concat', 'average'],
                        help='批次合并策略')
    
    args = options.parse_args(parser, inference=True)
    
    # 检查文件是否存在
    if not os.path.exists(args.noisy_h5_path):
        raise FileNotFoundError(f"噪声H5文件不存在: {args.noisy_h5_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 设置设备
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    print(f'使用设备: {device}')
    
    # 加载模型
    model = PointNet2Generator(device, args)
    model.load_state_dict(torch.load(str(args.model_path), map_location=device))
    model.to(device)
    model.eval()
    
    print(f'模型参数数量: {util.n_params(model)}')
    print(f'采样点数: {args.sample_points}')
    print(f'噪声H5文件: {args.noisy_h5_path}')
    print(f'GT H5文件: {args.gt_h5_path}')
    
    # 执行推理
    if args.sample_idx >= 0:
        # 处理单个样本
        inference_single_sample(args, model, device, args.sample_idx)
    else:
        # 处理所有样本
        inference_all_samples(args, model, device)
    
    print("推理完成！")


if __name__ == "__main__":
    main()
