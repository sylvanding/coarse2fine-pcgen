#!/usr/bin/env python3
"""
RefineNet CSV点云推理脚本

用于加载训练好的RefineNet模型对CSV格式的点云进行修正推理。
自动处理点云平移、边界计算和归一化参数设置。

归一化方法：采用与H5PairedDataset一致的归一化策略，
- 将点云坐标归一化到[-1, 1]范围
- 考虑volume_dims和padding参数
- Z轴使用Y轴的尺度进行归一化（保持训练时的一致性）
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入RefineNet模块
from src.refine_net import options
from src.refine_net.models import PointNet2Generator
import src.refine_net.util as util


class CSVPointCloudProcessor:
    """CSV点云处理器"""
    
    def __init__(self, csv_path: str, volume_dims: np.ndarray, padding: np.ndarray):
        """
        初始化CSV点云处理器
        
        Args:
            csv_path: CSV文件路径
            volume_dims: 体积维度 [x, y, z]
            padding: 边界填充 [x, y, z]
        """
        self.csv_path = Path(csv_path)
        self.original_pc = None
        self.translated_pc = None
        self.translation_offset = None
        self.volume_dims = volume_dims
        self.padding = padding
        
    def load_and_process(self):
        """加载并处理CSV点云数据"""
        print(f"加载CSV点云: {self.csv_path}")
        
        # 读取CSV文件
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # 检查列名
        if not all(col in df.columns for col in ['x [nm]', 'y [nm]', 'z [nm]']):
            raise ValueError("CSV文件必须包含 'x [nm]', 'y [nm]', 'z [nm]' 列")
        
        # 提取点云坐标
        self.original_pc = df[['x [nm]', 'y [nm]', 'z [nm]']].values.astype(np.float32)
        print(f"点云形状: {self.original_pc.shape}")
        
        # 计算边界
        min_coords = np.min(self.original_pc, axis=0)
        max_coords = np.max(self.original_pc, axis=0)
        
        print(f"原始边界:")
        print(f"  X: [{min_coords[0]:.2f}, {max_coords[0]:.2f}]")
        print(f"  Y: [{min_coords[1]:.2f}, {max_coords[1]:.2f}]")
        print(f"  Z: [{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
        
        # 计算平移偏移量，确保所有点都大于等于0
        self.translation_offset = -min_coords
        
        # 应用平移
        self.translated_pc = self.original_pc + self.translation_offset
        
        # 重新计算平移后的边界
        translated_min = np.min(self.translated_pc, axis=0)
        translated_max = np.max(self.translated_pc, axis=0)
        
        print(f"平移偏移量: [{self.translation_offset[0]:.2f}, {self.translation_offset[1]:.2f}, {self.translation_offset[2]:.2f}]")
        print(f"平移后边界:")
        print(f"  X: [{translated_min[0]:.2f}, {translated_max[0]:.2f}]")
        print(f"  Y: [{translated_min[1]:.2f}, {translated_max[1]:.2f}]")
        print(f"  Z: [{translated_min[2]:.2f}, {translated_max[2]:.2f}]")
        
        # # 设置volume_dims为最大值（向上取整到整数）
        # self.volume_dims = np.ceil(translated_max).astype(int) + self.padding
        # print(f"计算得到的volume_dims: [{self.volume_dims[0]}, {self.volume_dims[1]}, {self.volume_dims[2]}]")
        
        return self.translated_pc, self.volume_dims
    
    def save_results(self, refined_pc: np.ndarray, output_dir: Path):
        """
        保存推理结果
        
        Args:
            refined_pc: 修正后的点云 (N, 3)
            output_dir: 输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将修正后的点云平移回原始位置
        refined_original = refined_pc - self.translation_offset
        
        # 保存各种版本的点云
        pd.DataFrame(self.original_pc, columns=['x [nm]', 'y [nm]', 'z [nm]']).to_csv(
            output_dir / 'original.csv', index=False)
        
        # pd.DataFrame(self.translated_pc, columns=['x', 'y', 'z']).to_csv(
        #     output_dir / 'translated.csv', index=False)
        
        # pd.DataFrame(refined_pc, columns=['x', 'y', 'z']).to_csv(
        #     output_dir / 'refined_translated.csv', index=False)
        
        pd.DataFrame(refined_original, columns=['x [nm]', 'y [nm]', 'z [nm]']).to_csv(
            output_dir / 'refined_original.csv', index=False)
        
        # 保存处理信息
        info = {
            'original_shape': self.original_pc.shape,
            'translation_offset': self.translation_offset.tolist(),
            'volume_dims': self.volume_dims.tolist(),
            'original_bounds_min': np.min(self.original_pc, axis=0).tolist(),
            'original_bounds_max': np.max(self.original_pc, axis=0).tolist(),
            'translated_bounds_min': np.min(self.translated_pc, axis=0).tolist(),
            'translated_bounds_max': np.max(self.translated_pc, axis=0).tolist(),
        }
        
        pd.DataFrame([info]).to_csv(output_dir / 'processing_info.csv', index=False)
        
        print(f"结果保存到: {output_dir}")
        print(f"  - original.csv: 原始点云")
        # print(f"  - translated.csv: 平移后的点云")
        # print(f"  - refined_translated.csv: 修正后的点云（平移坐标系）")
        print(f"  - refined_original.csv: 修正后的点云（原始坐标系）")
        print(f"  - processing_info.csv: 处理信息")


def load_model(model_path, device, args):
    """加载训练好的模型"""
    model = PointNet2Generator(device, args)
    
    # 加载完整检查点
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model


def normalize_pointcloud(pc: np.ndarray, volume_dims: np.ndarray, padding: np.ndarray = None) -> tuple[np.ndarray, float]:
    """
    归一化点云到 [-1, 1] 范围
    参考H5PairedDataset的归一化方法
    
    Args:
        pc: 点云数据 (N, 3)
        volume_dims: 体积维度 [x, y, z]
        padding: 边界填充 [x, y, z]，默认为0
        
    Returns:
        归一化后的点云 (N, 3), z轴平均值
    """
    if padding is None:
        padding = np.zeros(3)
    
    # 计算实际空间范围: volume_dims ± padding
    # x, y, z 坐标范围分别为 [-padding[i], volume_dims[i] + padding[i]]
    x_min, x_max = -padding[0], volume_dims[0] + padding[0]
    y_min, y_max = -padding[1], volume_dims[1] + padding[1]
    z_min, z_max = -padding[2], volume_dims[2] + padding[2]
    
    # 将点云坐标从原始范围归一化到 [-1, 1]
    normalized = pc.copy()
    normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min) * 2 - 1  # [x_min, x_max] -> [-1, 1]
    normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min) * 2 - 1  # [y_min, y_max] -> [-1, 1]
    # 特殊处理：Z轴使用Y轴的尺度（与H5PairedDataset保持一致）
    normalized[:, 2] = (normalized[:, 2] - z_min) / (y_max - y_min) * 2 - 1  # [z_min, z_max] -> [-1, 1]，但使用Y轴尺度
    
    # # 确保z轴平均值为0
    # z_mean = np.mean(normalized[:, 2])
    # normalized[:, 2] = normalized[:, 2] - z_mean
    
    # return normalized.astype(np.float32), z_mean
    return normalized.astype(np.float32), np.zeros(3)


def denormalize_pointcloud(normalized_pc: np.ndarray, volume_dims: np.ndarray, padding: np.ndarray = None, z_mean: float = 0) -> np.ndarray:
    """
    反归一化点云从 [-1, 1] 范围
    参考H5PairedDataset的反归一化方法
    
    Args:
        normalized_pc: 归一化的点云 (N, 3)
        volume_dims: 体积维度 [x, y, z]
        padding: 边界填充 [x, y, z]，默认为0
        z_mean: z轴平均值
    Returns:
        反归一化后的点云 (N, 3)
    """
    if padding is None:
        padding = np.zeros(3)
    
    # 计算实际空间范围: volume_dims ± padding
    # x, y, z 坐标范围分别为 [-padding[i], volume_dims[i] + padding[i]]
    x_min, x_max = -padding[0], volume_dims[0] + padding[0]
    y_min, y_max = -padding[1], volume_dims[1] + padding[1]
    z_min, z_max = -padding[2], volume_dims[2] + padding[2]
    
    # # 把z轴移回中心
    # normalized_pc[:, 2] = normalized_pc[:, 2] + z_mean
    
    # 从 [-1, 1] 反归一化到原始坐标范围
    denormalized = normalized_pc.copy()
    denormalized[:, 0] = (denormalized[:, 0] + 1) / 2 * (x_max - x_min) + x_min  # [-1, 1] -> [x_min, x_max]
    denormalized[:, 1] = (denormalized[:, 1] + 1) / 2 * (y_max - y_min) + y_min  # [-1, 1] -> [y_min, y_max]
    # 特殊处理：Z轴使用Y轴的尺度（与H5PairedDataset保持一致）
    denormalized[:, 2] = (denormalized[:, 2] + 1) / 2 * (y_max - y_min) + z_min  # [-1, 1] -> [z_min, z_max]，但使用Y轴尺度
    
    return denormalized


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


def main_inference():
    """主推理函数"""
    parser = options.get_parser('RefineNet CSV点云修正推理')
    
    # 输入输出参数
    parser.add_argument('--model-path', type=str, default='/repos/datasets/exp-data-4pi-pc-mitochondria/3d_diffusion/iteration_580000.pt',
                        help='训练好的模型路径')
    parser.add_argument('--csv-path', type=str, default='/repos/datasets/exp-data-4pi-pc-mitochondria/3d_diffusion/generated_sample_03_points.csv',
                        help='输入CSV点云文件路径')
    parser.add_argument('--output-dir', type=str, default='/repos/datasets/exp-data-4pi-pc-mitochondria/3d_diffusion/generated_sample_03_points_refined',
                        help='推理结果输出目录')
    # `options` has defined these parameters
    # parser.add_argument('--sample-points', type=int, default=80000,
    #                     help='批处理采样点数')
    # parser.add_argument('--volume-dims', type=int, nargs=3, default=[8000, 8000, 300],
    #                     help='体积维度 [x, y, z]')
    # parser.add_argument('--padding', type=int, nargs=3, default=[0, 0, 0],
    #                     help='边界填充 [x, y, z]')
    
    args = parser.parse_args()
    
    args.volume_dims = [8000, 8000, 1200]
    
    # 检查输入文件
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {args.csv_path}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    print("=== RefineNet CSV推理配置 ===")
    print(f"模型路径: {args.model_path}")
    print(f"CSV文件: {args.csv_path}")
    print(f"体积维度: {args.volume_dims}")
    print(f"边界填充: {args.padding}")
    print(f"批处理点数: {args.sample_points}")
    print(f"体积维度: {args.volume_dims}")
    print(f"边界填充: {args.padding}")
    print(f"输出目录: {output_dir}")
    print("=" * 30)
    
    try:
        # 处理CSV点云
        print("\n处理CSV点云...")
        processor = CSVPointCloudProcessor(args.csv_path, np.array(args.volume_dims), np.array(args.padding))
        translated_pc, volume_dims = processor.load_and_process()
        
        # 动态设置归一化参数
        args.volume_dims = volume_dims.tolist()
        
        print(f"\n自动设置的参数:")
        print(f"  volume_dims: {args.volume_dims}")
        print(f"  padding: {args.padding}")
        print(f"  归一化方法: 与H5PairedDataset一致（Z轴使用Y轴尺度）")
        
        # 加载模型
        print("\n加载模型...")
        model = load_model(Path(args.model_path), device, args)
        print(f'模型参数数量: {util.n_params(model)}')
        
        # 准备输入数据
        print("\n准备输入数据...")
        num_points = len(translated_pc)
        print(f"点云点数: {num_points}")
        
        # 归一化点云
        normalized_pc, z_mean = normalize_pointcloud(translated_pc, np.array(args.volume_dims), np.array(args.padding))
        
        # 转换为模型输入格式: (N, 3) -> (3, N)
        normalized_tensor = torch.from_numpy(normalized_pc.T).to(device)  # (3, N)
        
        print(f"归一化后点云范围:")
        print(f"  X: [{normalized_pc[:, 0].min():.4f}, {normalized_pc[:, 0].max():.4f}]")
        print(f"  Y: [{normalized_pc[:, 1].min():.4f}, {normalized_pc[:, 1].max():.4f}]")
        print(f"  Z: [{normalized_pc[:, 2].min():.4f}, {normalized_pc[:, 2].max():.4f}]")
        
        # 推理
        print("\n执行推理...")
        with torch.no_grad():
            if num_points > args.sample_points:
                print(f"大规模点云，分批处理（批大小: {args.sample_points}）")
                refined_normalized = process_large_pointcloud(
                    model, normalized_tensor, args.sample_points, device
                )
            else:
                # 直接处理
                refined_normalized = model(normalized_tensor.unsqueeze(0))[0]  # (3, N)
        
        # 反归一化: (3, N) -> (N, 3)
        refined_normalized_np = refined_normalized.transpose(0, 1).cpu().numpy()
        refined_translated = denormalize_pointcloud(
            refined_normalized_np, np.array(args.volume_dims), np.array(args.padding), z_mean
        )
        
        print(f"反归一化后点云范围:")
        print(f"  X: [{refined_translated[:, 0].min():.2f}, {refined_translated[:, 0].max():.2f}]")
        print(f"  Y: [{refined_translated[:, 1].min():.2f}, {refined_translated[:, 1].max():.2f}]")
        print(f"  Z: [{refined_translated[:, 2].min():.2f}, {refined_translated[:, 2].max():.2f}]")
        
        # 保存结果
        print("\n保存结果...")
        processor.save_results(refined_translated, output_dir)
        
        print(f"\n🎉 推理完成！")
        print(f"原始点云: {args.csv_path}")
        print(f"处理点数: {num_points}")
        print(f"结果保存到: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main_inference()
    sys.exit(exit_code)
