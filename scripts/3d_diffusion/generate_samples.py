#!/usr/bin/env python
"""
3D Diffusion样本生成脚本

从训练好的3D扩散模型生成体素样本并保存为TIFF文件。
支持DDIM采样、批量生成和可视化。
"""

import argparse
import sys
from pathlib import Path
import logging
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.diffusion_lightning import DiffusionLightningModule, DiffusionLightningModuleWithEMA
from src.utils.config_loader import load_config_with_overrides
from src.voxel.converter import PointCloudToVoxel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从训练好的3D Diffusion模型生成样本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="模型检查点路径"
    )
    
    # 生成参数
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="生成样本数量"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="生成批次大小"
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="DDIM采样步数"
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM随机性参数 (0为确定性采样)"
    )
    
    # 输出设置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_samples",
        help="输出目录"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["tiff", "npy", "both"],
        default="tiff",
        help="保存格式"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generated_sample",
        help="输出文件前缀"
    )
    
    # 设备设置
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto, cpu, cuda, cuda:0 等)"
    )
    
    # 其他选项
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示生成进度"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径（可选，用于获取体素参数）"
    )
    
    return parser.parse_args()


def determine_device(device_arg: str) -> torch.device:
    """确定计算设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    
    logger.info(f"使用设备: {device}")
    return device


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """从检查点加载模型"""
    logger.info(f"从检查点加载模型: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 尝试确定模型类型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否为EMA模型
    if 'ema_decay' in checkpoint.get('hyper_parameters', {}):
        model = DiffusionLightningModuleWithEMA.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        logger.info("加载EMA模型")
    else:
        model = DiffusionLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        logger.info("加载标准模型")
    
    model.eval()
    model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_size()
    logger.info(f"模型参数: {model_info['total_parameters']:,}")
    logger.info(f"体素大小: {model.voxel_size}^3")
    
    return model


def generate_samples_in_batches(
    model,
    num_samples: int,
    batch_size: int,
    ddim_steps: int,
    ddim_eta: float,
    device: torch.device,
    show_progress: bool = True
) -> torch.Tensor:
    """批量生成样本"""
    logger.info(f"开始生成 {num_samples} 个样本 (批次大小: {batch_size})")
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    if show_progress:
        batch_iterator = tqdm(range(num_batches), desc="生成批次")
    else:
        batch_iterator = range(num_batches)
    
    with torch.no_grad():
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            # 生成当前批次
            shape = (current_batch_size, 1, model.voxel_size, model.voxel_size, model.voxel_size)
            
            # 使用DDIM采样
            batch_samples = model.diffusion.ddim_sample(
                model=model.model,
                shape=shape,
                device=device,
                eta=ddim_eta,
                num_inference_steps=ddim_steps,
                progress=show_progress and num_batches == 1  # 只在单批次时显示内部进度
            )
            
            # 后处理：将[-1, 1]映射到[0, 1]
            batch_samples = (batch_samples + 1.0) / 2.0
            batch_samples = torch.clamp(batch_samples, 0.0, 1.0)
            
            all_samples.append(batch_samples.cpu())
            
            if show_progress:
                batch_iterator.set_postfix({
                    'samples': f"{end_idx}/{num_samples}"
                })
    
    # 合并所有批次
    all_samples = torch.cat(all_samples, dim=0)
    logger.info(f"生成完成，总样本数: {len(all_samples)}")
    
    return all_samples


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    save_format: str,
    prefix: str,
    voxel_size: int
):
    """保存生成的样本"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_format in ["tiff", "both"]:
        # 创建体素转换器用于保存TIFF
        converter = PointCloudToVoxel(voxel_size=voxel_size)
        
        logger.info(f"保存TIFF文件到: {output_dir}")
        for i, sample in enumerate(samples):
            # 移除批次和通道维度
            voxel_np = sample.squeeze().numpy()
            
            # 保存为TIFF
            tiff_path = output_dir / f"{prefix}_{i:04d}.tiff"
            converter.save_as_tiff(voxel_np, str(tiff_path))
    
    if save_format in ["npy", "both"]:
        logger.info(f"保存NumPy文件到: {output_dir}")
        for i, sample in enumerate(samples):
            # 移除批次和通道维度
            voxel_np = sample.squeeze().numpy()
            
            # 保存为NPY
            npy_path = output_dir / f"{prefix}_{i:04d}.npy"
            np.save(npy_path, voxel_np)
    
    # 保存统计信息
    stats_path = output_dir / f"{prefix}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"生成样本统计信息\n")
        f.write(f"=================\n\n")
        f.write(f"样本数量: {len(samples)}\n")
        f.write(f"体素大小: {voxel_size}^3\n")
        f.write(f"数据类型: {samples.dtype}\n")
        f.write(f"数据形状: {samples.shape}\n\n")
        
        # 计算统计
        samples_np = samples.numpy()
        f.write(f"统计信息:\n")
        f.write(f"  均值: {np.mean(samples_np):.6f}\n")
        f.write(f"  标准差: {np.std(samples_np):.6f}\n")
        f.write(f"  最小值: {np.min(samples_np):.6f}\n")
        f.write(f"  最大值: {np.max(samples_np):.6f}\n")
        f.write(f"  占有率 (>0.5): {np.mean(samples_np > 0.5):.6f}\n")
    
    logger.info(f"已保存 {len(samples)} 个样本到: {output_dir}")


def compute_sample_statistics(samples: torch.Tensor):
    """计算并打印样本统计信息"""
    samples_np = samples.numpy()
    
    logger.info("=== 生成样本统计 ===")
    logger.info(f"样本数量: {len(samples)}")
    logger.info(f"体素形状: {samples.shape[2:]}")
    logger.info(f"数据类型: {samples.dtype}")
    logger.info(f"均值: {np.mean(samples_np):.6f}")
    logger.info(f"标准差: {np.std(samples_np):.6f}")
    logger.info(f"最小值: {np.min(samples_np):.6f}")
    logger.info(f"最大值: {np.max(samples_np):.6f}")
    logger.info(f"占有率 (>0.5): {np.mean(samples_np > 0.5):.6f}")
    
    # 计算每个样本的占有率
    occupancy_rates = []
    for sample in samples_np:
        occupancy = np.mean(sample.squeeze() > 0.5)
        occupancy_rates.append(occupancy)
    
    logger.info(f"占有率分布: 均值={np.mean(occupancy_rates):.4f}, "
                f"标准差={np.std(occupancy_rates):.4f}, "
                f"范围=[{np.min(occupancy_rates):.4f}, {np.max(occupancy_rates):.4f}]")


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=== 3D Diffusion样本生成 ===")
    logger.info(f"检查点: {args.checkpoint_path}")
    logger.info(f"生成样本数: {args.num_samples}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确定设备
    device = determine_device(args.device)
    
    try:
        # 加载模型
        model = load_model_from_checkpoint(args.checkpoint_path, device)
        
        # 生成样本
        samples = generate_samples_in_batches(
            model=model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            device=device,
            show_progress=args.show_progress
        )
        
        # 计算统计信息
        compute_sample_statistics(samples)
        
        # 保存样本
        output_dir = Path(args.output_dir)
        save_samples(
            samples=samples,
            output_dir=output_dir,
            save_format=args.save_format,
            prefix=args.prefix,
            voxel_size=model.voxel_size
        )
        
        logger.info("样本生成完成!")
        
    except KeyboardInterrupt:
        logger.info("生成被用户中断")
    except Exception as e:
        logger.error(f"生成过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
