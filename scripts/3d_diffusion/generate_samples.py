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
import random
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
        "--checkpoint_path",
        default="/repos/datasets/exp-data-4pi-pc-mt/ckpt/best-220-0.0006.ckpt",
        type=str,
        help="模型检查点路径"
    )
    
    # 生成参数
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="生成样本数量"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="生成批次大小"
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=1000,
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
        default="/repos/datasets/exp-data-4pi-pc-mt/3d_diffusion_generated_samples",
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
        default="mt_generated_sample",
        help="输出文件前缀"
    )
    
    # 设备设置
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备 (auto, cpu, cuda, cuda:0 等)"
    )
    
    # 其他选项
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 1000000),
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
        default="/repos/coarse2fine-pcgen/exp_record/mt/diffusion_config.yaml",
        help="配置文件路径（可选，用于获取体素参数）"
    )
    parser.add_argument(
        "--keep-samples-in-memory",
        action="store_true",
        help="在内存中保留所有样本用于统计（默认False以节省内存）"
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


def generate_and_save_samples_in_batches(
    model,
    num_samples: int,
    batch_size: int,
    ddim_steps: int,
    ddim_eta: float,
    device: torch.device,
    output_dir: Path,
    save_format: str,
    prefix: str,
    show_progress: bool = True,
    keep_in_memory: bool = False
) -> list:
    """批量生成样本并边生成边保存，返回numpy数组列表以节省显存"""
    logger.info(f"开始生成 {num_samples} 个样本 (批次大小: {batch_size})")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建体素转换器用于保存TIFF
    converter = None
    if save_format in ["tiff", "both"]:
        converter = PointCloudToVoxel(voxel_size=model.voxel_size)
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0
    
    if show_progress:
        batch_iterator = tqdm(range(num_batches), desc="生成并保存批次")
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
            
            # 立即转换为numpy并释放GPU内存
            batch_samples_np = batch_samples.cpu().numpy()
            del batch_samples  # 显式删除GPU张量
            torch.cuda.empty_cache()  # 清理GPU缓存
            
            # 立即保存当前批次
            for sample in batch_samples_np:
                voxel_np = sample.squeeze()
                
                # 保存为TIFF
                if save_format in ["tiff", "both"]:
                    tiff_path = output_dir / f"{prefix}_{sample_idx:04d}.tiff"
                    converter.save_as_tiff(voxel_np, str(tiff_path))
                
                # 保存为NPY
                if save_format in ["npy", "both"]:
                    npy_path = output_dir / f"{prefix}_{sample_idx:04d}.npy"
                    np.save(npy_path, voxel_np)
                
                sample_idx += 1
            
            # 保留样本用于统计（可选择性保留）
            if keep_in_memory:
                all_samples.append(batch_samples_np)
            else:
                # 只保留一小部分样本用于基本统计
                if len(all_samples) < 5:  # 最多保留5个批次用于统计
                    all_samples.append(batch_samples_np)
            
            if show_progress:
                batch_iterator.set_postfix({
                    'samples': f"{sample_idx}/{num_samples}",
                    'saved': f"{sample_idx}"
                })
    
    logger.info(f"生成并保存完成，总样本数: {sample_idx}")
    
    return all_samples


def compute_sample_statistics(samples: list):
    """计算并打印样本统计信息（numpy数组列表）"""
    if not samples:
        logger.warning("没有样本用于统计")
        return
        
    # 合并所有批次
    samples_np = np.concatenate([batch for batch in samples], axis=0)
    total_samples = len(samples_np)
    
    logger.info("=== 生成样本统计 ===")
    logger.info(f"统计样本数量: {total_samples} (可能是部分样本)")
    logger.info(f"体素形状: {samples_np.shape[2:]}")
    logger.info(f"数据类型: {samples_np.dtype}")
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


def save_statistics_file(samples: list, output_dir: Path, prefix: str, voxel_size: int):
    """保存统计信息到文件"""
    if not samples:
        logger.warning("没有样本用于保存统计信息")
        return
        
    # 合并所有批次计算统计
    all_samples_np = np.concatenate([batch for batch in samples], axis=0)
    total_samples = len(all_samples_np)
    
    stats_path = output_dir / f"{prefix}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"生成样本统计信息\n")
        f.write(f"=================\n\n")
        f.write(f"统计样本数量: {total_samples} (可能是部分样本)\n")
        f.write(f"体素大小: {voxel_size}^3\n")
        f.write(f"数据类型: {all_samples_np.dtype}\n")
        f.write(f"数据形状: {all_samples_np.shape}\n\n")
        
        f.write(f"统计信息:\n")
        f.write(f"  均值: {np.mean(all_samples_np):.6f}\n")
        f.write(f"  标准差: {np.std(all_samples_np):.6f}\n")
        f.write(f"  最小值: {np.min(all_samples_np):.6f}\n")
        f.write(f"  最大值: {np.max(all_samples_np):.6f}\n")
        f.write(f"  占有率 (>0.5): {np.mean(all_samples_np > 0.5):.6f}\n")
        
        # 计算每个样本的占有率分布
        occupancy_rates = []
        for sample in all_samples_np:
            occupancy = np.mean(sample.squeeze() > 0.5)
            occupancy_rates.append(occupancy)
        
        f.write(f"\n占有率分布:\n")
        f.write(f"  均值: {np.mean(occupancy_rates):.4f}\n")
        f.write(f"  标准差: {np.std(occupancy_rates):.4f}\n")
        f.write(f"  范围: [{np.min(occupancy_rates):.4f}, {np.max(occupancy_rates):.4f}]\n")
    
    logger.info(f"统计信息已保存到: {stats_path}")


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
        
        # 生成并保存样本（边生成边保存以节省显存）
        output_dir = Path(args.output_dir)
        samples = generate_and_save_samples_in_batches(
            model=model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            device=device,
            output_dir=output_dir,
            save_format=args.save_format,
            prefix=args.prefix,
            show_progress=args.show_progress,
            keep_in_memory=args.keep_samples_in_memory
        )
        
        # 计算统计信息
        compute_sample_statistics(samples)
        
        # 保存统计信息文件
        save_statistics_file(samples, output_dir, args.prefix, model.voxel_size)
        
        logger.info("样本生成完成!")
        
    except KeyboardInterrupt:
        logger.info("生成被用户中断")
    except Exception as e:
        logger.error(f"生成过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
