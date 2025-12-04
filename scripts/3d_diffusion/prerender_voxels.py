#!/usr/bin/env python
"""
预渲染体素脚本

将点云数据提前转换为体素并保存到H5文件，用于加速训练。
支持批量处理、精度控制、阈值过滤等功能。
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np
import h5py
from tqdm import tqdm
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.h5_loader import PointCloudH5Loader
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
        description="预渲染点云为体素数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出
    parser.add_argument(
        "--input-h5",
        type=str,
        required=True,
        help="输入点云H5文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--data-key",
        type=str,
        default="pointclouds",
        help="H5文件中的数据键名"
    )
    
    # 体素化参数
    parser.add_argument(
        "--voxel-size",
        type=int,
        default=128,
        help="体素网格分辨率"
    )
    parser.add_argument(
        "--voxelization-method",
        type=str,
        default="gaussian",
        choices=["occupancy", "density", "gaussian"],
        help="体素化方法"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="高斯体素化的标准差"
    )
    
    # 体积参数
    parser.add_argument(
        "--volume-dims",
        type=float,
        nargs=3,
        default=[28000.0, 28000.0, 1600.0],
        help="体积尺寸 [x, y, z] (nm)"
    )
    parser.add_argument(
        "--padding",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="边界填充 [x, y, z] (nm)"
    )
    
    # 处理参数
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试，None表示处理所有）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="每个输出H5文件包含的体素数量"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="体素值过滤阈值（低于此值的设为0）"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="保留的小数位数"
    )
    
    # 其他选项
    parser.add_argument(
        "--config",
        type=str,
        help="从配置文件读取参数（YAML格式）"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["none", "gzip", "lzf"],
        help="H5文件压缩方式"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """从配置文件加载参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def apply_threshold_and_precision(voxel_grid: np.ndarray, 
                                   threshold: float, 
                                   precision: int) -> np.ndarray:
    """
    应用阈值过滤和精度控制
    
    Args:
        voxel_grid: 体素网格
        threshold: 阈值（低于此值的设为0）
        precision: 保留的小数位数
        
    Returns:
        处理后的体素网格
    """
    # 应用阈值
    voxel_grid[voxel_grid < threshold] = 0.0
    
    # 控制精度
    voxel_grid = np.round(voxel_grid, decimals=precision)
    
    return voxel_grid


def prerender_voxels(
    input_h5_path: str,
    output_dir: str,
    data_key: str,
    voxel_size: int,
    voxelization_method: str,
    sigma: float,
    volume_dims: list,
    padding: list,
    max_samples: int,
    batch_size: int,
    threshold: float,
    precision: int,
    compression: str,
    overwrite: bool
):
    """
    预渲染体素主函数
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("预渲染体素配置:")
    logger.info(f"  输入文件: {input_h5_path}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  体素大小: {voxel_size}^3")
    logger.info(f"  体素化方法: {voxelization_method}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  过滤阈值: {threshold}")
    logger.info(f"  精度位数: {precision}")
    logger.info(f"  压缩方式: {compression}")
    logger.info("="*60)
    
    # 加载点云数据
    logger.info("加载点云数据...")
    loader = PointCloudH5Loader(input_h5_path, data_key)
    total_samples = loader.num_samples
    
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
        logger.info(f"限制处理样本数: {total_samples}")
    
    # 初始化体素转换器
    logger.info("初始化体素转换器...")
    converter = PointCloudToVoxel(
        voxel_size=voxel_size,
        method=voxelization_method,
        volume_dims=volume_dims,
        padding=padding
    )
    
    # 保存元数据
    metadata = {
        'source_h5': input_h5_path,
        'data_key': data_key,
        'voxel_size': voxel_size,
        'voxelization_method': voxelization_method,
        'sigma': sigma,
        'volume_dims': volume_dims,
        'padding': padding,
        'threshold': threshold,
        'precision': precision,
        'total_samples': total_samples,
        'batch_size': batch_size
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"已保存元数据: {metadata_path}")
    
    # 计算需要的批次数
    num_batches = (total_samples + batch_size - 1) // batch_size
    logger.info(f"总样本数: {total_samples}, 批次数: {num_batches}")
    
    # 设置压缩选项
    compression_opts = None
    if compression == "gzip":
        compression_opts = 9  # 最高压缩率
    elif compression == "none":
        compression = None
    
    # 分批处理
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_samples = end_idx - start_idx
        
        output_file = output_dir / f"voxels_batch_{batch_idx:04d}.h5"
        
        # 检查是否已存在
        if output_file.exists() and not overwrite:
            logger.info(f"批次 {batch_idx+1}/{num_batches}: 文件已存在，跳过")
            continue
        
        logger.info(f"处理批次 {batch_idx+1}/{num_batches}: 样本 [{start_idx}, {end_idx})")
        
        # 预分配数组
        voxel_batch = np.zeros(
            (batch_samples, voxel_size, voxel_size, voxel_size),
            dtype=np.float32
        )
        
        # 处理每个样本
        valid_count = 0
        for i, sample_idx in enumerate(tqdm(
            range(start_idx, end_idx),
            desc=f"批次{batch_idx+1}",
            ncols=80
        )):
            try:
                # 加载点云
                point_cloud = loader.load_single_cloud(sample_idx)
                
                # 转换为体素
                if voxelization_method == 'gaussian':
                    voxel_grid = converter.convert(point_cloud, sigma=sigma)
                else:
                    voxel_grid = converter.convert(point_cloud)
                
                # 应用阈值和精度控制
                voxel_grid = apply_threshold_and_precision(
                    voxel_grid, threshold, precision
                )
                
                # 存储
                voxel_batch[i] = voxel_grid.astype(np.float32)
                valid_count += 1
                
            except Exception as e:
                logger.warning(f"处理样本 {sample_idx} 失败: {e}")
                # 保留零数组
        
        # 保存批次
        logger.info(f"保存批次 {batch_idx+1}/{num_batches} 到 {output_file}")
        logger.info(f"  有效样本: {valid_count}/{batch_samples}")
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(
                'voxels',
                data=voxel_batch,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(1, voxel_size, voxel_size, voxel_size)
            )
            
            # 保存索引映射
            f.create_dataset(
                'indices',
                data=np.arange(start_idx, end_idx, dtype=np.int32)
            )
            
            # 保存元数据
            for key, value in metadata.items():
                f.attrs[key] = str(value)
        
        # 显示文件大小
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  文件大小: {file_size_mb:.2f} MB")
        
        # 显示稀疏度
        sparsity = np.sum(voxel_batch == 0) / voxel_batch.size * 100
        logger.info(f"  稀疏度: {sparsity:.2f}%")
    
    logger.info("="*60)
    logger.info("预渲染完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"批次文件数: {num_batches}")
    logger.info("="*60)


def main():
    """主函数"""
    args = parse_args()
    
    # 从配置文件加载参数（如果提供）
    if args.config:
        logger.info(f"从配置文件加载参数: {args.config}")
        config = load_config_from_file(args.config)
        
        # 应用配置文件中的参数（命令行参数优先）
        if 'prerender' in config:
            prerender_config = config['prerender']
            for key, value in prerender_config.items():
                # 将配置文件中的参数映射到命令行参数
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        
        # 应用data配置
        if 'data' in config:
            data_config = config['data']
            if not args.input_h5 and 'h5_file_path' in data_config:
                args.input_h5 = data_config['h5_file_path']
            if 'data_key' in data_config:
                args.data_key = data_config['data_key']
            if 'voxel_size' in data_config:
                args.voxel_size = data_config['voxel_size']
            if 'voxelization_method' in data_config:
                args.voxelization_method = data_config['voxelization_method']
            if 'sigma' in data_config:
                args.sigma = data_config['sigma']
            if 'volume_dims' in data_config:
                args.volume_dims = data_config['volume_dims']
            if 'padding' in data_config:
                args.padding = data_config['padding']
    
    # 验证输入文件
    if not Path(args.input_h5).exists():
        logger.error(f"输入文件不存在: {args.input_h5}")
        sys.exit(1)
    
    # 执行预渲染
    try:
        prerender_voxels(
            input_h5_path=args.input_h5,
            output_dir=args.output_dir,
            data_key=args.data_key,
            voxel_size=args.voxel_size,
            voxelization_method=args.voxelization_method,
            sigma=args.sigma,
            volume_dims=args.volume_dims,
            padding=args.padding,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            threshold=args.threshold,
            precision=args.precision,
            compression=args.compression,
            overwrite=args.overwrite
        )
    except Exception as e:
        logger.error(f"预渲染过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()

