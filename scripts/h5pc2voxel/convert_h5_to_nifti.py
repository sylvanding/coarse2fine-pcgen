"""
H5点云转NIfTI体素数据转换脚本

将H5格式的点云数据转换为NIfTI格式的体素数据，用于MONAI训练。
"""

import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import numpy as np
import nibabel as nib

# 添加项目根目录到路径
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


def create_affine_matrix(voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    创建NIfTI仿射变换矩阵
    
    Args:
        voxel_spacing: 体素间距 (x, y, z)
        
    Returns:
        4x4仿射矩阵
    """
    affine = np.eye(4)
    affine[0, 0] = voxel_spacing[0]
    affine[1, 1] = voxel_spacing[1]
    affine[2, 2] = voxel_spacing[2]
    return affine


def convert_h5_to_nifti(
    h5_file_path: str,
    output_dir: str,
    voxel_size: int = 64,
    voxelization_method: str = 'gaussian',
    sigma: float = 1.0,
    train_ratio: float = 0.8,
    data_key: str = 'point_clouds',
    volume_dims: list = None,
    padding: list = None,
    compression: bool = True
):
    """
    将H5点云文件转换为NIfTI体素文件
    
    Args:
        h5_file_path: 输入H5文件路径
        output_dir: 输出目录
        voxel_size: 体素网格分辨率
        voxelization_method: 体素化方法
        sigma: 高斯体素化的标准差
        train_ratio: 训练集比例
        data_key: H5文件中的数据键
        volume_dims: 体积尺寸 [x, y, z] (nm)
        padding: 体积边界填充 [x, y, z] (nm)
        compression: 是否使用压缩 (.nii.gz)
    """
    # 创建输出目录
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始转换: {h5_file_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"体素大小: {voxel_size}^3")
    logger.info(f"体素化方法: {voxelization_method}")
    
    # 初始化加载器和转换器
    loader = PointCloudH5Loader(h5_file_path, data_key)
    
    if volume_dims is None:
        volume_dims = [20000, 20000, 2500]
    if padding is None:
        padding = [0, 0, 100]
    
    converter = PointCloudToVoxel(
        voxel_size=voxel_size,
        method=voxelization_method,
        volume_dims=volume_dims,
        padding=padding
    )
    
    # 生成训练/验证集划分索引
    total_samples = loader.num_samples
    indices = np.arange(total_samples)
    np.random.seed(42)  # 固定随机种子以保证可重复性
    np.random.shuffle(indices)
    
    train_size = int(total_samples * train_ratio)
    train_indices = set(indices[:train_size].tolist())
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"训练集: {train_size}, 验证集: {total_samples - train_size}")
    
    # 创建仿射矩阵
    affine = create_affine_matrix(voxel_spacing=(1.0, 1.0, 1.0))
    
    # 转换所有样本
    extension = ".nii.gz" if compression else ".nii"
    
    train_count = 0
    val_count = 0
    
    for i in tqdm(range(total_samples), desc="转换点云"):
        try:
            # 加载点云
            point_cloud = loader.load_single_cloud(i)
            
            # 转换为体素
            if voxelization_method == 'gaussian':
                voxel_grid = converter.convert(point_cloud, sigma=sigma)
            else:
                voxel_grid = converter.convert(point_cloud)
            
            # 归一化到[0, 1]范围
            if voxel_grid.max() > voxel_grid.min():
                voxel_grid = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
            
            # 确保数据类型为float32
            voxel_grid = voxel_grid.astype(np.float32)
            
            # 添加通道维度 (C, D, H, W)
            voxel_grid = voxel_grid[np.newaxis, ...]
            
            # 创建NIfTI图像
            nifti_img = nib.Nifti1Image(voxel_grid, affine)
            
            # 保存到训练集或验证集
            if i in train_indices:
                output_file = train_dir / f"voxel_{i:05d}{extension}"
                train_count += 1
            else:
                output_file = val_dir / f"voxel_{i:05d}{extension}"
                val_count += 1
            
            nib.save(nifti_img, str(output_file))
            
        except Exception as e:
            logger.error(f"处理样本 {i} 时出错: {e}")
            continue
    
    logger.info(f"转换完成!")
    logger.info(f"训练集样本数: {train_count}")
    logger.info(f"验证集样本数: {val_count}")
    
    # 计算并报告文件大小统计
    train_files = list(train_dir.glob(f"*{extension}"))
    if train_files:
        avg_size = np.mean([f.stat().st_size for f in train_files[:10]]) / 1024 / 1024
        logger.info(f"平均文件大小: {avg_size:.2f} MB")
    
    # 保存数据集元信息
    metadata = {
        'h5_file': h5_file_path,
        'total_samples': total_samples,
        'train_samples': train_count,
        'val_samples': val_count,
        'voxel_size': voxel_size,
        'voxelization_method': voxelization_method,
        'sigma': sigma if voxelization_method == 'gaussian' else None,
        'volume_dims': volume_dims,
        'padding': padding,
        'compression': compression
    }
    
    import json
    metadata_file = output_path / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"数据集信息已保存到: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="将H5点云文件转换为NIfTI体素数据"
    )
    
    parser.add_argument(
        '--h5_file',
        type=str,
        required=True,
        help='输入H5文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--voxel_size',
        type=int,
        default=64,
        help='体素网格分辨率 (默认: 64)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='gaussian',
        choices=['occupancy', 'density', 'gaussian'],
        help='体素化方法 (默认: gaussian)'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='高斯体素化的标准差 (默认: 1.0)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )
    
    parser.add_argument(
        '--data_key',
        type=str,
        default='point_clouds',
        help='H5文件中的数据键 (默认: point_clouds)'
    )
    
    parser.add_argument(
        '--no_compression',
        action='store_true',
        help='不使用压缩 (.nii而不是.nii.gz)'
    )
    
    parser.add_argument(
        '--volume_dims',
        type=float,
        nargs=3,
        default=None,
        help='体积尺寸 [x y z] (nm)'
    )
    
    parser.add_argument(
        '--padding',
        type=float,
        nargs=3,
        default=None,
        help='体积边界填充 [x y z] (nm)'
    )
    
    args = parser.parse_args()
    
    convert_h5_to_nifti(
        h5_file_path=args.h5_file,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        voxelization_method=args.method,
        sigma=args.sigma,
        train_ratio=args.train_ratio,
        data_key=args.data_key,
        volume_dims=args.volume_dims,
        padding=args.padding,
        compression=not args.no_compression
    )


if __name__ == "__main__":
    main()

