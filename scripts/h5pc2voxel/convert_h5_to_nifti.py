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
import tarfile
import matplotlib.pyplot as plt
import random

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


def validate_nifti_file(nifti_path: str) -> dict:
    """
    验证NIfTI文件的基本信息，帮助调试ITK-SNAP显示问题
    
    Args:
        nifti_path: NIfTI文件路径
        
    Returns:
        包含文件信息的字典
    """
    try:
        # 加载NIfTI文件
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        info = {
            'file_path': nifti_path,
            'shape': data.shape,
            'dtype': data.dtype,
            'data_range': (float(data.min()), float(data.max())),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'non_zero_count': int(np.count_nonzero(data)),
            'total_voxels': int(data.size),
            'affine_matrix': nifti_img.affine.tolist(),
            'header_info': {
                'pixdim': nifti_img.header['pixdim'].tolist(),
                'datatype': int(nifti_img.header['datatype']),
                'bitpix': int(nifti_img.header['bitpix'])
            }
        }
        
        logger.info(f"NIfTI文件验证: {nifti_path}")
        logger.info(f"  形状: {info['shape']}")
        logger.info(f"  数据类型: {info['dtype']}")
        logger.info(f"  数值范围: [{info['data_range'][0]:.2f}, {info['data_range'][1]:.2f}]")
        logger.info(f"  平均值: {info['mean']:.2f}, 标准差: {info['std']:.2f}")
        logger.info(f"  非零体素: {info['non_zero_count']}/{info['total_voxels']} ({100*info['non_zero_count']/info['total_voxels']:.1f}%)")
        
        return info
        
    except Exception as e:
        logger.error(f"验证NIfTI文件失败: {e}")
        return None


def visualize_voxel_slice(voxel_grid: np.ndarray, output_path: str, slice_axis: int = 2, slice_idx: int = None):
    """
    可视化体素网格的某一层并保存为PNG
    
    Args:
        voxel_grid: 体素网格数据
        output_path: 输出PNG文件路径
        slice_axis: 切片轴 (0=X轴, 1=Y轴, 2=Z轴)
        slice_idx: 切片索引，如果为None则选择中间层
    """
    # 移除通道维度（如果存在）
    if voxel_grid.ndim == 4 and voxel_grid.shape[0] == 1:
        voxel_grid = voxel_grid[0]
    
    # 选择切片索引
    if slice_idx is None:
        slice_idx = voxel_grid.shape[slice_axis] // 2
    
    # 提取切片
    if slice_axis == 0:
        slice_data = voxel_grid[slice_idx, :, :]
        title = f"X axis slice (index: {slice_idx})"
    elif slice_axis == 1:
        slice_data = voxel_grid[:, slice_idx, :]
        title = f"Y axis slice (index: {slice_idx})"
    else:  # slice_axis == 2
        slice_data = voxel_grid[:, :, slice_idx]
        title = f"Z axis slice (index: {slice_idx})"
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(slice_data, cmap='viridis', origin='lower')
    plt.colorbar(label='voxel value')
    plt.title(title)
    plt.xlabel('voxel coordinate')
    plt.ylabel('voxel coordinate')
    
    # 保存图像
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"切片可视化已保存到: {output_path}")


def create_dataset_archive(output_dir: str, archive_name: str = None) -> str:
    """
    将数据集打包成tar.gz文件
    
    Args:
        output_dir: 数据集目录
        archive_name: 压缩包名称，如果为None则自动生成
        
    Returns:
        压缩包文件路径
    """
    output_path = Path(output_dir)
    
    if archive_name is None:
        archive_name = f"{output_path.name}_dataset.tar.gz"
    
    archive_path = output_path.parent / archive_name
    
    logger.info(f"开始打包数据集: {output_dir}")
    logger.info(f"压缩包路径: {archive_path}")
    
    with tarfile.open(archive_path, 'w:gz') as tar:
        # 添加训练集
        train_dir = output_path / "train"
        if train_dir.exists():
            for file_path in train_dir.glob("*.nii*"):
                arcname = f"train/{file_path.name}"
                tar.add(file_path, arcname=arcname)
        
        # 添加验证集
        val_dir = output_path / "val"
        if val_dir.exists():
            for file_path in val_dir.glob("*.nii*"):
                arcname = f"val/{file_path.name}"
                tar.add(file_path, arcname=arcname)
        
        # 添加元数据文件
        metadata_file = output_path / "dataset_info.json"
        if metadata_file.exists():
            tar.add(metadata_file, arcname="dataset_info.json")
    
    # 计算压缩包大小
    archive_size = archive_path.stat().st_size / 1024 / 1024 / 1024  # GB
    logger.info(f"打包完成! 压缩包大小: {archive_size:.2f} GB")
    
    return str(archive_path)


def test_single_conversion(
    h5_file_path: str,
    output_dir: str,
    voxel_size,
    voxelization_method: str = 'gaussian',
    sigma: float = 1.0,
    data_key: str = 'point_clouds',
    volume_dims: list = None,
    padding: list = None,
    sample_idx: int = None,
    nii_dtype: str = 'uint8'
):
    """
    测试单个点云的转换过程
    
    Args:
        h5_file_path: 输入H5文件路径
        output_dir: 输出目录
        voxel_size: 体素网格分辨率
        voxelization_method: 体素化方法
        sigma: 高斯体素化的标准差
        data_key: H5文件中的数据键
        volume_dims: 体积尺寸
        padding: 体积边界填充
        sample_idx: 样本索引，如果为None则随机选择
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理voxel_size显示
    if isinstance(voxel_size, (list, tuple)):
        voxel_size_str = f"{voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}"
    else:
        voxel_size_str = f"{voxel_size}^3"
    
    logger.info(f"开始测试转换: {h5_file_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"体素大小: {voxel_size_str}")
    logger.info(f"体素化方法: {voxelization_method}")
    
    # 初始化加载器和转换器
    loader = PointCloudH5Loader(h5_file_path, data_key)
    converter = PointCloudToVoxel(
        voxel_size=voxel_size,
        method=voxelization_method,
        volume_dims=volume_dims,
        padding=padding
    )
    
    # 选择样本
    total_samples = loader.num_samples
    if sample_idx is None:
        sample_idx = random.randint(0, total_samples - 1)
    else:
        sample_idx = min(sample_idx, total_samples - 1)
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"选择样本索引: {sample_idx}")
    
    try:
        # 加载点云
        point_cloud = loader.load_single_cloud(sample_idx)
        logger.info(f"点云形状: {point_cloud.shape}")
        logger.info(f"点云范围: X[{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}], "
                   f"Y[{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}], "
                   f"Z[{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")
        
        # 转换为体素
        if voxelization_method == 'gaussian':
            voxel_grid = converter.convert(point_cloud, sigma=sigma)
        else:
            voxel_grid = converter.convert(point_cloud)
        
        logger.info(f"体素网格形状: {voxel_grid.shape}")
        logger.info(f"体素值范围: [{voxel_grid.min():.4f}, {voxel_grid.max():.4f}]")
        logger.info(f"非零体素数: {np.count_nonzero(voxel_grid)}")
        
        # 调试信息：转换前的数据统计
        logger.info(f"转换前体素值统计: min={voxel_grid.min():.6f}, max={voxel_grid.max():.6f}, "
                   f"mean={voxel_grid.mean():.6f}, std={voxel_grid.std():.6f}")
        
        # 归一化到[0, 255]范围，便于ITK-SNAP显示
        if voxel_grid.max() > voxel_grid.min():
            voxel_grid = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
            voxel_grid = voxel_grid * 255.0  # 缩放到0-255范围
        
        # 确保数据类型为uint8，这是ITK-SNAP推荐的格式
        voxel_grid = voxel_grid.astype(eval(f"np.{nii_dtype}"))
        
        # 调试信息：转换后的数据统计
        logger.info(f"转换后体素值统计: min={voxel_grid.min()}, max={voxel_grid.max()}, "
                   f"mean={voxel_grid.mean():.2f}, std={voxel_grid.std():.2f}")
        logger.info(f"最终数据类型: {voxel_grid.dtype}")
        logger.info(f"最终数据形状: {voxel_grid.shape}")
        
        # 不添加通道维度，保持3D格式 (D, H, W)
        # ITK-SNAP期望标准的3D医学图像格式
        
        # 创建仿射矩阵，确保正确的空间定向
        affine = create_affine_matrix(voxel_spacing=(1.0, 1.0, 1.0))
        
        # 创建NIfTI图像 - 使用3D数据而不是4D
        nifti_img = nib.Nifti1Image(voxel_grid, affine)
        
        # 保存NIfTI文件
        nifti_file = output_path / f"test_voxel_{sample_idx:05d}.nii.gz"
        nib.save(nifti_img, str(nifti_file))
        
        file_size = nifti_file.stat().st_size / 1024 / 1024  # MB
        logger.info(f"NIfTI文件已保存: {nifti_file} ({file_size:.2f} MB)")
        
        # 验证生成的NIfTI文件
        validate_nifti_file(str(nifti_file))
        
        # 生成可视化图像
        # Z轴中间层
        z_slice_path = output_path / f"test_voxel_{sample_idx:05d}_z_slice.png"
        visualize_voxel_slice(voxel_grid, str(z_slice_path), slice_axis=2)
        
        # Y轴中间层
        y_slice_path = output_path / f"test_voxel_{sample_idx:05d}_y_slice.png"
        visualize_voxel_slice(voxel_grid, str(y_slice_path), slice_axis=1)
        
        # X轴中间层
        x_slice_path = output_path / f"test_voxel_{sample_idx:05d}_x_slice.png"
        visualize_voxel_slice(voxel_grid, str(x_slice_path), slice_axis=0)
        
        logger.info("测试转换完成!")
        
    except Exception as e:
        logger.error(f"测试转换失败: {e}")
        raise


def convert_h5_to_nifti(
    h5_file_path: str,
    output_dir: str,
    voxel_size = 64,
    voxelization_method: str = 'gaussian',
    sigma: float = 1.0,
    train_ratio: float = 0.8,
    data_key: str = 'point_clouds',
    volume_dims: list = None,
    padding: list = None,
    compression: bool = True,
    nii_dtype: str = 'uint8'
):
    """
    将H5点云文件转换为NIfTI体素文件
    
    Args:
        h5_file_path: 输入H5文件路径
        output_dir: 输出目录
        voxel_size: 体素网格分辨率，可以是整数（各向同性）或[X, Y, Z]列表（各向异性）
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
    
    # 处理voxel_size，支持整数或列表
    if isinstance(voxel_size, (list, tuple)):
        voxel_size_str = f"{voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}"
    else:
        voxel_size_str = f"{voxel_size}^3"
    
    logger.info(f"开始转换: {h5_file_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"体素大小: {voxel_size_str}")
    logger.info(f"体素化方法: {voxelization_method}")
    
    # 初始化加载器和转换器
    loader = PointCloudH5Loader(h5_file_path, data_key)
    
    # if volume_dims is None:
    #     volume_dims = [20000, 20000, 2500]
    # if padding is None:
    #     padding = [0, 0, 100]
    
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
            
            # 归一化到[0, 255]范围，便于ITK-SNAP显示
            if voxel_grid.max() > voxel_grid.min():
                voxel_grid = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
                voxel_grid = voxel_grid * 255.0  # 缩放到0-255范围
            
            # 确保数据类型为uint8，这是ITK-SNAP推荐的格式
            voxel_grid = voxel_grid.astype(eval(f"np.{nii_dtype}"))
            
            # 不添加通道维度，保持3D格式 (D, H, W)
            # ITK-SNAP期望标准的3D医学图像格式
            
            # 创建NIfTI图像 - 使用3D数据而不是4D
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
    
    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(
        description="将H5点云文件转换为NIfTI体素数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整转换并打包
  python convert_h5_to_nifti.py --h5_file data.h5 --output_dir output --create_archive
  
  # 仅测试单个样本
  python convert_h5_to_nifti.py --h5_file data.h5 --output_dir test_output --test_mode
  
  # 测试指定样本
  python convert_h5_to_nifti.py --h5_file data.h5 --output_dir test_output --test_mode --sample_idx 10
        """
    )
    
    parser.add_argument(
        '--h5_file',
        type=str,
        default='/repos/datasets/batch_simulation_microtubule_20251017_2048.h5',
        help='输入H5文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/repos/datasets/batch_simulation_microtubule_20251017_2048_nifti',
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--voxel_size',
        type=int,
        nargs='+',
        default=[1024, 1024, 128],
        help='体素网格分辨率，可以是单个整数（如64）或三个整数（如64 64 32表示X=64, Y=64, Z=32）'
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
        default=0.95,
        help='训练集比例 (默认: 0.95)'
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
        default=[20000, 20000, 600],
        help='体积尺寸 [x y z] (nm)'
    )
    
    parser.add_argument(
        '--padding',
        type=float,
        nargs=3,
        default=[0, 0, 200],
        help='体积边界填充 [x y z] (nm)'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        default=False,
        help='测试模式：只转换一个随机样本并生成可视化'
    )
    
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=None,
        help='测试模式下指定的样本索引（如果不指定则随机选择）'
    )
    
    parser.add_argument(
        '--create_archive',
        action='store_true',
        default=True,
        help='转换完成后创建tar.gz压缩包'
    )
    
    parser.add_argument(
        '--archive_name',
        type=str,
        default=None,
        help='压缩包名称（如果不指定则自动生成）'
    )
    
    parser.add_argument(
        '--nii_dtype',
        type=str,
        default='uint8',
        choices=['uint8', 'float32'],
        help='NIfTI数据类型 (默认: uint8, 可选: uint8, float32)'
    )
    
    args = parser.parse_args()
    
    # 处理voxel_size参数
    if isinstance(args.voxel_size, list):
        if len(args.voxel_size) == 1:
            voxel_size = args.voxel_size[0]
        elif len(args.voxel_size) == 3:
            voxel_size = args.voxel_size
        else:
            raise ValueError(f"voxel_size必须是1个或3个整数，但得到了{len(args.voxel_size)}个")
    else:
        voxel_size = args.voxel_size
    
    # 检查文件是否存在
    if not Path(args.h5_file).exists():
        logger.error(f"H5文件不存在: {args.h5_file}")
        return
    
    if args.test_mode:
        # 测试模式：只转换一个样本
        logger.info("运行测试模式...")
        test_single_conversion(
            h5_file_path=args.h5_file,
            output_dir=args.output_dir,
            voxel_size=voxel_size,
            voxelization_method=args.method,
            sigma=args.sigma,
            data_key=args.data_key,
            volume_dims=args.volume_dims,
            padding=args.padding,
            sample_idx=args.sample_idx,
            nii_dtype=args.nii_dtype
        )
    else:
        # 完整转换模式
        logger.info("运行完整转换模式...")
        train_count, val_count = convert_h5_to_nifti(
            h5_file_path=args.h5_file,
            output_dir=args.output_dir,
            voxel_size=voxel_size,
            voxelization_method=args.method,
            sigma=args.sigma,
            train_ratio=args.train_ratio,
            data_key=args.data_key,
            volume_dims=args.volume_dims,
            padding=args.padding,
            compression=not args.no_compression,
            nii_dtype=args.nii_dtype
        )
        
        # 如果需要创建压缩包
        if args.create_archive:
            logger.info("开始创建数据集压缩包...")
            archive_path = create_dataset_archive(
                output_dir=args.output_dir,
                archive_name=args.archive_name
            )
            logger.info(f"数据集已打包完成: {archive_path}")
            logger.info("现在可以将压缩包上传到服务器进行训练!")


if __name__ == "__main__":
    main()

