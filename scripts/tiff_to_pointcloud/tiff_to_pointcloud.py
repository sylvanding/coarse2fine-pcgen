#!/usr/bin/env python3
"""
3D TIFF文件到点云采样脚本

此脚本用于读取3D TIFF体素数据，将其采样转换为点云格式，并保存为CSV文件。
采样方法参照test_conversion脚本，支持多种采样策略以避免网格化效应。

使用方法:
    python scripts/tiff_to_pointcloud.py --input voxel_data.tiff --output points.csv
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
import tifffile
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.voxel.converter import PointCloudToVoxel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TiffToPointCloud:
    """
    TIFF到点云转换器
    
    专门用于处理3D TIFF文件到点云的采样转换
    """
    
    def __init__(self, volume_dims: list = None, padding: list = None):
        """
        初始化转换器
        
        Args:
            volume_dims (list): 体积尺寸 [x, y, z] (单位: nm)
            padding (list): 体积边界填充 [x, y, z] (单位: nm)
        """
        if volume_dims is None:
            volume_dims = [20000, 20000, 2500]
        if padding is None:
            padding = [0, 0, 100]
            
        # 创建一个临时的PointCloudToVoxel实例来使用其采样功能
        self.converter = PointCloudToVoxel(
            voxel_size=64,  # 临时值，会根据实际TIFF尺寸调整
            method='gaussian',
            volume_dims=volume_dims,
            padding=padding
        )
        
        # 保存体积参数
        self.volume_dims = np.array(volume_dims, dtype=np.float32)
        self.padding = np.array(padding, dtype=np.float32)
    
    def load_tiff(self, tiff_path: str) -> np.ndarray:
        """
        加载3D TIFF文件
        
        Args:
            tiff_path (str): TIFF文件路径
            
        Returns:
            np.ndarray: 3D体素网格数据
        """
        try:
            logger.info(f"正在加载TIFF文件: {tiff_path}")
            voxel_data = tifffile.imread(tiff_path)
            
            # 确保是3D数据
            if voxel_data.ndim != 3:
                raise ValueError(f"TIFF文件必须是3D数据，但得到了{voxel_data.ndim}D数据")
            
            logger.info(f"TIFF数据shape: {voxel_data.shape}")
            logger.info(f"数据类型: {voxel_data.dtype}")
            logger.info(f"数值范围: [{np.min(voxel_data)}, {np.max(voxel_data)}]")
            
            return voxel_data
            
        except Exception as e:
            logger.error(f"加载TIFF文件失败: {e}")
            raise
    
    def preprocess_voxel_data(self, voxel_data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        预处理体素数据
        
        Args:
            voxel_data (np.ndarray): 原始体素数据
            normalize (bool): 是否归一化到[0, 1]范围
            
        Returns:
            np.ndarray: 预处理后的体素数据
        """
        logger.info("开始预处理体素数据")
        
        # 转换为浮点数
        if voxel_data.dtype in [np.uint8, np.uint16, np.uint32]:
            voxel_data = voxel_data.astype(np.float32)
        
        # 归一化到[0, 1]范围
        if normalize:
            min_val = np.min(voxel_data)
            max_val = np.max(voxel_data)
            
            if max_val > min_val:
                voxel_data = (voxel_data - min_val) / (max_val - min_val)
                logger.info(f"数据已归一化到[0, 1]范围")
            else:
                logger.warning("数据的最大值等于最小值，跳过归一化")
        
        return voxel_data
    
    def sample_to_pointcloud(self, voxel_data: np.ndarray, threshold: float = 0.0,
                           num_points: int = None, method: str = 'probabilistic') -> np.ndarray:
        """
        将体素数据采样为点云
        
        Args:
            voxel_data (np.ndarray): 体素网格数据
            threshold (float): 体素值阈值
            num_points (int): 目标点数，None时自动确定
            method (str): 采样方法
            
        Returns:
            np.ndarray: 采样得到的点云，shape为(N, 3)
        """
        logger.info(f"开始体素到点云采样 (方法: {method}, 阈值: {threshold})")
        
        # 更新转换器的体素尺寸以匹配TIFF数据
        self.converter.voxel_size = max(voxel_data.shape)
        
        # 设置边界信息以正确反归一化坐标
        self.converter._last_min_bounds = -self.padding
        self.converter._last_max_bounds = self.volume_dims + self.padding
        
        # 使用转换器的采样方法
        point_cloud = self.converter.voxel_to_points(
            voxel_data,
            threshold=threshold,
            num_points=num_points,
            method=method
        )
        
        logger.info(f"采样完成，生成 {len(point_cloud)} 个点")
        return point_cloud
    
    def save_pointcloud(self, point_cloud: np.ndarray, output_path: str) -> None:
        """
        保存点云为CSV文件
        
        Args:
            point_cloud (np.ndarray): 点云数据
            output_path (str): 输出文件路径
        """
        # 确保输出路径是CSV格式
        if not output_path.lower().endswith('.csv'):
            output_path = output_path.rsplit('.', 1)[0] + '.csv'
        
        # 使用指定的表头格式保存CSV文件
        np.savetxt(output_path, point_cloud, fmt='%.6f', delimiter=',', 
                  header='x [nm],y [nm],z [nm]', comments='')
        
        logger.info(f"点云已保存为CSV文件: {output_path}")
        logger.info(f"保存的点云包含 {len(point_cloud)} 个点")


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='将3D TIFF体素数据采样为点云格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基本转换（默认概率采样）
    python scripts/tiff_to_pointcloud.py --input voxel_data.tiff --output points.csv
    
    # 指定采样点数和阈值
    python scripts/tiff_to_pointcloud.py --input voxel_data.tiff --output points.csv \\
        --num-points 100000 --threshold 0.1
    
    # 使用不同的采样方法
    python scripts/tiff_to_pointcloud.py --input voxel_data.tiff --output points.csv \\
        --method center --num-points 50000
    
    # 自定义体积参数
    python scripts/tiff_to_pointcloud.py --input voxel_data.tiff --output points.csv \\
        --volume-dims 15000 15000 3000 --padding 50 50 150
    
    # 批量处理（使用通配符）
    python scripts/tiff_to_pointcloud.py --input "data/*.tiff" --output-dir results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/repos/datasets/generated_sample_01.tiff',
        # required=True,
        help='输入TIFF文件路径（支持通配符模式如"*.tiff"）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/repos/datasets/generated_sample_01_points.csv',
        help='输出CSV文件路径（单文件模式时必需）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录（批量处理模式时必需）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='体素值阈值，低于此值的体素不参与采样 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=100000,
        help='目标采样点数，None时自动根据体素密度确定 (默认: 100000)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['probabilistic', 'center', 'random', 'weighted'],
        default='probabilistic',
        help='采样方法 (默认: probabilistic - 概率分布采样，避免网格化效应)'
    )
    
    parser.add_argument(
        '--volume-dims',
        type=float,
        nargs=3,
        default=[20000, 20000, 2500],
        help='体积尺寸 [x, y, z] (单位: nm) (默认: [20000, 20000, 2500])'
    )
    
    parser.add_argument(
        '--padding',
        type=float,
        nargs=3,
        default=[0, 0, 100],
        help='体积边界填充 [x, y, z] (单位: nm) (默认: [0, 0, 100])'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='是否将体素值归一化到[0, 1]范围'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='显示详细信息'
    )
    
    return parser.parse_args()


def analyze_voxel_grid(voxel_grid: np.ndarray) -> dict:
    """
    分析体素网格的基本统计信息
    
    Args:
        voxel_grid (np.ndarray): 体素网格
    
    Returns:
        dict: 包含统计信息的字典
    """
    stats = {
        'shape': voxel_grid.shape,
        'total_voxels': voxel_grid.size,
        'occupied_voxels': np.sum(voxel_grid > 0),
        'occupancy_ratio': np.sum(voxel_grid > 0) / voxel_grid.size,
        'min_value': np.min(voxel_grid),
        'max_value': np.max(voxel_grid),
        'mean_value': np.mean(voxel_grid),
        'std_value': np.std(voxel_grid),
        'non_zero_mean': np.mean(voxel_grid[voxel_grid > 0]) if np.any(voxel_grid > 0) else 0.0
    }
    
    return stats


def analyze_point_cloud(point_cloud: np.ndarray) -> dict:
    """
    分析点云的基本统计信息
    
    Args:
        point_cloud (np.ndarray): 点云数据
    
    Returns:
        dict: 包含统计信息的字典
    """
    if len(point_cloud) == 0:
        return {
            'num_points': 0,
            'min_coords': np.array([0, 0, 0]),
            'max_coords': np.array([0, 0, 0]),
            'mean_coords': np.array([0, 0, 0]),
            'std_coords': np.array([0, 0, 0]),
            'range_coords': np.array([0, 0, 0])
        }
    
    stats = {
        'num_points': len(point_cloud),
        'min_coords': np.min(point_cloud, axis=0),
        'max_coords': np.max(point_cloud, axis=0),
        'mean_coords': np.mean(point_cloud, axis=0),
        'std_coords': np.std(point_cloud, axis=0),
        'range_coords': np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0)
    }
    
    return stats


def print_statistics(voxel_stats: dict, point_stats: dict):
    """
    打印统计信息
    
    Args:
        voxel_stats (dict): 体素统计信息
        point_stats (dict): 点云统计信息
    """
    print("\n" + "="*60)
    print("体素网格统计信息:")
    print("="*60)
    print(f"网格shape: {voxel_stats['shape']}")
    print(f"总体素数: {voxel_stats['total_voxels']:,}")
    print(f"占有体素数: {voxel_stats['occupied_voxels']:,}")
    print(f"占有率: {voxel_stats['occupancy_ratio']:.4f}")
    print(f"体素值范围: [{voxel_stats['min_value']:.3f}, {voxel_stats['max_value']:.3f}]")
    print(f"平均体素值: {voxel_stats['mean_value']:.3f}")
    print(f"非零体素平均值: {voxel_stats['non_zero_mean']:.3f}")
    print(f"体素值标准差: {voxel_stats['std_value']:.3f}")
    
    print("\n" + "="*60)
    print("采样点云统计信息:")
    print("="*60)
    print(f"点数: {point_stats['num_points']:,}")
    if point_stats['num_points'] > 0:
        print("坐标范围:")
        print(f"  X: [{point_stats['min_coords'][0]:.3f}, {point_stats['max_coords'][0]:.3f}] (范围: {point_stats['range_coords'][0]:.3f})")
        print(f"  Y: [{point_stats['min_coords'][1]:.3f}, {point_stats['max_coords'][1]:.3f}] (范围: {point_stats['range_coords'][1]:.3f})")
        print(f"  Z: [{point_stats['min_coords'][2]:.3f}, {point_stats['max_coords'][2]:.3f}] (范围: {point_stats['range_coords'][2]:.3f})")
        print(f"平均坐标: ({point_stats['mean_coords'][0]:.3f}, {point_stats['mean_coords'][1]:.3f}, {point_stats['mean_coords'][2]:.3f})")
    
    print("="*60)


def validate_inputs(args):
    """
    验证输入参数
    
    Args:
        args: 命令行参数
    
    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 参数值无效
    """
    # 检查输入模式
    if '*' in args.input or '?' in args.input:
        # 批量处理模式
        if not args.output_dir:
            raise ValueError("批量处理模式需要指定 --output-dir 参数")
        if not os.path.exists(args.output_dir):
            logger.info(f"创建输出目录: {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)
    else:
        # 单文件模式
        if not args.output:
            raise ValueError("单文件模式需要指定 --output 参数")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input}")
        
        # 检查输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"创建输出目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
    
    # 验证参数范围
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("threshold必须在[0, 1]范围内")
    
    if args.num_points is not None and args.num_points <= 0:
        raise ValueError("num_points必须大于0")


def process_single_file(input_path: str, output_path: str, args):
    """
    处理单个TIFF文件
    
    Args:
        input_path (str): 输入TIFF文件路径
        output_path (str): 输出CSV文件路径
        args: 命令行参数
    """
    try:
        # 1. 创建转换器
        converter = TiffToPointCloud(
            volume_dims=args.volume_dims,
            padding=args.padding
        )
        
        # 2. 加载TIFF文件
        voxel_data = converter.load_tiff(input_path)
        
        # 3. 预处理数据
        voxel_data = converter.preprocess_voxel_data(voxel_data, normalize=args.normalize)
        
        # 4. 分析体素网格
        voxel_stats = analyze_voxel_grid(voxel_data)
        
        # 5. 采样为点云
        point_cloud = converter.sample_to_pointcloud(
            voxel_data,
            threshold=args.threshold,
            num_points=args.num_points,
            method=args.method
        )
        
        # 6. 分析点云
        point_stats = analyze_point_cloud(point_cloud)
        
        # 7. 保存点云
        if len(point_cloud) > 0:
            converter.save_pointcloud(point_cloud, output_path)
        else:
            logger.warning("采样得到的点云为空，跳过保存")
            return False
        
        # 8. 保存统计信息
        info_file = output_path.replace('.csv', '_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("TIFF到点云转换信息\n")
            f.write("="*50 + "\n\n")
            f.write(f"输入文件: {input_path}\n")
            f.write(f"输出文件: {output_path}\n\n")
            
            f.write("转换参数:\n")
            f.write(f"  threshold: {args.threshold}\n")
            f.write(f"  num_points: {args.num_points}\n")
            f.write(f"  method: {args.method}\n")
            f.write(f"  volume_dims: {args.volume_dims}\n")
            f.write(f"  padding: {args.padding}\n")
            f.write(f"  normalize: {args.normalize}\n\n")
            
            f.write("体素网格统计:\n")
            for key, value in voxel_stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n点云统计:\n")
            for key, value in point_stats.items():
                f.write(f"  {key}: {value}\n")
        
        # 9. 显示统计信息
        if args.verbose:
            print_statistics(voxel_stats, point_stats)
        
        # 10. 输出总结
        print(f"\n✅ 转换完成!")
        print(f"📦 输入体素网格: {voxel_stats['shape']}")
        print(f"📊 采样点云: {point_stats['num_points']:,} 个点")
        print(f"💾 CSV文件: {output_path}")
        print(f"📋 信息文件: {info_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_path} 时发生错误: {e}")
        raise


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志级别
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 验证输入
        validate_inputs(args)
        
        # 判断处理模式
        if '*' in args.input or '?' in args.input:
            # 批量处理模式
            import glob
            input_files = glob.glob(args.input)
            
            if not input_files:
                logger.error(f"没有找到匹配的文件: {args.input}")
                sys.exit(1)
            
            logger.info(f"找到 {len(input_files)} 个文件进行批量处理")
            
            success_count = 0
            for input_file in input_files:
                try:
                    # 生成输出文件名
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    output_file = os.path.join(args.output_dir, f"{base_name}_points.csv")
                    
                    logger.info(f"\n处理文件: {input_file}")
                    if process_single_file(input_file, output_file, args):
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"处理文件 {input_file} 失败: {e}")
                    continue
            
            print(f"\n批量处理完成: {success_count}/{len(input_files)} 个文件成功转换")
            
        else:
            # 单文件处理模式
            process_single_file(args.input, args.output, args)
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
