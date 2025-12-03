#!/usr/bin/env python3
"""
点云到体素和体素到点云转换测试脚本

此脚本用于测试点云到体素和体素到点云的转换功能，加载H5文件中的第一个点云样本，
将其转换为3D体素网格，并保存为TIFF文件以便查看转换效果。随后，将体素网格采样回点云，
并保存为新的点云文件。

使用方法:
    python scripts/test_conversion.py --input data/sample.h5 --output test_voxel.tiff
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.data.h5_loader import PointCloudH5Loader
from src.voxel.converter import PointCloudToVoxel

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='测试点云到体素的转换功能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基本转换（占有网格）
    python scripts/test_conversion.py --input data/sample.h5 --output test_occupancy.tiff
    
    # 密度网格转换
    python scripts/test_conversion.py --input data/sample.h5 --output test_density.tiff --method density --voxel-size 128
    
    # 高斯密度网格转换
    python scripts/test_conversion.py --input data/sample.h5 --output test_gaussian.tiff --method gaussian --sigma 1.5
    
    # 体素网格上采样
    python scripts/test_conversion.py --input data/sample.h5 --output test_upsampled.tiff --upsample --upsample-factor 2.0 --upsample-method linear
    
    # 体素采样回点云（新的概率分布方法）
    python scripts/test_conversion.py --input data/sample.h5 --output test_sampled.tiff --sample-back --sample-num-points 100000 --sample-method probabilistic
    
    # 完整流程：转换→上采样→采样回点云（推荐使用概率采样）
    python scripts/test_conversion.py --input data/sample.h5 --output test_full.tiff --method gaussian --sigma 1.5 --upsample --upsample-factor 2.0 --sample-back --sample-num-points 200000 --sample-method probabilistic
    
    # 自定义体积参数转换
    python scripts/test_conversion.py --input data/sample.h5 --output test_custom.tiff --volume-dims 15000 15000 3000 --padding 50 50 150
    
    # 保存原始点云
    python scripts/test_conversion.py --input data/sample.h5 --output test_voxel.tiff --save-original --original-output original_pointcloud.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='/repos/datasets/tissue-datasets/pointclouds-clean-check.h5',
        type=str,
        help='输入H5文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='/repos/datasets/tissue-datasets/superresolution_results/test_voxel.tiff',
        type=str,
        help='输出TIFF文件路径'
    )
    
    parser.add_argument(
        '--save-original',
        default=True,
        action='store_true',
        help='是否保存原始点云为CSV文件'
    )
    
    parser.add_argument(
        '--original-output',
        type=str,
        default=None,
        help='原始点云输出文件路径（默认基于--output参数生成）'
    )
    
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='要转换的点云样本索引 (默认: 0)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['occupancy', 'density', 'gaussian'],
        default='gaussian',
        help='体素化方法 (默认: occupancy)'
    )
    
    parser.add_argument(
        '--voxel-size',
        type=int,
        default=128,
        help='体素网格分辨率 (默认: 64)'
    )
    
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='高斯方法的标准差 (默认: 1.5)'
    )
    
    parser.add_argument(
        '--data-key',
        type=str,
        default='pointclouds',
        help='H5文件中数据的键名 (默认: data)'
    )
    
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.00,
        help='边界扩展比例 (默认: 0.00)'
    )
    
    parser.add_argument(
        '--volume-dims',
        type=float,
        nargs=3,
        default=[28000.0, 28000.0, 1600.0],
        help='体积尺寸 [x, y, z] (单位: nm) (默认: [20000, 20000, 2500])'
    )
    
    parser.add_argument(
        '--padding',
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help='体积边界填充 [x, y, z] (单位: nm) (默认: [0, 0, 100])'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        default=True,
        action='store_true',
        help='显示详细信息'
    )
    
    # 体素采样回点云参数
    parser.add_argument(
        '--sample-back',
        default=True,
        action='store_true',
        help='是否将体素网格采样回点云'
    )
    
    parser.add_argument(
        '--sample-num-points',
        type=int,
        default=int(1e5),
        help='采样回点云的目标点数，None时返回所有超过阈值的体素'
    )
    
    parser.add_argument(
        '--sample-threshold',
        type=float,
        default=0.0,
        help='体素值阈值，高于此值的体素被视为包含点 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--sample-method',
        type=str,
        choices=['probabilistic', 'center', 'random', 'weighted'],
        default='probabilistic',
        help='采样方法 (默认: probabilistic - 概率分布采样，避免网格化效应)'
    )
    
    # 体素上采样参数
    parser.add_argument(
        '--upsample',
        default=False,
        action='store_true',
        help='是否对体素网格进行上采样'
    )
    
    parser.add_argument(
        '--upsample-factor',
        type=float,
        default=2.0,
        help='上采样倍数 (默认: 2.0)'
    )
    
    parser.add_argument(
        '--upsample-method',
        type=str,
        choices=['linear', 'nearest', 'cubic'],
        default='linear',
        help='上采样插值方法 (默认: linear)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """
    验证输入参数
    
    Args:
        args: 命令行参数
    
    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 参数值无效
    """
    # 检查输入文件
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 检查输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # 验证参数范围
    if args.voxel_size <= 0:
        raise ValueError("体素大小必须大于0")
    
    if args.index < 0:
        raise ValueError("样本索引必须非负")
    
    if not 0 <= args.padding_ratio <= 1:
        raise ValueError("padding_ratio必须在[0, 1]范围内")
    
    if args.sigma <= 0:
        raise ValueError("sigma必须大于0")
    
    # 验证采样参数
    if args.sample_back:
        if args.sample_threshold < 0 or args.sample_threshold > 1:
            raise ValueError("sample_threshold必须在[0, 1]范围内")
        if args.sample_num_points is not None and args.sample_num_points <= 0:
            raise ValueError("sample_num_points必须大于0")
    
    # 验证上采样参数
    if args.upsample:
        if args.upsample_factor <= 1.0:
            raise ValueError("upsample_factor必须大于1.0")


def analyze_point_cloud(point_cloud: np.ndarray) -> dict:
    """
    分析点云的基本统计信息
    
    Args:
        point_cloud (np.ndarray): 点云数据
    
    Returns:
        dict: 包含统计信息的字典
    """
    stats = {
        'num_points': len(point_cloud),
        'min_coords': np.min(point_cloud, axis=0),
        'max_coords': np.max(point_cloud, axis=0),
        'mean_coords': np.mean(point_cloud, axis=0),
        'std_coords': np.std(point_cloud, axis=0),
        'range_coords': np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0)
    }
    
    return stats


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
        'std_value': np.std(voxel_grid)
    }
    
    return stats


def compare_point_clouds(original_stats: dict, sampled_stats: dict) -> dict:
    """
    比较原始点云和采样点云的统计信息
    
    Args:
        original_stats (dict): 原始点云统计信息
        sampled_stats (dict): 采样点云统计信息
    
    Returns:
        dict: 比较结果
    """
    comparison = {
        'point_count_ratio': sampled_stats['num_points'] / original_stats['num_points'],
        'range_diff': sampled_stats['range_coords'] - original_stats['range_coords'],
        'mean_diff': sampled_stats['mean_coords'] - original_stats['mean_coords'],
        'std_diff': sampled_stats['std_coords'] - original_stats['std_coords']
    }
    
    return comparison


def print_statistics(point_stats: dict, voxel_stats: dict, 
                    sampled_stats: dict = None, comparison: dict = None,
                    upsampled_stats: dict = None):
    """
    打印统计信息
    
    Args:
        point_stats (dict): 原始点云统计信息
        voxel_stats (dict): 体素统计信息
        sampled_stats (dict, optional): 采样点云统计信息
        comparison (dict, optional): 点云比较结果
        upsampled_stats (dict, optional): 上采样体素统计信息
    """
    print("\n" + "="*60)
    print("原始点云统计信息:")
    print("="*60)
    print(f"点数: {point_stats['num_points']:,}")
    print("坐标范围:")
    print(f"  X: [{point_stats['min_coords'][0]:.3f}, {point_stats['max_coords'][0]:.3f}] (范围: {point_stats['range_coords'][0]:.3f})")
    print(f"  Y: [{point_stats['min_coords'][1]:.3f}, {point_stats['max_coords'][1]:.3f}] (范围: {point_stats['range_coords'][1]:.3f})")
    print(f"  Z: [{point_stats['min_coords'][2]:.3f}, {point_stats['max_coords'][2]:.3f}] (范围: {point_stats['range_coords'][2]:.3f})")
    print(f"平均坐标: ({point_stats['mean_coords'][0]:.3f}, {point_stats['mean_coords'][1]:.3f}, {point_stats['mean_coords'][2]:.3f})")
    
    print("\n" + "="*60)
    print("体素网格统计信息:")
    print("="*60)
    print(f"网格shape: {voxel_stats['shape']}")
    print(f"总体素数: {voxel_stats['total_voxels']:,}")
    print(f"占有体素数: {voxel_stats['occupied_voxels']:,}")
    print(f"占有率: {voxel_stats['occupancy_ratio']:.4f}")
    print(f"体素值范围: [{voxel_stats['min_value']:.3f}, {voxel_stats['max_value']:.3f}]")
    print(f"平均体素值: {voxel_stats['mean_value']:.3f}")
    print(f"体素值标准差: {voxel_stats['std_value']:.3f}")
    
    if upsampled_stats:
        print("\n" + "="*60)
        print("上采样体素网格统计信息:")
        print("="*60)
        print(f"网格shape: {upsampled_stats['shape']}")
        print(f"总体素数: {upsampled_stats['total_voxels']:,}")
        print(f"占有体素数: {upsampled_stats['occupied_voxels']:,}")
        print(f"占有率: {upsampled_stats['occupancy_ratio']:.4f}")
        print(f"体素值范围: [{upsampled_stats['min_value']:.3f}, {upsampled_stats['max_value']:.3f}]")
        print(f"平均体素值: {upsampled_stats['mean_value']:.3f}")
        print(f"体素值标准差: {upsampled_stats['std_value']:.3f}")
    
    if sampled_stats:
        print("\n" + "="*60)
        print("采样点云统计信息:")
        print("="*60)
        print(f"点数: {sampled_stats['num_points']:,}")
        print("坐标范围:")
        print(f"  X: [{sampled_stats['min_coords'][0]:.3f}, {sampled_stats['max_coords'][0]:.3f}] (范围: {sampled_stats['range_coords'][0]:.3f})")
        print(f"  Y: [{sampled_stats['min_coords'][1]:.3f}, {sampled_stats['max_coords'][1]:.3f}] (范围: {sampled_stats['range_coords'][1]:.3f})")
        print(f"  Z: [{sampled_stats['min_coords'][2]:.3f}, {sampled_stats['max_coords'][2]:.3f}] (范围: {sampled_stats['range_coords'][2]:.3f})")
        print(f"平均坐标: ({sampled_stats['mean_coords'][0]:.3f}, {sampled_stats['mean_coords'][1]:.3f}, {sampled_stats['mean_coords'][2]:.3f})")
    
    if comparison:
        print("\n" + "="*60)
        print("点云对比分析:")
        print("="*60)
        print(f"点数比例: {comparison['point_count_ratio']:.4f}")
        print("坐标范围变化:")
        print(f"  ΔX: {comparison['range_diff'][0]:.3f}")
        print(f"  ΔY: {comparison['range_diff'][1]:.3f}")
        print(f"  ΔZ: {comparison['range_diff'][2]:.3f}")
        print("平均坐标偏移:")
        print(f"  ΔX: {comparison['mean_diff'][0]:.3f}")
        print(f"  ΔY: {comparison['mean_diff'][1]:.3f}")
        print(f"  ΔZ: {comparison['mean_diff'][2]:.3f}")
    
    print("="*60)


def test_conversion_pipeline(args):
    """
    执行完整的转换测试流程
    
    Args:
        args: 命令行参数
    """
    try:
        # 1. 加载H5数据
        logger.info("正在加载H5数据文件...")
        loader = PointCloudH5Loader(args.input, data_key=args.data_key)
        
        # 检查样本索引是否有效
        if args.index >= loader.num_samples:
            raise IndexError(
                f"样本索引 {args.index} 超出范围，"
                f"文件中共有 {loader.num_samples} 个样本"
            )
        
        # 2. 加载指定的点云样本
        logger.info(f"正在加载样本 {args.index}...")
        point_cloud = loader.load_single_cloud(args.index)
        
        # 3. 保存原始点云（如果需要）
        if args.save_original:
            # 确定原始点云输出路径
            if args.original_output:
                original_output = args.original_output
            else:
                # 基于输出文件路径生成原始点云文件名
                original_output = args.output.replace('.tiff', '_original.csv').replace('.tif', '_original.csv')
            
            logger.info(f"正在保存原始点云到: {original_output}")
            
            # 确保输出目录存在
            original_output_dir = os.path.dirname(original_output)
            if original_output_dir and not os.path.exists(original_output_dir):
                os.makedirs(original_output_dir, exist_ok=True)
            
            # 保存原始点云
            converter_temp = PointCloudToVoxel()  # 临时创建转换器用于保存功能
            converter_temp.save_point_cloud(point_cloud, original_output)
        
        # 4. 分析点云
        point_stats = analyze_point_cloud(point_cloud)
        
        # 5. 创建体素转换器
        logger.info(f"创建体素转换器 (方法: {args.method}, 大小: {args.voxel_size})")
        logger.info(f"体积尺寸: {args.volume_dims} nm")
        logger.info(f"填充: {args.padding} nm")
        converter = PointCloudToVoxel(
            voxel_size=args.voxel_size,
            method=args.method,
            padding_ratio=args.padding_ratio,
            volume_dims=args.volume_dims,
            padding=args.padding
        )
        
        # 6. 执行转换
        logger.info("正在执行点云到体素的转换...")
        if args.method == 'gaussian':
            voxel_grid = converter.convert(point_cloud, sigma=args.sigma)
        else:
            voxel_grid = converter.convert(point_cloud)
        
        # 7. 分析体素网格
        voxel_stats = analyze_voxel_grid(voxel_grid)
        
        # 8. 体素上采样（如果需要）
        upsampled_grid = None
        upsampled_stats = None
        if args.upsample:
            logger.info(f"正在进行体素上采样 (倍数: {args.upsample_factor}, 方法: {args.upsample_method})")
            upsampled_grid = converter.upsample_voxel_grid(
                voxel_grid, 
                scale_factor=args.upsample_factor, 
                method=args.upsample_method
            )
            upsampled_stats = analyze_voxel_grid(upsampled_grid)
            
            # 保存上采样的体素网格
            upsampled_output = args.output.replace('.tiff', '_upsampled.tiff')
            logger.info(f"正在保存上采样体素网格到: {upsampled_output}")
            converter.save_as_tiff(upsampled_grid, upsampled_output)
        
        # 9. 体素采样回点云（如果需要）
        sampled_point_cloud = None
        sampled_stats = None
        comparison = None
        if args.sample_back:
            logger.info(f"正在将体素采样回点云 (方法: {args.sample_method}, 阈值: {args.sample_threshold})")
            # 选择要采样的体素网格（上采样后的或原始的）
            grid_to_sample = upsampled_grid if upsampled_grid is not None else voxel_grid
            
            sampled_point_cloud = converter.voxel_to_points(
                grid_to_sample,
                threshold=args.sample_threshold,
                num_points=args.sample_num_points,
                method=args.sample_method
            )
            
            if len(sampled_point_cloud) > 0:
                sampled_stats = analyze_point_cloud(sampled_point_cloud)
                comparison = compare_point_clouds(point_stats, sampled_stats)
                
                # 保存采样的点云为CSV格式
                sampled_output = args.output.replace('.tiff', '_sampled.csv')
                logger.info(f"正在保存采样点云到: {sampled_output}")
                converter.save_point_cloud(sampled_point_cloud, sampled_output)
            else:
                logger.warning("采样得到的点云为空")
        
        # 10. 保存原始体素网格TIFF文件
        logger.info(f"正在保存体素网格到: {args.output}")
        converter.save_as_tiff(voxel_grid, args.output)
        
        # 11. 显示统计信息
        if args.verbose:
            print_statistics(point_stats, voxel_stats, sampled_stats, comparison, upsampled_stats)
        
        # 12. 保存转换信息
        conversion_info = converter.get_conversion_info()
        info_file = args.output.replace('.tiff', '_info.txt').replace('.tif', '_info.txt')
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("点云到体素转换信息\n")
            f.write("="*50 + "\n\n")
            f.write(f"输入文件: {args.input}\n")
            f.write(f"样本索引: {args.index}\n")
            f.write(f"输出文件: {args.output}\n")
            
            if args.save_original:
                if args.original_output:
                    original_output = args.original_output
                else:
                    original_output = args.output.replace('.tiff', '_original.csv').replace('.tif', '_original.csv')
                f.write(f"原始点云文件: {original_output}\n")
            
            f.write("\n")
            
            f.write("转换参数:\n")
            for key, value in conversion_info.items():
                f.write(f"  {key}: {value}\n")
            
            if args.upsample:
                f.write("\n上采样参数:\n")
                f.write(f"  upsample_factor: {args.upsample_factor}\n")
                f.write(f"  upsample_method: {args.upsample_method}\n")
            
            if args.sample_back:
                f.write("\n采样参数:\n")
                f.write(f"  sample_threshold: {args.sample_threshold}\n")
                f.write(f"  sample_num_points: {args.sample_num_points}\n")
                f.write(f"  sample_method: {args.sample_method}\n")
            
            f.write("\n原始点云统计:\n")
            for key, value in point_stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n体素统计:\n")
            for key, value in voxel_stats.items():
                f.write(f"  {key}: {value}\n")
            
            if upsampled_stats:
                f.write("\n上采样体素统计:\n")
                for key, value in upsampled_stats.items():
                    f.write(f"  {key}: {value}\n")
            
            if sampled_stats:
                f.write("\n采样点云统计:\n")
                for key, value in sampled_stats.items():
                    f.write(f"  {key}: {value}\n")
            
            if comparison:
                f.write("\n点云对比结果:\n")
                for key, value in comparison.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info(f"转换信息已保存到: {info_file}")
        
        # 13. 输出总结
        print("\n✅ 转换完成!")
        print(f"📊 输入: {point_stats['num_points']:,} 个点")
        print(f"📦 体素网格: {voxel_stats['shape']}")
        print(f"💾 TIFF文件: {args.output}")
        
        if args.save_original:
            if args.original_output:
                original_output = args.original_output
            else:
                original_output = args.output.replace('.tiff', '_original.csv').replace('.tif', '_original.csv')
            print(f"📄 原始点云: {original_output}")
        
        if args.upsample and upsampled_grid is not None:
            upsampled_output = args.output.replace('.tiff', '_upsampled.tiff').replace('.tif', '_upsampled.tif')
            print(f"🔍 上采样网格: {upsampled_stats['shape']}")
            print(f"💾 上采样TIFF: {upsampled_output}")
        
        if args.sample_back and sampled_point_cloud is not None and len(sampled_point_cloud) > 0:
            sampled_output = args.output.replace('.tiff', '_sampled.csv')
            print(f"🎯 采样点云: {len(sampled_point_cloud):,} 个点")
            print(f"💾 采样CSV: {sampled_output}")
        
        print(f"📋 信息文件: {info_file}")
        
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
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
        
        # 执行转换测试
        test_conversion_pipeline(args)
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
