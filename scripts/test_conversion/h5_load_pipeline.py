#!/usr/bin/env python3
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

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        default="/repos/datasets/batch_simulation_mitochondria.h5",
        type=str,
        help="输入H5文件路径",
    )

    parser.add_argument(
        "--verbose", "-v", default=True, action="store_true", help="显示详细信息"
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
        "num_points": len(point_cloud),
        "min_coords": np.min(point_cloud, axis=0),
        "max_coords": np.max(point_cloud, axis=0),
        "mean_coords": np.mean(point_cloud, axis=0),
        "std_coords": np.std(point_cloud, axis=0),
        "range_coords": np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0),
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
        "shape": voxel_grid.shape,
        "total_voxels": voxel_grid.size,
        "occupied_voxels": np.sum(voxel_grid > 0),
        "occupancy_ratio": np.sum(voxel_grid > 0) / voxel_grid.size,
        "min_value": np.min(voxel_grid),
        "max_value": np.max(voxel_grid),
        "mean_value": np.mean(voxel_grid),
        "std_value": np.std(voxel_grid),
    }

    return stats


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

        # 3. 分析点云
        point_stats = analyze_point_cloud(point_cloud)

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


if __name__ == "__main__":
    main()
