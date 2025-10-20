#!/usr/bin/env python3
"""
批量H5点云转换脚本(for 实验数据)

此脚本用于批量处理H5文件中的所有点云样本，将每个点云转换为体素网格，
然后采样回点云，最终将所有生成的点云保存为新的H5文件。

使用方法:
    python scripts/batch_h5_conversion.py --input data/input.h5 --output data/output.h5
"""

import argparse
import sys
import os
import time
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.data.h5_loader import PointCloudH5Loader
from src.voxel.converter import PointCloudToVoxel

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
        description="批量处理H5文件中的所有点云样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基本批量转换（高斯方法）
    python scripts/batch_h5_conversion.py --input data/input.h5 --output data/output.h5
    
    # 自定义体素化参数
    python scripts/batch_h5_conversion.py --input data/input.h5 --output data/output.h5 --method density --voxel-size 128
    
    # 包含上采样的完整流程
    python scripts/batch_h5_conversion.py --input data/input.h5 --output data/output.h5 --upsample --upsample-factor 2.0 --sample-num-points 200000
    
    # 自定义体积参数
    python scripts/batch_h5_conversion.py --input data/input.h5 --output data/output.h5 --volume-dims 15000 15000 3000 --padding 50 50 150
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        default="/repos/datasets/batch_simulation_microtubule_20251017_2048.h5",
        type=str,
        help="输入H5文件路径",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="/repos/datasets/batch_simulation_microtubule_20251017_2048_noised.h5",
        type=str,
        help="输出H5文件路径",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["occupancy", "density", "gaussian"],
        default="gaussian",
        help="体素化方法 (默认: gaussian)",
    )

    parser.add_argument(
        "--voxel-size", type=int, default=256, help="体素网格分辨率 (默认: 256)"
    )

    parser.add_argument(
        "--sigma", type=float, default=1.0, help="高斯方法的标准差 (默认: 1.0)"
    )

    parser.add_argument(
        "--data-key",
        type=str,
        default="point_clouds",
        help="H5文件中数据的键名 (默认: point_clouds)",
    )

    parser.add_argument(
        "--output-key",
        type=str,
        default="point_clouds",
        help="输出H5文件中数据的键名 (默认: point_clouds)",
    )

    parser.add_argument(
        "--padding-ratio", type=float, default=0.00, help="边界扩展比例 (默认: 0.00)"
    )

    parser.add_argument(
        "--volume-dims",
        type=float,
        nargs=3,
        default=[20000, 20000, 600],
        help="体积尺寸 [x, y, z] (单位: nm) (默认: [20000, 20000, 2500])",
    )

    parser.add_argument(
        "--padding",
        type=float,
        nargs=3,
        default=[0, 0, 200],
        help="体积边界填充 [x, y, z] (单位: nm) (默认: [0, 0, 100])",
    )

    # 体素采样回点云参数
    parser.add_argument(
        "--sample-num-points",
        type=int,
        default=100000,
        help="采样回点云的目标点数 (默认: 100000)",
    )

    parser.add_argument(
        "--sample-threshold",
        type=float,
        default=0.0,
        help="体素值阈值，高于此值的体素被视为包含点 (默认: 0.0)",
    )

    parser.add_argument(
        "--sample-method",
        type=str,
        choices=["probabilistic", "center", "random", "weighted"],
        default="probabilistic",
        help="采样方法 (默认: probabilistic)",
    )

    # 体素上采样参数
    parser.add_argument(
        "--upsample", action="store_true", help="是否对体素网格进行上采样"
    )

    parser.add_argument(
        "--upsample-factor", type=float, default=1.0, help="上采样倍数 (默认: 2.0)"
    )

    parser.add_argument(
        "--upsample-method",
        type=str,
        choices=["linear", "nearest", "cubic"],
        default="linear",
        help="上采样插值方法 (默认: linear)",
    )

    # 处理参数
    parser.add_argument(
        "--batch-size", type=int, default=1, help="批处理大小，用于内存管理 (默认: 1)"
    )

    parser.add_argument(
        "--start-index", type=int, default=0, help="开始处理的样本索引 (默认: 0)"
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="结束处理的样本索引，None表示处理到最后 (默认: None)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    parser.add_argument(
        "--skip-errors", action="store_true", help="跳过错误的样本继续处理"
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

    if not 0 <= args.padding_ratio <= 1:
        raise ValueError("padding_ratio必须在[0, 1]范围内")

    if args.sigma <= 0:
        raise ValueError("sigma必须大于0")

    if args.sample_threshold < 0 or args.sample_threshold > 1:
        raise ValueError("sample_threshold必须在[0, 1]范围内")

    if args.sample_num_points <= 0:
        raise ValueError("sample_num_points必须大于0")

    if args.upsample and args.upsample_factor <= 1.0:
        raise ValueError("upsample_factor必须大于1.0")

    if args.batch_size <= 0:
        raise ValueError("batch_size必须大于0")

    if args.start_index < 0:
        raise ValueError("start_index必须非负")


def process_single_point_cloud(
    point_cloud: np.ndarray, converter: PointCloudToVoxel, args
) -> np.ndarray:
    """
    处理单个点云样本

    Args:
        point_cloud (np.ndarray): 输入点云
        converter (PointCloudToVoxel): 体素转换器
        args: 命令行参数

    Returns:
        np.ndarray: 生成的点云
    """
    # 1. 点云转体素
    if args.method == "gaussian":
        voxel_grid = converter.convert(point_cloud, sigma=args.sigma)
    else:
        voxel_grid = converter.convert(point_cloud)

    # 2. 体素上采样（如果需要）
    if args.upsample:
        voxel_grid = converter.upsample_voxel_grid(
            voxel_grid, scale_factor=args.upsample_factor, method=args.upsample_method
        )

    # 3. 体素采样回点云
    generated_point_cloud = converter.voxel_to_points(
        voxel_grid,
        threshold=args.sample_threshold,
        num_points=args.sample_num_points,
        method=args.sample_method,
    )

    return generated_point_cloud


def get_max_points_count(generated_clouds: list) -> int:
    """
    获取生成点云中的最大点数，用于创建统一的数组形状

    Args:
        generated_clouds (list): 生成的点云列表

    Returns:
        int: 最大点数
    """
    return max(len(cloud) for cloud in generated_clouds if len(cloud) > 0)


def pad_point_clouds(generated_clouds: list, max_points: int) -> np.ndarray:
    """
    将点云填充到统一长度

    Args:
        generated_clouds (list): 生成的点云列表
        max_points (int): 目标点数

    Returns:
        np.ndarray: 填充后的点云数组 (num_samples, max_points, 3)
    """
    num_samples = len(generated_clouds)
    padded_clouds = np.zeros((num_samples, max_points, 3), dtype=np.float32)

    for i, cloud in enumerate(generated_clouds):
        if len(cloud) > 0:
            # 如果点云长度超过max_points，随机采样
            if len(cloud) > max_points:
                indices = np.random.choice(len(cloud), max_points, replace=False)
                padded_clouds[i] = cloud[indices]
            else:
                # 否则直接填充，多余位置保持0
                padded_clouds[i, : len(cloud)] = cloud

    return padded_clouds


def batch_h5_conversion(args):
    """
    执行批量H5点云转换

    Args:
        args: 命令行参数
    """
    try:
        # 1. 加载输入H5数据
        logger.info(f"正在加载H5数据文件: {args.input}")
        loader = PointCloudH5Loader(args.input, data_key=args.data_key)

        # 尝试读取sample_ids
        sample_ids = None
        try:
            with h5py.File(args.input, "r") as f:
                if "sample_ids" in f:
                    sample_ids = f["sample_ids"][:]
                    logger.info(f"发现sample_ids数据，共 {len(sample_ids)} 个样本ID")
                else:
                    logger.warning("输入文件中没有找到sample_ids，将使用索引作为ID")
        except Exception as e:
            logger.warning(f"读取sample_ids时出错: {e}")
            sample_ids = None

        # 确定处理范围
        total_samples = loader.num_samples
        total_samples = 3
        start_idx = args.start_index
        end_idx = args.end_index if args.end_index is not None else total_samples
        end_idx = min(end_idx, total_samples)

        if start_idx >= total_samples:
            raise ValueError(f"start_index {start_idx} 超出样本数量 {total_samples}")

        logger.info(f"数据集包含 {total_samples} 个样本")
        logger.info(
            f"处理范围: {start_idx} 到 {end_idx - 1} (共 {end_idx - start_idx} 个样本)"
        )

        # 2. 创建体素转换器
        logger.info(f"创建体素转换器 (方法: {args.method}, 大小: {args.voxel_size})")
        converter = PointCloudToVoxel(
            voxel_size=args.voxel_size,
            method=args.method,
            padding_ratio=args.padding_ratio,
            volume_dims=args.volume_dims,
            padding=args.padding,
        )

        # 3. 批量处理点云
        logger.info("开始批量处理点云...")
        generated_clouds = []
        failed_indices = []
        processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "empty_results": 0,
        }

        # 使用tqdm显示进度
        progress_bar = tqdm(
            range(start_idx, end_idx, args.batch_size), desc="处理中", unit="batch"
        )

        for batch_start in progress_bar:
            batch_end = min(batch_start + args.batch_size, end_idx)
            batch_indices = list(range(batch_start, batch_end))

            for idx in batch_indices:
                processing_stats["total_processed"] += 1

                try:
                    # 加载点云
                    point_cloud = loader.load_single_cloud(idx)

                    # 处理点云
                    generated_cloud = process_single_point_cloud(
                        point_cloud, converter, args
                    )

                    if len(generated_cloud) == 0:
                        processing_stats["empty_results"] += 1
                        if args.verbose:
                            logger.warning(f"样本 {idx} 生成的点云为空")

                        if not args.skip_errors:
                            raise ValueError(f"样本 {idx} 生成的点云为空")

                    generated_clouds.append(generated_cloud)
                    processing_stats["successful"] += 1

                    # 更新进度条描述
                    progress_bar.set_postfix(
                        {
                            "Success": processing_stats["successful"],
                            "Failed": processing_stats["failed"],
                            "Empty": processing_stats["empty_results"],
                        }
                    )

                except Exception as e:
                    processing_stats["failed"] += 1
                    failed_indices.append(idx)

                    if args.skip_errors:
                        logger.warning(f"处理样本 {idx} 时出错，跳过: {e}")
                        # 添加空的点云占位
                        generated_clouds.append(np.empty((0, 3)))
                        continue
                    else:
                        logger.error(f"处理样本 {idx} 时出错: {e}")
                        raise

        progress_bar.close()

        # 4. 检查是否有成功的结果
        if processing_stats["successful"] == 0:
            raise RuntimeError("没有成功处理的样本")

        logger.info(
            f"处理完成！成功: {processing_stats['successful']}, "
            f"失败: {processing_stats['failed']}, "
            f"空结果: {processing_stats['empty_results']}"
        )

        if failed_indices:
            logger.warning(f"失败的样本索引: {failed_indices}")

        # 5. 准备保存数据
        logger.info("正在准备输出数据...")

        # 获取最大点数并填充
        valid_clouds = [cloud for cloud in generated_clouds if len(cloud) > 0]
        if not valid_clouds:
            raise RuntimeError("没有有效的生成点云")

        max_points = get_max_points_count(valid_clouds)
        logger.info(f"最大点数: {max_points}")

        # 填充点云到统一形状
        padded_clouds = pad_point_clouds(generated_clouds, max_points)
        # padded_clouds = generated_clouds

        # 6. 准备对应的sample_ids
        output_sample_ids = None
        if sample_ids is not None:
            # 提取处理范围内的sample_ids
            output_sample_ids = sample_ids[start_idx:end_idx]
            logger.info(f"提取对应的sample_ids: {len(output_sample_ids)} 个")
        else:
            # 使用索引作为ID
            output_sample_ids = np.arange(start_idx, end_idx)
            logger.info(f"使用索引作为sample_ids: {start_idx} 到 {end_idx - 1}")

        # 7. 保存到H5文件
        logger.info(f"正在保存到H5文件: {args.output}")

        with h5py.File(args.output, "w") as f:
            # 保存主要数据
            dataset = f.create_dataset(
                args.output_key,
                data=padded_clouds,
                compression="gzip",
                compression_opts=9,
            )

            # 保存sample_ids（保持与原始文件的对应关系）
            f.create_dataset(
                "sample_ids",
                data=output_sample_ids,
                compression="gzip",
                compression_opts=9,
            )

            # 保存元数据
            f.attrs["original_file"] = args.input
            f.attrs["original_data_key"] = args.data_key
            f.attrs["total_samples"] = len(generated_clouds)
            f.attrs["max_points_per_sample"] = max_points
            f.attrs["processing_range"] = f"{start_idx}-{end_idx - 1}"
            f.attrs["successful_samples"] = processing_stats["successful"]
            f.attrs["failed_samples"] = processing_stats["failed"]
            f.attrs["empty_results"] = processing_stats["empty_results"]

            # 保存转换参数
            conv_params = f.create_group("conversion_parameters")
            conv_params.attrs["method"] = args.method
            conv_params.attrs["voxel_size"] = args.voxel_size
            conv_params.attrs["sigma"] = args.sigma
            conv_params.attrs["padding_ratio"] = args.padding_ratio
            conv_params.attrs["volume_dims"] = args.volume_dims
            conv_params.attrs["padding"] = args.padding
            conv_params.attrs["sample_threshold"] = args.sample_threshold
            conv_params.attrs["sample_num_points"] = args.sample_num_points
            conv_params.attrs["sample_method"] = args.sample_method
            conv_params.attrs["upsample"] = args.upsample
            conv_params.attrs["upsample_factor"] = args.upsample_factor
            conv_params.attrs["upsample_method"] = args.upsample_method

            # 保存失败索引（如果有）
            if failed_indices:
                f.create_dataset("failed_indices", data=failed_indices)

            # 保存实际点数信息
            actual_point_counts = [len(cloud) for cloud in generated_clouds]
            f.create_dataset("actual_point_counts", data=actual_point_counts)

        # 8. 输出总结
        print("\n" + "=" * 60)
        print("✅ 批量转换完成!")
        print("=" * 60)
        print(f"📁 输入文件: {args.input}")
        print(f"📁 输出文件: {args.output}")
        print(f"📊 处理样本: {processing_stats['total_processed']}")
        print(f"✅ 成功: {processing_stats['successful']}")
        print(f"❌ 失败: {processing_stats['failed']}")
        print(f"🔸 空结果: {processing_stats['empty_results']}")
        print(f"📦 输出形状: {padded_clouds.shape}")
        print(f"🎯 最大点数: {max_points}")
        print(
            f"🔗 Sample IDs: {'保留原始对应关系' if sample_ids is not None else '使用索引ID'}"
        )

        if failed_indices:
            print(
                f"⚠️  失败索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
            )

        print("=" * 60)

        # 保存处理报告
        report_file = args.output.replace(".h5", "_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("批量H5点云转换报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {args.input}\n")
            f.write(f"输出文件: {args.output}\n")
            f.write(f"处理范围: {start_idx} 到 {end_idx - 1}\n\n")

            f.write("处理统计:\n")
            f.write(f"  总处理样本: {processing_stats['total_processed']}\n")
            f.write(f"  成功样本: {processing_stats['successful']}\n")
            f.write(f"  失败样本: {processing_stats['failed']}\n")
            f.write(f"  空结果样本: {processing_stats['empty_results']}\n\n")

            f.write("转换参数:\n")
            f.write(f"  method: {args.method}\n")
            f.write(f"  voxel_size: {args.voxel_size}\n")
            f.write(f"  sigma: {args.sigma}\n")
            f.write(f"  sample_method: {args.sample_method}\n")
            f.write(f"  sample_num_points: {args.sample_num_points}\n")
            f.write(f"  upsample: {args.upsample}\n")
            if args.upsample:
                f.write(f"  upsample_factor: {args.upsample_factor}\n")
                f.write(f"  upsample_method: {args.upsample_method}\n")

            f.write(f"\n输出数据形状: {padded_clouds.shape}\n")
            f.write(f"最大点数: {max_points}\n")
            f.write(
                f"Sample IDs: {'保留原始对应关系' if sample_ids is not None else '使用索引ID'}\n"
            )

            if failed_indices:
                f.write(f"\n失败的样本索引:\n")
                for idx in failed_indices:
                    f.write(f"  {idx}\n")

            if output_sample_ids is not None:
                f.write(
                    f"\nSample IDs范围: {output_sample_ids[0]} 到 {output_sample_ids[-1]}\n"
                )

        logger.info(f"处理报告已保存到: {report_file}")

    except Exception as e:
        logger.error(f"批量转换过程中发生错误: {e}")
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

        # 执行批量转换
        batch_h5_conversion(args)

    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


def test_h5_correspondence(
    gt_file: str,
    noisy_file: str,
    sample_index: int = 0,
    output_dir: str = "test_correspondence",
    data_key: str = "point_clouds",
    args: argparse.Namespace = None,
):
    """
    测试GT和转换后H5文件的对应关系

    Args:
        gt_file (str): 原始GT H5文件路径
        noisy_file (str): 转换后的H5文件路径
        sample_index (int): 要测试的样本索引
        output_dir (str): 输出目录
        data_key (str): 数据键名
    """
    try:
        import tifffile

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取GT文件
        logger.info(f"读取GT文件: {gt_file}")
        gt_loader = PointCloudH5Loader(gt_file, data_key=args.data_key)

        # 检查样本索引
        if sample_index >= gt_loader.num_samples:
            raise ValueError(
                f"样本索引 {sample_index} 超出GT文件样本数 {gt_loader.num_samples}"
            )

        # 读取GT点云
        gt_point_cloud = gt_loader.load_single_cloud(sample_index)

        # 读取GT的sample_ids
        gt_sample_ids = None
        try:
            with h5py.File(gt_file, "r") as f:
                if "sample_ids" in f:
                    gt_sample_ids = f["sample_ids"][:]
                    logger.info(f"GT文件sample_ids数量: {len(gt_sample_ids)}")
        except Exception as e:
            logger.warning(f"读取GT sample_ids失败: {e}")

        # 读取转换后文件
        logger.info(f"读取转换后文件: {noisy_file}")
        noisy_loader = PointCloudH5Loader(noisy_file, data_key=args.output_key)

        # 检查样本索引
        if sample_index >= noisy_loader.num_samples:
            raise ValueError(
                f"样本索引 {sample_index} 超出转换后文件样本数 {noisy_loader.num_samples}"
            )

        # 读取转换后点云
        noisy_point_cloud = noisy_loader.load_single_cloud(sample_index)
        noisy_point_cloud = np.clip(
            noisy_point_cloud,
            [0 - args.padding[0], 0 - args.padding[1], 0 - args.padding[2]],
            [
                args.volume_dims[0] + args.padding[0] - 1,
                args.volume_dims[1] + args.padding[1] - 1,
                args.volume_dims[2] + args.padding[2] - 1,
            ],
        )

        # 读取转换后的sample_ids
        noisy_sample_ids = None
        try:
            with h5py.File(noisy_file, "r") as f:
                if "sample_ids" in f:
                    noisy_sample_ids = f["sample_ids"][:]
                    logger.info(f"转换后文件sample_ids数量: {len(noisy_sample_ids)}")
        except Exception as e:
            logger.warning(f"读取转换后 sample_ids失败: {e}")

        # 验证sample_ids对应关系
        if gt_sample_ids is not None and noisy_sample_ids is not None:
            if sample_index < len(gt_sample_ids) and sample_index < len(
                noisy_sample_ids
            ):
                gt_id = gt_sample_ids[sample_index]
                noisy_id = noisy_sample_ids[sample_index]
                if gt_id == noisy_id:
                    logger.info(f"✅ Sample ID对应关系正确: {gt_id}")
                else:
                    logger.warning(
                        f"⚠️ Sample ID不匹配! GT: {gt_id}, 转换后: {noisy_id}"
                    )
            else:
                logger.warning("Sample ID索引超出范围")
        else:
            logger.warning("无法验证Sample ID对应关系")

        # 保存点云为CSV文件
        logger.info("保存点云为CSV文件...")

        # 保存GT点云
        gt_csv_path = os.path.join(output_dir, f"gt_sample_{sample_index}.csv")
        with open(gt_csv_path, "w", encoding="utf-8") as f:
            f.write("x[nm],y[nm],z[nm]\n")
            for point in gt_point_cloud:
                f.write(f"{point[0]:.1f},{point[1]:.1f},{point[2]:.1f}\n")
        logger.info(f"GT点云CSV已保存: {gt_csv_path}")

        # 保存转换后点云
        if len(noisy_point_cloud) > 0:
            noisy_csv_path = os.path.join(
                output_dir, f"noisy_sample_{sample_index}.csv"
            )
            with open(noisy_csv_path, "w", encoding="utf-8") as f:
                f.write("x[nm],y[nm],z[nm]\n")
                for point in noisy_point_cloud:
                    f.write(f"{point[0]:.1f},{point[1]:.1f},{point[2]:.1f}\n")
            logger.info(f"转换后点云CSV已保存: {noisy_csv_path}")
        else:
            logger.warning("转换后点云为空，无法生成CSV文件")

        # 转换点云为体素用于可视化
        logger.info("转换点云为体素网格用于可视化...")
        converter = PointCloudToVoxel(
            voxel_size=args.voxel_size,
            method="gaussian",
            padding_ratio=args.padding_ratio,
            volume_dims=args.volume_dims,
            padding=args.padding,
        )

        # GT点云转体素
        gt_voxel = converter.convert(gt_point_cloud, sigma=args.sigma)
        gt_output = os.path.join(output_dir, f"gt_sample_{sample_index}.tiff")
        converter.save_as_tiff(gt_voxel, gt_output)
        logger.info(f"GT体素已保存: {gt_output}")

        # 转换后点云转体素
        if len(noisy_point_cloud) > 0:
            noisy_voxel = converter.convert(noisy_point_cloud, sigma=args.sigma)
            noisy_output = os.path.join(output_dir, f"noisy_sample_{sample_index}.tiff")
            converter.save_as_tiff(noisy_voxel, noisy_output)
            logger.info(f"转换后体素已保存: {noisy_output}")
        else:
            logger.warning("转换后点云为空，无法生成体素")

        # 输出统计信息
        print("\n" + "=" * 60)
        print(f"📊 样本 {sample_index} 对应关系测试结果")
        print("=" * 60)
        print(f"GT点云点数: {len(gt_point_cloud):,}")
        print(f"转换后点云点数: {len(noisy_point_cloud):,}")

        if gt_sample_ids is not None and noisy_sample_ids is not None:
            if sample_index < len(gt_sample_ids) and sample_index < len(
                noisy_sample_ids
            ):
                print(f"GT Sample ID: {gt_sample_ids[sample_index]}")
                print(f"转换后 Sample ID: {noisy_sample_ids[sample_index]}")
                print(
                    f"ID匹配: {'✅' if gt_sample_ids[sample_index] == noisy_sample_ids[sample_index] else '❌'}"
                )

        # 计算基本统计差异
        gt_stats = {
            "min": np.min(gt_point_cloud, axis=0),
            "max": np.max(gt_point_cloud, axis=0),
            "mean": np.mean(gt_point_cloud, axis=0),
            "std": np.std(gt_point_cloud, axis=0),
        }

        if len(noisy_point_cloud) > 0:
            noisy_stats = {
                "min": np.min(noisy_point_cloud, axis=0),
                "max": np.max(noisy_point_cloud, axis=0),
                "mean": np.mean(noisy_point_cloud, axis=0),
                "std": np.std(noisy_point_cloud, axis=0),
            }

            print(f"\n坐标范围变化:")
            for i, axis in enumerate(["X", "Y", "Z"]):
                gt_range = gt_stats["max"][i] - gt_stats["min"][i]
                noisy_range = noisy_stats["max"][i] - noisy_stats["min"][i]
                print(
                    f"  {axis}: GT={gt_range:.2f}, 转换后={noisy_range:.2f}, 差异={noisy_range - gt_range:.2f}"
                )

            print(f"\n平均坐标偏移:")
            mean_diff = noisy_stats["mean"] - gt_stats["mean"]
            for i, axis in enumerate(["X", "Y", "Z"]):
                print(f"  {axis}: {mean_diff[i]:.2f}")

        print(f"\n📁 输出文件保存在: {output_dir}")
        print(f"  - GT点云CSV: gt_sample_{sample_index}.csv")
        if len(noisy_point_cloud) > 0:
            print(f"  - 转换后点云CSV: noisy_sample_{sample_index}.csv")
        print(f"  - GT体素TIFF: gt_sample_{sample_index}.tiff")
        if len(noisy_point_cloud) > 0:
            print(f"  - 转换后体素TIFF: noisy_sample_{sample_index}.tiff")
        print("=" * 60)

        # 保存测试报告
        report_file = os.path.join(
            output_dir, f"correspondence_test_sample_{sample_index}.txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"H5文件对应关系测试报告 - 样本 {sample_index}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GT文件: {gt_file}\n")
            f.write(f"转换后文件: {noisy_file}\n")
            f.write(f"测试样本索引: {sample_index}\n\n")

            f.write("点云统计:\n")
            f.write(f"  GT点数: {len(gt_point_cloud):,}\n")
            f.write(f"  转换后点数: {len(noisy_point_cloud):,}\n")
            f.write(
                f"  点数比例: {len(noisy_point_cloud) / len(gt_point_cloud):.4f}\n\n"
            )

            if gt_sample_ids is not None and noisy_sample_ids is not None:
                if sample_index < len(gt_sample_ids) and sample_index < len(
                    noisy_sample_ids
                ):
                    f.write("Sample ID验证:\n")
                    f.write(f"  GT Sample ID: {gt_sample_ids[sample_index]}\n")
                    f.write(f"  转换后 Sample ID: {noisy_sample_ids[sample_index]}\n")
                    f.write(
                        f"  ID匹配: {'是' if gt_sample_ids[sample_index] == noisy_sample_ids[sample_index] else '否'}\n\n"
                    )

            f.write("输出文件:\n")
            f.write(f"  GT点云CSV: gt_sample_{sample_index}.csv\n")
            if len(noisy_point_cloud) > 0:
                f.write(f"  转换后点云CSV: noisy_sample_{sample_index}.csv\n")
            f.write(f"  GT体素: gt_sample_{sample_index}.tiff\n")
            if len(noisy_point_cloud) > 0:
                f.write(f"  转换后体素: noisy_sample_{sample_index}.tiff\n")

        logger.info(f"测试报告已保存: {report_file}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    import sys

    # --- test mode ---
    sys.argv.append("test")
    sys.argv.append("/repos/datasets/batch_simulation_microtubule_20251017_2048.h5")
    sys.argv.append(
        "/repos/datasets/batch_simulation_microtubule_20251017_2048_noised.h5"
    )
    sys.argv.append("1")

    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式：python batch_h5_conversion.py test <gt_file> <noisy_file> [sample_index]
        if len(sys.argv) < 4:
            print(
                "测试用法: python batch_h5_conversion.py test <gt_file> <noisy_file> [sample_index]"
            )
            print("示例: python batch_h5_conversion.py test gt.h5 noisy.h5 0")
            sys.exit(1)

        gt_file = sys.argv[2]
        noisy_file = sys.argv[3]
        sample_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0

        # 备份原始 sys.argv，只解析正常参数
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]  # 只保留脚本名
        args = parse_arguments()
        sys.argv = original_argv  # 恢复原始 argv

        logger.info(f"开始测试H5文件对应关系...")
        test_h5_correspondence(gt_file, noisy_file, sample_index, args=args)
    else:
        # 正常批量转换模式
        main()
