"""
快速运行脚本 - 使用配置文件运行数据处理

这个脚本从配置文件读取参数，自动完成整个数据处理流程。

使用方法:
    1. 复制 config_example.py 为 config.py
    2. 修改 config.py 中的参数
    3. 运行: python scripts/exp-data-process/run_processing.py

Author: AI Assistant
Date: 2025-10-05
"""

import sys
from pathlib import Path
import logging
import argparse

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入配置文件
try:
    from scripts.exp_data_process import config_mt as config

    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    print("警告: 未找到 config.py 文件")
    print("请复制 config_example.py 为 config.py 并修改相关参数")
    print()

from scripts.exp_data_process.process_pointcloud_data import (
    find_all_csv_files,
    generate_samples,
    save_samples_to_h5,
)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    配置日志系统

    Args:
        log_level (str): 日志级别
        log_file (str): 日志文件路径
    """
    # 创建日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 配置根日志记录器
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )


def run_with_config():
    """使用配置文件运行数据处理"""

    if not CONFIG_LOADED:
        raise RuntimeError("未找到配置文件，无法运行")

    # 配置日志
    log_file = config.LOG_FILE if config.SAVE_LOG_TO_FILE else None
    setup_logging(config.LOG_LEVEL, log_file)

    logger = logging.getLogger(__name__)

    # 显示配置信息
    logger.info("=" * 70)
    logger.info("点云数据处理 - 使用配置文件运行")
    logger.info("=" * 70)
    logger.info("")
    logger.info("配置参数:")
    logger.info(f"  输入目录: {config.INPUT_DIR}")
    logger.info(f"  输出文件: {config.OUTPUT_H5}")
    logger.info(f"  样本数量: {config.NUM_SAMPLES}")
    logger.info(f"  区域大小: {config.REGION_SIZE} nm")
    logger.info(f"  期望点数: {config.TARGET_POINTS}")
    logger.info(f"  最小点数: {config.MIN_POINTS}")
    logger.info(f"  最大尝试: {config.MAX_ATTEMPTS}")
    logger.info(f"  坐标精度: {config.DECIMALS} 位小数")
    logger.info(f"  随机种子: {config.RANDOM_SEED}")
    logger.info("")

    try:
        # 步骤 1: 查找CSV文件
        logger.info("=" * 70)
        logger.info("步骤 1: 查找CSV文件")
        logger.info("=" * 70)
        csv_files = find_all_csv_files(config.INPUT_DIR)

        # 步骤 2: 生成样本
        logger.info("")
        logger.info("=" * 70)
        logger.info("步骤 2: 生成点云样本")
        logger.info("=" * 70)
        samples = generate_samples(
            csv_files=csv_files,
            num_samples=config.NUM_SAMPLES,
            region_size=config.REGION_SIZE,
            target_points=config.TARGET_POINTS,
            min_points=config.MIN_POINTS,
            max_attempts=config.MAX_ATTEMPTS,
            decimals=config.DECIMALS,
            random_seed=config.RANDOM_SEED,
        )

        # 步骤 3: 保存到H5文件
        logger.info("")
        logger.info("=" * 70)
        logger.info("步骤 3: 保存到H5文件")
        logger.info("=" * 70)
        save_samples_to_h5(
            samples=samples,
            output_path=config.OUTPUT_H5,
            dataset_name=config.DATASET_NAME,
            compression=config.COMPRESSION,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("处理完成！")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}", exc_info=True)
        return False


def run_with_args(args):
    """使用命令行参数运行数据处理"""

    # 配置日志
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # 显示参数
    logger.info("=" * 70)
    logger.info("点云数据处理 - 使用命令行参数运行")
    logger.info("=" * 70)
    logger.info("")
    logger.info("参数:")
    logger.info(f"  输入目录: {args.input_dir}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"  样本数量: {args.num_samples}")
    logger.info(f"  区域大小: ({args.region_x}, {args.region_y}, {args.region_z}) nm")
    logger.info(f"  期望点数: {args.target_points}")
    logger.info(f"  最小点数: {args.min_points}")
    logger.info(f"  最大尝试: {args.max_attempts}")
    logger.info(f"  坐标精度: {args.decimals} 位小数")
    logger.info(f"  随机种子: {args.seed}")
    logger.info("")

    try:
        # 步骤 1: 查找CSV文件
        logger.info("=" * 70)
        logger.info("步骤 1: 查找CSV文件")
        logger.info("=" * 70)
        csv_files = find_all_csv_files(args.input_dir)

        # 步骤 2: 生成样本
        logger.info("")
        logger.info("=" * 70)
        logger.info("步骤 2: 生成点云样本")
        logger.info("=" * 70)
        samples = generate_samples(
            csv_files=csv_files,
            num_samples=args.num_samples,
            region_size=(args.region_x, args.region_y, args.region_z),
            target_points=args.target_points,
            min_points=args.min_points,
            max_attempts=args.max_attempts,
            decimals=args.decimals,
            random_seed=args.seed,
        )

        # 步骤 3: 保存到H5文件
        logger.info("")
        logger.info("=" * 70)
        logger.info("步骤 3: 保存到H5文件")
        logger.info("=" * 70)
        save_samples_to_h5(
            samples=samples,
            output_path=args.output,
            dataset_name="pointclouds",
            compression="gzip",
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("处理完成！")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}", exc_info=True)
        return False


def main():
    """主函数"""

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="点云数据处理工具 - 从CSV文件生成训练样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件运行
  python scripts/exp-data-process/run_processing.py
  
  # 使用命令行参数运行
  python scripts/exp-data-process/run_processing.py \\
    --input-dir /path/to/csv/files \\
    --output outputs/samples.h5 \\
    --num-samples 100 \\
    --region-size 1000 1000 500
        """,
    )

    parser.add_argument("--input-dir", "-i", type=str, help="输入CSV文件目录")

    parser.add_argument("--output", "-o", type=str, help="输出H5文件路径")

    parser.add_argument(
        "--num-samples", "-n", type=int, default=100, help="生成样本数量 (默认: 100)"
    )

    parser.add_argument(
        "--region-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="提取区域大小 (单位: nm)",
    )

    parser.add_argument(
        "--region-x", type=float, default=1000.0, help="X方向区域大小 (默认: 1000 nm)"
    )

    parser.add_argument(
        "--region-y", type=float, default=1000.0, help="Y方向区域大小 (默认: 1000 nm)"
    )

    parser.add_argument(
        "--region-z", type=float, default=500.0, help="Z方向区域大小 (默认: 500 nm)"
    )

    parser.add_argument(
        "--target-points",
        type=int,
        default=1024,
        help="期望的点数，每个样本的目标点数 (默认: 1024)",
    )

    parser.add_argument(
        "--min-points", type=int, default=100, help="每个样本的最小点数 (默认: 100)"
    )

    parser.add_argument(
        "--max-attempts", type=int, default=10, help="每个CSV的最大尝试次数 (默认: 10)"
    )

    parser.add_argument(
        "--decimals", type=int, default=2, help="坐标保留的小数位数 (默认: 2)"
    )

    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 处理 region-size 参数
    if args.region_size is not None:
        args.region_x, args.region_y, args.region_z = args.region_size

    # 判断使用哪种模式
    if args.input_dir is None and args.output is None:
        # 使用配置文件模式
        if not CONFIG_LOADED:
            print("错误: 未找到配置文件，且未提供命令行参数")
            print("请执行以下操作之一:")
            print("  1. 复制 config_example.py 为 config.py 并修改参数")
            print("  2. 使用命令行参数运行 (使用 --help 查看参数说明)")
            sys.exit(1)

        success = run_with_config()
    else:
        # 使用命令行参数模式
        if args.input_dir is None or args.output is None:
            print("错误: 使用命令行参数模式时，必须同时提供 --input-dir 和 --output")
            print("使用 --help 查看完整的参数说明")
            sys.exit(1)

        success = run_with_args(args)

    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
