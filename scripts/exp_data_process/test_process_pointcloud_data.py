"""
点云数据处理测试脚本

测试点云数据处理脚本的功能，将生成的样本保存为CSV文件和PNG可视化图片，
便于检查数据质量。

功能：
    - 调用主脚本生成点云样本
    - 将每个样本保存为独立的CSV文件
    - 将每个样本在XY平面上可视化（用Z轴值表示颜色）
    - 生成统计报告

Usage:
    python scripts/exp-data-process/test_process_pointcloud_data.py

Author: AI Assistant
Date: 2025-10-05
"""

import random
import sys
from pathlib import Path
from typing import List, Optional
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

# 导入主脚本的函数
from scripts.exp_data_process.process_pointcloud_data import (
    find_all_csv_files,
    generate_samples_with_source
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_pointcloud_as_csv(
    points: np.ndarray,
    output_path: str,
    source_file: Optional[Path] = None,
    include_header: bool = True
) -> None:
    """
    将点云保存为CSV文件
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        output_path (str): 输出CSV文件路径
        source_file (Optional[Path]): 来源CSV文件路径
        include_header (bool): 是否包含表头
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(
        points,
        columns=['x [nm]', 'y [nm]', 'z [nm]']
    )
    
    # 保存为CSV
    df.to_csv(output_path, index=False, header=include_header)
    
    # 如果提供了来源文件信息，创建一个元数据文件
    if source_file is not None:
        metadata_path = output_path.with_suffix('.metadata.txt')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"来源文件: {source_file}\n")
            f.write(f"样本点数: {len(points)}\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")
    
    logger.debug(f"已保存CSV: {output_path.name}")


def visualize_pointcloud_xy(
    points: np.ndarray,
    output_path: str,
    figsize: tuple = (8, 8),
    dpi: int = 200,
    cmap: str = 'jet',
    point_size: float = 0.1,
    show_colorbar: bool = True,
    show_stats: bool = False
) -> None:
    """
    在XY平面上可视化点云，用Z轴值表示颜色
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        output_path (str): 输出PNG文件路径
        figsize (tuple): 图片大小
        dpi (int): 图片分辨率
        cmap (str): 颜色映射
        point_size (float): 点的大小
        show_colorbar (bool): 是否显示颜色条
        show_stats (bool): 是否显示统计信息
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 提取坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 绘制散点图（用Z值作为颜色）
    scatter = ax.scatter(
        x, y, c=z,
        cmap=cmap,
        s=point_size,
        alpha=1,
        edgecolors='none'
    )
    
    # 设置坐标轴
    ax.set_xlabel('X [nm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [nm]', fontsize=12, fontweight='bold')
    # ax.set_title('Point Cloud XY Projection', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z [nm]', fontsize=12, fontweight='bold')
    
    # 添加统计信息
    if show_stats:
        stats_text = (
            f"Points: {len(points)}\n"
            f"X: [{x.min():.1f}, {x.max():.1f}] nm\n"
            f"Y: [{y.min():.1f}, {y.max():.1f}] nm\n"
            f"Z: [{z.min():.1f}, {z.max():.1f}] nm"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.debug(f"已保存可视化: {output_path.name}")


def create_summary_visualization(
    samples: List[np.ndarray],
    output_path: str,
    max_samples: int = 16
) -> None:
    """
    创建多个样本的汇总可视化
    
    Args:
        samples (List[np.ndarray]): 点云样本列表
        output_path (str): 输出PNG文件路径
        max_samples (int): 最多显示的样本数
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 限制显示的样本数
    display_samples = samples[:max_samples]
    num_display = len(display_samples)
    
    # 计算子图布局
    ncols = int(np.ceil(np.sqrt(num_display)))
    nrows = int(np.ceil(num_display / ncols))
    
    # 创建图形
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), dpi=100)
    
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    
    # 绘制每个样本
    for idx, (ax, points) in enumerate(zip(axes.flat, display_samples)):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        scatter = ax.scatter(
            x, y, c=z,
            cmap='viridis',
            s=0.5,
            alpha=0.8,
            edgecolors='none'
        )
        
        ax.set_title(f'Sample {idx + 1} ({len(points)} pts)', fontsize=10)
        ax.set_xlabel('X [nm]', fontsize=8)
        ax.set_ylabel('Y [nm]', fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(num_display, nrows * ncols):
        axes.flat[idx].axis('off')
    
    # 添加总标题
    fig.suptitle(
        f'Point Cloud Samples Overview (showing {num_display}/{len(samples)})',
        fontsize=16,
        fontweight='bold'
    )
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"已保存汇总可视化: {output_path}")


def generate_statistics_report(
    samples: List[np.ndarray],
    source_files: List[Path],
    output_path: str
) -> None:
    """
    生成样本统计报告
    
    Args:
        samples (List[np.ndarray]): 点云样本列表
        source_files (List[Path]): 每个样本对应的源文件列表
        output_path (str): 输出文本文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    num_samples = len(samples)
    point_counts = [len(sample) for sample in samples]
    
    # 统计来源文件
    unique_sources = list(set(source_files))
    source_counts = {source: source_files.count(source) for source in unique_sources}
    
    # 计算每个样本的边界框大小
    bbox_sizes = []
    for sample in samples:
        min_coords = np.min(sample, axis=0)
        max_coords = np.max(sample, axis=0)
        bbox_size = max_coords - min_coords
        bbox_sizes.append(bbox_size)
    
    bbox_sizes = np.array(bbox_sizes)
    
    # 生成报告
    report = []
    report.append("=" * 70)
    report.append("点云样本统计报告")
    report.append("=" * 70)
    report.append("")
    
    report.append(f"样本数量: {num_samples}")
    report.append(f"来源文件数量: {len(unique_sources)}")
    report.append("")
    
    report.append("来源文件统计:")
    for source_file, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  - {source_file.name}: {count} 个样本")
    report.append("")
    
    report.append("点数统计:")
    report.append(f"  - 最小值: {np.min(point_counts)}")
    report.append(f"  - 最大值: {np.max(point_counts)}")
    report.append(f"  - 平均值: {np.mean(point_counts):.2f}")
    report.append(f"  - 中位数: {np.median(point_counts):.2f}")
    report.append(f"  - 标准差: {np.std(point_counts):.2f}")
    report.append("")
    
    report.append("边界框尺寸统计 (nm):")
    report.append("  X方向:")
    report.append(f"    - 最小值: {np.min(bbox_sizes[:, 0]):.2f}")
    report.append(f"    - 最大值: {np.max(bbox_sizes[:, 0]):.2f}")
    report.append(f"    - 平均值: {np.mean(bbox_sizes[:, 0]):.2f}")
    report.append("")
    report.append("  Y方向:")
    report.append(f"    - 最小值: {np.min(bbox_sizes[:, 1]):.2f}")
    report.append(f"    - 最大值: {np.max(bbox_sizes[:, 1]):.2f}")
    report.append(f"    - 平均值: {np.mean(bbox_sizes[:, 1]):.2f}")
    report.append("")
    report.append("  Z方向:")
    report.append(f"    - 最小值: {np.min(bbox_sizes[:, 2]):.2f}")
    report.append(f"    - 最大值: {np.max(bbox_sizes[:, 2]):.2f}")
    report.append(f"    - 平均值: {np.mean(bbox_sizes[:, 2]):.2f}")
    report.append("")
    
    report.append("=" * 70)
    report.append("详细样本信息")
    report.append("=" * 70)
    report.append("")
    
    for i, (sample, point_count, bbox_size, source_file) in enumerate(
        zip(samples, point_counts, bbox_sizes, source_files)
    ):
        report.append(f"样本 {i + 1}:")
        report.append(f"  - 来源文件: {source_file.name}")
        report.append(f"  - 点数: {point_count}")
        report.append(
            f"  - 边界框: "
            f"X={bbox_size[0]:.2f} nm, "
            f"Y={bbox_size[1]:.2f} nm, "
            f"Z={bbox_size[2]:.2f} nm"
        )
        
        # 坐标范围
        min_coords = np.min(sample, axis=0)
        max_coords = np.max(sample, axis=0)
        report.append(
            f"  - X范围: [{min_coords[0]:.2f}, {max_coords[0]:.2f}] nm"
        )
        report.append(
            f"  - Y范围: [{min_coords[1]:.2f}, {max_coords[1]:.2f}] nm"
        )
        report.append(
            f"  - Z范围: [{min_coords[2]:.2f}, {max_coords[2]:.2f}] nm"
        )
        report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"已保存统计报告: {output_path}")
    
    # 同时打印到控制台
    print("\n" + report_text)


def generate_source_mapping_csv(
    samples: List[np.ndarray],
    source_files: List[Path],
    output_path: str
) -> None:
    """
    生成样本来源映射表CSV文件
    
    Args:
        samples (List[np.ndarray]): 点云样本列表
        source_files (List[Path]): 每个样本对应的源文件列表
        output_path (str): 输出CSV文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建映射数据
    mapping_data = []
    for i, (sample, source_file) in enumerate(zip(samples, source_files)):
        min_coords = np.min(sample, axis=0)
        max_coords = np.max(sample, axis=0)
        bbox_size = max_coords - min_coords
        
        mapping_data.append({
            'sample_id': f"sample_{i + 1:04d}",
            'sample_index': i + 1,
            'source_file': str(source_file),
            'source_filename': source_file.name,
            'point_count': len(sample),
            'bbox_x_nm': bbox_size[0],
            'bbox_y_nm': bbox_size[1],
            'bbox_z_nm': bbox_size[2],
            'min_x_nm': min_coords[0],
            'min_y_nm': min_coords[1],
            'min_z_nm': min_coords[2],
            'max_x_nm': max_coords[0],
            'max_y_nm': max_coords[1],
            'max_z_nm': max_coords[2]
        })
    
    # 保存为CSV
    df = pd.DataFrame(mapping_data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"已保存样本来源映射表: {output_path}")
    logger.info(f"  - 包含 {len(mapping_data)} 个样本的详细信息")
    logger.info(f"  - 涉及 {len(set(source_files))} 个不同的源文件")


def test_pointcloud_processing(
    input_dir: str,
    output_dir: str,
    num_samples: int = 10,
    region_size: tuple = (1000.0, 1000.0, 500.0),
    target_points: int = 1024,
    min_points: int = 100,
    max_attempts: int = 10,
    decimals: int = 2,
    random_seed: int = 42
) -> None:
    """
    测试点云处理流程的主函数
    
    Args:
        input_dir (str): 输入CSV文件目录
        output_dir (str): 输出目录
        num_samples (int): 生成样本数量
        region_size (tuple): 提取区域大小
        target_points (int): 期望点数
        min_points (int): 最小点数要求
        max_attempts (int): 最大尝试次数
        decimals (int): 坐标精度
        random_seed (int): 随机种子
    """
    output_path = Path(output_dir)
    
    # 创建输出子目录
    csv_output_dir = output_path / "csv_samples"
    png_output_dir = output_path / "png_visualizations"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    png_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 查找CSV文件
        logger.info("=" * 70)
        logger.info("步骤 1: 查找CSV文件")
        logger.info("=" * 70)
        csv_files = find_all_csv_files(input_dir)
        
        # 2. 生成样本
        logger.info("\n" + "=" * 70)
        logger.info("步骤 2: 生成点云样本")
        logger.info("=" * 70)
        samples, source_files = generate_samples_with_source(
            csv_files=csv_files,
            num_samples=num_samples,
            region_size=region_size,
            target_points=target_points,
            min_points=min_points,
            max_attempts=max_attempts,
            decimals=decimals,
            random_seed=random_seed
        )
        
        # 3. 保存CSV和可视化
        logger.info("\n" + "=" * 70)
        logger.info("步骤 3: 保存CSV文件和可视化")
        logger.info("=" * 70)
        
        for i, (sample, source_file) in enumerate(tqdm(zip(samples, source_files), desc="保存文件", total=len(samples))):
            # 保存CSV
            csv_path = csv_output_dir / f"sample_{i + 1:04d}.csv"
            save_pointcloud_as_csv(sample, str(csv_path), source_file)
            
            # 保存可视化
            png_path = png_output_dir / f"sample_{i + 1:04d}.png"
            visualize_pointcloud_xy(sample, str(png_path))
        
        # 4. 生成统计报告
        logger.info("\n" + "=" * 70)
        logger.info("步骤 5: 生成统计报告")
        logger.info("=" * 70)
        report_path = output_path / "statistics_report.txt"
        generate_statistics_report(samples, source_files, str(report_path))
        
        # 5. 生成样本来源映射表
        logger.info("\n" + "=" * 70)
        logger.info("步骤 5: 生成样本来源映射表")
        logger.info("=" * 70)
        mapping_path = output_path / "sample_source_mapping.csv"
        generate_source_mapping_csv(samples, source_files, str(mapping_path))
        
        # 完成
        logger.info("\n" + "=" * 70)
        logger.info("测试完成！")
        logger.info("=" * 70)
        logger.info(f"输出目录: {output_path}")
        logger.info(f"  - CSV文件: {csv_output_dir}")
        logger.info(f"  - PNG可视化: {png_output_dir}")
        logger.info(f"  - 统计报告: {report_path}")
        logger.info(f"  - 来源映射表: {mapping_path}")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        raise


def main():
    """主函数：运行测试"""
    
    # =========================
    # 配置参数
    # =========================
    
    # 输入目录（包含CSV文件）
    INPUT_DIR = "/repos/datasets/exp-data-4pi-pc-mt"
    
    # 输出目录
    OUTPUT_DIR = "/repos/datasets/exp-data-4pi-pc-mt/mt_exp_pointcloud_samples_test"
    
    # 生成样本数量（测试用，建议使用较小的值）
    NUM_SAMPLES = 50
    
    # 提取区域大小（单位：nm）
    REGION_SIZE = (8000.0, 8000.0, 300.0)  # x, y, z
    
    # 期望点数
    TARGET_POINTS = 20000 * 4
    
    # 最小点数要求
    MIN_POINTS = 20000 * 4
    
    # 每个CSV的最大尝试次数
    MAX_ATTEMPTS = 20
    
    # 坐标精度（小数位数）
    DECIMALS = 2
    
    # 随机种子
    RANDOM_SEED = random.randint(0, 1000000)
    
    # =========================
    # 运行测试
    # =========================
    
    test_pointcloud_processing(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        region_size=REGION_SIZE,
        target_points=TARGET_POINTS,
        min_points=MIN_POINTS,
        max_attempts=MAX_ATTEMPTS,
        decimals=DECIMALS,
        random_seed=RANDOM_SEED
    )


if __name__ == "__main__":
    main()
