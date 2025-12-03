"""
H5点云数据可视化与分析工具

用于读取HDF5格式的点云数据文件，进行基本统计分析，
并随机选择样本生成XY平面投影的可视化图片。

功能：
    - 加载H5点云数据
    - 统计样本数、点数、坐标范围
    - 随机抽取样本进行可视化（XY平面投影，Z轴颜色映射）
    - 生成统计报告

Usage:
    python scripts/exp_data_process/viz_h5.py --input path/to/data.h5 --output path/to/output --num-samples 10
"""

import sys
import argparse
import logging
from pathlib import Path
import random
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.data.h5_loader import PointCloudH5Loader  # noqa: E402

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_pointcloud_xy(
    points: np.ndarray,
    output_path: str,
    figsize: tuple = (8, 8),
    dpi: int = 200,
    cmap: str = 'jet',
    point_size: float = 0.5,
    show_colorbar: bool = True,
    title: Optional[str] = None
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
        title (Optional[str]): 图片标题
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
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z [nm]', fontsize=12, fontweight='bold')
    
    # 添加统计信息文本框
    stats_text = (
        f"Points: {len(points)}\n"
        f"X: [{x.min():.1f}, {x.max():.1f}]\n"
        f"Y: [{y.min():.1f}, {y.max():.1f}]\n"
        f"Z: [{z.min():.1f}, {z.max():.1f}]"
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


def generate_report(
    loader: PointCloudH5Loader,
    sampled_indices: List[int],
    output_path: str
) -> None:
    """
    生成分析报告
    
    Args:
        loader (PointCloudH5Loader): H5加载器实例
        sampled_indices (List[int]): 被抽样可视化的样本索引
        output_path (str): 报告保存路径
    """
    output_path = Path(output_path)
    
    # 获取整体统计信息
    num_samples = loader.num_samples
    num_points = loader.num_points
    
    # 计算整体边界框（使用随机采样的子集来估算，避免加载全部数据）
    sample_subset_size = min(100, num_samples)
    subset_indices = random.sample(range(num_samples), sample_subset_size)
    min_bounds, max_bounds = loader.get_point_cloud_bounds(subset_indices)
    
    report = []
    report.append("=" * 70)
    report.append(f"H5文件分析报告: {loader.file_path}")
    report.append("=" * 70)
    report.append("")
    
    report.append("基本信息:")
    report.append(f"  - 样本总数: {num_samples}")
    report.append(f"  - 每样本点数: {num_points}")
    report.append(f"  - 数据Shape: {loader.data_shape}")
    report.append("")
    
    report.append(f"整体坐标范围 (基于 {sample_subset_size} 个随机样本估算):")
    report.append(f"  - X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}] nm")
    report.append(f"  - Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}] nm")
    report.append(f"  - Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}] nm")
    report.append("")
    
    report.append("=" * 70)
    report.append("抽样样本详细信息")
    report.append("=" * 70)
    report.append("")
    
    for idx in sampled_indices:
        cloud = loader.load_single_cloud(idx)
        mins = np.min(cloud, axis=0)
        maxs = np.max(cloud, axis=0)
        dims = maxs - mins
        
        report.append(f"样本 Index {idx}:")
        report.append(f"  - 点数: {len(cloud)}")
        report.append(f"  - 尺寸: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} nm")
        report.append(f"  - X范围: [{mins[0]:.1f}, {maxs[0]:.1f}]")
        report.append(f"  - Y范围: [{mins[1]:.1f}, {maxs[1]:.1f}]")
        report.append(f"  - Z范围: [{mins[2]:.1f}, {maxs[2]:.1f}]")
        report.append("")
        
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    logger.info(f"已保存分析报告: {output_path}")
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="H5点云数据可视化与分析")
    parser.add_argument("--input", "-i", type=str, default="/repos/datasets/tissue-datasets/pointclouds-clean-check.h5", help="输入H5文件路径")
    parser.add_argument("--output", "-o", type=str, default="/repos/datasets/tissue-datasets/pointclouds-clean-check-viz", help="输出目录路径")
    parser.add_argument("--num-samples", "-n", type=int, default=10, help="随机抽取可视化的样本数量")
    parser.add_argument("--seed", "-s", type=int, default=42, help="随机种子")
    parser.add_argument("--point-size", type=float, default=0.5, help="可视化点的大小")
    parser.add_argument("--key", type=str, default="pointclouds", help="H5文件中的数据键名")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化子目录
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        logger.info(f"正在加载H5文件: {input_path}")
        loader = PointCloudH5Loader(str(input_path), data_key=args.key)
        
        # 2. 选择样本
        total_samples = loader.num_samples
        num_viz = min(args.num_samples, total_samples)
        selected_indices = sorted(random.sample(range(total_samples), num_viz))
        
        logger.info(f"随机选择了 {num_viz} 个样本进行可视化")
        
        # 3. 生成可视化
        for idx in tqdm(selected_indices, desc="生成可视化图片"):
            cloud = loader.load_single_cloud(idx)
            output_path = viz_dir / f"sample_{idx:05d}.png"
            
            visualize_pointcloud_xy(
                points=cloud,
                output_path=str(output_path),
                point_size=args.point_size,
                title=f"Sample {idx}"
            )
            
        # # 4. 生成报告
        # report_path = output_dir / "analysis_report.txt"
        # generate_report(loader, selected_indices, str(report_path))
        
        logger.info("处理完成！")
        logger.info(f"可视化结果保存在: {viz_dir}")
        # logger.info(f"分析报告保存在: {report_path}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

