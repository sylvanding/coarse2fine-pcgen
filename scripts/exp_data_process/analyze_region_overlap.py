"""
点云区域重叠率分析脚本

分析从同一CSV文件中随机提取多个区域时的重叠程度，用于诊断数据增强策略是否存在问题。

功能：
    - 从每个CSV文件中多次随机提取区域
    - 计算区域边界框的IoU（Intersection over Union）
    - 统计平均重叠率、最大重叠率、重叠分布
    - 生成可视化报告

Usage:
    python scripts/exp_data_process/analyze_region_overlap.py

Author: AI Assistant
Date: 2025-10-10
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_csv_files(root_dir: str) -> List[Path]:
    """
    递归查找目录下所有必要列存在的CSV文件
    
    Args:
        root_dir (str): 根目录路径
        
    Returns:
        List[Path]: 必需列存在的CSV文件路径列表
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise ValueError(f"目录不存在: {root_dir}")
    
    if not root_path.is_dir():
        raise ValueError(f"路径不是目录: {root_dir}")
    
    # 递归查找所有CSV文件
    csv_files = list(root_path.rglob("*.csv"))
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {root_dir} 及其子目录中未找到CSV文件")
    
    # 只保留必需的列存在的csv文件
    required_columns = ["x [nm]", "y [nm]", "z [nm]"]
    valid_csv_files = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file, nrows=1)  # 只读第一行来检查列
            if all(col in df.columns for col in required_columns):
                valid_csv_files.append(file)
        except Exception as e:
            logger.warning(f"无法读取文件 {file}: {e}")
    
    if len(valid_csv_files) == 0:
        raise ValueError(f"在 {root_dir} 及其子目录中未找到必需的列: {required_columns}")
    
    logger.info(f"找到 {len(valid_csv_files)} 个符合要求的CSV文件")
    return valid_csv_files


def load_csv_pointcloud(csv_path: Path) -> Optional[np.ndarray]:
    """
    从CSV文件加载点云数据
    
    Args:
        csv_path (Path): CSV文件路径
        
    Returns:
        Optional[np.ndarray]: 形状为(N, 3)的点云数组，如果加载失败返回None
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["x [nm]", "y [nm]", "z [nm]"]
        
        if not all(col in df.columns for col in required_columns):
            return None
        
        points = df[required_columns].values.astype(np.float64)
        
        if len(points) == 0:
            return None
        
        # 移除NaN值
        if np.any(np.isnan(points)):
            points = points[~np.isnan(points).any(axis=1)]
            if len(points) == 0:
                return None
        
        return points
        
    except Exception as e:
        logger.error(f"加载文件 {csv_path.name} 时出错: {e}")
        return None


def extract_random_region_with_bounds(
    points: np.ndarray,
    region_size: Tuple[float, float, float],
    min_points: int = 100,
    max_attempts: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从点云中随机提取一个指定大小的区域，并返回区域边界
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云数据
        region_size (Tuple[float, float, float]): 区域大小(x_size, y_size, z_size)
        min_points (int): 区域内最少点数要求
        max_attempts (int): 最大尝试次数
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: 
            - 提取的点云区域（N, 3）
            - 区域边界框（2, 3），[min_coords, max_coords]
            如果失败返回None
    """
    if len(points) < min_points:
        return None
    
    # 计算点云边界
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    x_size, y_size, z_size = region_size
    
    # 检查点云是否足够大
    point_cloud_size = max_coords - min_coords
    if (point_cloud_size[0] < x_size or 
        point_cloud_size[1] < y_size or 
        point_cloud_size[2] < z_size):
        return None
    
    # 尝试随机提取区域
    for attempt in range(max_attempts):
        # 随机选择区域的起始点
        x_start = np.random.uniform(min_coords[0], max_coords[0] - x_size)
        y_start = np.random.uniform(min_coords[1], max_coords[1] - y_size)
        z_start = np.random.uniform(min_coords[2], max_coords[2] - z_size)
        
        # 定义区域边界
        x_end = x_start + x_size
        y_end = y_start + y_size
        z_end = z_start + z_size
        
        # 筛选在区域内的点
        mask = (
            (points[:, 0] >= x_start) & (points[:, 0] <= x_end) &
            (points[:, 1] >= y_start) & (points[:, 1] <= y_end) &
            (points[:, 2] >= z_start) & (points[:, 2] <= z_end)
        )
        
        region_points = points[mask]
        
        # 检查点数是否满足要求
        if len(region_points) >= min_points:
            # 返回点云和边界框
            bounds = np.array([
                [x_start, y_start, z_start],
                [x_end, y_end, z_end]
            ])
            return region_points, bounds
    
    return None


def calculate_bbox_iou_3d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    计算两个3D边界框的IoU（Intersection over Union）
    
    Args:
        bbox1 (np.ndarray): 边界框1，形状为(2, 3)，[min_coords, max_coords]
        bbox2 (np.ndarray): 边界框2，形状为(2, 3)，[min_coords, max_coords]
        
    Returns:
        float: IoU值，范围[0, 1]
    """
    # 解析边界框坐标
    min1, max1 = bbox1[0], bbox1[1]
    min2, max2 = bbox2[0], bbox2[1]
    
    # 计算交集的边界
    intersect_min = np.maximum(min1, min2)
    intersect_max = np.minimum(max1, max2)
    
    # 检查是否有交集
    if np.any(intersect_min >= intersect_max):
        return 0.0
    
    # 计算交集体积
    intersect_volume = np.prod(intersect_max - intersect_min)
    
    # 计算各自的体积
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    
    # 计算并集体积
    union_volume = volume1 + volume2 - intersect_volume
    
    # 计算IoU
    iou = intersect_volume / union_volume if union_volume > 0 else 0.0
    
    return iou


def extract_multiple_regions_from_csv(
    csv_path: Path,
    num_extractions: int,
    region_size: Tuple[float, float, float],
    min_points: int = 100,
    max_attempts: int = 10
) -> List[np.ndarray]:
    """
    从单个CSV文件中多次提取区域
    
    Args:
        csv_path (Path): CSV文件路径
        num_extractions (int): 提取次数
        region_size (Tuple[float, float, float]): 区域大小
        min_points (int): 最少点数
        max_attempts (int): 每次提取的最大尝试次数
        
    Returns:
        List[np.ndarray]: 提取的区域边界框列表，每个边界框形状为(2, 3)
    """
    # 加载点云
    points = load_csv_pointcloud(csv_path)
    if points is None:
        logger.warning(f"无法加载文件: {csv_path}")
        return []
    
    bounds_list = []
    
    for _ in range(num_extractions):
        result = extract_random_region_with_bounds(
            points, region_size, min_points, max_attempts
        )
        
        if result is not None:
            _, bounds = result
            bounds_list.append(bounds)
    
    return bounds_list


def calculate_overlap_statistics(bounds_list: List[np.ndarray]) -> Dict:
    """
    计算一组区域边界框的重叠统计信息
    
    Args:
        bounds_list (List[np.ndarray]): 边界框列表
        
    Returns:
        Dict: 包含各种统计信息的字典
    """
    if len(bounds_list) < 2:
        return {
            'num_regions': len(bounds_list),
            'num_pairs': 0,
            'mean_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'median_iou': 0.0,
            'iou_values': [],
            'high_overlap_ratio': 0.0  # IoU > 0.5的比例
        }
    
    # 计算所有区域对的IoU
    iou_values = []
    num_regions = len(bounds_list)
    
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            iou = calculate_bbox_iou_3d(bounds_list[i], bounds_list[j])
            iou_values.append(iou)
    
    iou_values = np.array(iou_values)
    
    # 统计信息
    stats = {
        'num_regions': num_regions,
        'num_pairs': len(iou_values),
        'mean_iou': float(np.mean(iou_values)),
        'max_iou': float(np.max(iou_values)),
        'min_iou': float(np.min(iou_values)),
        'median_iou': float(np.median(iou_values)),
        'std_iou': float(np.std(iou_values)),
        'iou_values': iou_values.tolist(),
        'high_overlap_ratio': float(np.sum(iou_values > 0.5) / len(iou_values))
    }
    
    return stats


def analyze_single_csv(
    csv_path: Path,
    num_extractions: int,
    region_size: Tuple[float, float, float],
    min_points: int = 100,
    max_attempts: int = 10
) -> Dict:
    """
    分析单个CSV文件的区域重叠情况
    
    Args:
        csv_path (Path): CSV文件路径
        num_extractions (int): 提取次数
        region_size (Tuple[float, float, float]): 区域大小
        min_points (int): 最少点数
        max_attempts (int): 每次提取的最大尝试次数
        
    Returns:
        Dict: 分析结果
    """
    logger.info(f"分析文件: {csv_path.name}")
    
    # 提取多个区域
    bounds_list = extract_multiple_regions_from_csv(
        csv_path, num_extractions, region_size, min_points, max_attempts
    )
    
    if len(bounds_list) == 0:
        logger.warning(f"文件 {csv_path.name} 无法提取任何有效区域")
        return None
    
    # 计算统计信息
    stats = calculate_overlap_statistics(bounds_list)
    stats['csv_file'] = str(csv_path)
    stats['extraction_success_rate'] = len(bounds_list) / num_extractions
    
    logger.info(f"  成功提取: {len(bounds_list)}/{num_extractions} 个区域")
    logger.info(f"  平均IoU: {stats['mean_iou']:.4f}")
    logger.info(f"  最大IoU: {stats['max_iou']:.4f}")
    logger.info(f"  高重叠率(>0.5): {stats['high_overlap_ratio']:.2%}")
    
    return stats


def plot_iou_distribution(all_stats: List[Dict], output_dir: Path):
    """
    绘制IoU分布图
    
    Args:
        all_stats (List[Dict]): 所有CSV文件的统计信息
        output_dir (Path): 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 绘制所有IoU值的直方图
    plt.figure(figsize=(12, 5))
    
    # 子图1: 所有IoU值的分布
    plt.subplot(1, 2, 1)
    all_ious = []
    for stats in all_stats:
        if stats and 'iou_values' in stats:
            all_ious.extend(stats['iou_values'])
    
    if len(all_ious) > 0:
        plt.hist(all_ious, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('IoU value')
        plt.ylabel('Frequency')
        plt.title('IoU distribution of all region pairs')
        plt.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5 threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 子图2: 每个CSV的平均IoU
    plt.subplot(1, 2, 2)
    mean_ious = [stats['mean_iou'] for stats in all_stats if stats]
    if len(mean_ious) > 0:
        plt.hist(mean_ious, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Average IoU')
        plt.ylabel('CSV file number')
        plt.title('Average IoU distribution of each CSV file')
        plt.axvline(x=np.mean(mean_ious), color='r', linestyle='--', 
                   label=f'Overall average: {np.mean(mean_ious):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"IoU分布图已保存到: {output_dir / 'iou_distribution.png'}")
    plt.close()
    
    # 2. 绘制各CSV文件的统计箱线图
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    csv_names = []
    iou_data = []
    for stats in all_stats[:20]:  # 最多显示20个文件
        if stats and len(stats['iou_values']) > 0:
            csv_names.append(Path(stats['csv_file']).stem[:20])  # 截断文件名
            iou_data.append(stats['iou_values'])
    
    if len(iou_data) > 0:
        plt.boxplot(iou_data, labels=csv_names)
        plt.xlabel('CSV file')
        plt.ylabel('IoU value')
        plt.title('IoU distribution of each CSV file (boxplot)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='IoU=0.5')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'iou_boxplot.png', dpi=300, bbox_inches='tight')
        logger.info(f"IoU boxplot has been saved to: {output_dir / 'iou_boxplot.png'}")
    
    plt.close()


def save_analysis_report(all_stats: List[Dict], output_path: Path):
    """
    保存分析报告
    
    Args:
        all_stats (List[Dict]): 所有CSV文件的统计信息
        output_path (Path): 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 计算总体统计
    valid_stats = [s for s in all_stats if s]
    
    if len(valid_stats) == 0:
        logger.warning("No valid statistics data")
        return
    
    mean_ious = [s['mean_iou'] for s in valid_stats]
    max_ious = [s['max_iou'] for s in valid_stats]
    high_overlap_ratios = [s['high_overlap_ratio'] for s in valid_stats]
    
    summary = {
        'total_csv_files': len(all_stats),
        'valid_csv_files': len(valid_stats),
        'overall_statistics': {
            'mean_of_mean_ious': float(np.mean(mean_ious)),
            'std_of_mean_ious': float(np.std(mean_ious)),
            'median_of_mean_ious': float(np.median(mean_ious)),
            'max_of_max_ious': float(np.max(max_ious)),
            'mean_high_overlap_ratio': float(np.mean(high_overlap_ratios))
        },
        'per_file_statistics': valid_stats
    }
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis report has been saved to: {output_path}")
    
    # 同时保存一个人类可读的文本报告
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("点云区域重叠率分析报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  - 分析的CSV文件数: {len(valid_stats)}\n")
        f.write(f"  - 平均IoU的均值: {summary['overall_statistics']['mean_of_mean_ious']:.4f}\n")
        f.write(f"  - 平均IoU的标准差: {summary['overall_statistics']['std_of_mean_ious']:.4f}\n")
        f.write(f"  - 平均IoU的中位数: {summary['overall_statistics']['median_of_mean_ious']:.4f}\n")
        f.write(f"  - 所有文件中的最大IoU: {summary['overall_statistics']['max_of_max_ious']:.4f}\n")
        f.write(f"  - 高重叠(IoU>0.5)区域对的平均比例: {summary['overall_statistics']['mean_high_overlap_ratio']:.2%}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        
        f.write("各CSV文件详细统计:\n")
        f.write("-" * 70 + "\n")
        
        for i, stats in enumerate(valid_stats, 1):
            f.write(f"\n{i}. {Path(stats['csv_file']).name}\n")
            f.write(f"   提取成功率: {stats['extraction_success_rate']:.2%}\n")
            f.write(f"   提取区域数: {stats['num_regions']}\n")
            f.write(f"   区域对数: {stats['num_pairs']}\n")
            f.write(f"   平均IoU: {stats['mean_iou']:.4f}\n")
            f.write(f"   最大IoU: {stats['max_iou']:.4f}\n")
            f.write(f"   中位数IoU: {stats['median_iou']:.4f}\n")
            f.write(f"   IoU标准差: {stats['std_iou']:.4f}\n")
            f.write(f"   高重叠率(>0.5): {stats['high_overlap_ratio']:.2%}\n")
    
    logger.info(f"文本报告已保存到: {txt_path}")


def main():
    """主函数：执行重叠率分析"""
    
    # =========================
    # 配置参数
    # =========================
    
    # 输入目录（包含CSV文件）
    INPUT_DIR = "/repos/datasets/exp-data-4pi-pc-mitochondria/Mitochondria_for_Agent"
    
    # 输出目录
    OUTPUT_DIR = "/repos/datasets/exp-data-4pi-pc-mitochondria/overlap_analysis"
    
    # 每个CSV文件的提取次数
    NUM_EXTRACTIONS = 100
    
    # 提取区域大小（单位：nm）- 与process_pointcloud_data.py中的配置保持一致
    REGION_SIZE = (8000.0, 8000.0, 1200.0)  # x, y, z
    
    # 最小点数要求
    MIN_POINTS = 40000
    
    # 每次提取的最大尝试次数
    MAX_ATTEMPTS = 20
    
    # 分析的CSV文件数量上限（可选，用于加快测试）
    MAX_CSV_FILES = None  # None表示分析所有文件，或设置为具体数字如20
    
    # =========================
    # 执行分析流程
    # =========================
    
    try:
        logger.info("=" * 70)
        logger.info("开始点云区域重叠率分析")
        logger.info("=" * 70)
        logger.info(f"配置参数:")
        logger.info(f"  - 输入目录: {INPUT_DIR}")
        logger.info(f"  - 输出目录: {OUTPUT_DIR}")
        logger.info(f"  - 每个CSV提取次数: {NUM_EXTRACTIONS}")
        logger.info(f"  - 区域大小: {REGION_SIZE}")
        logger.info(f"  - 最小点数: {MIN_POINTS}")
        logger.info(f"  - 最大尝试次数: {MAX_ATTEMPTS}")
        logger.info("=" * 70 + "\n")
        
        # 1. 查找所有CSV文件
        csv_files = find_all_csv_files(INPUT_DIR)
        
        # 限制文件数量（可选）
        if MAX_CSV_FILES is not None and len(csv_files) > MAX_CSV_FILES:
            logger.info(f"限制分析文件数量: {MAX_CSV_FILES}")
            csv_files = csv_files[:MAX_CSV_FILES]
        
        # 2. 分析每个CSV文件
        all_stats = []
        
        for csv_file in tqdm(csv_files, desc="分析CSV文件"):
            stats = analyze_single_csv(
                csv_file,
                NUM_EXTRACTIONS,
                REGION_SIZE,
                MIN_POINTS,
                MAX_ATTEMPTS
            )
            all_stats.append(stats)
        
        # 3. 生成报告和可视化
        output_path = Path(OUTPUT_DIR)
        
        # 保存分析报告
        save_analysis_report(all_stats, output_path / "overlap_analysis_report.json")
        
        # 绘制可视化图表
        plot_iou_distribution(all_stats, output_path)
        
        logger.info("\n" + "=" * 70)
        logger.info("分析完成！")
        logger.info("=" * 70)
        
        # 打印关键发现
        valid_stats = [s for s in all_stats if s]
        if valid_stats:
            mean_ious = [s['mean_iou'] for s in valid_stats]
            high_overlap_ratios = [s['high_overlap_ratio'] for s in valid_stats]
            
            logger.info("\n关键发现:")
            logger.info(f"  - 总体平均IoU: {np.mean(mean_ious):.4f}")
            logger.info(f"  - 平均高重叠率(>0.5): {np.mean(high_overlap_ratios):.2%}")
            logger.info(f"\n建议:")
            
            avg_iou = np.mean(mean_ious)
            avg_high_overlap = np.mean(high_overlap_ratios)
            
            if avg_iou > 0.3 or avg_high_overlap > 0.2:
                logger.warning("  ⚠️  检测到较高的区域重叠率！")
                logger.warning("  可能的解决方案:")
                logger.warning("    1. 增大REGION_SIZE，使每次提取的区域更大")
                logger.warning("    2. 减小MAX_ATTEMPTS，避免在同一区域多次提取")
                logger.warning("    3. 考虑使用非重叠采样策略")
                logger.warning("    4. 增加更多不同的CSV源文件")
            else:
                logger.info("  ✓ 区域重叠率在可接受范围内")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

