"""
H5点云数据统计分析工具

用于读取HDF5格式的点云数据文件，并输出详细的统计信息。

功能：
    - 加载H5点云数据
    - 统计样本数、点数、数据类型
    - 计算坐标范围（最小值、最大值、均值、标准差）
    - 分析点云密度分布
    - 检测异常值和空点云
    - 生成详细的统计报告

Usage:
    python scripts/exp_data_process/check_h5.py --input path/to/data.h5
    python scripts/exp_data_process/check_h5.py --input path/to/data.h5 --key pointclouds --detailed
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
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


def compute_statistics(loader: PointCloudH5Loader, detailed: bool = False, 
                      sample_size: Optional[int] = None, outlier_threshold: float = 3.0) -> Dict[str, Any]:
    """
    计算点云数据的统计信息
    
    Args:
        loader (PointCloudH5Loader): H5加载器实例
        detailed (bool): 是否进行详细分析（会加载所有数据，较慢）
        sample_size (Optional[int]): 抽样大小，None表示使用全部数据
        outlier_threshold (float): 异常检测阈值（标准差的倍数）
        
    Returns:
        Dict[str, Any]: 包含统计信息的字典
    """
    stats = {}
    
    # 基本信息
    stats['num_samples'] = loader.num_samples
    stats['num_points_per_sample'] = loader.num_points
    stats['data_shape'] = loader.data_shape
    
    # 确定分析的样本数量
    if sample_size is None or sample_size >= loader.num_samples:
        sample_indices = list(range(loader.num_samples))
        stats['is_full_analysis'] = True
    else:
        sample_indices = np.random.choice(loader.num_samples, sample_size, replace=False).tolist()
        stats['is_full_analysis'] = False
    
    stats['analyzed_samples'] = len(sample_indices)
    stats['sample_indices'] = sample_indices  # 保存分析的样本索引
    
    # 初始化统计变量
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    point_counts = []
    empty_samples = []
    sample_data = []  # 保存每个样本的详细信息
    
    # 逐样本统计
    logger.info(f"正在分析 {len(sample_indices)} 个样本...")
    for idx in tqdm(sample_indices, desc="计算统计信息"):
        cloud = loader.load_single_cloud(idx)
        
        # 检测空点云
        if len(cloud) == 0 or np.all(cloud == 0):
            empty_samples.append(idx)
            continue
        
        # 统计有效点数（排除全零点）
        valid_mask = ~np.all(cloud == 0, axis=1)
        valid_cloud = cloud[valid_mask]
        point_counts.append(len(valid_cloud))
        
        if len(valid_cloud) > 0:
            min_coords = np.min(valid_cloud, axis=0)
            max_coords = np.max(valid_cloud, axis=0)
            mean_coords = np.mean(valid_cloud, axis=0)
            std_coords = np.std(valid_cloud, axis=0)
            
            all_mins.append(min_coords)
            all_maxs.append(max_coords)
            all_means.append(mean_coords)
            all_stds.append(std_coords)
            
            # 保存样本详细信息
            sample_data.append({
                'index': idx,
                'min': min_coords,
                'max': max_coords,
                'mean': mean_coords,
                'std': std_coords,
                'point_count': len(valid_cloud)
            })
    
    # 汇总统计
    if len(all_mins) > 0:
        all_mins = np.array(all_mins)
        all_maxs = np.array(all_maxs)
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        
        # 全局坐标范围
        stats['global_min'] = np.min(all_mins, axis=0)
        stats['global_max'] = np.max(all_maxs, axis=0)
        stats['global_range'] = stats['global_max'] - stats['global_min']
        
        # 平均坐标统计
        stats['mean_coords'] = np.mean(all_means, axis=0)
        stats['std_coords'] = np.mean(all_stds, axis=0)
        
        # 计算所有样本中心点的总体标准差（用于异常检测）
        stats['mean_coords_std'] = np.std(all_means, axis=0)
        
        # 点数统计
        stats['point_count_mean'] = np.mean(point_counts)
        stats['point_count_std'] = np.std(point_counts)
        stats['point_count_min'] = np.min(point_counts)
        stats['point_count_max'] = np.max(point_counts)
        
        # 异常检测
        stats['empty_samples'] = empty_samples
        stats['num_empty_samples'] = len(empty_samples)
        
        # 密度分析（点云尺寸）
        dimensions = all_maxs - all_mins
        stats['avg_dimensions'] = np.mean(dimensions, axis=0)
        stats['std_dimensions'] = np.std(dimensions, axis=0)
        
        # 计算平均密度（点数/体积）
        volumes = np.prod(dimensions, axis=1)
        valid_volumes = volumes[volumes > 0]
        if len(valid_volumes) > 0:
            densities = np.array(point_counts)[volumes > 0] / valid_volumes
            stats['avg_density'] = np.mean(densities)
            stats['std_density'] = np.std(densities)
        
        # 检测偏离均值很远的异常点云
        stats['outlier_samples'] = detect_outliers(
            sample_data, 
            stats['mean_coords'], 
            stats['mean_coords_std'],
            threshold=outlier_threshold
        )
        stats['sample_data'] = sample_data  # 保存所有样本数据
        stats['outlier_threshold'] = outlier_threshold  # 保存阈值
        
    else:
        logger.warning("没有找到有效的点云数据！")
        stats['global_min'] = None
        stats['global_max'] = None
        stats['outlier_samples'] = []
        stats['sample_data'] = []
    
    return stats


def detect_outliers(sample_data: list, mean_coords: np.ndarray, std_coords: np.ndarray, 
                    threshold: float = 3.0) -> list:
    """
    检测偏离均值很远的异常点云
    
    Args:
        sample_data (list): 样本数据列表
        mean_coords (np.ndarray): 全局平均坐标
        std_coords (np.ndarray): 全局坐标标准差
        threshold (float): 异常阈值（标准差的倍数）
        
    Returns:
        list: 异常样本列表，每个元素包含样本索引和偏离信息
    """
    outliers = []
    
    for sample in sample_data:
        sample_mean = sample['mean']
        
        # 计算每个维度偏离均值的标准差倍数
        deviations = np.abs(sample_mean - mean_coords) / (std_coords + 1e-10)
        
        # 计算总体偏离程度（欧氏距离）
        total_deviation = np.linalg.norm(deviations)
        
        # 检查是否有任何维度超过阈值
        max_deviation = np.max(deviations)
        
        if max_deviation > threshold or total_deviation > threshold * np.sqrt(3):
            outliers.append({
                'index': sample['index'],
                'mean': sample_mean,
                'min': sample['min'],
                'max': sample['max'],
                'point_count': sample['point_count'],
                'deviations': deviations,
                'max_deviation': max_deviation,
                'total_deviation': total_deviation
            })
    
    # 按偏离程度排序
    outliers.sort(key=lambda x: x['total_deviation'], reverse=True)
    
    return outliers


def print_statistics_report(stats: Dict[str, Any], file_path: str) -> None:
    """
    打印统计报告
    
    Args:
        stats (Dict[str, Any]): 统计信息字典
        file_path (str): H5文件路径
    """
    print("\n" + "=" * 80)
    print("H5点云数据统计报告")
    print("=" * 80)
    print(f"文件路径: {file_path}")
    print()
    
    # 基本信息
    print("【基本信息】")
    print(f"  样本总数:        {stats['num_samples']}")
    print(f"  每样本点数:      {stats['num_points_per_sample']}")
    print(f"  数据Shape:       {stats['data_shape']}")
    print(f"  分析样本数:      {stats['analyzed_samples']}")
    print(f"  是否全量分析:    {'是' if stats['is_full_analysis'] else '否'}")
    print()
    
    if stats['global_min'] is not None:
        # 坐标范围
        print("【全局坐标范围】")
        print(f"  X轴: [{stats['global_min'][0]:>10.2f}, {stats['global_max'][0]:>10.2f}] nm  (范围: {stats['global_range'][0]:>10.2f} nm)")
        print(f"  Y轴: [{stats['global_min'][1]:>10.2f}, {stats['global_max'][1]:>10.2f}] nm  (范围: {stats['global_range'][1]:>10.2f} nm)")
        print(f"  Z轴: [{stats['global_min'][2]:>10.2f}, {stats['global_max'][2]:>10.2f}] nm  (范围: {stats['global_range'][2]:>10.2f} nm)")
        print()
        
        # 平均坐标
        print("【平均坐标中心】")
        print(f"  X: {stats['mean_coords'][0]:>10.2f} ± {stats['std_coords'][0]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][0]:>8.2f})")
        print(f"  Y: {stats['mean_coords'][1]:>10.2f} ± {stats['std_coords'][1]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][1]:>8.2f})")
        print(f"  Z: {stats['mean_coords'][2]:>10.2f} ± {stats['std_coords'][2]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][2]:>8.2f})")
        print()
        
        # 点数统计
        print("【点数统计】")
        print(f"  平均点数:        {stats['point_count_mean']:>10.1f} ± {stats['point_count_std']:>8.1f}")
        print(f"  最少点数:        {stats['point_count_min']:>10d}")
        print(f"  最多点数:        {stats['point_count_max']:>10d}")
        print()
        
        # 尺寸统计
        print("【平均点云尺寸】")
        print(f"  X方向: {stats['avg_dimensions'][0]:>10.2f} ± {stats['std_dimensions'][0]:>8.2f} nm")
        print(f"  Y方向: {stats['avg_dimensions'][1]:>10.2f} ± {stats['std_dimensions'][1]:>8.2f} nm")
        print(f"  Z方向: {stats['avg_dimensions'][2]:>10.2f} ± {stats['std_dimensions'][2]:>8.2f} nm")
        print()
        
        # 密度统计
        if 'avg_density' in stats:
            print("【点云密度】")
            print(f"  平均密度:        {stats['avg_density']:>10.6f} ± {stats['std_density']:>8.6f} 点/nm³")
            print()
        
        # 异常检测
        print("【数据质量】")
        print(f"  空样本数:        {stats['num_empty_samples']}")
        if stats['num_empty_samples'] > 0:
            print(f"  空样本索引:      {stats['empty_samples'][:10]}" + 
                  ("..." if len(stats['empty_samples']) > 10 else ""))
        
        # 异常点云检测
        outliers = stats.get('outlier_samples', [])
        threshold = stats.get('outlier_threshold', 3.0)
        print(f"  异常样本数:      {len(outliers)} (偏离均值>{threshold}σ)")
        print()
    else:
        print("【警告】未找到有效的点云数据！")
        print()
    
    # 打印异常点云详细信息
    outliers = stats.get('outlier_samples', [])
    if len(outliers) > 0:
        threshold = stats.get('outlier_threshold', 3.0)
        print("=" * 80)
        print(f"【异常点云详细信息】（偏离均值>{threshold}σ的样本，共{len(outliers)}个）")
        print("=" * 80)
        print()
        
        mean_coords = stats['mean_coords']
        
        for i, outlier in enumerate(outliers[:20], 1):  # 最多显示20个
            idx = outlier['index']
            mean = outlier['mean']
            mins = outlier['min']
            maxs = outlier['max']
            deviations = outlier['deviations']
            
            print(f"异常样本 #{i}: Index {idx}")
            print(f"  点数:            {outlier['point_count']}")
            print(f"  偏离程度:        {outlier['total_deviation']:.2f}σ (最大单维度: {outlier['max_deviation']:.2f}σ)")
            print(f"  中心坐标:        X={mean[0]:>10.2f}, Y={mean[1]:>10.2f}, Z={mean[2]:>10.2f} nm")
            print(f"  与均值差异:      ΔX={mean[0]-mean_coords[0]:>+10.2f} ({deviations[0]:>5.2f}σ), "
                  f"ΔY={mean[1]-mean_coords[1]:>+10.2f} ({deviations[1]:>5.2f}σ), "
                  f"ΔZ={mean[2]-mean_coords[2]:>+10.2f} ({deviations[2]:>5.2f}σ)")
            print("  坐标范围:")
            print(f"    X: [{mins[0]:>10.2f}, {maxs[0]:>10.2f}] nm  (尺寸: {maxs[0]-mins[0]:>10.2f})")
            print(f"    Y: [{mins[1]:>10.2f}, {maxs[1]:>10.2f}] nm  (尺寸: {maxs[1]-mins[1]:>10.2f})")
            print(f"    Z: [{mins[2]:>10.2f}, {maxs[2]:>10.2f}] nm  (尺寸: {maxs[2]-mins[2]:>10.2f})")
            print()
        
        if len(outliers) > 20:
            print(f"... 还有 {len(outliers)-20} 个异常样本未显示")
            print()
    
    print("=" * 80)


def save_statistics_report(stats: Dict[str, Any], file_path: str, output_path: str) -> None:
    """
    保存统计报告到文件
    
    Args:
        stats (Dict[str, Any]): 统计信息字典
        file_path (str): H5文件路径
        output_path (str): 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("H5点云数据统计报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"文件路径: {file_path}\n\n")
        
        # 基本信息
        f.write("【基本信息】\n")
        f.write(f"  样本总数:        {stats['num_samples']}\n")
        f.write(f"  每样本点数:      {stats['num_points_per_sample']}\n")
        f.write(f"  数据Shape:       {stats['data_shape']}\n")
        f.write(f"  分析样本数:      {stats['analyzed_samples']}\n")
        f.write(f"  是否全量分析:    {'是' if stats['is_full_analysis'] else '否'}\n\n")
        
        if stats['global_min'] is not None:
            # 坐标范围
            f.write("【全局坐标范围】\n")
            f.write(f"  X轴: [{stats['global_min'][0]:>10.2f}, {stats['global_max'][0]:>10.2f}] nm  (范围: {stats['global_range'][0]:>10.2f} nm)\n")
            f.write(f"  Y轴: [{stats['global_min'][1]:>10.2f}, {stats['global_max'][1]:>10.2f}] nm  (范围: {stats['global_range'][1]:>10.2f} nm)\n")
            f.write(f"  Z轴: [{stats['global_min'][2]:>10.2f}, {stats['global_max'][2]:>10.2f}] nm  (范围: {stats['global_range'][2]:>10.2f} nm)\n\n")
            
            # 平均坐标
            f.write("【平均坐标中心】\n")
            f.write(f"  X: {stats['mean_coords'][0]:>10.2f} ± {stats['std_coords'][0]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][0]:>8.2f})\n")
            f.write(f"  Y: {stats['mean_coords'][1]:>10.2f} ± {stats['std_coords'][1]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][1]:>8.2f})\n")
            f.write(f"  Z: {stats['mean_coords'][2]:>10.2f} ± {stats['std_coords'][2]:>8.2f} nm  (样本间std: {stats['mean_coords_std'][2]:>8.2f})\n\n")
            
            # 点数统计
            f.write("【点数统计】\n")
            f.write(f"  平均点数:        {stats['point_count_mean']:>10.1f} ± {stats['point_count_std']:>8.1f}\n")
            f.write(f"  最少点数:        {stats['point_count_min']:>10d}\n")
            f.write(f"  最多点数:        {stats['point_count_max']:>10d}\n\n")
            
            # 尺寸统计
            f.write("【平均点云尺寸】\n")
            f.write(f"  X方向: {stats['avg_dimensions'][0]:>10.2f} ± {stats['std_dimensions'][0]:>8.2f} nm\n")
            f.write(f"  Y方向: {stats['avg_dimensions'][1]:>10.2f} ± {stats['std_dimensions'][1]:>8.2f} nm\n")
            f.write(f"  Z方向: {stats['avg_dimensions'][2]:>10.2f} ± {stats['std_dimensions'][2]:>8.2f} nm\n\n")
            
            # 密度统计
            if 'avg_density' in stats:
                f.write("【点云密度】\n")
                f.write(f"  平均密度:        {stats['avg_density']:>10.6f} ± {stats['std_density']:>8.6f} 点/nm³\n\n")
            
            # 异常检测
            f.write("【数据质量】\n")
            f.write(f"  空样本数:        {stats['num_empty_samples']}\n")
            if stats['num_empty_samples'] > 0:
                f.write(f"  空样本索引:      {stats['empty_samples']}\n")
            
            # 异常点云检测
            outliers = stats.get('outlier_samples', [])
            threshold = stats.get('outlier_threshold', 3.0)
            f.write(f"  异常样本数:      {len(outliers)} (偏离均值>{threshold}σ)\n")
            f.write("\n")
        else:
            f.write("【警告】未找到有效的点云数据！\n\n")
    
    # 保存异常点云详细信息
    outliers = stats.get('outlier_samples', [])
    if len(outliers) > 0:
        threshold = stats.get('outlier_threshold', 3.0)
        f.write("=" * 80 + "\n")
        f.write(f"【异常点云详细信息】（偏离均值>{threshold}σ的样本，共{len(outliers)}个）\n")
        f.write("=" * 80 + "\n\n")
        
        mean_coords = stats['mean_coords']
        
        for i, outlier in enumerate(outliers, 1):
            idx = outlier['index']
            mean = outlier['mean']
            mins = outlier['min']
            maxs = outlier['max']
            deviations = outlier['deviations']
            
            f.write(f"异常样本 #{i}: Index {idx}\n")
            f.write(f"  点数:            {outlier['point_count']}\n")
            f.write(f"  偏离程度:        {outlier['total_deviation']:.2f}σ (最大单维度: {outlier['max_deviation']:.2f}σ)\n")
            f.write(f"  中心坐标:        X={mean[0]:>10.2f}, Y={mean[1]:>10.2f}, Z={mean[2]:>10.2f} nm\n")
            f.write(f"  与均值差异:      ΔX={mean[0]-mean_coords[0]:>+10.2f} ({deviations[0]:>5.2f}σ), "
                    f"ΔY={mean[1]-mean_coords[1]:>+10.2f} ({deviations[1]:>5.2f}σ), "
                    f"ΔZ={mean[2]-mean_coords[2]:>+10.2f} ({deviations[2]:>5.2f}σ)\n")
            f.write("  坐标范围:\n")
            f.write(f"    X: [{mins[0]:>10.2f}, {maxs[0]:>10.2f}] nm  (尺寸: {maxs[0]-mins[0]:>10.2f})\n")
            f.write(f"    Y: [{mins[1]:>10.2f}, {maxs[1]:>10.2f}] nm  (尺寸: {maxs[1]-mins[1]:>10.2f})\n")
            f.write(f"    Z: [{mins[2]:>10.2f}, {maxs[2]:>10.2f}] nm  (尺寸: {maxs[2]-mins[2]:>10.2f})\n")
            f.write("\n")
    
    f.write("=" * 80 + "\n")
    
    logger.info(f"统计报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="H5点云数据统计分析工具")
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="/repos/datasets/tissue-datasets/pointclouds-clean-check.h5",
        help="输入H5文件路径"
    )
    parser.add_argument(
        "--key", "-k", 
        type=str, 
        default="pointclouds",
        help="H5文件中的数据键名"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default=None,
        help="输出报告文件路径 (可选，不指定则只打印到控制台)"
    )
    parser.add_argument(
        "--detailed", "-d", 
        action="store_true",
        help="进行详细分析（分析所有样本，较慢）"
    )
    parser.add_argument(
        "--sample-size", "-s", 
        type=int, 
        default=None,
        help="抽样分析的样本数量 (默认: None，使用全部数据)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--outlier-threshold", "-t", 
        type=float, 
        default=3.0,
        help="异常检测阈值（标准差的倍数，默认: 3.0）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    input_path = Path(args.input)
    
    # 检查文件是否存在
    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        sys.exit(1)
    
    try:
        # 1. 加载数据
        logger.info(f"正在加载H5文件: {input_path}")
        loader = PointCloudH5Loader(str(input_path), data_key=args.key)
        
        # 2. 计算统计信息
        sample_size = None if args.detailed else args.sample_size
        if sample_size is None and not args.detailed:
            # 默认抽样策略：如果样本数超过1000，则抽样1000个
            sample_size = min(1000, loader.num_samples)
        
        stats = compute_statistics(
            loader, 
            detailed=args.detailed, 
            sample_size=sample_size,
            outlier_threshold=args.outlier_threshold
        )
        
        # 3. 打印报告
        print_statistics_report(stats, str(input_path))
        
        # 4. 保存报告（如果指定了输出路径）
        if args.output:
            save_statistics_report(stats, str(input_path), args.output)
        
        logger.info("分析完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

