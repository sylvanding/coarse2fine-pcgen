"""
生成测试CSV数据

用于快速生成一些测试用的CSV点云文件，便于测试数据处理脚本。

生成的点云包括：
    - 球形点云
    - 立方体点云
    - 圆柱形点云
    - 随机点云

Usage:
    python scripts/exp-data-process/generate_test_csv.py

Author: AI Assistant
Date: 2025-10-05
"""

import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sphere_pointcloud(
    center: tuple = (5000, 5000, 2500),
    radius: float = 2000.0,
    num_points: int = 10000
) -> np.ndarray:
    """
    生成球形点云
    
    Args:
        center (tuple): 球心坐标 (x, y, z)
        radius (float): 半径
        num_points (int): 点数
        
    Returns:
        np.ndarray: 形状为(N, 3)的点云
    """
    # 使用球坐标生成均匀分布的点
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    r = radius * np.cbrt(np.random.uniform(0, 1, num_points))
    
    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)
    
    return np.column_stack([x, y, z])


def generate_cube_pointcloud(
    center: tuple = (5000, 5000, 2500),
    size: float = 3000.0,
    num_points: int = 10000
) -> np.ndarray:
    """
    生成立方体点云
    
    Args:
        center (tuple): 立方体中心坐标 (x, y, z)
        size (float): 立方体边长
        num_points (int): 点数
        
    Returns:
        np.ndarray: 形状为(N, 3)的点云
    """
    half_size = size / 2
    
    x = np.random.uniform(center[0] - half_size, center[0] + half_size, num_points)
    y = np.random.uniform(center[1] - half_size, center[1] + half_size, num_points)
    z = np.random.uniform(center[2] - half_size, center[2] + half_size, num_points)
    
    return np.column_stack([x, y, z])


def generate_cylinder_pointcloud(
    center: tuple = (5000, 5000, 2500),
    radius: float = 1500.0,
    height: float = 3000.0,
    num_points: int = 10000
) -> np.ndarray:
    """
    生成圆柱形点云
    
    Args:
        center (tuple): 圆柱中心坐标 (x, y, z)
        radius (float): 圆柱半径
        height (float): 圆柱高度
        num_points (int): 点数
        
    Returns:
        np.ndarray: 形状为(N, 3)的点云
    """
    # 在圆柱内均匀采样
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    z = np.random.uniform(center[2] - height/2, center[2] + height/2, num_points)
    
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    
    return np.column_stack([x, y, z])


def generate_random_pointcloud(
    bounds: tuple = ((0, 10000), (0, 10000), (0, 5000)),
    num_points: int = 10000
) -> np.ndarray:
    """
    生成随机点云
    
    Args:
        bounds (tuple): 坐标边界 ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        num_points (int): 点数
        
    Returns:
        np.ndarray: 形状为(N, 3)的点云
    """
    x = np.random.uniform(bounds[0][0], bounds[0][1], num_points)
    y = np.random.uniform(bounds[1][0], bounds[1][1], num_points)
    z = np.random.uniform(bounds[2][0], bounds[2][1], num_points)
    
    return np.column_stack([x, y, z])


def generate_clustered_pointcloud(
    num_clusters: int = 5,
    points_per_cluster: int = 2000,
    cluster_std: float = 500.0,
    bounds: tuple = ((0, 10000), (0, 10000), (0, 5000))
) -> np.ndarray:
    """
    生成聚类点云（多个高斯分布的簇）
    
    Args:
        num_clusters (int): 簇的数量
        points_per_cluster (int): 每个簇的点数
        cluster_std (float): 簇的标准差
        bounds (tuple): 坐标边界
        
    Returns:
        np.ndarray: 形状为(N, 3)的点云
    """
    all_points = []
    
    for _ in range(num_clusters):
        # 随机选择簇中心
        center_x = np.random.uniform(bounds[0][0], bounds[0][1])
        center_y = np.random.uniform(bounds[1][0], bounds[1][1])
        center_z = np.random.uniform(bounds[2][0], bounds[2][1])
        
        # 生成高斯分布的点
        x = np.random.normal(center_x, cluster_std, points_per_cluster)
        y = np.random.normal(center_y, cluster_std, points_per_cluster)
        z = np.random.normal(center_z, cluster_std, points_per_cluster)
        
        cluster_points = np.column_stack([x, y, z])
        all_points.append(cluster_points)
    
    return np.vstack(all_points)


def save_pointcloud_csv(
    points: np.ndarray,
    output_path: str
) -> None:
    """
    保存点云为CSV文件
    
    Args:
        points (np.ndarray): 形状为(N, 3)的点云
        output_path (str): 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(
        points,
        columns=['x [nm]', 'y [nm]', 'z [nm]']
    )
    
    # 保存为CSV
    df.to_csv(output_path, index=False)
    logger.info(f"已保存: {output_path}")


def generate_test_dataset(
    output_dir: str = "test_data/csv_pointclouds",
    num_samples_per_type: int = 5
) -> None:
    """
    生成完整的测试数据集
    
    Args:
        output_dir (str): 输出目录
        num_samples_per_type (int): 每种类型生成的样本数
    """
    output_path = Path(output_dir)
    
    logger.info("=" * 70)
    logger.info("生成测试CSV数据集")
    logger.info("=" * 70)
    logger.info(f"输出目录: {output_path}")
    logger.info(f"每种类型样本数: {num_samples_per_type}")
    logger.info("")
    
    # 定义生成函数和参数
    generators = {
        "sphere": (generate_sphere_pointcloud, {
            "radius": 2000.0,
            "num_points": 10000
        }),
        "cube": (generate_cube_pointcloud, {
            "size": 3000.0,
            "num_points": 10000
        }),
        "cylinder": (generate_cylinder_pointcloud, {
            "radius": 1500.0,
            "height": 3000.0,
            "num_points": 10000
        }),
        "random": (generate_random_pointcloud, {
            "num_points": 10000
        }),
        "clustered": (generate_clustered_pointcloud, {
            "num_clusters": 5,
            "points_per_cluster": 2000
        })
    }
    
    # 生成每种类型的样本
    total_samples = len(generators) * num_samples_per_type
    
    with tqdm(total=total_samples, desc="生成样本") as pbar:
        for shape_type, (generator_func, params) in generators.items():
            # 创建子目录
            shape_dir = output_path / shape_type
            
            for i in range(num_samples_per_type):
                # 为每个样本随机化中心位置
                center = (
                    np.random.uniform(5000, 15000),
                    np.random.uniform(5000, 15000),
                    np.random.uniform(2000, 4000)
                )
                
                if "center" in params or shape_type in ["sphere", "cube", "cylinder"]:
                    params_with_center = params.copy()
                    if shape_type != "random" and shape_type != "clustered":
                        params_with_center["center"] = center
                else:
                    params_with_center = params
                
                # 生成点云
                points = generator_func(**params_with_center)
                
                # 保存为CSV
                output_file = shape_dir / f"{shape_type}_{i+1:03d}.csv"
                save_pointcloud_csv(points, str(output_file))
                
                pbar.update(1)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("测试数据生成完成！")
    logger.info("=" * 70)
    logger.info(f"总共生成 {total_samples} 个CSV文件")
    logger.info(f"输出目录: {output_path}")
    logger.info("")
    logger.info("目录结构:")
    for shape_type in generators.keys():
        shape_dir = output_path / shape_type
        num_files = len(list(shape_dir.glob("*.csv"))) if shape_dir.exists() else 0
        logger.info(f"  {shape_type}/: {num_files} 个文件")


def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成测试CSV点云数据"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_data/csv_pointclouds",
        help="输出目录 (默认: test_data/csv_pointclouds)"
    )
    
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=5,
        help="每种类型生成的样本数 (默认: 5)"
    )
    
    args = parser.parse_args()
    
    # 生成测试数据
    generate_test_dataset(
        output_dir=args.output_dir,
        num_samples_per_type=args.num_samples
    )
    
    # 打印使用提示
    print("\n" + "=" * 70)
    print("测试数据已生成！")
    print("=" * 70)
    print("\n现在你可以运行数据处理脚本进行测试:")
    print("\n1. 编辑配置文件:")
    print("   - 复制 config_example.py 为 config.py")
    print(f"   - 设置 INPUT_DIR = '{args.output_dir}'")
    print("   - 设置其他参数\n")
    print("2. 运行测试脚本:")
    print("   python scripts/exp-data-process/test_process_pointcloud_data.py\n")
    print("3. 或直接运行处理脚本:")
    print("   python scripts/exp-data-process/run_processing.py\n")


if __name__ == "__main__":
    main()
