#!/usr/bin/env python3
"""
将outputs/batch_simulation_mitochondria下的所有CSV点云文件转换为H5格式
每个CSV文件包含x,y,z坐标数据，最终保存为[样本数, 点数, 3]的H5数据集
"""

import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_point_count(points, target_count):
    """
    统一点云的点数

    Parameters:
    -----------
    points : numpy.ndarray
        形状为 (n_points, 3) 的点云数据
    target_count : int
        目标点数

    Returns:
    --------
    numpy.ndarray
        形状为 (target_count, 3) 的标准化点云数据
    """
    current_count = len(points)

    if current_count == target_count:
        return points
    elif current_count > target_count:
        # 随机删除点
        indices = np.random.choice(current_count, target_count, replace=False)
        return points[indices]
    else:
        # 随机重复点
        additional_needed = target_count - current_count
        indices = np.random.choice(current_count, additional_needed, replace=True)
        additional_points = points[indices]
        return np.vstack([points, additional_points])


def load_csv_file(csv_path):
    """
    加载CSV文件并返回xyz坐标数据

    Parameters:
    -----------
    csv_path : str or Path
        CSV文件路径

    Returns:
    --------
    numpy.ndarray
        形状为 (n_points, 3) 的坐标数据
    """
    try:
        df = pd.read_csv(csv_path)
        # 提取xyz列
        points = df[["x [nm]", "y [nm]", "z [nm]"]].values
        return points.astype(np.float32)
    except Exception as e:
        print(f"读取文件 {csv_path} 时出错: {e}")
        return None


def get_folder_numbers(base_dir):
    """
    获取所有文件夹编号，按数字顺序排序

    Parameters:
    -----------
    base_dir : str or Path
        基础目录路径

    Returns:
    --------
    list
        排序后的编号列表
    """
    folders = []
    pattern = re.compile(r"mitochondria_(\d+)_\d+k")

    for folder_name in os.listdir(base_dir):
        match = pattern.match(folder_name)
        if match:
            folder_number = int(match.group(1))
            # if 1 <= folder_number <= 1024:  # 只处理001-1024编号的文件夹
            folders.append((folder_number, folder_name))

    # 按编号排序
    folders.sort(key=lambda x: x[0])
    return folders


def main():
    """主函数"""
    # 设置随机种子以保证结果可重复
    np.random.seed(42)

    # 定义路径
    input_dir = Path("/repos/biolocsim/outputs/batch_simulation_mitochondria")
    output_file = Path("/repos/biolocsim/outputs/batch_simulation_mitochondria_8192.h5")

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    # 获取所有文件夹
    folders = get_folder_numbers(input_dir)
    print(f"找到 {len(folders)} 个有效文件夹")

    if len(folders) == 0:
        print("未找到有效的文件夹")
        return

    # 第一遍扫描：确定目标点数
    print("正在分析点云数据，确定目标点数...")
    point_counts = []
    valid_files = []

    for folder_num, folder_name in tqdm(folders, desc="扫描文件"):
        csv_path = input_dir / folder_name / "mitochondria_simulation.csv"
        if csv_path.exists():
            points = load_csv_file(csv_path)
            if points is not None:
                point_counts.append(len(points))
                valid_files.append((folder_num, folder_name, csv_path))

    if len(valid_files) == 0:
        print("没有找到有效的CSV文件")
        return

    # 计算目标点数（使用中位数作为目标）
    target_point_count = int(np.median(point_counts))
    print(f"点数统计: 最小={min(point_counts)}, 最大={max(point_counts)}, 中位数={target_point_count}")
    print(f"将所有点云标准化为 {target_point_count} 个点")

    # 第二遍扫描：加载数据并标准化
    print("正在加载和标准化点云数据...")
    all_point_clouds = []
    sample_ids = []

    for folder_num, folder_name, csv_path in tqdm(valid_files, desc="处理文件"):
        points = load_csv_file(csv_path)
        if points is not None:
            normalized_points = normalize_point_count(points, target_point_count)
            all_point_clouds.append(normalized_points)
            sample_ids.append(folder_num)

    # 转换为numpy数组
    point_cloud_array = np.array(all_point_clouds, dtype=np.float16)
    sample_ids = np.array(sample_ids, dtype=np.int32)

    print(f"最终数据形状: {point_cloud_array.shape}")
    print(f"数据类型: {point_cloud_array.dtype}")

    # 保存到H5文件
    print(f"保存数据到: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        # 保存点云数据
        f.create_dataset("point_clouds", data=point_cloud_array, compression="gzip", compression_opts=9)

        # 保存样本ID
        f.create_dataset("sample_ids", data=sample_ids, compression="gzip", compression_opts=9)

        # 保存元数据
        f.attrs["num_samples"] = len(all_point_clouds)
        f.attrs["points_per_sample"] = target_point_count
        f.attrs["num_features"] = 3  # x, y, z
        f.attrs["feature_names"] = ["x [nm]", "y [nm]", "z [nm]"]
        f.attrs["data_source"] = "batch_simulation_mitochondria"
        f.attrs["description"] = "Mitochondria point cloud data from biolocsim batch simulation"

    print("转换完成！")
    print(f"输出文件: {output_file}")
    print(f"数据集形状: [{len(all_point_clouds)}, {target_point_count}, 3]")

    # 验证保存的数据
    print("\n验证保存的数据...")
    with h5py.File(output_file, "r") as f:
        print("H5文件中的数据集:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}, {f[key].dtype}")
        print("元数据:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
