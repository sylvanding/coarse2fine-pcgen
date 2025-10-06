"""
点云数据处理工具包

用于从CSV文件批量生成点云训练样本，包括数据增强、区域提取和格式转换。
"""

from .process_pointcloud_data import (
    find_all_csv_files,
    load_csv_pointcloud,
    rotate_pointcloud_z,
    extract_random_region,
    translate_to_origin,
    round_coordinates,
    generate_samples,
    save_samples_to_h5
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__all__ = [
    "find_all_csv_files",
    "load_csv_pointcloud",
    "rotate_pointcloud_z",
    "extract_random_region",
    "translate_to_origin",
    "round_coordinates",
    "generate_samples",
    "save_samples_to_h5"
]
