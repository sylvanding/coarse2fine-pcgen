"""
CSV点云数据格式化转换脚本

功能：
1. 读取指定文件夹下的所有 .csv 文件
2. 忽略原文件第一行（表头）
3. 读取以空格分隔的 xyz 坐标
4. 保留小数点后两位
5. 输出为以逗号分隔的新 CSV 文件
6. 添加新表头: "x [nm], y [nm], z[ ]"

Usage:
    python scripts/exp_data_process/check_csv.py --input_dir /path/to/input --output_dir /path/to/output
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List

def process_single_file(input_path: Path, output_path: Path):
    """处理单个CSV文件"""
    try:
        # 读取数据
        data_rows = []
        with open(input_path, 'r', encoding='utf-8') as f:
            # 读取所有行
            lines = f.readlines()
            
            # 从第二行开始处理 (忽略第一行)
            for line in lines[1:]:
                # 去除首尾空白并按空格分割
                parts = line.strip().split()
                if len(parts) >= 3:
                    # 取前三个值作为x, y, z
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        # 格式化为保留两位小数的字符串
                        data_rows.append([f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])
                    except ValueError:
                        print(f"警告: 文件 {input_path.name} 中存在无法解析的数据行: {line.strip()}")
                        continue

        # 写入新文件
        if data_rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入新表头
                writer.writerow(["x [nm]", "y [nm]", "z [nm]"])
                # 写入数据
                writer.writerows(data_rows)
            print(f"成功处理: {input_path.name} -> {output_path}")
        else:
            print(f"警告: 文件 {input_path.name} 没有有效数据")

    except Exception as e:
        print(f"处理文件 {input_path.name} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="批量处理点云CSV文件格式")
    parser.add_argument("--input_dir", type=str, default="/repos/datasets/tissue-datasets/pointclouds-clean", help="输入CSV文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/repos/datasets/tissue-datasets/pointclouds-clean-check", help="输出CSV文件夹路径")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        return
        
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有csv文件
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"在 '{input_dir}' 中没有找到CSV文件")
        return
        
    print(f"找到 {len(csv_files)} 个CSV文件，开始处理...")
    
    # 批量处理
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        process_single_file(csv_file, output_file)
        
    print("所有文件处理完成。")

if __name__ == "__main__":
    main()

