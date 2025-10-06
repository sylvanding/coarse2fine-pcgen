"""
CSV坐标范围分析器

读取指定文件夹下所有子文件夹中的CSV文件，分析每个文件中xyz坐标的范围。
支持表头为"x[nm],y[nm],z[nm]"格式的CSV文件。

Usage:
    python scripts/test_conversion/analyze_csv_ranges.py <目标文件夹路径>

Example:
    python scripts/test_conversion/analyze_csv_ranges.py data/csv_files/
    python scripts/test_conversion/analyze_csv_ranges.py /path/to/csv/folder/
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVRangeAnalyzer:
    """
    CSV文件坐标范围分析器
    
    用于分析包含3D坐标数据的CSV文件，计算x、y、z坐标的范围统计信息。
    支持递归搜索子文件夹中的所有CSV文件。
    """
    
    def __init__(self, target_directory: str):
        """
        初始化分析器
        
        Args:
            target_directory (str): 目标文件夹路径
        """
        self.target_directory = Path(target_directory)
        if not self.target_directory.exists():
            raise FileNotFoundError(f"目标文件夹不存在: {target_directory}")
    
    def find_csv_files(self) -> Dict[str, List[Path]]:
        """
        递归查找所有CSV文件，按文件夹分组
        
        Returns:
            Dict[str, List[Path]]: 文件夹路径到CSV文件列表的映射
        """
        csv_files_by_folder = {}
        
        logger.info(f"开始搜索CSV文件: {self.target_directory}")
        
        # 递归搜索所有CSV文件
        for csv_file in self.target_directory.rglob("*.csv"):
            folder_path = str(csv_file.parent.relative_to(self.target_directory))
            if folder_path == ".":
                folder_path = "根目录"
            
            if folder_path not in csv_files_by_folder:
                csv_files_by_folder[folder_path] = []
            csv_files_by_folder[folder_path].append(csv_file)
        
        logger.info(f"找到 {sum(len(files) for files in csv_files_by_folder.values())} 个CSV文件")
        return csv_files_by_folder
    
    def analyze_csv_file(self, csv_file: Path) -> Dict[str, Tuple[float, float]]:
        """
        分析单个CSV文件的坐标范围
        
        Args:
            csv_file (Path): CSV文件路径
            
        Returns:
            Dict[str, Tuple[float, float]]: 坐标轴到(最小值, 最大值)的映射
            
        Raises:
            ValueError: 当CSV文件格式不正确时
            FileNotFoundError: 当文件无法读取时
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查列名
            expected_columns = ["x [nm]", "y [nm]", "z [nm]"]
            if not all(col in df.columns for col in expected_columns):
                # 尝试其他可能的列名格式
                possible_x_cols = [col for col in df.columns if 'x' in col.lower()]
                possible_y_cols = [col for col in df.columns if 'y' in col.lower()]
                possible_z_cols = [col for col in df.columns if 'z' in col.lower()]
                
                if len(possible_x_cols) == 1 and len(possible_y_cols) == 1 and len(possible_z_cols) == 1:
                    x_col, y_col, z_col = possible_x_cols[0], possible_y_cols[0], possible_z_cols[0]
                    logger.warning(f"文件 {csv_file.name} 使用非标准列名: {x_col}, {y_col}, {z_col}")
                else:
                    raise ValueError(f"CSV文件列名不匹配。期望: {expected_columns}, 实际: {list(df.columns)}")
            else:
                x_col, y_col, z_col = expected_columns
            
            # 检查数据是否为空
            if df.empty:
                raise ValueError("CSV文件为空")
            
            # 计算坐标范围
            ranges = {}
            for axis, col in [('x', x_col), ('y', y_col), ('z', z_col)]:
                data = df[col].dropna()  # 移除NaN值
                if len(data) == 0:
                    ranges[axis] = (np.nan, np.nan)
                else:
                    ranges[axis] = (float(data.min()), float(data.max()))
            
            return ranges
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV文件为空或格式错误")
        except FileNotFoundError:
            raise FileNotFoundError(f"无法读取文件: {csv_file}")
        except Exception as e:
            raise ValueError(f"分析文件时出错: {str(e)}")
    
    def analyze_all_files(self) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
        """
        分析所有找到的CSV文件
        
        Returns:
            Dict: 嵌套字典，结构为 {文件夹: {文件名: {坐标轴: (最小值, 最大值)}}}
        """
        csv_files_by_folder = self.find_csv_files()
        results = {}
        
        for folder_path, csv_files in csv_files_by_folder.items():
            results[folder_path] = {}
            
            logger.info(f"分析文件夹: {folder_path} ({len(csv_files)} 个文件)")
            
            for csv_file in csv_files:
                try:
                    ranges = self.analyze_csv_file(csv_file)
                    results[folder_path][csv_file.name] = ranges
                    
                    # 记录文件信息
                    point_count = self._get_point_count(csv_file)
                    logger.info(f"  ✓ {csv_file.name}: {point_count} 个点")
                    
                except Exception as e:
                    logger.error(f"  ✗ {csv_file.name}: {str(e)}")
                    results[folder_path][csv_file.name] = {"error": str(e)}
        
        return results
    
    def _get_point_count(self, csv_file: Path) -> int:
        """获取CSV文件中的点数"""
        try:
            df = pd.read_csv(csv_file)
            return len(df)
        except:
            return 0
    
    def print_results(self, results: Dict) -> None:
        """
        打印分析结果
        
        Args:
            results: analyze_all_files()的返回结果
        """
        print("\n" + "="*80)
        print("CSV文件坐标范围分析结果")
        print("="*80)
        
        total_files = 0
        successful_files = 0
        total_points = 0
        
        for folder_path, folder_results in results.items():
            print(f"\n📁 文件夹: {folder_path}")
            print("-" * 60)
            
            if not folder_results:
                print("  (无CSV文件)")
                continue
            
            for file_name, file_results in folder_results.items():
                total_files += 1
                
                if "error" in file_results:
                    print(f"  ❌ {file_name}: {file_results['error']}")
                else:
                    successful_files += 1
                    # 获取点数
                    file_path = self.target_directory / folder_path.replace("根目录", ".") / file_name
                    point_count = self._get_point_count(file_path)
                    total_points += point_count
                    
                    print(f"  📄 {file_name} (点数: {point_count:,}):")
                    
                    for axis in ['x', 'y', 'z']:
                        if axis in file_results:
                            min_val, max_val = file_results[axis]
                            if np.isnan(min_val) or np.isnan(max_val):
                                print(f"    {axis.upper()}: 无有效数据")
                            else:
                                range_val = max_val - min_val
                                print(f"    {axis.upper()}: [{min_val:.6f}, {max_val:.6f}] (范围: {range_val:.6f} nm)")
                    print()
        
        # 打印统计摘要
        print("="*80)
        print(f"分析完成: {successful_files}/{total_files} 个文件成功分析")
        print(f"总点数: {total_points:,} 个点")
        if total_files > successful_files:
            print(f"失败文件数: {total_files - successful_files}")
        print("="*80)
    
    def save_results_to_csv(self, results: Dict, output_file: str = "csv_range_analysis.csv") -> None:
        """
        将结果保存为CSV文件
        
        Args:
            results: 分析结果
            output_file: 输出文件名
        """
        output_data = []
        
        for folder_path, folder_results in results.items():
            for file_name, file_results in folder_results.items():
                if "error" not in file_results:
                    # 获取点数
                    file_path = self.target_directory / folder_path.replace("根目录", ".") / file_name
                    point_count = self._get_point_count(file_path)
                    
                    row = {
                        'folder': folder_path,
                        'file': file_name,
                        'point_count': point_count,
                        'x_min': file_results.get('x', (np.nan, np.nan))[0],
                        'x_max': file_results.get('x', (np.nan, np.nan))[1],
                        'y_min': file_results.get('y', (np.nan, np.nan))[0],
                        'y_max': file_results.get('y', (np.nan, np.nan))[1],
                        'z_min': file_results.get('z', (np.nan, np.nan))[0],
                        'z_max': file_results.get('z', (np.nan, np.nan))[1],
                    }
                    # 计算范围
                    for axis in ['x', 'y', 'z']:
                        min_val = row[f'{axis}_min']
                        max_val = row[f'{axis}_max']
                        if not np.isnan(min_val) and not np.isnan(max_val):
                            row[f'{axis}_range'] = max_val - min_val
                        else:
                            row[f'{axis}_range'] = np.nan
                    
                    output_data.append(row)
        
        if output_data:
            df = pd.DataFrame(output_data)
            output_path = self.target_directory / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"结果已保存到: {output_path}")
        else:
            logger.warning("没有有效数据可保存")


def main():
    sys.argv.append("/repos/datasets/exp-data-4pi-pc-mt")
    
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python scripts/test_conversion/analyze_csv_ranges.py <目标文件夹路径>")
        print("\n示例:")
        print("  python scripts/test_conversion/analyze_csv_ranges.py data/csv_files/")
        print("  python scripts/test_conversion/analyze_csv_ranges.py /path/to/csv/folder/")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    
    try:
        # 创建分析器
        analyzer = CSVRangeAnalyzer(target_directory)
        
        # 执行分析
        results = analyzer.analyze_all_files()
        
        # 打印结果
        analyzer.print_results(results)
        
        # 保存结果到CSV文件
        analyzer.save_results_to_csv(results)
        
    except FileNotFoundError as e:
        logger.error(f"文件夹不存在: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
