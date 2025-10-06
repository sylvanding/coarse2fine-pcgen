# 点云数据处理工具

这个目录包含用于从CSV文件批量生成点云训练样本的工具脚本。

## 功能概述

### 主脚本: `process_pointcloud_data.py`

从CSV点云文件中提取区域，进行数据增强，并保存为HDF5格式。

**核心功能：**
- 递归扫描目录查找所有CSV文件（表头格式：`x [nm], y [nm], z [nm]`）
- 随机旋转点云（绕Z轴）
- 随机提取指定大小的点云区域
- 过滤点数不足的区域（自动重试）
- 坐标归一化和精度控制（保留2位小数）
- 批量生成样本并保存为H5文件

### 测试脚本: `test_process_pointcloud_data.py`

用于验证和检查生成的样本质量。

**功能：**
- 调用主脚本生成样本
- 将每个样本保存为CSV文件
- 生成XY平面可视化图片（用Z轴值表示颜色）
- 创建多样本汇总可视化
- 生成详细的统计报告

## 快速开始

### 1. 安装依赖

```bash
# 确保已激活conda环境
conda activate c2f-pcgen

# 安装必要的依赖
pip install pandas h5py matplotlib tqdm
```

### 2. 准备数据

确保你的CSV文件满足以下要求：
- 表头必须包含：`x [nm]`, `y [nm]`, `z [nm]`
- 坐标单位为纳米(nm)
- 文件格式为标准CSV

**示例CSV格式：**
```csv
x [nm],y [nm],z [nm]
100.23,200.45,50.12
101.34,201.56,51.23
...
```

### 3. 运行主脚本

编辑 `process_pointcloud_data.py` 中的配置参数：

```python
# 输入目录（包含CSV文件）
INPUT_DIR = "/path/to/your/csv/files"

# 输出H5文件路径
OUTPUT_H5 = "outputs/processed_data/pointcloud_samples.h5"

# 生成样本数量
NUM_SAMPLES = 100

# 提取区域大小（单位：nm）
REGION_SIZE = (1000.0, 1000.0, 500.0)  # x, y, z

# 最小点数要求
MIN_POINTS = 100

# 每个CSV的最大尝试次数
MAX_ATTEMPTS = 10
```

运行脚本：
```bash
python scripts/exp-data-process/process_pointcloud_data.py
```

### 4. 运行测试脚本（可选）

测试脚本用于验证生成的样本质量，建议先用少量样本测试：

```python
# 编辑 test_process_pointcloud_data.py
INPUT_DIR = "/path/to/your/csv/files"
OUTPUT_DIR = "outputs/test_processed_data"
NUM_SAMPLES = 10  # 测试用，使用较小的值
```

运行测试：
```bash
python scripts/exp-data-process/test_process_pointcloud_data.py
```

测试输出包括：
- `csv_samples/`: 每个样本的CSV文件
- `png_visualizations/`: 每个样本的XY投影可视化
- `samples_overview.png`: 多样本汇总图
- `statistics_report.txt`: 详细统计报告

## 参数说明

### 区域提取参数

- **`region_size`**: 提取区域的尺寸 (x, y, z)，单位为nm
  - 建议根据点云密度和所需样本大小设置
  - 例如: `(1000, 1000, 500)` 表示 1000nm × 1000nm × 500nm 的区域

- **`min_points`**: 每个样本的最小点数要求
  - 确保每个样本包含足够的点用于训练
  - 如果提取的区域点数不足，会自动重试

- **`max_attempts`**: 每个CSV文件的最大尝试次数
  - 如果连续10次都无法找到满足条件的区域，放弃该CSV文件
  - 避免在点云密度过低的文件上浪费时间

### 坐标处理参数

- **`decimals`**: 坐标保留的小数位数（默认：2）
  - 减少存储空间和计算精度需求
  - 2位小数对于nm级别的数据通常足够

- **`random_seed`**: 随机种子（可选）
  - 设置后可以确保结果可重复
  - 用于调试和实验对比

## 输出格式

### H5文件结构

```
pointcloud_samples.h5
└── pointclouds (dataset)
    ├── shape: (num_samples, max_points, 3)
    ├── dtype: float32
    └── attributes:
        ├── num_samples: 样本数量
        ├── max_points: 最大点数
        ├── min_points: 最小点数
        └── avg_points: 平均点数
```

**注意：** 
- 由于每个样本的点数不同，使用NaN填充到统一长度
- 使用gzip压缩减少文件大小

### 测试输出

测试脚本生成以下文件：

```
outputs/test_processed_data/
├── csv_samples/
│   ├── sample_0001.csv
│   ├── sample_0002.csv
│   └── ...
├── png_visualizations/
│   ├── sample_0001.png
│   ├── sample_0002.png
│   └── ...
├── samples_overview.png
└── statistics_report.txt
```

## 使用技巧

### 1. 确定合适的区域大小

建议先运行测试脚本，通过可视化检查：
- 区域是否太大（包含过多背景）
- 区域是否太小（丢失结构信息）
- 点云密度是否合适

### 2. 调整最小点数要求

根据应用需求设置：
- **稀疏点云任务**: 设置较小的值（如50-100）
- **密集点云任务**: 设置较大的值（如500-1000）

### 3. 处理失败情况

如果生成样本时遇到很多失败：
- 检查CSV文件格式是否正确
- 降低`min_points`要求
- 减小`region_size`
- 增加`max_attempts`

### 4. 性能优化

- 如果CSV文件很大，可以考虑在`load_csv_pointcloud`中添加随机采样
- 对于大规模数据集，可以使用多进程并行处理

## 常见问题

### Q1: 为什么某些CSV文件被跳过？

**可能原因：**
- CSV文件表头格式不正确
- 文件包含NaN值
- 点云尺寸小于所需的区域大小
- 连续多次尝试都找不到满足点数要求的区域

**解决方法：**
- 检查CSV文件格式
- 调整`region_size`和`min_points`参数
- 查看日志了解具体原因

### Q2: 生成的H5文件很大怎么办？

**优化建议：**
- 减少样本数量
- 降低坐标精度（增加`decimals`参数）
- 使用更强的压缩（修改`compression`参数）
- 减小区域大小或降低最小点数要求

### Q3: 如何验证生成的样本质量？

**建议步骤：**
1. 先用测试脚本生成少量样本（10-20个）
2. 查看可视化图片，确认：
   - 点云形态是否合理
   - 旋转是否正确
   - 区域提取是否符合预期
3. 检查统计报告，确认：
   - 点数分布是否合理
   - 边界框尺寸是否符合预期
4. 满意后再运行主脚本生成完整数据集

## 进阶使用

### 作为Python模块使用

```python
from scripts.exp_data_process.process_pointcloud_data import (
    find_all_csv_files,
    generate_samples,
    save_samples_to_h5
)

# 查找CSV文件
csv_files = find_all_csv_files("/path/to/csv")

# 生成样本
samples = generate_samples(
    csv_files=csv_files,
    num_samples=100,
    region_size=(1000, 1000, 500),
    target_points=1024,  # 期望的点数
    min_points=100
)

# 保存到H5
save_samples_to_h5(samples, "output.h5")
```

### 自定义数据处理流程

可以修改以下函数来适应特定需求：
- `rotate_pointcloud_z()`: 修改旋转策略
- `extract_random_region()`: 修改区域提取逻辑
- `translate_to_origin()`: 修改坐标归一化方式

## 联系与支持

如有问题或建议，请联系开发团队或提交Issue。

---

**最后更新**: 2025-10-05
**版本**: 1.0.0
