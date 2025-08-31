# TIFF到点云采样脚本

将3D TIFF体素数据采样转换为点云格式的工具脚本。

## 功能特点

- **多种采样方法**: 支持概率分布采样、中心采样、随机采样和加权采样
- **避免网格化效应**: 默认使用概率分布采样方法，生成更自然的点云
- **灵活的参数控制**: 可调节采样阈值、目标点数、体积参数等
- **批量处理**: 支持通配符模式批量处理多个TIFF文件
- **详细统计分析**: 提供体素网格和点云的统计信息
- **完整的日志记录**: 详细的处理过程记录

## 采样方法说明

### 1. probabilistic (概率分布采样) - 推荐
- **优势**: 避免传统方法的网格化效应，生成更自然的点云
- **原理**: 基于体素值构建概率分布，进行连续空间采样
- **适用**: 需要高质量、自然分布点云的场景

### 2. center (中心采样)
- **原理**: 在每个超过阈值的体素中心生成点
- **适用**: 需要规则分布的点云

### 3. random (随机采样)
- **原理**: 在每个超过阈值的体素内随机生成点
- **适用**: 需要一定随机性但保持体素结构的点云

### 4. weighted (加权采样)
- **原理**: 基于体素值进行概率加权的采样
- **适用**: 需要考虑体素强度权重的点云

## 使用方法

### 基本用法

```bash
# 基本转换（默认概率采样）
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv

# 指定采样点数和阈值
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv \\
    --num-points 100000 \\
    --threshold 0.1
```

### 不同采样方法

```bash
# 概率分布采样（推荐）
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv \\
    --method probabilistic \\
    --num-points 200000

# 体素中心采样
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv \\
    --method center \\
    --num-points 50000

# 体素内随机采样
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv \\
    --method random \\
    --num-points 75000
```

### 自定义参数

```bash
# 自定义体积参数和采样阈值
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input voxel_data.tiff \\
    --output points.csv \\
    --volume-dims 15000 15000 3000 \\
    --padding 50 50 150 \\
    --threshold 0.05 \\
    --num-points 150000
```

### 批量处理

```bash
# 批量处理多个TIFF文件
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input "data/*.tiff" \\
    --output-dir results/ \\
    --method probabilistic \\
    --num-points 100000

# 处理特定模式的文件
python scripts/tiff_to_pointcloud/tiff_to_pointcloud.py \\
    --input "experiments/exp_*.tiff" \\
    --output-dir point_clouds/ \\
    --threshold 0.1
```

## 参数说明

### 必需参数
- `--input, -i`: 输入TIFF文件路径（支持通配符如"*.tiff"）
- `--output, -o`: 输出CSV文件路径（单文件模式）
- `--output-dir`: 输出目录（批量处理模式）

### 采样参数
- `--threshold`: 体素值阈值，低于此值的体素不参与采样（默认: 0.0）
- `--num-points`: 目标采样点数（默认: 100000，None时自动确定）
- `--method`: 采样方法（默认: probabilistic）

### 体积参数
- `--volume-dims`: 体积尺寸 [x, y, z] (单位: nm)（默认: [20000, 20000, 2500]）
- `--padding`: 体积边界填充 [x, y, z] (单位: nm)（默认: [0, 0, 100]）

### 处理选项
- `--normalize`: 是否将体素值归一化到[0, 1]范围（默认: True）
- `--verbose, -v`: 显示详细信息（默认: True）

## 输出文件

### 点云CSV文件
- 格式: `x [nm],y [nm],z [nm]`
- 包含采样得到的所有点的3D坐标

### 信息文件 (*_info.txt)
- 转换参数记录
- 体素网格统计信息
- 点云统计信息
- 便于后续分析和复现

## 性能优化

### 大数据集处理
- 自动使用别名采样算法处理大量采样需求
- 内存友好的批处理策略
- 详细的进度日志

### 采样质量
- 概率分布采样避免网格化效应
- 混合分布的子体素偏移增加自然性
- 多种采样算法适应不同需求

## 示例输出

```
======================================================
体素网格统计信息:
======================================================
网格shape: (256, 256, 64)
总体素数: 4,194,304
占有体素数: 125,847
占有率: 0.0300
体素值范围: [0.000, 1.000]
平均体素值: 0.045
非零体素平均值: 0.523
体素值标准差: 0.156

======================================================
采样点云统计信息:
======================================================
点数: 100,000
坐标范围:
  X: [-50.000, 20050.000] (范围: 20100.000)
  Y: [-50.000, 20050.000] (范围: 20100.000)  
  Z: [50.000, 2650.000] (范围: 2600.000)
平均坐标: (10000.234, 10005.678, 1325.432)

✅ 转换完成!
📦 输入体素网格: (256, 256, 64)
📊 采样点云: 100,000 个点
💾 CSV文件: output/points.csv
📋 信息文件: output/points_info.txt
```

## 注意事项

1. **内存使用**: 大体素网格可能占用较多内存，建议根据系统配置调整参数
2. **采样质量**: 概率分布采样质量最佳但计算开销稍大
3. **阈值选择**: 阈值过低可能产生过多噪声点，过高可能遗漏细节
4. **坐标系统**: 输出坐标基于输入的体积参数设置
