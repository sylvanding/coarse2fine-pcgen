# 体素采样和上采样功能说明

本文档介绍了 Coarse2Fine-PCGen 项目中新增的体素采样回点云和体素上采样功能。

## 功能概述

### 1. 体素采样回点云 (Voxel-to-Points Sampling)

将体素网格重新采样为点云数据，支持多种采样策略和自定义点数。

**主要特性：**
- 支持自定义目标点数
- 多种采样方法：中心采样、随机采样、加权采样
- 可调节体素阈值
- 自动处理坐标反归一化

### 2. 体素上采样 (Voxel Upsampling)

提升体素网格的分辨率，通过插值方法增加体素数量。

**主要特性：**
- 支持任意倍数缩放
- 多种插值方法：线性、最近邻、三次插值
- 保持原始数据分布特征

## API 使用方法

### PointCloudToVoxel 类新方法

#### 1. voxel_to_points()

```python
def voxel_to_points(self, voxel_grid: np.ndarray, threshold: float = 0.5, 
                   num_points: Optional[int] = None, method: str = 'center') -> np.ndarray:
    """
    将体素网格转换回点云
    
    Args:
        voxel_grid: 输入体素网格
        threshold: 体素值阈值 (默认: 0.5)
        num_points: 目标点数，None时返回所有超过阈值的体素
        method: 采样方法 ('center', 'random', 'weighted')
    
    Returns:
        采样得到的点云数据
    """
```

**采样方法说明：**
- `center`: 体素中心采样，生成位于体素中心的点
- `random`: 体素内随机采样，在体素内随机生成点位置
- `weighted`: 加权概率采样，根据体素值进行概率采样

#### 2. upsample_voxel_grid()

```python
def upsample_voxel_grid(self, voxel_grid: np.ndarray, scale_factor: float = 2.0, 
                       method: str = 'linear') -> np.ndarray:
    """
    对体素网格进行上采样
    
    Args:
        voxel_grid: 输入体素网格
        scale_factor: 放大倍数 (默认: 2.0)
        method: 插值方法 ('linear', 'nearest', 'cubic')
    
    Returns:
        上采样后的体素网格
    """
```

**插值方法说明：**
- `linear`: 线性插值，平滑过渡
- `nearest`: 最近邻插值，保持原始值
- `cubic`: 三次插值，更平滑的结果

#### 3. save_point_cloud()

```python
def save_point_cloud(self, point_cloud: np.ndarray, output_path: str) -> None:
    """
    保存点云数据到CSV文件
    
    Args:
        point_cloud: 点云数据
        output_path: 输出文件路径（CSV格式）
    """
```

**CSV格式说明：**
- 文件格式：逗号分隔值(CSV)
- 表头：`x [nm],y [nm],z [nm]`
- 数据精度：6位小数
- 自动添加.csv扩展名（如果未指定）

## 命令行使用方法

### test_conversion.py 新增参数

#### 体素采样回点云参数

```bash
--sample-back                    # 启用体素采样回点云
--sample-num-points INT          # 目标点数
--sample-threshold FLOAT         # 体素阈值 (默认: 0.5)
--sample-method {center,random,weighted}  # 采样方法 (默认: center)
```

#### 体素上采样参数

```bash
--upsample                       # 启用体素上采样
--upsample-factor FLOAT          # 上采样倍数 (默认: 2.0)
--upsample-method {linear,nearest,cubic}  # 插值方法 (默认: linear)
```

### 使用示例

#### 1. 基本体素采样回点云

```bash
python scripts/test_conversion.py \
    --input data/sample.h5 \
    --output test_sampled.tiff \
    --sample-back \
    --sample-num-points 10000 \
    --sample-method weighted
```

#### 2. 体素上采样

```bash
python scripts/test_conversion.py \
    --input data/sample.h5 \
    --output test_upsampled.tiff \
    --upsample \
    --upsample-factor 2.0 \
    --upsample-method linear
```

#### 3. 完整流程：转换→上采样→采样回点云

```bash
python scripts/test_conversion.py \
    --input data/sample.h5 \
    --output test_full.tiff \
    --method gaussian \
    --sigma 1.5 \
    --upsample \
    --upsample-factor 2.0 \
    --sample-back \
    --sample-num-points 50000 \
    --sample-method random
```

## 输出文件

当启用新功能时，会生成以下额外文件：

### 体素上采样输出
- `*_upsampled.tiff`: 上采样后的体素网格

### 体素采样回点云输出
- `*_sampled.csv`: 采样的点云数据 (CSV格式，表头: "x [nm],y [nm],z [nm]")

### 统计信息
- `*_info.txt`: 包含所有转换参数和统计信息的详细报告

## 性能考虑

### 内存使用
- 上采样会显著增加内存使用量（按倍数的三次方增长）
- 建议对大体素网格使用较小的上采样倍数

### 处理时间
- 上采样时间与目标体素数量成正比
- 加权采样比其他方法稍慢（需要计算概率）

## 应用场景

### 1. 数据增强
通过上采样提升体素分辨率，用于训练高分辨率生成模型

### 2. 质量评估
将生成的体素网格采样回点云，与原始点云对比评估质量

### 3. 可视化
生成不同密度的点云用于可视化展示

### 4. 后处理
对生成的体素进行精细化处理，获得更高质量的点云输出

## 演示脚本

运行演示脚本查看新功能：

```bash
python scripts/demo_new_features.py
```

该脚本会演示完整的转换→上采样→采样流程，并保存结果到 `output/demo/` 目录。
