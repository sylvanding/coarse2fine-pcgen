# 🚀 新功能更新：体素采样和上采样

本次更新为 Coarse2Fine-PCGen 项目添加了两个重要的新功能：

## ✨ 新增功能

### 1. 📦➡️☁️ 体素采样回点云 (Voxel-to-Points Sampling)

将生成的体素网格重新采样为点云数据，支持：

- **自定义点数**：可指定目标点数或使用所有超过阈值的体素
- **多种采样方法**：
  - `center`: 体素中心采样
  - `random`: 体素内随机采样  
  - `weighted`: 基于体素值的加权概率采样
- **可调阈值**：灵活控制采样的体素范围

### 2. 🔍 体素上采样 (Voxel Upsampling)

提升体素网格分辨率，支持：

- **任意倍数缩放**：如 2x, 4x 等
- **多种插值方法**：
  - `linear`: 线性插值（平滑）
  - `nearest`: 最近邻插值（保持原值）
  - `cubic`: 三次插值（最平滑）

## 🛠️ 技术实现

### 核心类方法扩展

**`PointCloudToVoxel` 类新增方法：**

```python
# 体素采样回点云
voxel_to_points(voxel_grid, threshold=0.5, num_points=None, method='center')

# 体素上采样
upsample_voxel_grid(voxel_grid, scale_factor=2.0, method='linear')

# 点云保存
save_point_cloud(point_cloud, output_path, format='txt')
```

### 命令行接口扩展

**新增参数：**

```bash
# 体素采样参数
--sample-back                    # 启用采样功能
--sample-num-points INT          # 目标点数
--sample-threshold FLOAT         # 体素阈值
--sample-method {center,random,weighted}  # 采样方法

# 体素上采样参数
--upsample                       # 启用上采样
--upsample-factor FLOAT          # 放大倍数
--upsample-method {linear,nearest,cubic}  # 插值方法
```

## 🎯 使用场景

1. **质量评估**: 将生成的体素转回点云，与原始数据对比
2. **数据增强**: 通过上采样获得更高分辨率的体素表示
3. **可视化**: 生成不同密度的点云用于展示
4. **后处理**: 对生成结果进行精细化处理

## 📖 快速开始

### 基础使用

```bash
# 体素采样回点云
python scripts/test_conversion.py --input data/sample.h5 --output result.tiff \
    --sample-back --sample-num-points 10000 --sample-method weighted

# 体素上采样
python scripts/test_conversion.py --input data/sample.h5 --output result.tiff \
    --upsample --upsample-factor 2.0 --upsample-method linear
```

### 完整流程

```bash
# 点云→体素→上采样→采样回点云
python scripts/test_conversion.py --input data/sample.h5 --output full_result.tiff \
    --method gaussian --sigma 1.5 \
    --upsample --upsample-factor 2.0 \
    --sample-back --sample-num-points 50000 --sample-method random
```

### 演示脚本

```bash
# 运行演示脚本查看所有新功能
python scripts/demo_new_features.py
```

## 📊 输出文件

运行后会生成：

- `result.tiff`: 原始体素网格
- `result_upsampled.tiff`: 上采样的体素网格（如果启用）
- `result_sampled.csv`: 采样的点云（CSV格式，表头: "x [nm],y [nm],z [nm]"）
- `result_info.txt`: 详细统计信息和参数记录

## 📈 性能考虑

- **内存使用**: 上采样会按倍数的三次方增加内存需求
- **处理时间**: 与目标分辨率和点数成正比
- **建议**: 对大数据集使用适中的上采样倍数（如2.0-4.0）

## 📚 文档

详细文档请查看：
- [体素采样和上采样功能说明](docs/voxel_sampling_upsampling.md)
- [演示脚本](scripts/demo_new_features.py)

## 🔧 代码质量

- ✅ 完整的类型注解
- ✅ 详细的文档字符串
- ✅ 错误处理和日志记录
- ✅ 参数验证
- ✅ 兼容原有API

---

💡 **提示**: 这些新功能可以独立使用，也可以组合使用，为点云生成工作流提供了更大的灵活性。
