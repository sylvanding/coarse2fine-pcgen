# 惰性加载数据集使用指南

## 概述

新的惰性加载数据集实现了高效的HDF5数据读取机制，专为大规模点云数据集设计。相比传统的全量加载方式，惰性加载具有以下优势：

- **内存高效**: 只在需要时读取单个样本，大幅减少内存占用
- **多进程支持**: 每个worker进程独立打开文件，支持高效的多进程数据加载
- **性能优化**: 支持LRU缓存和性能监控
- **向下兼容**: 保持与原有接口的完全兼容性

## 数据集类型

### 1. H5PairedDataset (基本惰性加载)

最基础的惰性加载实现，适用于大多数场景。

```python
from src.refine_net.data_handler import get_dataset

# 创建基本惰性加载数据集
dataset_class = get_dataset('h5_paired')
train_dataset = dataset_class(None, device, args, split='train')
```

**特点:**
- 初始化时只获取文件元信息，不加载实际数据
- 在`__getitem__`中按需读取单个样本
- 支持多进程数据加载
- 内存占用最小

### 2. H5InferenceDataset (推理惰性加载)

专用于模型推理的数据集，返回详细的样本信息。

```python
dataset_class = get_dataset('h5_inference')
inference_dataset = dataset_class(None, device, args)
```

**特点:**
- 继承自H5PairedDataset的所有优点
- 返回包含原始和归一化点云的详细字典
- 固定使用验证集前4个样本
- 保持原始点数，不进行采样

### 3. LazyH5Dataset (高性能缓存版本)

带有LRU缓存和性能监控的高级版本，推荐用于生产环境。

```python
dataset_class = get_dataset('h5_lazy_cached')
train_dataset = dataset_class(None, device, args, split='train')
```

**特点:**
- 包含基本惰性加载的所有功能
- LRU缓存机制，减少重复I/O操作
- 性能统计和监控
- 可配置的缓存大小

## 配置参数

### 基本文件配置

```python
class Args:
    # 必需参数
    gt_h5_path = "/path/to/gt_data.h5"      # GT点云文件路径
    gt_data_key = "gt_clouds"               # GT数据键名
    noisy_h5_path = "/path/to/noisy_data.h5" # 噪声点云文件路径  
    noisy_data_key = "noisy_clouds"         # 噪声数据键名
    
    # 可选的体素指导
    use_voxel_guidance = False              # 是否使用体素指导
    voxel_h5_path = "/path/to/voxel.h5"     # 体素文件路径
    voxel_data_key = "voxel_grids"          # 体素数据键名
```

### 数据处理配置

```python
class Args:
    # 数据集分割
    train_ratio = 0.8                       # 训练集占比
    
    # 点云采样
    sample_points = 2048                    # 采样点数
    iterations = 10000                      # 训练迭代次数
    
    # 归一化参数
    volume_dims = [20000, 20000, 2500]      # 空间尺寸 (x,y,z)
    padding = [0, 0, 100]                   # 边界padding (x,y,z)
```

### 缓存配置 (仅LazyH5Dataset)

```python
class Args:
    # 缓存设置
    enable_sample_cache = True              # 启用缓存
    sample_cache_size = 100                 # 缓存样本数
```

## 使用示例

### 基本使用

```python
import torch
from torch.utils.data import DataLoader
from src.refine_net.data_handler import get_dataset

# 创建配置
class Config:
    gt_h5_path = "/data/gt_pointclouds.h5"
    gt_data_key = "clouds"
    noisy_h5_path = "/data/noisy_pointclouds.h5"
    noisy_data_key = "clouds"
    train_ratio = 0.8
    sample_points = 2048
    iterations = 10000
    volume_dims = [20000, 20000, 2500]
    padding = [0, 0, 100]

args = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据集
dataset_class = get_dataset('h5_paired')
train_dataset = dataset_class(None, device, args, split='train')
val_dataset = dataset_class(None, device, args, split='val')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# 训练循环
for epoch in range(10):
    for batch_idx, (gt_pc, noisy_pc) in enumerate(train_loader):
        # gt_pc: [batch_size, 3, num_points]
        # noisy_pc: [batch_size, 3, num_points]
        # 进行训练...
        pass
```

### 带缓存的高性能版本

```python
# 使用带缓存的版本以获得更好性能
class PerformanceConfig(Config):
    enable_sample_cache = True
    sample_cache_size = 200  # 根据内存情况调整

args = PerformanceConfig()

dataset_class = get_dataset('h5_lazy_cached')
train_dataset = dataset_class(None, device, args, split='train')

# 训练后查看缓存统计
stats = train_dataset.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"平均加载时间: {stats['avg_load_time']:.4f}秒")
```

### 推理使用

```python
# 推理数据集
dataset_class = get_dataset('h5_inference')
inference_dataset = dataset_class(None, device, args)

for i in range(len(inference_dataset)):
    sample = inference_dataset[i]
    
    # 获取数据
    gt_normalized = sample['gt_normalized']      # 归一化的GT点云
    noisy_normalized = sample['noisy_normalized'] # 归一化的噪声点云
    gt_original = sample['gt_original']          # 原始GT点云
    noisy_original = sample['noisy_original']    # 原始噪声点云
    sample_idx = sample['sample_idx']            # 样本索引
    num_points = sample['num_points']            # 点数
    
    # 进行推理...
    # predicted = model(noisy_normalized)
    
    # 反归一化结果
    # predicted_original = inference_dataset.denormalize_pointcloud(predicted)
```

### 体素指导模式

```python
class VoxelConfig(Config):
    use_voxel_guidance = True
    voxel_h5_path = "/data/voxel_grids.h5"
    voxel_data_key = "grids"

args = VoxelConfig()
dataset_class = get_dataset('h5_lazy_cached')
train_dataset = dataset_class(None, device, args, split='train')

for gt_pc, noisy_pc, voxel_grid in train_dataset:
    # gt_pc: [3, num_points]
    # noisy_pc: [3, num_points]  
    # voxel_grid: [1, D, H, W] 或 [C, D, H, W]
    # 使用体素指导进行训练...
    pass
```

## 性能优化建议

### 1. 缓存大小调优

```python
# 根据可用内存调整缓存大小
import psutil

available_memory_gb = psutil.virtual_memory().available / (1024**3)

if available_memory_gb > 16:
    cache_size = 500  # 大内存系统
elif available_memory_gb > 8:
    cache_size = 200  # 中等内存系统
else:
    cache_size = 50   # 小内存系统

args.sample_cache_size = cache_size
```

### 2. 多进程配置

```python
# 根据CPU核心数配置worker数量
import multiprocessing

num_workers = min(multiprocessing.cpu_count(), 8)  # 最多8个worker
train_loader = DataLoader(dataset, batch_size=16, num_workers=num_workers)
```

### 3. 内存管理

```python
# 定期清理缓存（在epoch结束时）
if hasattr(train_dataset, 'clear_cache'):
    train_dataset.clear_cache()
```

## 故障排除

### 常见错误和解决方案

1. **文件不存在错误**
   ```
   FileNotFoundError: GT文件不存在: /path/to/file.h5
   ```
   - 检查文件路径是否正确
   - 确保文件具有读取权限

2. **数据键不存在错误**
   ```
   ValueError: 数据键 'clouds' 不存在。可用键: ['data', 'metadata']
   ```
   - 使用h5py查看文件中的实际键名
   - 更新配置中的data_key参数

3. **多进程序列化错误**
   ```
   RuntimeError: h5py objects cannot be pickled
   ```
   - 确保使用新的惰性加载数据集
   - 检查num_workers设置

4. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 减少batch_size
   - 降低sample_cache_size
   - 减少sample_points

### 性能监控

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 监控缓存性能
stats = dataset.get_cache_stats()
if stats['hit_rate'] < 0.5:
    print("警告: 缓存命中率较低，考虑增加缓存大小")

# 监控内存使用
import psutil
memory_percent = psutil.virtual_memory().percent
if memory_percent > 80:
    print("警告: 内存使用率过高，考虑减少缓存大小")
```

## 迁移指南

### 从旧版本迁移

如果你正在使用旧的数据集实现，迁移到新的惰性加载版本非常简单：

```python
# 旧版本
# dataset = OldH5Dataset(args)

# 新版本 - 只需要更改get_dataset调用
dataset_class = get_dataset('h5_paired')  # 或 'h5_lazy_cached'
dataset = dataset_class(None, device, args, split='train')
```

所有现有的接口和返回格式都保持不变，无需修改训练代码。

### 性能对比

| 功能 | 旧版本 | 惰性加载版本 | 缓存版本 |
|------|--------|--------------|----------|
| 初始化时间 | 慢 (加载全部数据) | 快 (只读元信息) | 快 |
| 内存占用 | 高 (全部数据) | 低 (单个样本) | 中等 (缓存) |
| 访问速度 | 快 (内存) | 中等 (磁盘I/O) | 快 (缓存命中) |
| 多进程支持 | 有限 | 完全支持 | 完全支持 |

选择建议：
- **开发和调试**: 使用`h5_paired`
- **生产训练**: 使用`h5_lazy_cached`
- **模型推理**: 使用`h5_inference`
