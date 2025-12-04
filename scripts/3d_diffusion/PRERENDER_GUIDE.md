# 预渲染体素使用指南

## 概述

预渲染功能可以大幅加速训练速度。通过提前将点云转换为体素并保存，训练时直接加载预渲染的体素数据，避免了每次训练时的重复计算。

## 优势

1. **训练速度提升**：跳过耗时的点云到体素转换过程
2. **内存优化**：通过精度控制和阈值过滤减小数据大小
3. **灵活性**：支持批量保存，防止单个文件过大
4. **测试便利**：可以只预渲染部分数据用于测试

## 使用流程

### 步骤1: 预渲染体素数据

#### 方法A: 使用配置文件（推荐）

```bash
python scripts/3d_diffusion/prerender_voxels.py \
    --config exp_record/tissue/diffusion_config.yaml \
    --output-dir /path/to/output
```

#### 方法B: 使用命令行参数

```bash
python scripts/3d_diffusion/prerender_voxels.py \
    --input-h5 /root/shared-nvme/pointclouds-clean-check.h5 \
    --output-dir /root/shared-nvme/prerendered_voxels_128 \
    --voxel-size 128 \
    --voxelization-method gaussian \
    --sigma 1.0 \
    --volume-dims 28000.0 28000.0 1600.0 \
    --padding 0 0 0 \
    --batch-size 100 \
    --threshold 0.001 \
    --precision 10 \
    --compression gzip
```

#### 测试模式（只处理前N个样本）

```bash
python scripts/3d_diffusion/prerender_voxels.py \
    --config exp_record/tissue/diffusion_config.yaml \
    --output-dir /root/shared-nvme/prerendered_voxels_test \
    --max-samples 50
```

### 步骤2: 修改配置文件

编辑 `exp_record/tissue/diffusion_config.yaml`:

```yaml
data:
  # 启用预渲染数据
  use_prerendered: true
  prerendered_dir: "/root/shared-nvme/prerendered_voxels_128"
```

### 步骤3: 正常训练

```bash
python scripts/3d_diffusion/train_diffusion.py \
    --config exp_record/tissue/diffusion_config.yaml
```

## 配置参数详解

### 预渲染参数（prerender节）

```yaml
prerender:
  output_dir: "/root/shared-nvme/prerendered_voxels_128"  # 输出目录
  max_samples: null      # 最大处理样本数 (null=全部, 数字=仅处理前N个)
  batch_size: 100        # 每个H5文件的体素数 (防止文件过大)
  threshold: 0.001       # 过滤阈值 (低于此值设为0)
  precision: 10          # 小数位数 (减小文件大小)
  compression: "gzip"    # 压缩方式
  overwrite: false       # 是否覆盖已存在文件
```

### 数据参数（data节）

```yaml
data:
  use_prerendered: true  # 是否使用预渲染数据
  prerendered_dir: "/path/to/prerendered"  # 预渲染数据目录
```

## 参数调优建议

### threshold（过滤阈值）

- **作用**：将低于阈值的体素值设为0，减小文件大小
- **推荐值**：
  - `0.001` - 适度过滤，保留大部分细节
  - `0.01` - 激进过滤，大幅减小文件
  - `0.0` - 不过滤

### precision（精度）

- **作用**：控制小数点后保留位数
- **推荐值**：
  - `10` - 高精度，适合大多数场景
  - `6` - 中等精度，进一步减小文件
  - `4` - 低精度，极限压缩

### batch_size（批次大小）

- **作用**：控制每个H5文件包含的体素数
- **推荐值**：
  - `50-100` - 适合128³体素
  - `100-200` - 适合64³体素
  - `20-50` - 适合256³体素

### max_samples（测试用）

- **测试阶段**：设置为 `10`, `50`, `100` 等小值
- **生产环境**：设置为 `null`（处理全部）

## 文件结构

预渲染后的目录结构：

```
prerendered_voxels_128/
├── metadata.yaml              # 元数据文件
├── voxels_batch_0000.h5       # 批次0
├── voxels_batch_0001.h5       # 批次1
├── voxels_batch_0002.h5       # 批次2
└── ...
```

每个H5文件包含：
- `voxels` - 体素数据 (N, D, H, W)
- `indices` - 原始索引映射
- `attrs` - 元数据属性

## 性能对比

| 模式 | 数据加载时间 | 训练速度 | 适用场景 |
|------|------------|---------|---------|
| 动态渲染 | ~1-5秒/批次 | 慢 | 小数据集、调试 |
| 预渲染 | ~0.01秒/批次 | 快 | 生产环境、大数据集 |

## 常见问题

### Q: 预渲染需要多少存储空间？

**A**: 取决于体素大小和参数设置：
- 128³体素，无压缩：~8MB/样本
- 128³体素，gzip压缩+阈值过滤：~0.5-2MB/样本
- 64³体素：约为128³的1/8

### Q: 预渲染需要多长时间？

**A**: 取决于数据集大小：
- 100个样本：~2-5分钟
- 1000个样本：~20-50分钟
- 建议在后台或服务器上运行

### Q: 如何切换回动态渲染？

**A**: 修改配置文件：
```yaml
data:
  use_prerendered: false
```

### Q: 预渲染数据可以复用吗？

**A**: 可以！只要体素化参数（voxel_size, method, sigma等）相同，预渲染的数据可以在不同实验中复用。

### Q: 如何验证预渲染是否成功？

**A**: 检查输出目录：
1. 确认存在 `metadata.yaml`
2. 确认存在多个 `voxels_batch_*.h5` 文件
3. 查看日志中的"预渲染完成"消息

## 最佳实践

1. **首次使用**：先用 `--max-samples 10` 测试，确认参数正确
2. **批量大小**：根据可用存储空间调整 `batch_size`
3. **压缩选项**：推荐使用 `gzip`，平衡压缩率和速度
4. **阈值设置**：观察第一批次的稀疏度统计，调整阈值
5. **定期清理**：删除不需要的预渲染数据以节省空间

## 示例工作流

```bash
# 1. 测试预渲染（10个样本）
python scripts/3d_diffusion/prerender_voxels.py \
    --config exp_record/tissue/diffusion_config.yaml \
    --output-dir /tmp/test_prerender \
    --max-samples 10

# 2. 检查结果
ls -lh /tmp/test_prerender

# 3. 满意后，预渲染全部数据
python scripts/3d_diffusion/prerender_voxels.py \
    --config exp_record/tissue/diffusion_config.yaml \
    --output-dir /root/shared-nvme/prerendered_voxels_128

# 4. 修改配置启用预渲染
# 编辑 exp_record/tissue/diffusion_config.yaml
# 设置 use_prerendered: true

# 5. 开始训练
python scripts/3d_diffusion/train_diffusion.py \
    --config exp_record/tissue/diffusion_config.yaml
```

## 故障排查

### 错误: "元数据文件不存在"

**原因**: 预渲染未完成或目录错误

**解决**: 检查 `prerendered_dir` 路径，确认预渲染已完成

### 错误: "未找到批次文件"

**原因**: 预渲染数据损坏或被删除

**解决**: 重新运行预渲染脚本

### 训练速度没有提升

**检查**:
1. 确认 `use_prerendered: true`
2. 查看日志确认使用了 `PrerenderedVoxelDataset`
3. 检查 `num_workers` 设置（建议 4-8）

