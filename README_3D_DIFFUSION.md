# 3D Diffusion 体素生成模型

基于DDPM (Denoising Diffusion Probabilistic Models) 的3D体素生成系统，使用PyTorch Lightning框架实现。

## 🌟 特性

- **3D UNet架构**: 专为体素数据设计的3D卷积神经网络
- **DDPM扩散过程**: 高质量的生成式建模
- **DDIM采样**: 快速确定性采样支持
- **可配置体素大小**: 支持32³, 64³, 128³等分辨率
- **自动验证**: 每N个epoch自动生成验证样本并保存TIFF
- **PyTorch Lightning**: 现代化训练框架
- **配置驱动**: YAML配置文件系统
- **EMA支持**: 指数移动平均提升生成质量

## 📋 系统要求

### 硬件要求
- **GPU**: 8GB+ VRAM (64³体素) / 4GB+ (32³体素)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 软件依赖
```bash
# 核心依赖
torch>=1.12.0
pytorch-lightning>=1.8.0
numpy>=1.21.0
h5py>=3.7.0
tifffile>=2022.5.4
scipy>=1.9.0
PyYAML>=6.0
tqdm>=4.64.0

# 可选依赖 (推荐)
tensorboard>=2.9.0
matplotlib>=3.5.0
```

## 🚀 快速开始

### 1. 环境设置
```bash
# 克隆项目
cd /repos/coarse2fine-pcgen

# 安装依赖
pip install torch pytorch-lightning numpy h5py tifffile scipy PyYAML tqdm

# 检查系统
python scripts/run_example.py --check-deps --test-system
```

### 2. 数据准备
确保您的数据为HDF5格式，结构如下：
```
data.h5
├── point_clouds: (N, M, 3)  # N个样本，每个M个点，xyz坐标
```

示例数据创建：
```bash
python scripts/test_conversion/create_sample_data.py
```

### 3. 配置模型
编辑 `configs/diffusion_config.yaml`：
```yaml
data:
  h5_file_path: "/path/to/your/data.h5"
  voxel_size: 64
  voxelization_method: "gaussian"

model:
  model_channels: 128
  num_timesteps: 1000

training:
  batch_size: 4
  learning_rate: 0.0001
  max_epochs: 200
```

### 4. 开始训练
```bash
# 基础训练
python scripts/train_diffusion.py --config configs/diffusion_config.yaml

# 自定义参数
python scripts/train_diffusion.py \
  --data-path /path/to/data.h5 \
  --voxel-size 64 \
  --batch-size 4 \
  --max-epochs 100
```

### 5. 生成样本
```bash
# 从检查点生成样本
python scripts/generate_samples.py \
  experiments/3d_diffusion_voxels/version_0/checkpoints/best-epoch=XX-val_loss=X.XXXX.ckpt \
  --num-samples 8 \
  --output-dir generated_samples
```

## 📊 数据流程

```
点云数据 → 体素化 → 3D Diffusion训练 → 生成体素 → (可选)转换回点云
   ↓           ↓            ↓              ↓
 H5文件    64³体素网格   UNet噪声预测    TIFF文件
```

### 体素化方法
- **occupancy**: 二值占有网格 (0/1)
- **density**: 密度网格 (点数统计)
- **gaussian**: 高斯密度分布 (推荐)

## 🔧 配置详解

### 关键配置项
| 配置项 | 说明 | 推荐值 |
|--------|------|---------|
| `data.voxel_size` | 体素分辨率 | 64 |
| `data.voxelization_method` | 体素化方法 | gaussian |
| `model.model_channels` | 模型基础通道 | 128 |
| `training.batch_size` | 批次大小 | 4-8 |
| `training.learning_rate` | 学习率 | 1e-4 |
| `validation.sample_interval` | 验证间隔 | 10 |

### 环境变量覆盖
```bash
export DIFFUSION_DATA_VOXEL_SIZE=128
export DIFFUSION_TRAINING_BATCH_SIZE=2
export DIFFUSION_MODEL_MODEL_CHANNELS=64
```

## 📈 训练监控

### TensorBoard
```bash
tensorboard --logdir experiments/3d_diffusion_voxels/version_0/logs
```

### 关键指标
- **train_loss**: 训练损失
- **val_loss**: 验证损失
- **gen_occupancy**: 生成体素占有率
- **learning_rate**: 学习率变化

## 💡 优化建议

### 内存优化
```yaml
# 减少批次大小
training:
  batch_size: 2
  accumulate_grad_batches: 2  # 等效batch_size=4

# 使用混合精度
system:
  precision: "16-mixed"

# 减少工作进程
system:
  num_workers: 2
```

### 性能优化
```yaml
# 启用模型编译 (PyTorch 2.0+)
model:
  compile_model: true

# 数据缓存
data:
  cache_voxels: true
  max_cache_size: 1000

# 持久化工作进程
system:
  persistent_workers: true
```

## 🐛 调试指南

### 常见问题
1. **内存不足**: 减少batch_size或voxel_size
2. **训练慢**: 检查GPU使用率，启用混合精度
3. **生成质量差**: 增加训练时间，调整学习率
4. **验证TIFF为空**: 检查体素化参数和数据质量

### 调试模式
```bash
# 快速测试模式
python scripts/train_diffusion.py --fast-dev-run

# 过拟合少量数据
python scripts/train_diffusion.py --overfit-batches 5

# 限制训练数据
python scripts/train_diffusion.py --limit-train-batches 0.1
```

## 📁 输出文件

训练过程中会生成以下文件：
```
experiments/3d_diffusion_voxels/version_0/
├── checkpoints/              # 模型检查点
│   ├── best-epoch=XX-val_loss=X.XXXX.ckpt
│   └── last.ckpt
├── logs/                     # TensorBoard日志
├── validation_outputs/       # 验证TIFF文件
│   ├── epoch_0010/
│   ├── epoch_0020/
│   └── ...
└── config.yaml              # 实际使用的配置
```

## 🔄 后续处理

### 体素到点云转换
生成体素后可转换回点云：
```python
from src.voxel.converter import PointCloudToVoxel

converter = PointCloudToVoxel(voxel_size=64)
point_cloud = converter.voxel_to_points(
    voxel_grid, 
    threshold=0.5, 
    method='probabilistic'
)
```

## 📚 技术细节

### 模型架构
- **编码器**: 3D卷积下采样，通道数递增
- **解码器**: 3D转置卷积上采样，跳跃连接
- **注意力**: 多尺度自注意力机制
- **时间嵌入**: 正弦位置编码 + MLP

### 扩散过程
- **前向过程**: 逐步添加高斯噪声
- **反向过程**: UNet预测噪声
- **损失函数**: MSE噪声预测损失
- **采样**: DDIM确定性采样

### 数据处理
- **归一化**: 体素值映射到[-1,1]
- **增强**: 旋转、翻转、噪声
- **缓存**: 内存缓存加速训练

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于MIT许可证开源。
