# 条件3D扩散模型 (Conditional 3D Diffusion Model)

使用2D投影图像作为条件来指导3D体素的生成。

## 📋 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [文件结构](#文件结构)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [性能优化](#性能优化)

## 🎯 概述

该模块实现了一个条件3D扩散模型，能够基于2D投影图像生成对应的3D体素数据。主要特点：

- ✨ **条件生成**: 使用2D投影图像作为条件，指导3D体素生成
- 🔄 **灵活投影**: 支持沿任意轴（x/y/z）生成2D投影
- 🚀 **高效训练**: 基于MONAI Generative Models，支持混合精度训练
- 📊 **可视化**: 集成TensorBoard，实时监控训练过程
- 💾 **易于使用**: 完整的配置文件和脚本，开箱即用

## 💻 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐)
- MONAI >= 1.3
- MONAI Generative Models

安装依赖：

```bash
cd /repos/coarse2fine-pcgen
pip install -r monai_diffusion/requirements.txt
```

## 🚀 快速开始

### 1. 准备数据

确保你的数据组织如下：

```
/repos/datasets/your_dataset/
├── train/
│   ├── sample_001.nii.gz
│   ├── sample_002.nii.gz
│   └── ...
└── val/
    ├── sample_001.nii.gz
    ├── sample_002.nii.gz
    └── ...
```

### 2. 修改配置文件

编辑 `monai_diffusion/config/conditional_diffusion_config.yaml`:

```yaml
data:
  train_data_dir: "/path/to/your/train/data"
  val_data_dir: "/path/to/your/val/data"
  voxel_size: [96, 96, 48]  # 根据显存调整
```

### 3. 训练模型

```bash
python monai_diffusion/3d_diffusion/train_conditional_diffusion.py \
    --config monai_diffusion/config/conditional_diffusion_config.yaml
```

### 4. 生成样本

```bash
python monai_diffusion/3d_diffusion/generate_conditional_samples.py \
    --config monai_diffusion/config/conditional_diffusion_config.yaml \
    --checkpoint outputs/conditional_diffusion/checkpoints/best_model.pt \
    --condition path/to/condition_image.png \
    --output outputs/conditional_diffusion/samples/ \
    --num_samples 4 \
    --use_ddim
```

## 🔬 工作原理

### 数据流程

1. **加载3D体素** → 从NIfTI文件加载 (H, W, D)
2. **生成2D投影** → 沿z轴累加得到 (H, W)
3. **归一化** → 投影图像归一化到 [0, 1]

### 训练流程

```
输入:
  - 3D体素 (B, 1, H, W, D)
  - 2D条件图像 (B, 1, H, W)

1. 2D条件编码:
   2D Conv → (B, 64, H/2, W/2)
   2D Conv → (B, 128, H/4, W/4)
   2D Conv → (B, 256, H/8, W/8)
   Global Pool → (B, 256)
   Linear → (B, condition_embed_dim)

2. 条件融合:
   condition_embed + time_embed → combined_embed

3. 3D U-Net:
   noisy_voxel + combined_embed → predicted_noise

4. 损失计算:
   MSE(predicted_noise, true_noise)
```

### 生成流程

```
输入:
  - 2D条件图像 (1, 1, H, W)
  - 随机噪声 (1, 1, H, W, D)

迭代去噪 (T=1000 → 0):
  1. 编码条件图像
  2. 预测噪声
  3. 去除噪声
  4. 更新3D体素

输出:
  - 生成的3D体素 (1, 1, H, W, D)
```

## 📁 文件结构

```
monai_diffusion/3d_diffusion/
├── __init__.py                        # 模块初始化
├── README.md                          # 本文档
├── conditional_dataset.py             # 条件数据集实现
├── train_conditional_diffusion.py     # 训练脚本
└── generate_conditional_samples.py    # 样本生成脚本

monai_diffusion/config/
└── conditional_diffusion_config.yaml  # 配置文件
```

## ⚙️ 配置说明

### 数据配置

```yaml
data:
  train_data_dir: "/path/to/train"      # 训练数据目录
  val_data_dir: "/path/to/val"          # 验证数据目录
  voxel_size: [96, 96, 48]              # 体素分辨率 [X, Y, Z]
  voxel_resize: [128, 128, 64]          # 预处理resize (可选)
  cache_rate: 0.2                       # 数据缓存比例
  num_workers: 4                        # DataLoader工作进程数
```

### 模型配置

```yaml
diffusion:
  spatial_dims: 3                       # 3D模型
  in_channels: 1                        # 输入通道数
  out_channels: 1                       # 输出通道数
  condition_channels: 1                 # 2D条件通道数
  condition_embed_dim: 256              # 条件嵌入维度
  projection_axis: 2                    # 投影轴 (0=x, 1=y, 2=z)
  num_channels: [64, 128, 256]          # U-Net通道数
  attention_levels: [false, false, true] # 注意力层级
```

### 训练配置

```yaml
training:
  n_epochs: 150                         # 训练轮数
  learning_rate: 5.0e-5                 # 学习率
  batch_size: 4                         # 批次大小
  val_interval: 10                      # 验证间隔
  save_interval: 25                     # 保存间隔
  fast_dev_run: false                   # 快速测试模式
```

### 调度器配置

```yaml
scheduler:
  num_train_timesteps: 1000             # 扩散步数
  schedule: "linear_beta"               # 调度方式
  beta_start: 0.0005                    # 起始噪声
  beta_end: 0.0195                      # 结束噪声
```

## 📚 使用示例

### 示例1：基础训练

```bash
# 使用默认配置训练
python monai_diffusion/3d_diffusion/train_conditional_diffusion.py \
    --config monai_diffusion/config/conditional_diffusion_config.yaml
```

### 示例2：快速测试

修改配置文件中的 `fast_dev_run`:

```yaml
training:
  fast_dev_run: true
  fast_dev_run_batches: 2
```

然后运行训练脚本，只会运行2个batch用于快速验证代码。

### 示例3：从checkpoint恢复训练

修改配置文件：

```yaml
checkpoints:
  resume_from: "outputs/conditional_diffusion/checkpoints/latest_checkpoint.pt"
```

### 示例4：生成多个样本

```bash
python monai_diffusion/3d_diffusion/generate_conditional_samples.py \
    --config monai_diffusion/config/conditional_diffusion_config.yaml \
    --checkpoint outputs/conditional_diffusion/checkpoints/best_model.pt \
    --condition condition.png \
    --output samples/ \
    --num_samples 10 \
    --num_inference_steps 250 \
    --use_ddim \
    --save_projections
```

### 示例5：使用NIfTI文件作为条件

```bash
# 从真实的3D体素生成投影作为条件
python monai_diffusion/3d_diffusion/generate_conditional_samples.py \
    --config monai_diffusion/config/conditional_diffusion_config.yaml \
    --checkpoint best_model.pt \
    --condition real_sample.nii.gz \
    --output samples/ \
    --num_samples 1
```

## 🔧 性能优化

### 显存优化

如果遇到显存不足 (Out of Memory)，尝试以下方法：

1. **减小体素分辨率**:
   ```yaml
   voxel_size: [64, 64, 32]  # 从 [96, 96, 48] 降低
   ```

2. **减小批次大小**:
   ```yaml
   batch_size: 2  # 从 4 降低
   ```

3. **减小模型通道数**:
   ```yaml
   num_channels: [32, 64, 128]  # 从 [64, 128, 256] 降低
   ```

4. **使用预resize**:
   ```yaml
   voxel_resize: [96, 96, 48]  # 预先缩小数据
   ```

5. **减少工作进程**:
   ```yaml
   num_workers: 2  # 从 4 降低
   ```

### 训练加速

1. **启用混合精度**:
   ```yaml
   device:
     mixed_precision: true
   ```

2. **使用DDIM采样** (生成时更快):
   ```bash
   --use_ddim  # 可以将1000步减少到50-250步
   ```

3. **增加批次大小** (如果显存允许):
   ```yaml
   batch_size: 8
   ```

4. **使用数据缓存**:
   ```yaml
   cache_rate: 1.0  # 缓存所有数据（需要足够内存）
   ```

### 质量优化

1. **增加推理步数**:
   ```bash
   --num_inference_steps 1000  # 更多步数，质量更好但更慢
   ```

2. **调整噪声调度**:
   ```yaml
   scheduler:
     schedule: "scaled_linear_beta"  # 尝试不同调度方式
   ```

3. **增加训练轮数**:
   ```yaml
   n_epochs: 300  # 更多训练
   ```

## 📊 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir outputs/conditional_diffusion/logs
```

在浏览器中打开 `http://localhost:6006`，可以查看：

- **训练损失曲线**
- **验证损失曲线**
- **生成样本对比** (条件图像 | 真实投影 | 生成投影)

## 🐛 常见问题

### Q1: 生成的3D体素全是噪声？

**A**: 可能是模型还没训练好，需要：
- 增加训练轮数
- 检查TensorBoard中的损失曲线
- 确保数据正确加载和归一化

### Q2: 生成结果很模糊？

**A**: 尝试：
- 增加推理步数 (如1000)
- 调整beta_start和beta_end
- 使用DDPM而不是DDIM

### Q3: 训练速度很慢？

**A**: 
- 启用混合精度训练
- 减小体素分辨率
- 使用更少的num_inference_steps进行验证

### Q4: 条件图像的影响很弱？

**A**: 
- 增加condition_embed_dim
- 调整条件编码器的架构
- 训练更长时间

## 📖 参考文献

1. Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM)
2. Song et al. "Denoising Diffusion Implicit Models" (DDIM)
3. Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
4. MONAI Generative Models Documentation

## 📝 开发计划

- [ ] 实现Classifier-Free Guidance
- [ ] 支持多条件输入（多个2D投影）
- [ ] 集成预训练的2D编码器（如ResNet）
- [ ] 实现潜在扩散版本（与VQVAE结合）
- [ ] 添加评估指标（SSIM、PSNR等）

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循项目根目录的许可证。

