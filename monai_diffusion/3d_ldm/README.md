# 3D Latent Diffusion Model Training

基于MONAI Generative Models的3D Latent Diffusion Model训练管道。

## 快速开始

### 1. 数据准备

首先将H5点云数据转换为NIfTI格式：

```bash
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_nifti \
    --voxel_size 64 \
    --method gaussian \
    --sigma 1.0
```

### 2. 配置文件

编辑 `monai_diffusion/config/ldm_config.yaml` 设置训练参数：

```yaml
data:
  train_data_dir: "data/voxels_nifti/train"
  val_data_dir: "data/voxels_nifti/val"
  voxel_size: 64

autoencoder:
  training:
    n_epochs: 500
    batch_size: 2
    
diffusion:
  training:
    n_epochs: 500
    batch_size: 2
```

### 3. 训练AutoencoderKL（阶段1）

```bash
python monai_diffusion/3d_ldm/train_autoencoder.py \
    --config monai_diffusion/config/ldm_config.yaml
```

快速开发模式（调试用）：

```yaml
# 在配置文件中设置
autoencoder:
  training:
    fast_dev_run: true
    fast_dev_run_batches: 2
```

### 4. 训练Diffusion Model（阶段2）

确保AutoencoderKL训练完成后：

```bash
python monai_diffusion/3d_ldm/train_diffusion.py \
    --config monai_diffusion/config/ldm_config.yaml
```

### 5. 生成样本

```bash
python monai_diffusion/3d_ldm/generate_samples.py \
    --config monai_diffusion/config/ldm_config.yaml \
    --output_dir outputs/samples
```

## 训练恢复

如果训练中断，可以从checkpoint恢复：

```yaml
# 在配置文件中设置
autoencoder:
  checkpoints:
    resume_from: "outputs/autoencoder/latest_checkpoint.pt"

diffusion:
  checkpoints:
    resume_from: "outputs/diffusion/latest_checkpoint.pt"
```

然后重新运行训练脚本。

## 监控训练

使用TensorBoard查看训练进度：

```bash
# AutoencoderKL训练日志
tensorboard --logdir logs/autoencoder

# Diffusion Model训练日志
tensorboard --logdir logs/diffusion
```

## 配置参数说明

### 数据配置

- `voxel_size`: 训练时的体素分辨率（会自动调整输入数据）
- `cache_rate`: 数据缓存比例，0-1之间，越大越占内存但速度越快
- `num_workers`: 数据加载线程数

### AutoencoderKL配置

- `n_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `adv_weight`: 对抗损失权重
- `perceptual_weight`: 感知损失权重
- `kl_weight`: KL散度权重
- `autoencoder_warm_up_n_epochs`: 前N个epoch不使用对抗损失

### Diffusion配置

- `num_train_timesteps`: 训练时的时间步数（通常1000）
- `num_inference_steps`: 采样时的时间步数（越大质量越好但速度越慢）
- `generate_samples_interval`: 每N个epoch生成样本用于监控

### 快速开发模式

用于快速验证代码和调试：

```yaml
training:
  fast_dev_run: true
  fast_dev_run_batches: 2  # 每个epoch只运行2个batch
```

## 输出文件结构

```
outputs/
├── autoencoder/
│   ├── best_model.pt          # 最佳AutoencoderKL模型
│   └── latest_checkpoint.pt   # 最新checkpoint
├── diffusion/
│   ├── best_model.pt          # 最佳Diffusion模型
│   └── latest_checkpoint.pt   # 最新checkpoint
└── samples/
    ├── nifti/                 # NIfTI格式样本
    │   └── sample_0000.nii.gz
    └── pointcloud/            # 点云格式样本
        └── sample_0000.csv

logs/
├── autoencoder/               # AutoencoderKL训练日志
└── diffusion/                 # Diffusion训练日志
```

## 常见问题

### Q: 显存不足怎么办？

A: 减小 `batch_size` 或 `voxel_size`，或者减小模型通道数：

```yaml
autoencoder:
  num_channels: [32, 64, 64]  # 改为 [16, 32, 32]
```

### Q: 训练很慢怎么办？

A: 
1. 增加 `num_workers`
2. 提高 `cache_rate`
3. 使用混合精度训练（默认已启用）

### Q: 如何调整体素分辨率？

A: 只需修改配置文件中的 `voxel_size`，数据集会自动调整：

```yaml
data:
  voxel_size: 32  # 或 64, 128
```

### Q: 生成的样本质量不好？

A: 
1. 增加训练epoch数
2. 检查AutoencoderKL是否训练充分
3. 增加采样步数 `num_inference_steps`
4. 调整 `scale_factor`（自动计算，但可以手动调整）

