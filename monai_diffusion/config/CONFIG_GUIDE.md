# 配置文件指南

本文档详细说明 `ldm_config.yaml` 中所有配置参数的含义和使用方法。

## 数据配置 (data)

### 基本路径

```yaml
data:
  train_data_dir: "data/voxels_nifti/train"  # 训练数据目录
  val_data_dir: "data/voxels_nifti/val"      # 验证数据目录
```

- **train_data_dir**: 包含NIfTI训练文件的目录
- **val_data_dir**: 包含NIfTI验证文件的目录
- 两个目录都应该包含 `.nii` 或 `.nii.gz` 文件

### 体素设置

```yaml
  voxel_size: 64  # 训练时的目标体素分辨率
```

- **voxel_size**: 训练时使用的体素网格分辨率
- 常用值: 32, 64, 128
- 会自动调整输入数据到此分辨率
- 更大的值需要更多显存但质量更好

### 数据加载

```yaml
  cache_rate: 0.25    # 缓存比例
  num_workers: 4      # 数据加载线程数
  pin_memory: true    # 是否使用固定内存
```

- **cache_rate**: 缓存数据的比例 (0.0-1.0)
  - 0.0 = 不缓存
  - 1.0 = 全部缓存
  - 推荐: 0.25-0.5 (平衡速度和内存)
  
- **num_workers**: 并行数据加载的进程数
  - 推荐: 4-8
  - 过大可能导致内存问题
  
- **pin_memory**: 使用固定内存加速GPU传输
  - GPU训练时设为 `true`
  - CPU训练时设为 `false`

### 数据增强

```yaml
  augmentation:
    enabled: true              # 是否启用数据增强
    random_flip_prob: 0.5      # 随机翻转概率
    random_rotate_prob: 0.5    # 随机旋转概率
```

- **enabled**: 是否使用数据增强（仅训练集）
- **random_flip_prob**: 随机翻转的概率
- **random_rotate_prob**: 随机90度旋转的概率
- 验证集始终不使用增强

---

## AutoencoderKL配置 (autoencoder)

### 网络架构

```yaml
autoencoder:
  spatial_dims: 3              # 空间维度（3D）
  in_channels: 1               # 输入通道数
  out_channels: 1              # 输出通道数
  num_channels: [32, 64, 64]   # 各层通道数
  latent_channels: 3           # 潜在空间通道数
  num_res_blocks: 1            # 残差块数量
  norm_num_groups: 16          # GroupNorm组数
  attention_levels: [false, false, true]  # 哪些层使用注意力
```

**关键参数说明**：

- **num_channels**: 编码器各层的通道数
  - 越大模型容量越大，但显存占用越多
  - 显存不足时可减少: `[16, 32, 32]`
  - 追求质量可增加: `[64, 128, 128]`

- **latent_channels**: 潜在空间的通道数
  - 影响压缩率和表达能力
  - 典型值: 3-8
  - 必须与Diffusion的`in_channels`匹配

- **attention_levels**: 控制注意力机制的使用
  - `true`的位置使用自注意力
  - 通常在较深层使用
  - 增加计算成本但提升质量

### 训练配置

```yaml
  training:
    n_epochs: 500                         # 训练轮数
    batch_size: 2                         # 批次大小
    learning_rate: 1.0e-4                 # 学习率
    
    # 损失权重
    adv_weight: 0.01                      # 对抗损失权重
    perceptual_weight: 0.001              # 感知损失权重
    kl_weight: 1.0e-6                     # KL散度权重
    
    autoencoder_warm_up_n_epochs: 5       # Warm-up轮数
```

**训练参数调优**：

- **n_epochs**: 训练轮数
  - 推荐: 100-500
  - 观察验证损失决定是否提前停止

- **batch_size**: 每个批次的样本数
  - 受显存限制
  - RTX 3090 (24GB): 可用 4-8
  - RTX 3060 (12GB): 建议 2-4

- **learning_rate**: 学习率
  - 默认 1e-4 通常效果很好
  - 训练不稳定可减小到 5e-5
  - 收敛太慢可增大到 2e-4

**损失权重平衡**：

- **adv_weight**: 对抗损失的权重
  - 太大: 训练不稳定，mode collapse
  - 太小: 生成质量差
  - 推荐: 0.005-0.02

- **perceptual_weight**: 感知损失的权重
  - 帮助保持高层语义特征
  - 推荐: 0.0005-0.002

- **kl_weight**: KL散度正则化权重
  - 控制潜在空间的正则性
  - 太大: 输出模糊
  - 太小: 潜在空间不规则
  - 推荐: 1e-7 到 1e-5

- **autoencoder_warm_up_n_epochs**: 
  - 前N个epoch不使用对抗损失
  - 让autoencoder先学习基本重建
  - 推荐: 3-10

### 判别器配置

```yaml
    discriminator:
      num_layers_d: 3        # 判别器层数
      num_channels: 32       # 判别器通道数
```

- **num_layers_d**: 判别器的卷积层数
  - 更多层 = 更强判别能力
  - 推荐: 2-4

- **num_channels**: 判别器的基础通道数
  - 影响判别器容量
  - 推荐: 16-64

### 验证和保存

```yaml
    val_interval: 10      # 验证间隔（epoch）
    save_interval: 10     # 保存间隔（epoch）
```

- **val_interval**: 每N个epoch进行一次验证
- **save_interval**: 每N个epoch保存一次checkpoint
- 频繁验证/保存会降低训练速度

### 快速开发模式

```yaml
    fast_dev_run: false           # 快速开发模式
    fast_dev_run_batches: 2       # 每epoch运行的batch数
```

- **fast_dev_run**: 启用后只运行少量batch
  - 用于快速验证代码修改
  - 类似PyTorch Lightning的fast_dev_run
  - 生产训练时设为 `false`

- **fast_dev_run_batches**: fast_dev_run模式下的batch数
  - 推荐: 2-5

### Checkpoint配置

```yaml
  checkpoints:
    output_dir: "outputs/autoencoder"    # 输出目录
    save_best: true                      # 是否保存最佳模型
    resume_from: null                    # 恢复训练的路径
```

- **output_dir**: checkpoint保存目录
- **save_best**: 是否额外保存验证损失最小的模型
- **resume_from**: 
  - `null`: 从头开始训练
  - 路径: 从checkpoint恢复训练
  - 示例: `"outputs/autoencoder/latest_checkpoint.pt"`

### 日志配置

```yaml
  logging:
    log_dir: "logs/autoencoder"  # TensorBoard日志目录
    log_interval: 10              # 日志记录间隔（batch）
```

- **log_dir**: TensorBoard日志保存位置
- **log_interval**: 每N个batch记录一次到TensorBoard

---

## Diffusion Model配置 (diffusion)

### UNet架构

```yaml
diffusion:
  spatial_dims: 3
  in_channels: 3                          # 必须匹配autoencoder的latent_channels
  out_channels: 3
  num_channels: [32, 64, 64]
  attention_levels: [false, true, true]
  num_head_channels: [0, 64, 64]          # 多头注意力通道数
  num_res_blocks: 1
```

**重要**: `in_channels` 和 `out_channels` 必须等于 `autoencoder.latent_channels`!

- **num_head_channels**: 多头注意力的头数
  - 0 = 不使用注意力
  - 与attention_levels配合使用
  - 推荐: 32, 64, 或 128

### 调度器配置

```yaml
  scheduler:
    num_train_timesteps: 1000           # 训练时间步数
    schedule: "scaled_linear_beta"      # beta调度方式
    beta_start: 0.0015                  # 起始beta值
    beta_end: 0.0195                    # 结束beta值
```

- **num_train_timesteps**: 扩散过程的时间步数
  - 标准值: 1000
  - 更多步数 = 更平滑的扩散过程

- **schedule**: beta值的调度策略
  - `"linear_beta"`: 线性调度
  - `"scaled_linear_beta"`: 缩放线性（推荐）
  - `"squaredcos_cap_v2"`: 余弦调度

- **beta_start/beta_end**: 噪声调度的起止值
  - 控制扩散过程的速度
  - 默认值通常效果很好

### 训练配置

```yaml
  training:
    n_epochs: 500                          # 训练轮数
    batch_size: 2                          # 批次大小
    learning_rate: 1.0e-4                  # 学习率
    
    val_interval: 10                       # 验证间隔
    save_interval: 10                      # 保存间隔
    
    generate_samples_interval: 20          # 生成样本间隔
    num_samples_to_generate: 4             # 生成样本数量
    
    fast_dev_run: false
    fast_dev_run_batches: 2
```

- **generate_samples_interval**: 每N个epoch生成样本
  - 用于监控训练进度
  - 生成样本较慢，不宜太频繁
  - 推荐: 10-50

- **num_samples_to_generate**: 每次生成的样本数
  - 会保存到TensorBoard
  - 推荐: 4-8

### Checkpoint配置

```yaml
  checkpoints:
    autoencoder_path: "outputs/autoencoder/best_model.pt"  # AutoencoderKL路径
    output_dir: "outputs/diffusion"
    save_best: true
    resume_from: null
```

**重要**: `autoencoder_path` 必须指向已训练好的AutoencoderKL!

### 日志配置

```yaml
  logging:
    log_dir: "logs/diffusion"
    log_interval: 10
```

---

## 采样配置 (sampling)

```yaml
sampling:
  num_inference_steps: 1000     # 采样步数
  num_samples: 16               # 生成样本数量
  
  output_dir: "outputs/samples"
  save_format: "nifti"          # 保存格式: "nifti", "pointcloud", "both"
```

- **num_inference_steps**: 生成时的去噪步数
  - 更多步数 = 更好质量但更慢
  - 可以少于训练时的timesteps
  - 推荐: 50-1000

- **num_samples**: 一次生成多少个样本

- **save_format**: 输出格式
  - `"nifti"`: 只保存体素
  - `"pointcloud"`: 只保存点云
  - `"both"`: 两者都保存

### 点云转换配置

```yaml
  pointcloud:
    num_points: 10000              # 目标点数
    threshold: 0.1                 # 体素阈值
    method: "probabilistic"        # 采样方法
```

- **num_points**: 转换为点云时的点数
- **threshold**: 低于此值的体素不参与采样
- **method**: 采样方法
  - `"probabilistic"`: 概率采样（推荐，自然）
  - `"center"`: 体素中心
  - `"random"`: 体素内随机
  - `"weighted"`: 加权采样

---

## 设备配置 (device)

```yaml
device:
  use_cuda: true              # 是否使用GPU
  gpu_id: 0                   # GPU设备ID
  mixed_precision: true       # 是否使用混合精度
```

- **use_cuda**: 
  - `true`: 使用GPU（如果可用）
  - `false`: 强制使用CPU

- **gpu_id**: 使用哪个GPU
  - 0, 1, 2, ... (从0开始编号)
  - 多GPU环境下指定

- **mixed_precision**: 
  - `true`: 使用FP16/FP32混合精度
  - 节省显存，加速训练
  - 推荐开启

---

## 随机种子 (seed)

```yaml
seed: 42
```

- 固定随机种子保证可重复性
- 影响数据划分、数据增强、模型初始化

---

## 配置示例

### 低显存配置 (8GB)

```yaml
data:
  voxel_size: 32
  cache_rate: 0.0

autoencoder:
  num_channels: [16, 32, 32]
  training:
    batch_size: 1
    n_epochs: 100

diffusion:
  num_channels: [16, 32, 32]
  training:
    batch_size: 1
    n_epochs: 150
```

### 高质量配置 (24GB+)

```yaml
data:
  voxel_size: 128
  cache_rate: 1.0

autoencoder:
  num_channels: [64, 128, 128]
  latent_channels: 8
  training:
    batch_size: 4
    n_epochs: 500

diffusion:
  num_channels: [64, 128, 128]
  training:
    batch_size: 4
    n_epochs: 500

sampling:
  num_inference_steps: 1000
```

### 快速测试配置

```yaml
autoencoder:
  training:
    n_epochs: 10
    fast_dev_run: true
    fast_dev_run_batches: 2

diffusion:
  training:
    n_epochs: 10
    fast_dev_run: true
    fast_dev_run_batches: 2
```

---

## 配置文件管理

### 创建实验配置

```bash
# 复制默认配置
cp monai_diffusion/config/ldm_config.yaml experiments/exp1_config.yaml

# 修改配置
vim experiments/exp1_config.yaml

# 使用自定义配置训练
python monai_diffusion/3d_ldm/train_autoencoder.py \
    --config experiments/exp1_config.yaml
```

### 版本控制

推荐使用git追踪配置文件：

```bash
git add monai_diffusion/config/ldm_config.yaml
git commit -m "Update training config for experiment X"
```

### 配置验证

训练开始时会打印所有配置参数，检查是否正确：

```
INFO - 初始化NIfTI数据集:
INFO -   数据目录: data/voxels_nifti/train
INFO -   样本数量: 800
INFO -   目标体素大小: 64^3
INFO -   缓存比例: 0.25
INFO -   数据增强: True
```

---

## 故障排除

### 配置文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
```

解决: 确保配置文件路径正确，或使用绝对路径。

### 参数类型错误

```
TypeError: 'NoneType' object is not subscriptable
```

解决: 检查配置文件中是否有拼写错误或缺失必需参数。

### 显存溢出

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

解决: 减小 `batch_size`, `voxel_size`, 或 `num_channels`。

---

## 总结

配置文件是控制整个训练流程的中枢。理解每个参数的作用，可以：

1. **针对硬件优化**: 调整batch_size和cache_rate
2. **平衡质量和速度**: 调整voxel_size和num_channels
3. **快速实验迭代**: 使用fast_dev_run模式
4. **追求最佳质量**: 增加训练epoch和采样步数

建议从默认配置开始，逐步调整以适应具体需求。

