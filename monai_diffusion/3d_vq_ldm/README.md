# VQ-Latent Diffusion Model (VQ-LDM)

基于MONAI Generative Models的3D VQ-Latent Diffusion Model实现。

## 📖 概述

VQ-LDM是一个两阶段的生成模型：

1. **阶段1 - VQVAE**: 学习将3D体素数据压缩到离散的潜在空间（codebook）
2. **阶段2 - Diffusion**: 在VQVAE的潜在空间上训练扩散模型

与传统的Latent Diffusion Model (LDM)相比，VQ-LDM使用**离散量化**的潜在空间，这可以：
- 提供更紧凑的表示
- 更容易与自回归模型（如Transformer）结合
- 在某些任务上有更好的重建质量

## 🏗️ 架构对比

### VQ-LDM vs LDM

| 特性 | VQ-LDM | LDM |
|------|--------|-----|
| 第一阶段 | VQVAE | AutoencoderKL |
| 潜在空间 | 离散量化（Codebook） | 连续（高斯分布） |
| 损失函数 | 重建损失 + 量化损失 | 重建损失 + KL散度 + 对抗损失 + 感知损失 |
| 训练复杂度 | 相对简单 | 需要判别器和感知损失网络 |
| 应用 | 适合结合自回归模型 | 适合纯扩散生成 |

### VQVAE结构

```
输入图像 (B, 1, 64, 64, 64)
    ↓
Encoder (Conv3D + ResBlocks)
    ↓
潜在表示 (B, 32, 16, 16, 16)  # 假设4倍下采样
    ↓
Vector Quantization (查找最近的codebook向量)
    ↓
量化潜在表示 (B, 32, 16, 16, 16)
    ↓
Decoder (ConvTranspose3D + ResBlocks)
    ↓
重建图像 (B, 1, 64, 64, 64)
```

### Diffusion结构

```
量化潜在表示 (B, 32, 16, 16, 16)
    ↓
添加噪声 (逐步加噪到t时刻)
    ↓
噪声潜在表示 (B, 32, 16, 16, 16)
    ↓
DiffusionModelUNet (预测噪声)
    ↓
预测的噪声 (B, 32, 16, 16, 16)
    ↓
去噪 (DDPM/DDIM采样)
    ↓
干净潜在表示 → VQVAE解码 → 生成图像
```

## 🚀 快速开始

### 1. 准备数据

确保数据已经准备好并放在正确的目录下（参考`data/`目录）。

### 2. 配置文件

编辑配置文件 `monai_diffusion/config/vq_ldm_config_local.yaml`：

```yaml
# 关键参数
data:
  train_data_dir: "data/microtubules/nifti/train"
  val_data_dir: "data/microtubules/nifti/val"
  patch_size: [64, 64, 64]
  batch_size: 4

vqvae:
  num_embeddings: 256  # Codebook大小
  embedding_dim: 32     # 嵌入维度
  num_channels: [64, 128, 256]

diffusion:
  in_channels: 32  # 必须等于vqvae.embedding_dim
  num_channels: [64, 128, 256, 512]
```

### 3. 训练VQVAE（阶段1）

```bash
python monai_diffusion/3d_vq_ldm/train_vqvae.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml
```

训练完成后，最佳模型将保存在：
```
outputs/vq_ldm/vqvae_checkpoints/best_model.pt
```

### 4. 训练Diffusion（阶段2）

```bash
python monai_diffusion/3d_vq_ldm/train_diffusion.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml
```

训练完成后，最佳模型将保存在：
```
outputs/vq_ldm/diffusion_checkpoints/best_model.pt
```

### 5. 监控训练

使用TensorBoard查看训练进度：

```bash
# VQVAE训练日志
tensorboard --logdir outputs/vq_ldm/vqvae_logs

# Diffusion训练日志
tensorboard --logdir outputs/vq_ldm/diffusion_logs
```

## 📊 训练监控指标

### VQVAE阶段

- **total_loss**: 总损失（重建损失 + 量化损失）
- **recon_loss**: 重建损失（L1或MSE）
- **quant_loss**: 量化损失（VQ损失）
- **可视化**: 输入图像 vs 重建图像

### Diffusion阶段

- **loss**: 噪声预测MSE损失
- **可视化**: 真实样本 vs 生成样本

## ⚙️ 关键参数说明

### VQVAE参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_embeddings` | Codebook大小（离散编码数量） | 256-1024 |
| `embedding_dim` | 每个编码的维度 | 16-64 |
| `num_channels` | 编码器/解码器通道数 | [64, 128, 256] |
| `num_res_layers` | 残差块数量 | 2-3 |
| `learning_rate` | 学习率 | 1e-4 |

**Codebook大小的影响**：
- 越大：表示能力越强，但训练越慢，可能过拟合
- 越小：训练快，但表示能力有限，重建质量可能下降

### Diffusion参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `in_channels` | 输入通道数（必须等于embedding_dim） | 32 |
| `num_train_timesteps` | 扩散步数 | 1000 |
| `num_channels` | UNet通道数 | [64, 128, 256, 512] |
| `attention_levels` | 哪些层使用注意力 | [false, false, true, true] |
| `learning_rate` | 学习率 | 1e-4 |

## 🔧 显存优化

如果遇到显存不足（OOM），可以尝试：

### 数据层面
```yaml
data:
  patch_size: [32, 32, 32]  # 从64降到32
  batch_size: 2             # 减小batch_size
  voxel_resize: [256, 256, 256]  # 预先缩小体素
```

### 模型层面
```yaml
vqvae:
  num_channels: [32, 64, 128]  # 减小通道数
  num_embeddings: 128          # 减小codebook

diffusion:
  num_channels: [32, 64, 128, 256]  # 减小UNet通道数
```

### 训练层面
```yaml
device:
  mixed_precision: true  # 启用混合精度（必须）
```

## 📈 预期性能

### VQVAE训练

- **训练时间**（64³ patch，batch_size=4）：
  - 单个epoch: 约5-10分钟（取决于数据量）
  - 收敛: 约50-100 epochs
  
- **显存占用**：
  - 约6-10GB（取决于配置）

### Diffusion训练

- **训练时间**（潜在空间16³，batch_size=8）：
  - 单个epoch: 约2-5分钟
  - 收敛: 约200-500 epochs
  
- **显存占用**：
  - 约4-8GB（潜在空间比原始空间小得多）

## 🆚 何时使用VQ-LDM vs LDM

### 使用VQ-LDM当：
- ✅ 需要离散的潜在表示
- ✅ 计划使用自回归模型（如Transformer）
- ✅ 希望训练过程更简单（不需要判别器）
- ✅ 数据量较少，担心对抗训练不稳定

### 使用LDM当：
- ✅ 需要连续的潜在空间
- ✅ 追求最高的生成质量
- ✅ 有足够的数据和计算资源
- ✅ 希望利用感知损失和对抗训练

## 🐛 常见问题

### Q1: VQVAE重建效果很差，输出全黑/全白？

**可能原因**：
- 数据归一化问题
- 学习率过大/过小
- 模型容量不足

**解决方案**：
1. 检查数据是否正确归一化到[-1, 1]或[0, 1]
2. 降低学习率（如1e-5）
3. 增加`num_embeddings`或`embedding_dim`
4. 使用加权重建损失（参考3d_ldm的实现）

### Q2: Diffusion训练loss不下降？

**可能原因**：
- VQVAE没有训练好
- 学习率不合适
- scale_factor计算有问题

**解决方案**：
1. 确保VQVAE已经收敛（重建效果好）
2. 调整学习率
3. 检查潜在空间分布是否合理

### Q3: 生成的样本质量不好？

**可能原因**：
- 训练epoch不够
- 采样步数太少
- VQVAE压缩比过高

**解决方案**：
1. 增加训练epoch（至少300+）
2. 增加推理步数（num_inference_steps）
3. 调整VQVAE架构，减少下采样倍数

## 📚 参考资料

- [Neural Discrete Representation Learning (VQVAE)](https://arxiv.org/abs/1711.00937) - van den Oord et al., 2017
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Rombach et al., 2022
- [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)

## 📝 代码结构

```
monai_diffusion/3d_vq_ldm/
├── train_vqvae.py          # VQVAE训练脚本
├── train_diffusion.py      # Diffusion训练脚本（基于VQVAE）
└── README.md               # 本文档

monai_diffusion/config/
└── vq_ldm_config_local.yaml  # VQ-LDM配置文件

outputs/vq_ldm/
├── vqvae_checkpoints/      # VQVAE检查点
├── vqvae_logs/             # VQVAE TensorBoard日志
├── diffusion_checkpoints/  # Diffusion检查点
└── diffusion_logs/         # Diffusion TensorBoard日志
```

## 🎯 下一步

完成VQ-LDM训练后，可以：

1. **生成新样本**：编写推理脚本，从随机噪声生成新的3D体素
2. **条件生成**：添加条件信息（如类别标签、文本描述）到Diffusion模型
3. **自回归建模**：在VQVAE的离散潜在空间上训练Transformer
4. **插值与编辑**：利用潜在空间进行语义插值和图像编辑

---

**注意**: VQ-LDM是实验性实现，性能可能随数据集和超参数变化。建议先在小规模数据上测试配置。

