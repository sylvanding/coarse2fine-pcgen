# 去噪自编码器 (Denoising Autoencoder) 使用指南

## 概述

去噪自编码器是一种强大的训练策略，通过让模型学习从噪声中恢复干净图像，迫使它理解数据的深层特征，而不是简单地复制输入。

## 核心原理

### 标准自编码器的问题
- **任务**: 输入 → 模型 → 输出 (输入 = 输出)
- **问题**: 模型可能只学会"完美复制"，输出平均值或简单模式
- **结果**: 无法学到数据的本质特征

### 去噪自编码器的解决方案
- **任务**: 噪声输入 → 模型 → 干净输出
- **优势**: 必须真正"理解"数据结构才能从噪声中恢复
- **结果**: 学习到更深层次的特征表示

## 实现方式

### 1. 训练流程

```python
# 原始干净图像（值域范围：[0, 1]）
clean_images = batch["image"]

# 添加噪声
noisy_images = add_noise(clean_images, 
                        noise_type="mixed",
                        noise_std=0.15, 
                        dropout_prob=0.1)

# 模型输入：带噪声的图像（值域范围：[0, 1]）
# 模型输出：应该恢复出干净图像（值域范围：[0, 1]）
reconstruction = autoencoder(noisy_images)

# 损失计算：重建结果 vs 原始干净图像
loss = criterion(reconstruction, clean_images)
```

**重要说明**: 本项目中，图像数据归一化到**[0, 1]**范围，噪声参数应相应调整。

### 2. 支持的噪声类型

#### Gaussian Noise (高斯噪声)
```yaml
denoising:
  enabled: true
  noise_type: "gaussian"
  noise_std: 0.15  # 噪声标准差
```
- **适用场景**: 模拟传感器噪声、测量误差
- **效果**: 添加随机波动，考验模型的鲁棒性

#### Dropout Noise (丢弃噪声)
```yaml
denoising:
  enabled: true
  noise_type: "dropout"
  dropout_prob: 0.1  # 丢弃概率
```
- **适用场景**: 模拟数据缺失、遮挡
- **效果**: 随机将一些体素置零，考验模型的补全能力

#### Mixed Noise (混合噪声) ⭐推荐
```yaml
denoising:
  enabled: true
  noise_type: "mixed"
  noise_std: 0.15
  dropout_prob: 0.1
```
- **适用场景**: 综合训练，最具挑战性
- **效果**: 同时应用高斯噪声和dropout，模型必须处理多种退化

## 数据范围说明 ⚠️

本项目的图像数据归一化范围是 **[0, 1]**，这与某些教程中使用的 [-1, 1] 范围不同。

### 归一化配置

在数据加载时，通过以下配置控制：

```python
# monai_diffusion/datasets/voxel_nifti_dataset.py
normalize_to_minus_one_one=False  # False表示归一化到[0, 1]
```

在配置文件中体现为：

```yaml
# 数据增强部分的ScaleIntensityRanged变换
transforms.ScaleIntensityRanged(
    keys="image",
    a_min=0.0,      # 输入最小值
    a_max=255.0,    # 输入最大值
    b_min=0.0,      # 输出最小值 [0, 1] ✅
    b_max=1.0,      # 输出最大值
    clip=True
)
```

### 噪声参数的含义

- **noise_std=0.15** 表示添加标准差为0.15的高斯噪声
  - 相对于[0, 1]范围，这是15%的波动
  - 噪声后会clamp到[0, 1]范围内
  
- **dropout_prob=0.1** 表示随机将10%的体素置零
  - 置零后值为0.0（在[0, 1]范围的最小值）

### 如何验证数据范围

可以在训练脚本中添加验证代码：

```python
# 在训练循环中
images = batch["image"].to(device)
print(f"Images min: {images.min().item():.4f}, max: {images.max().item():.4f}")
# 输出应该是: Images min: 0.0000, max: 1.0000（或接近）
```

## 配置说明

### 快速开始

在 `ldm_config_local.yaml` 中添加：

```yaml
autoencoder:
  training:
    # ... 其他配置 ...
    
    # 去噪自编码器配置
    denoising:
      enabled: true  # 启用去噪模式
      noise_type: "mixed"  # 噪声类型
      noise_std: 0.15  # 高斯噪声标准差
      dropout_prob: 0.1  # 体素丢弃概率
```

### 参数调节建议

#### 初期训练（温和模式）
```yaml
denoising:
  enabled: true
  noise_type: "gaussian"
  noise_std: 0.1
  dropout_prob: 0.05
```
- 适合模型快速收敛
- 噪声较小，学习难度较低

#### 标准训练（推荐）⭐
```yaml
denoising:
  enabled: true
  noise_type: "mixed"
  noise_std: 0.15
  dropout_prob: 0.1
```
- 平衡训练难度和模型性能
- 混合噪声提供更全面的训练

#### 高难度训练（进阶）
```yaml
denoising:
  enabled: true
  noise_type: "mixed"
  noise_std: 0.2
  dropout_prob: 0.15
```
- 最具挑战性
- 可能需要更长的训练时间
- 能学到更鲁棒的特征

### 完整配置示例

```yaml
autoencoder:
  # 网络架构
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  num_channels: [32, 64, 64]
  latent_channels: 16
  
  training:
    n_epochs: 500
    batch_size: 1
    learning_rate: 1.0e-4
    
    # 去噪自编码器
    denoising:
      enabled: true
      noise_type: "mixed"
      noise_std: 0.15
      dropout_prob: 0.1
    
    # 其他配置...
```

## 运行训练

```bash
# 使用默认配置（已启用去噪模式）
python monai_diffusion/3d_ldm/train_autoencoder.py \
    --config monai_diffusion/config/ldm_config_local.yaml
```

## 可视化结果

启用去噪模式后，TensorBoard中会显示：

### Patch-based重建
- **标签**: `patch_denoising/sample_X`
- **显示**: 噪声输入 | 模型重建 | 干净目标
- **解读**: 
  - 左侧：带噪声的输入
  - 中间：模型的去噪重建结果
  - 右侧：原始干净图像（目标）

### 误差图
- **标签**: `patch_denoising/sample_X_error`
- **显示**: |重建 - 目标| 的绝对误差
- **解读**: 
  - 越暗表示重建越准确
  - 亮区表示重建误差较大的区域

## 预期效果

### 训练过程中观察到的现象

1. **初期 (0-50 epochs)**
   - 误差较大
   - 重建结果模糊
   - 模型正在学习基本结构

2. **中期 (50-200 epochs)**
   - 误差逐渐减小
   - 重建结果清晰度提升
   - 细节开始恢复

3. **后期 (200+ epochs)**
   - 误差稳定在较低水平
   - 重建结果接近干净图像
   - 模型学会了鲁棒的特征表示

### 与标准自编码器对比

| 特性 | 标准自编码器 | 去噪自编码器 |
|------|-------------|-------------|
| 训练难度 | 简单 | 中等到困难 |
| 收敛速度 | 快 | 较慢 |
| 特征质量 | 可能学到表层特征 | 学到深层鲁棒特征 |
| 泛化能力 | 一般 | 更强 |
| 抗噪能力 | 较弱 | 强 |
| 下游任务性能 | 一般 | 更好 |

## 理论依据

### 为什么去噪更有效？

1. **信息瓶颈理论**
   - 噪声迫使模型压缩信息
   - 只有本质特征能通过瓶颈
   - 噪声细节无法通过，自然被过滤

2. **正则化效应**
   - 噪声相当于一种正则化
   - 防止模型过拟合
   - 提升泛化能力

3. **鲁棒性训练**
   - 模型必须处理各种退化
   - 学习到的特征更稳定
   - 对真实世界的噪声更有抵抗力

## 调试建议

### 如果重建效果不好

1. **降低噪声强度**
   ```yaml
   noise_std: 0.1  # 从0.15降到0.1
   dropout_prob: 0.05  # 从0.1降到0.05
   ```

2. **切换噪声类型**
   ```yaml
   noise_type: "gaussian"  # 从"mixed"改为单一类型
   ```

3. **延长训练时间**
   - 去噪任务更复杂，可能需要更多epoch

4. **检查数据质量**
   - 确保输入数据已正确归一化到**[0, 1]**
   - 检查 `normalize_to_minus_one_one=False` 是否设置正确

### 如果训练不稳定

1. **降低学习率**
   ```yaml
   learning_rate: 5.0e-5  # 从1e-4降到5e-5
   ```

2. **增加warm-up期**
   ```yaml
   autoencoder_warm_up_n_epochs: 10  # 更长的warm-up
   ```

3. **禁用感知损失**
   ```yaml
   use_perceptual_loss: false
   ```

## 进阶应用

### 1. 渐进式噪声增强

在训练过程中逐渐增加噪声强度：

```python
# 在训练循环中动态调整噪声
epoch_ratio = epoch / n_epochs
noise_std = 0.05 + 0.15 * epoch_ratio  # 从0.05逐渐增加到0.2
```

### 2. 自适应噪声

根据重建误差动态调整噪声：

```python
# 如果重建误差小，增加噪声难度
if val_loss < threshold:
    noise_std *= 1.1  # 增加10%
```

### 3. 多尺度噪声

对不同尺度的特征添加不同强度的噪声。

## 参考文献

- Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders", ICML 2008
- Vincent et al., "Stacked Denoising Autoencoders", JMLR 2010

## 总结

去噪自编码器通过简单的策略——**给输入加噪声，让模型恢复干净输出**——就能显著提升模型的特征学习能力。这是一种经典且有效的训练技巧，强烈推荐在AutoencoderKL训练中使用！

🔥 **核心思想**: 不要让模型做"完美复制"，给它增加点难度，它会学得更好！

