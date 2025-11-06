# AutoencoderKL 损失函数使用指南

## 概述

为了解决模型预测全黑图像的问题，我们实现了多种加权损失函数，特别针对前景像素（微管）较少的不平衡数据集。

## 问题背景

在微管体素数据中：
- **背景像素（黑色）**: 占据大部分体素（例如 95%）
- **前景像素（微管）**: 只占少量体素（例如 5%）

使用标准L1/MSE损失时，模型容易学会预测全黑图像，因为这样能最小化大部分像素的误差。

## 损失函数类型

### 1. 标准L1损失（`l1`）
**适用场景**: 前景和背景较为平衡的数据

```yaml
loss:
  reconstruction_loss_type: "l1"
```

**优点**: 简单、稳定
**缺点**: 对不平衡数据效果不佳

---

### 2. 标准MSE损失（`mse`）
**适用场景**: 需要对大误差更敏感的场景

```yaml
loss:
  reconstruction_loss_type: "mse"
```

**优点**: 对大误差惩罚更重
**缺点**: 对不平衡数据效果不佳

---

### 3. 加权重建损失（`weighted`）⭐ **推荐用于不平衡数据**
**适用场景**: 前景像素较少，需要强调前景重建质量

```yaml
loss:
  reconstruction_loss_type: "weighted"
  weighted:
    loss_type: "l1"              # 或 "mse"
    foreground_weight: 20.0      # 前景像素权重 ⭐
    background_weight: 1.0       # 背景像素权重
    threshold: 0.1               # 前景阈值
```

**工作原理**:
- 将目标图像中像素值 > `threshold` 的像素视为前景
- 前景像素的误差乘以 `foreground_weight`
- 背景像素的误差乘以 `background_weight`

**参数调节建议**:
- **轻度不平衡** (前景约10%): `foreground_weight=10.0`
- **中度不平衡** (前景约5%): `foreground_weight=20.0` ⭐ 推荐
- **严重不平衡** (前景约1%): `foreground_weight=50.0` 或更高

**优点**: 
- 简单直接，迫使模型关注前景
- 权重可灵活调节

**缺点**: 
- 需要手动调节阈值和权重

---

### 4. Dice Loss（`dice`）
**适用场景**: 分割任务，关注前景区域的整体重叠度

```yaml
loss:
  reconstruction_loss_type: "dice"
  dice:
    smooth: 1.0e-5
    sigmoid: false
```

**工作原理**:
Dice Loss = 1 - Dice系数，其中:
```
Dice系数 = 2 × |预测 ∩ 目标| / (|预测| + |目标|)
```

**优点**: 
- 直接优化前景重叠度
- 对类别不平衡天然鲁棒

**缺点**: 
- 可能忽略逐像素的细节
- 训练初期可能不稳定

---

### 5. 组合损失（`combined`）⭐⭐ **最推荐**
**适用场景**: 需要同时关注前景重叠度和逐像素细节

```yaml
loss:
  reconstruction_loss_type: "combined"
  combined:
    dice_weight: 1.0            # Dice Loss权重
    recon_weight: 1.0           # 重建损失权重
    recon_loss_type: "l1"       # "l1" 或 "mse"
    foreground_weight: 20.0     # 前景像素权重 ⭐
    background_weight: 1.0      # 背景像素权重
    threshold: 0.1              # 前景阈值
    dice_smooth: 1.0e-5         # Dice平滑项
```

**总损失 = dice_weight × Dice Loss + recon_weight × 加权重建损失**

**优点**: 
- 结合两种损失的优势
- Dice Loss关注整体前景重叠
- 加权重建损失关注逐像素细节
- 最适合不平衡数据

**权重调节建议**:
- **平衡两者**: `dice_weight=1.0, recon_weight=1.0` ⭐ 推荐
- **强调重叠度**: `dice_weight=2.0, recon_weight=1.0`
- **强调细节**: `dice_weight=1.0, recon_weight=2.0`

---

## 使用示例

### 示例1: 使用组合损失（推荐配置）

```yaml
# monai_diffusion/config/ldm_config_local.yaml
autoencoder:
  training:
    loss:
      reconstruction_loss_type: "combined"
      combined:
        dice_weight: 1.0
        recon_weight: 1.0
        recon_loss_type: "l1"
        foreground_weight: 20.0    # 根据数据集调整
        background_weight: 1.0
        threshold: 0.1              # 根据归一化范围调整
        dice_smooth: 1.0e-5
```

### 示例2: 纯加权L1损失（简单有效）

```yaml
autoencoder:
  training:
    loss:
      reconstruction_loss_type: "weighted"
      weighted:
        loss_type: "l1"
        foreground_weight: 30.0     # 对于极度不平衡的数据
        background_weight: 1.0
        threshold: 0.1
```

### 示例3: 对比测试标准L1和加权损失

运行两次训练，对比效果：

```yaml
# 第一次: 标准L1
loss:
  reconstruction_loss_type: "l1"

# 第二次: 加权损失
loss:
  reconstruction_loss_type: "weighted"
  weighted:
    foreground_weight: 20.0
```

---

## 参数调节指南

### 1. 前景阈值（`threshold`）

根据数据的归一化范围设置：

- **归一化到 [0, 1]**: `threshold=0.1` 或 `0.05`
- **归一化到 [-1, 1]**: `threshold=0.0` 或 `0.1`
- **原始强度值**: 根据实际值设置

**如何确定**:
1. 可视化几个样本的直方图
2. 观察前景像素的典型值范围
3. 将阈值设置在前景和背景之间

### 2. 前景权重（`foreground_weight`）

根据前景占比设置：

| 前景占比 | 推荐权重 | 说明 |
|---------|---------|------|
| ~20%    | 5-10    | 轻度不平衡 |
| ~10%    | 10-20   | 中度不平衡 |
| ~5%     | 20-30   | 严重不平衡 ⭐ |
| ~1%     | 50-100  | 极度不平衡 |

**调节策略**:
1. 从较小的权重开始（如10.0）
2. 如果模型仍预测全黑，逐步增加权重
3. 观察TensorBoard中的重建可视化
4. 权重过大可能导致前景过拟合，需平衡

### 3. Dice和重建损失的平衡

在使用组合损失时：

```yaml
combined:
  dice_weight: 1.0      # Dice Loss权重
  recon_weight: 1.0     # 重建损失权重
```

**调节建议**:
- 如果整体形状不好，增加 `dice_weight`
- 如果细节模糊，增加 `recon_weight`
- 监控TensorBoard中两个损失的数值范围，确保它们在相近的量级

---

## 训练监控

### 在TensorBoard中观察

```bash
tensorboard --logdir outputs/logs/autoencoder
```

**关键指标**:
1. **重建可视化**: `patch_reconstruction` 或 `patch_denoising`
   - 对比输入和重建，看是否还是全黑
   
2. **误差图**: `*_error`
   - 前景区域的误差是否在降低
   
3. **损失曲线**: `train/step/recon_loss`
   - 是否在下降
   - 是否稳定

### 判断效果

**成功的标志**:
- ✅ 重建图像中能看到微管结构
- ✅ 误差图显示前景区域误差较小
- ✅ 重建损失持续下降

**失败的标志**:
- ❌ 重建图像仍然全黑
- ❌ 误差图显示前景区域误差很大
- ❌ 损失不下降或下降很慢

**解决方案**:
1. 增加 `foreground_weight`（如从20增加到50）
2. 切换到 `combined` 损失类型
3. 检查数据归一化是否正确
4. 降低 `kl_weight`，先让重建做好

---

## 常见问题

### Q1: 模型预测全黑怎么办？

**A**: 
1. 使用 `weighted` 或 `combined` 损失
2. 增加 `foreground_weight` 到 30-50
3. 降低 `kl_weight` 到 1e-7 或更小
4. 确保数据增强不会让前景消失

### Q2: 前景出现了但很模糊？

**A**:
1. 使用 `combined` 损失
2. 增加 `recon_weight` 相对于 `dice_weight`
3. 确保 `foreground_weight` 足够大

### Q3: 前景过亮或过饱和？

**A**:
1. 降低 `foreground_weight`
2. 确保数据归一化正确
3. 检查激活函数是否合适

### Q4: 损失数值很大或很小？

**A**:
1. 调节 `dice_weight` 和 `recon_weight` 使两个损失在相近量级
2. Dice Loss通常在 0-1 范围
3. L1 Loss的数值取决于数据范围

---

## 完整配置示例

针对微管数据（前景约5%）的推荐配置：

```yaml
# monai_diffusion/config/ldm_config_local.yaml
autoencoder:
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  num_channels: [32, 64, 128]
  latent_channels: 16
  
  training:
    n_epochs: 500
    batch_size: 1
    learning_rate: 1.0e-4
    
    # ⭐ 核心损失配置
    loss:
      reconstruction_loss_type: "combined"
      combined:
        dice_weight: 1.0
        recon_weight: 1.0
        recon_loss_type: "l1"
        foreground_weight: 20.0     # ⭐ 关键参数
        background_weight: 1.0
        threshold: 0.1              # 根据归一化调整
        dice_smooth: 1.0e-5
    
    # 其他损失权重
    adv_weight: 0.01
    perceptual_weight: 0.001
    kl_weight: 1.0e-6               # 较小的KL权重
    
    # 优化配置
    use_gradient_checkpointing: true
    autoencoder_warm_up_n_epochs: 50  # 前50个epoch不用对抗损失
    
    # 去噪配置（可选，进一步增强鲁棒性）
    denoising:
      enabled: true
      noise_type: "mixed"
      noise_std: 0.03
      dropout_prob: 0.1
    
    val_interval: 5
    save_interval: 10
```

---

## 实验建议

### 阶段1: 基础测试（快速验证）
```yaml
training:
  fast_dev_run: true
  fast_dev_run_batches: 10
  n_epochs: 10
```

测试不同损失函数：
1. `l1` (基线)
2. `weighted` (foreground_weight=20)
3. `combined` (foreground_weight=20)

### 阶段2: 参数搜索
选择效果最好的损失类型，调节 `foreground_weight`:
- 10.0, 20.0, 30.0, 50.0

### 阶段3: 完整训练
使用最佳参数进行完整训练（500 epochs）

---

## 总结

**最佳实践**:

1. **首选**: 使用 `combined` 损失，`foreground_weight=20-30`
2. **备选**: 使用 `weighted` 损失，`foreground_weight=30-50`
3. **监控**: 使用TensorBoard密切观察重建可视化
4. **调节**: 根据前景占比动态调整权重
5. **平衡**: 保持Dice和重建损失在相近量级

**记住**: 权重没有绝对的"正确"值，需要根据具体数据集进行实验调节！

