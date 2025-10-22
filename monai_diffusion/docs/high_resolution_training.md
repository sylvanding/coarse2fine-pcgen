# 高分辨率体素训练指南

本文档介绍如何在有限的 GPU 显存（如 24GB）下训练高分辨率体素（如 256×256×64）的 AutoencoderKL。

## 快速开始

### 1. 使用预配置的高分辨率配置文件

我们提供了专门针对高分辨率优化的配置文件：

```bash
python monai_diffusion/3d_ldm/train_autoencoder.py \
  --config monai_diffusion/config/ldm_config_high_res.yaml
```

这个配置文件已经包含了所有推荐的优化设置。

### 2. 在开始训练前测试显存占用

使用我们提供的显存测试工具：

```bash
# 测试默认配置 (256x256x64, batch_size=1, channels=[32,64,64])
python monai_diffusion/tools/test_memory_usage.py

# 测试自定义配置
python monai_diffusion/tools/test_memory_usage.py \
  --resolution 256,256,64 \
  --batch-size 1 \
  --channels 16,32,64
```

这会测试不同优化策略下的显存占用，帮助你选择合适的配置。

## 关键优化

### 必须启用的优化（否则肯定 OOM）

#### 1. 混合精度训练（AMP）

**显存节省：约 50%**

在 `ldm_config.yaml` 中确保：

```yaml
device:
  mixed_precision: true  # 必须为 true！
```

代码会自动使用 `torch.cuda.amp.autocast` 和 `GradScaler`。

#### 2. Batch Size = 1

**显存节省：线性减少**

```yaml
autoencoder:
  training:
    batch_size: 1  # 高分辨率下必须为1
```

#### 3. 禁用感知损失（或后期启用）

**显存节省：约 300-500MB**

```yaml
autoencoder:
  training:
    use_perceptual_loss: false  # 初期禁用
```

感知损失需要额外的 SqueezeNet，占用大量显存。建议：
- 训练前期（0-300 epochs）：禁用
- 训练后期微调：启用（如果显存允许）

### 推荐的优化

#### 4. 梯度累积

**效果：模拟更大的 batch size，不增加显存**

```yaml
autoencoder:
  training:
    batch_size: 1
    gradient_accumulation_steps: 4  # 等效 batch_size=4
```

代码会在 N 个 batch 后才更新参数，提高训练稳定性。

#### 5. 减少通道数

**显存节省：30-50%**

```yaml
autoencoder:
  num_channels: [16, 32, 64]  # 从 [32, 64, 64] 减少
  norm_num_groups: 8  # 相应调整（必须能整除通道数）
```

对重建质量影响较小，但显存节省明显。

#### 6. 简化判别器

**显存节省：约 200-400MB**

```yaml
autoencoder:
  training:
    discriminator:
      num_layers_d: 2  # 从 3 减少到 2
      num_channels: 16  # 从 32 减少到 16
```

### 可选的优化

#### 7. 梯度检查点

**显存节省：30-40%，但训练速度降低 20-30%**

```yaml
autoencoder:
  training:
    use_gradient_checkpointing: true
```

⚠️ **注意**：需要 MONAI Generative 版本支持 `enable_gradient_checkpointing` 方法。

## 配置示例

### 配置 A：24GB GPU + 256×256×64（推荐）

文件：`monai_diffusion/config/ldm_config_high_res.yaml`

```yaml
data:
  voxel_size: [256, 256, 64]
  cache_rate: 0

autoencoder:
  num_channels: [16, 32, 64]
  norm_num_groups: 8
  attention_levels: [false, false, true]
  
  training:
    batch_size: 1
    use_perceptual_loss: false
    use_gradient_checkpointing: true
    gradient_accumulation_steps: 4
    
    discriminator:
      num_layers_d: 2
      num_channels: 16

device:
  mixed_precision: true
```

**预期显存占用**：约 10-14GB

### 配置 B：32GB GPU + 256×256×64（平衡）

```yaml
autoencoder:
  num_channels: [24, 48, 64]
  
  training:
    batch_size: 2
    use_perceptual_loss: true
    use_gradient_checkpointing: false
    gradient_accumulation_steps: 2

device:
  mixed_precision: true
```

**预期显存占用**：约 18-22GB

### 配置 C：40GB+ GPU + 256×256×64（最高质量）

```yaml
autoencoder:
  num_channels: [32, 64, 128]
  
  training:
    batch_size: 4
    use_perceptual_loss: true
    use_gradient_checkpointing: false
    gradient_accumulation_steps: 1

device:
  mixed_precision: true
```

## 代码修改说明

### 主要修改点

1. **添加混合精度支持**：
   - 导入 `torch.cuda.amp.autocast` 和 `GradScaler`
   - 在训练循环中使用 `autocast` 上下文
   - 使用 `scaler.scale()` 和 `scaler.step()`

2. **添加梯度累积**：
   - 损失除以累积步数：`loss / gradient_accumulation_steps`
   - 只在累积完成后更新参数

3. **可选的感知损失**：
   - 根据 `use_perceptual_loss` 配置决定是否创建 PerceptualLoss
   - 节省显存的关键优化

4. **梯度检查点支持**：
   - 检查模型是否有 `enable_gradient_checkpointing` 方法
   - 如果有则调用启用

### 训练循环核心逻辑

```python
# 梯度清零（在累积开始时）
if step % gradient_accumulation_steps == 0:
    optimizer_g.zero_grad(set_to_none=True)

# 混合精度前向传播
with autocast(enabled=use_amp):
    reconstruction, z_mu, z_sigma = autoencoder(images)
    recons_loss = l1_loss(reconstruction, images)
    
    # 可选的感知损失
    if use_perceptual_loss:
        p_loss = loss_perceptual(reconstruction, images)
        loss_g = recons_loss + perceptual_weight * p_loss
    else:
        loss_g = recons_loss
    
    # 梯度累积：除以累积步数
    loss_g = loss_g / gradient_accumulation_steps

# 反向传播
if use_amp:
    scaler_g.scale(loss_g).backward()
    # 在累积结束时更新参数
    if (step + 1) % gradient_accumulation_steps == 0:
        scaler_g.step(optimizer_g)
        scaler_g.update()
else:
    loss_g.backward()
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer_g.step()
```

## 故障排查

### Q1: 训练时出现 CUDA OOM 错误

**解决步骤**：

1. **确认混合精度已启用**：
   ```bash
   # 查看训练日志，应该看到：
   # "启用混合精度训练（AMP）"
   ```

2. **减少 batch_size 到 1**：
   ```yaml
   batch_size: 1
   ```

3. **禁用感知损失**：
   ```yaml
   use_perceptual_loss: false
   ```

4. **进一步减少通道数**：
   ```yaml
   num_channels: [12, 24, 48]
   ```

5. **启用梯度检查点**：
   ```yaml
   use_gradient_checkpointing: true
   ```

6. **最后手段：降低分辨率**：
   ```yaml
   voxel_size: [128, 128, 64]
   ```

### Q2: 训练速度太慢

**可能原因和解决方法**：

1. **梯度检查点导致**：
   - 如果启用了 `use_gradient_checkpointing`，会降低 20-30% 速度
   - 如果显存充足，可以禁用

2. **I/O 瓶颈**：
   ```yaml
   data:
     num_workers: 4  # 增加数据加载线程
     pin_memory: true
   ```

3. **感知损失计算**：
   - PerceptualLoss 需要额外的前向传播
   - 如果不需要高质量重建，可以禁用

### Q3: 训练不稳定，loss 出现 NaN

**解决方法**：

1. **检查混合精度设置**：
   - 代码已经使用 GradScaler，应该不会有问题
   - 如果仍然出现，尝试降低学习率

2. **降低学习率**：
   ```yaml
   learning_rate: 5.0e-5  # 从 1e-4 降低
   ```

3. **检查 KL 权重**：
   ```yaml
   kl_weight: 1.0e-7  # 从 1e-6 降低
   ```

### Q4: 重建质量不佳

**可能原因**：

1. **通道数太少**：
   - 如果使用了 `[16, 32, 64]`，可以增加到 `[24, 48, 64]`
   - 需要更多显存

2. **感知损失被禁用**：
   - 如果显存允许，启用感知损失：
   ```yaml
   use_perceptual_loss: true
   ```

3. **训练时间不够**：
   - 高分辨率需要更长的训练时间
   - 建议至少 300-500 epochs

## 监控训练

### 使用 TensorBoard

```bash
tensorboard --logdir outputs/logs/autoencoder_high_res
```

**关键指标**：
- `train/epoch/recon_loss`：重建损失，应该持续下降
- `val/epoch/recon_loss`：验证损失，监控过拟合
- `reconstruction/sample_*`：重建结果可视化

### 监控 GPU 使用

```bash
# 实时监控
watch -n 0.5 nvidia-smi

# 或在另一个终端运行
nvidia-smi dmon -s u
```

### 预期训练时间

| 分辨率 | Batch Size | GPU | 时间/Epoch |
|--------|------------|-----|------------|
| 64×64×32 | 4 | RTX 3090 | ~2分钟 |
| 128×128×64 | 2 | RTX 3090 | ~8分钟 |
| 256×256×64 | 1 | RTX 3090 | ~20分钟 |
| 256×256×64 | 1+检查点 | RTX 3090 | ~30分钟 |

## 最佳实践

### 训练策略

1. **分阶段训练**：
   ```
   阶段1 (0-100 epochs):
     - 低分辨率 (128×128×64)
     - 禁用感知损失
     - 快速收敛
   
   阶段2 (100-300 epochs):
     - 高分辨率 (256×256×64)
     - 禁用感知损失
     - 学习主要结构
   
   阶段3 (300-500 epochs):
     - 高分辨率 (256×256×64)
     - 启用感知损失（如果显存允许）
     - 微调细节
   ```

2. **渐进式增加复杂度**：
   - 先用简单配置训练到收敛
   - 然后逐步增加模型复杂度
   - 从已有 checkpoint 继续训练

3. **定期验证**：
   ```yaml
   val_interval: 10  # 每10个epoch验证一次
   visualize_interval: 10  # 可视化重建结果
   ```

### 配置调优优先级

1. **必须启用**：
   - ✅ mixed_precision: true
   - ✅ batch_size: 1
   - ✅ gradient_accumulation_steps: 4

2. **优先调整**：
   - num_channels (影响最大)
   - use_perceptual_loss
   - discriminator 复杂度

3. **最后考虑**：
   - use_gradient_checkpointing
   - 降低分辨率

## 参考资料

- [显存优化详细指南](memory_optimization.md)
- [配置文件：高分辨率](../config/ldm_config_high_res.yaml)
- [配置文件：标准分辨率](../config/ldm_config.yaml)
- [显存测试工具](../tools/test_memory_usage.py)

## 总结

训练高分辨率体素 AutoencoderKL 的关键：

1. **必须启用混合精度**（节省 50% 显存）
2. **batch_size=1 + 梯度累积**（模拟大 batch）
3. **禁用感知损失**（节省 300-500MB）
4. **减少通道数**（[16,32,64] 而不是 [32,64,64]）
5. **简化判别器**（影响小，显存节省明显）

遵循这些原则，在 24GB GPU 上训练 256×256×64 是完全可行的！

