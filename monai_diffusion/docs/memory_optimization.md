# 显存优化指南

当使用高分辨率体素（如 256×256×64）训练 AutoencoderKL 时，显存占用会显著增加。本文档提供了多种优化策略。

## 显存占用分析

### 为什么 256×256×64 比 64×64×32 占用显存多得多？

1. **数据体积增长**：
   - 64×64×32 = 131,072 个体素
   - 256×256×64 = 4,194,304 个体素
   - **增长了 32 倍**！

2. **中间激活值**：
   - 3D 卷积在每一层都会保存激活值用于反向传播
   - 激活值的总量与输入尺寸和通道数成正比
   - 假设有 3 层，通道数为 [32, 64, 64]：
     - 第一层：4,194,304 × 32 = 134M 个float
     - 第二层：约 67M 个float（下采样后）
     - 第三层：约 33M 个float
   - 仅激活值就占用约 **900+ MB**（FP32）

3. **注意力机制**：
   - Self-attention 的复杂度是 O(n²)，其中 n 是空间位置数
   - 在 256×256×64 的最后一层，即使下采样到 32×32×8，仍有 8,192 个位置
   - Attention map 需要 8,192² ≈ **67M** 的存储

4. **判别器**：
   - PatchDiscriminator 也需要处理 256×256×64 的输入
   - 额外占用约 **500+ MB**

5. **PerceptualLoss**：
   - 使用 SqueezeNet 计算感知损失
   - 需要额外的前向传播，占用约 **300+ MB**

6. **梯度**：
   - 每个参数都需要存储梯度
   - 梯度占用与参数量相同的显存

**总计**：不优化的情况下，256×256×64 可能需要 **15-20GB** 显存！

---

## 优化策略

### 1. 混合精度训练（AMP）★★★★★

**效果**：节省约 **50%** 显存

**原理**：使用 FP16 进行前向和反向传播，FP32 只用于参数更新

**配置**：
```yaml
device:
  mixed_precision: true  # 必须启用！
```

**显存节省**：
- 激活值：900MB → 450MB
- 梯度：减半
- **总节省**：约 5-8GB

---

### 2. 禁用感知损失（PerceptualLoss）★★★★

**效果**：节省约 **300-500MB** 显存

**原理**：PerceptualLoss 使用额外的网络（SqueezeNet）计算特征，占用大量显存

**配置**：
```yaml
autoencoder:
  training:
    use_perceptual_loss: false  # 禁用感知损失
```

**影响**：
- ✅ 显存大幅减少
- ❌ 重建质量可能略微下降
- 建议：训练初期禁用，后期微调时启用

---

### 3. 梯度累积（Gradient Accumulation）★★★★

**效果**：模拟更大的 batch size，不增加显存

**原理**：多个 batch 累积梯度后再更新参数

**配置**：
```yaml
autoencoder:
  training:
    batch_size: 1
    gradient_accumulation_steps: 4  # 模拟 batch_size=4
```

**优点**：
- 训练稳定性提升
- 不增加显存占用
- 等效更大的 batch size

---

### 4. 梯度检查点（Gradient Checkpointing）★★★

**效果**：节省约 **30-40%** 显存

**原理**：不保存所有中间激活，反向传播时重新计算

**配置**：
```yaml
autoencoder:
  training:
    use_gradient_checkpointing: true
```

**权衡**：
- ✅ 显存大幅减少
- ❌ 训练速度降低约 20-30%
- 建议：显存紧张时启用

**注意**：需要 MONAI Generative 支持此功能。如果模型没有 `enable_gradient_checkpointing` 方法，此选项无效。

---

### 5. 减少通道数（num_channels）★★★★

**效果**：节省 **30-50%** 显存

**原理**：减少卷积层的通道数，直接减少参数和激活值

**配置**：
```yaml
autoencoder:
  num_channels: [16, 32, 64]  # 从 [32, 64, 64] 减少到 [16, 32, 64]
  norm_num_groups: 8  # 相应减少（必须能整除通道数）
```

**影响**：
- ✅ 显存显著减少
- ❌ 模型容量减少，可能影响重建质量
- 建议：先尝试，观察重建效果

---

### 6. 减少判别器复杂度★★★

**效果**：节省约 **200-400MB** 显存

**配置**：
```yaml
autoencoder:
  training:
    discriminator:
      num_layers_d: 2  # 从 3 减少到 2
      num_channels: 16  # 从 32 减少到 16
```

**影响**：
- ✅ 显存减少
- ❌ 判别器能力减弱，可能影响对抗训练效果
- 建议：判别器本身影响较小，可以减少

---

### 7. 减少注意力层★★

**效果**：节省约 **500MB+** 显存

**原理**：注意力机制复杂度高，占用大量显存

**配置**：
```yaml
autoencoder:
  attention_levels: [false, false, false]  # 完全禁用注意力
  # 或
  attention_levels: [false, false, true]  # 只在最后一层使用
```

**影响**：
- ✅ 显存显著减少
- ❌ 可能影响长程依赖建模
- 建议：高分辨率下只在最后一层使用

---

### 8. 减少 batch_size★★★★★

**效果**：线性减少显存

**配置**：
```yaml
autoencoder:
  training:
    batch_size: 1  # 高分辨率下使用 batch_size=1
    gradient_accumulation_steps: 4  # 配合梯度累积
```

**影响**：
- ✅ 显存显著减少
- ⚠️ 需要配合梯度累积以保证训练稳定性

---

### 9. 不缓存数据集★★

**效果**：节省 CPU 内存（不是 GPU 显存）

**配置**：
```yaml
data:
  cache_rate: 0  # 不缓存数据
```

**说明**：
- 高分辨率数据缓存到内存会占用大量 RAM
- 建议使用硬盘 I/O，配合 `num_workers` 并行加载

---

## 推荐配置组合

### 组合 A：最大限度节省显存（适用于 24GB GPU + 256×256×64）

```yaml
autoencoder:
  num_channels: [16, 32, 64]
  norm_num_groups: 8
  attention_levels: [false, false, true]
  
  training:
    batch_size: 1
    use_perceptual_loss: false  # 禁用
    use_gradient_checkpointing: true  # 启用
    gradient_accumulation_steps: 4
    
    discriminator:
      num_layers_d: 2
      num_channels: 16

device:
  mixed_precision: true  # 必须启用
```

**预期显存占用**：约 **10-14GB**

---

### 组合 B：平衡性能与显存（适用于 32GB+ GPU）

```yaml
autoencoder:
  num_channels: [24, 48, 64]
  norm_num_groups: 8
  attention_levels: [false, false, true]
  
  training:
    batch_size: 2
    use_perceptual_loss: true  # 启用
    use_gradient_checkpointing: false  # 不启用
    gradient_accumulation_steps: 2
    
    discriminator:
      num_layers_d: 3
      num_channels: 24

device:
  mixed_precision: true
```

**预期显存占用**：约 **18-22GB**

---

### 组合 C：最高质量（适用于 40GB+ GPU）

```yaml
autoencoder:
  num_channels: [32, 64, 128]
  norm_num_groups: 16
  attention_levels: [false, true, true]
  
  training:
    batch_size: 4
    use_perceptual_loss: true
    use_gradient_checkpointing: false
    gradient_accumulation_steps: 1
    
    discriminator:
      num_layers_d: 3
      num_channels: 32

device:
  mixed_precision: true
```

---

## 实际使用示例

### 场景 1：我有 24GB GPU，想训练 256×256×64

**步骤**：
1. 使用配置文件：`ldm_config_high_res.yaml`
2. 确认配置：
   ```yaml
   voxel_size: [256, 256, 64]
   batch_size: 1
   mixed_precision: true
   use_perceptual_loss: false
   gradient_accumulation_steps: 4
   ```
3. 运行：
   ```bash
   python monai_diffusion/3d_ldm/train_autoencoder.py \
     --config monai_diffusion/config/ldm_config_high_res.yaml
   ```

---

### 场景 2：训练中出现 CUDA OOM 错误

**诊断步骤**：
1. 确认是否启用了混合精度：
   ```yaml
   device:
     mixed_precision: true
   ```

2. 进一步减少 batch_size：
   ```yaml
   training:
     batch_size: 1  # 如果已经是1，无法再减
   ```

3. 禁用感知损失：
   ```yaml
   training:
     use_perceptual_loss: false
   ```

4. 减少通道数：
   ```yaml
   num_channels: [12, 24, 48]  # 进一步减少
   ```

5. 启用梯度检查点：
   ```yaml
   training:
     use_gradient_checkpointing: true
   ```

6. 如果仍然 OOM，考虑降低分辨率：
   ```yaml
   voxel_size: [128, 128, 64]  # 从 256 降到 128
   ```

---

## 性能对比

| 配置 | 分辨率 | batch_size | 通道数 | AMP | Perceptual | 显存占用 | 相对速度 |
|------|--------|------------|--------|-----|------------|----------|----------|
| 默认 | 64³ | 4 | [32,64,64] | ❌ | ✅ | ~8GB | 1.0x |
| 默认+AMP | 64³ | 4 | [32,64,64] | ✅ | ✅ | ~5GB | 1.1x |
| 高分辨率-激进 | 256×256×64 | 1 | [16,32,64] | ✅ | ❌ | ~12GB | 0.3x |
| 高分辨率-平衡 | 256×256×64 | 1 | [24,48,64] | ✅ | ✅ | ~18GB | 0.25x |
| 高分辨率+检查点 | 256×256×64 | 1 | [16,32,64] | ✅ | ❌ | ~9GB | 0.2x |

---

## 监控显存使用

### 实时监控
```bash
watch -n 0.5 nvidia-smi
```

### Python 代码监控
```python
import torch

# 在训练循环中添加
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
```

---

## 故障排查

### Q: 为什么启用 AMP 后训练不稳定？

**A**: 某些操作在 FP16 下数值不稳定。尝试：
1. 使用梯度缩放（代码已自动启用）
2. 检查 loss 是否出现 NaN
3. 如果持续不稳定，在 autocast 外计算某些损失

### Q: 梯度检查点没有生效？

**A**: 检查 MONAI Generative 版本：
```python
import generative
print(generative.__version__)  # 需要较新版本
```

如果不支持，可以手动实现：
```python
from torch.utils.checkpoint import checkpoint

# 在模型中包装需要检查点的层
```

### Q: 即使用了所有优化，仍然 OOM？

**A**: 考虑：
1. 降低分辨率（256→128 或 128→64）
2. 使用多 GPU 训练（DDP）
3. 使用 CPU offloading（需要自定义实现）
4. 分块处理（将大体素分成多个小块）

---

## 总结

| 优化方法 | 显存节省 | 速度影响 | 质量影响 | 推荐度 |
|----------|----------|----------|----------|--------|
| 混合精度训练 | ★★★★★ | +10% | 无 | ★★★★★ |
| 禁用感知损失 | ★★★★ | +15% | -轻微 | ★★★★ |
| 梯度累积 | ★★★★★ | 无 | 无 | ★★★★★ |
| 梯度检查点 | ★★★★ | -20% | 无 | ★★★ |
| 减少通道数 | ★★★★ | +5% | -中等 | ★★★★ |
| 减少判别器 | ★★★ | +5% | -轻微 | ★★★★ |
| 减少注意力 | ★★★ | +10% | -轻微 | ★★★ |
| batch_size=1 | ★★★★★ | 无 | 无 | ★★★★★ |

**关键建议**：
1. **必须启用**：混合精度训练（AMP）
2. **强烈推荐**：batch_size=1 + 梯度累积
3. **高分辨率建议**：禁用感知损失 + 减少通道数
4. **显存极度紧张**：启用梯度检查点

---

## 参考资料

- [MONAI Generative Models 文档](https://docs.monai.io/projects/generative/)
- [PyTorch AMP 教程](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing 原理](https://github.com/cybertronai/gradient-checkpointing)

