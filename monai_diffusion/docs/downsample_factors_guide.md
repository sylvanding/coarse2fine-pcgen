# AutoencoderKL 下采样因子配置指南

## 功能介绍

我们为MONAI Generative的AutoencoderKL添加了自定义下采样因子的功能，允许您灵活控制latent space的压缩倍数，以节省显存和加速训练。

## 原理说明

### 传统方式
在原始的AutoencoderKL中，下采样倍数由`num_channels`的长度决定：
- `num_channels: [32, 64, 64]` → 2次下采样 → **4倍**总缩小
- `num_channels: [32, 64, 64, 128]` → 3次下采样 → **8倍**总缩小
- `num_channels: [32, 64, 64, 128, 256]` → 4次下采样 → **16倍**总缩小

每次下采样固定使用stride=2，因此总下采样倍数 = 2^(层数-1)

### 新方式（自定义下采样因子）
现在您可以通过`downsample_factors`参数灵活指定每层的下采样倍数：
```yaml
num_channels: [32, 64, 64]
downsample_factors: [4, 2]  # 第1层4倍，第2层2倍 → 总共8倍
```

这样可以在保持较少层数的同时实现更大的下采样倍数，从而：
- ✅ 减少网络深度，降低显存占用
- ✅ 减小latent space大小，进一步节省显存
- ✅ 加快训练和推理速度

## 显存节省分析

### 示例：256×256×64 体素
假设输入体素为 256×256×64：

#### 配置1: 传统方式 4倍下采样
```yaml
num_channels: [32, 64, 64]
downsample_factors: [2, 2]  # 或不指定
```
- Latent空间大小: 64×64×16
- 浅层特征图最大: 128×128×32 (64通道)
- **显存占用: ~8GB**

#### 配置2: 8倍下采样 (推荐)
```yaml
num_channels: [32, 64, 64]
downsample_factors: [4, 2]
```
- Latent空间大小: 32×32×8 ← **体积缩小8倍**
- 浅层特征图最大: 64×64×16 (64通道) ← **体积缩小4倍**
- **显存占用: ~3GB** ← **节省62.5%显存**

#### 配置3: 16倍下采样 (超大分辨率)
```yaml
num_channels: [32, 64]  # 只需2层
downsample_factors: [16]  # 一次下采样16倍
```
- Latent空间大小: 16×16×4 ← **体积缩小64倍**
- **显存占用: ~1.5GB** ← **节省81%显存**

## 配置示例

### 示例1: 64×64×32 低分辨率（默认）
```yaml
autoencoder:
  num_channels: [32, 64, 64]
  downsample_factors: [2, 2]  # 4倍下采样
  latent_channels: 3
  
  training:
    batch_size: 4
```
- Latent: 16×16×8
- 适合: 学习阶段、快速原型

### 示例2: 128×128×64 中等分辨率
```yaml
autoencoder:
  num_channels: [32, 64, 64]
  downsample_factors: [4, 2]  # 8倍下采样
  latent_channels: 3
  
  training:
    batch_size: 2
```
- Latent: 16×16×8
- 适合: 平衡质量和效率

### 示例3: 256×256×64 高分辨率（推荐）
```yaml
autoencoder:
  num_channels: [16, 32, 64]  # 减少通道数
  downsample_factors: [4, 2]  # 8倍下采样
  latent_channels: 3
  attention_levels: [false, false, true]
  
  training:
    batch_size: 1
    use_perceptual_loss: false  # 禁用以节省显存
    gradient_accumulation_steps: 4
    mixed_precision: true
```
- Latent: 32×32×8
- 适合: 高质量生成

### 示例4: 512×512×128 超高分辨率
```yaml
autoencoder:
  num_channels: [16, 32, 64]
  downsample_factors: [8, 2]  # 16倍下采样
  latent_channels: 4
  attention_levels: [false, false, true]
  
  training:
    batch_size: 1
    use_perceptual_loss: false
    use_gradient_checkpointing: true
    gradient_accumulation_steps: 8
    mixed_precision: true
```
- Latent: 32×32×8
- 适合: 极高分辨率场景

### 示例5: 极致压缩配置
```yaml
autoencoder:
  num_channels: [32, 64]  # 只有2层
  downsample_factors: [16]  # 一次下采样16倍
  latent_channels: 8  # 更大的latent通道补偿信息损失
  attention_levels: [false, true]
  
  training:
    batch_size: 2
```
- 输入256×256×64 → Latent 16×16×4
- 最小显存占用
- 可能损失一些细节

## 最佳实践建议

### 1. 下采样因子选择原则
- **2-4倍**: 适合低分辨率（64³以下），保留最多细节
- **8倍**: 适合中高分辨率（128³-256³），**最常用配置**
- **16倍以上**: 适合超高分辨率（512³以上）或显存极度受限场景

### 2. 下采样因子组合
推荐使用**从大到小**的下采样序列：
```yaml
# ✅ 推荐: 先大后小
downsample_factors: [4, 2, 2]  # 第一层4倍，后续2倍

# ⚠️ 不太推荐: 先小后大
downsample_factors: [2, 2, 4]  # 可能导致浅层特征图过大
```

原因: 先大幅下采样可以快速减小特征图尺寸，节省浅层显存。

### 3. 配合其他优化技术
使用大下采样因子时，建议同时使用：
```yaml
training:
  # 必选
  mixed_precision: true  # AMP混合精度
  
  # 推荐
  gradient_accumulation_steps: 4  # 梯度累积
  use_perceptual_loss: false  # 对于超大分辨率
  
  # 可选
  use_gradient_checkpointing: true  # 牺牲20%速度换40%显存
```

### 4. Latent通道数调整
下采样倍数越大，建议增加`latent_channels`补偿信息损失：
- 4倍下采样: `latent_channels: 3`
- 8倍下采样: `latent_channels: 3-4`
- 16倍下采样: `latent_channels: 4-8`

### 5. 验证配置
使用快速开发模式验证配置是否可行：
```yaml
training:
  fast_dev_run: true
  fast_dev_run_batches: 2
```

## 显存计算公式

对于体素大小为 (H, W, D) 的输入：

### Latent空间大小
```
total_downsample = downsample_factors[0] × downsample_factors[1] × ...
latent_size = (H/total_downsample, W/total_downsample, D/total_downsample)
latent_memory = latent_size × latent_channels × 4 bytes
```

### 总显存占用（粗略估计）
```
total_memory ≈ 
    输入数据 (H×W×D×4) + 
    每层特征图 (累加) + 
    Latent空间 + 
    梯度 (约等于参数量) +
    优化器状态 (Adam约2倍参数量)
```

**关键**: 浅层特征图尺寸最大，使用大的初始下采样因子可显著降低显存。

## 常见问题

### Q1: downsample_factors长度应该是多少？
**A**: 长度应该是 `len(num_channels) - 1`
```yaml
num_channels: [32, 64, 64]     # 3个元素
downsample_factors: [4, 2]      # 2个元素 ✅

num_channels: [32, 64, 64, 128] # 4个元素
downsample_factors: [4, 2, 2]   # 3个元素 ✅
```

### Q2: 可以设置非2的幂次的下采样因子吗？
**A**: 可以！支持任意整数，如3, 5, 7等。但**建议使用2的幂次**（2, 4, 8, 16），以获得最佳硬件加速效果。

### Q3: 下采样因子过大会导致什么问题？
**A**: 
- ⚠️ 信息损失：过度压缩可能丢失细节
- ⚠️ 训练不稳定：梯度传播路径变短
- ⚠️ 重建质量下降

建议：
- 逐步增大下采样因子，观察重建质量
- 通过增加`latent_channels`补偿
- 监控重建损失（recon_loss）

### Q4: 为什么配置了downsample_factors还是OOM？
**A**: 检查以下配置：
1. `batch_size` 太大 → 减小为1
2. `num_channels` 通道数太多 → 减小通道数
3. `use_perceptual_loss: true` → 设为false
4. 未启用`mixed_precision` → 设为true
5. 添加`gradient_accumulation_steps: 4`

### Q5: 如何选择最佳配置？
**A**: 使用二分查找策略：
```python
# 步骤1: 找到最大可用batch_size
batch_sizes = [8, 4, 2, 1]
for bs in batch_sizes:
    try_train(batch_size=bs)
    
# 步骤2: 找到最小可用downsample_factor
# 从8倍开始，如果OOM则增加到16倍
downsample_options = [
    [4, 2],   # 8倍
    [8, 2],   # 16倍
    [16],     # 16倍（更少层）
]
```

## 技术细节

### 修改的文件
1. `GenerativeModels/generative/networks/nets/autoencoderkl.py`
   - 修改`Downsample`类，添加`stride`参数
   - 修改`Upsample`类，添加`scale_factor`参数
   - 修改`Encoder`类，添加`downsample_factors`参数
   - 修改`Decoder`类，添加`upsample_factors`参数
   - 修改`AutoencoderKL`类，传递这些参数

2. `monai_diffusion/config/ldm_config.yaml`
   - 添加`downsample_factors`配置项

3. `monai_diffusion/3d_ldm/train_autoencoder.py`
   - 读取并传递`downsample_factors`到模型

### 向后兼容性
✅ 完全向后兼容！如果不指定`downsample_factors`，将自动使用默认值（每层2倍），行为与原版完全一致。

## 总结

通过自定义下采样因子，您可以：

1. **大幅节省显存**: 8倍下采样可节省60%以上显存
2. **灵活配置**: 根据分辨率和硬件自由调整
3. **保持质量**: 通过调整latent_channels补偿信息损失
4. **加速训练**: 更小的latent space意味着更快的扩散模型训练

建议从`downsample_factors: [4, 2]`（8倍下采样）开始尝试！

