# VQ-LDM 快速开始指南

这是一个5分钟快速开始指南，帮助你快速上手VQ-Latent Diffusion Model。

## 📋 前置要求

```bash
# 1. 确保已安装依赖
pip install torch torchvision monai nibabel pyyaml tensorboard pillow

# 2. 确保GenerativeModels已克隆到项目根目录
cd /repos/coarse2fine-pcgen
git clone https://github.com/Project-MONAI/GenerativeModels.git

# 3. 确保数据已准备好
# 数据应该在: data/microtubules/nifti/train 和 data/microtubules/nifti/val
```

## 🚀 三步完成训练

### Step 1: 检查配置

编辑配置文件：`monai_diffusion/config/vq_ldm_config_local.yaml`

```yaml
# 最关键的参数
data:
  patch_size: [64, 64, 64]  # 根据显存调整
  batch_size: 4

vqvae:
  num_embeddings: 256  # Codebook大小
  embedding_dim: 32

diffusion:
  in_channels: 32  # 必须等于vqvae.embedding_dim
```

### Step 2: 训练VQVAE（阶段1）

```bash
# 训练VQVAE
python monai_diffusion/3d_vq_ldm/train_vqvae.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml

# 监控训练
tensorboard --logdir outputs/vq_ldm/vqvae_logs
```

**预期时间**：根据数据量，通常需要1-3小时

**如何判断训练好了**：
- 重建损失（recon_loss）稳定下降
- TensorBoard中的重建图像清晰可见
- 验证损失不再显著下降

### Step 3: 训练Diffusion（阶段2）

```bash
# 训练Diffusion Model
python monai_diffusion/3d_vq_ldm/train_diffusion.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml

# 监控训练
tensorboard --logdir outputs/vq_ldm/diffusion_logs
```

**预期时间**：根据数据量，通常需要3-6小时

**如何判断训练好了**：
- 损失（loss）稳定在较低水平
- TensorBoard中的生成样本质量越来越好
- 至少训练300+ epochs

## 🎯 完整训练命令

```bash
# 一键执行完整训练流程（如果你的设备可以长时间运行）
cd /repos/coarse2fine-pcgen

# 阶段1: VQVAE
echo "开始训练VQVAE..."
python monai_diffusion/3d_vq_ldm/train_vqvae.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml

# 阶段2: Diffusion
echo "开始训练Diffusion..."
python monai_diffusion/3d_vq_ldm/train_diffusion.py \
    --config monai_diffusion/config/vq_ldm_config_local.yaml

echo "训练完成！"
```

## 🧪 快速测试（仅验证代码正确性）

如果你只想验证代码是否能运行，不想等待完整训练：

```bash
# 快速测试（只运行几个batch）
python monai_diffusion/3d_vq_ldm/quick_test.py --stage both

# 或分别测试
python monai_diffusion/3d_vq_ldm/quick_test.py --stage vqvae
python monai_diffusion/3d_vq_ldm/quick_test.py --stage diffusion
```

这个快速测试会：
- 自动启用fast_dev_run模式
- 每个epoch只运行2个batch
- 只训练2个epoch
- 验证代码能否正常运行

## 📊 查看训练结果

### TensorBoard可视化

```bash
# VQVAE训练曲线和重建图像
tensorboard --logdir outputs/vq_ldm/vqvae_logs --port 6006

# Diffusion训练曲线和生成样本
tensorboard --logdir outputs/vq_ldm/diffusion_logs --port 6007
```

打开浏览器访问：
- VQVAE: http://localhost:6006
- Diffusion: http://localhost:6007

### 检查Checkpoint

```bash
# VQVAE checkpoints
ls -lh outputs/vq_ldm/vqvae_checkpoints/
# 应该看到: best_model.pt, latest_checkpoint.pt

# Diffusion checkpoints
ls -lh outputs/vq_ldm/diffusion_checkpoints/
# 应该看到: best_model.pt, latest_checkpoint.pt
```

## ⚡ 显存优化（如果遇到OOM）

### 方法1: 减小Patch大小

```yaml
data:
  patch_size: [32, 32, 32]  # 从64降到32
  batch_size: 2             # 同时减小batch size
```

### 方法2: 减小模型大小

```yaml
vqvae:
  num_channels: [32, 64, 128]  # 从[64, 128, 256]减小
  num_embeddings: 128          # 从256减小

diffusion:
  num_channels: [32, 64, 128, 256]  # 减小通道数
```

### 方法3: 启用混合精度（必须）

```yaml
device:
  mixed_precision: true  # 确保启用
```

### 方法4: 预先缩小体素

```yaml
data:
  voxel_resize: [128, 128, 128]  # 在裁剪patch前先resize
```

## 🐛 常见问题快速解决

### Q1: 找不到数据

```bash
# 检查数据路径
ls data/microtubules/nifti/train/*.nii.gz
ls data/microtubules/nifti/val/*.nii.gz

# 如果没有数据，需要先准备数据
# 参考项目主README的数据准备部分
```

### Q2: ImportError: No module named 'generative'

```bash
# 确保GenerativeModels在正确位置
cd /repos/coarse2fine-pcgen
ls GenerativeModels/generative/

# 如果不存在，克隆仓库
git clone https://github.com/Project-MONAI/GenerativeModels.git
```

### Q3: CUDA out of memory

参考上面的"显存优化"部分，依次尝试：
1. 减小patch_size
2. 减小batch_size
3. 减小模型通道数
4. 启用mixed_precision

### Q4: VQVAE重建效果差

可能原因和解决方案：
```yaml
# 1. 增加训练epoch
vqvae:
  training:
    n_epochs: 200  # 从100增加到200

# 2. 增加模型容量
vqvae:
  num_embeddings: 512  # 从256增加
  num_channels: [64, 128, 256, 512]  # 添加更多层

# 3. 降低学习率
vqvae:
  training:
    learning_rate: 5e-5  # 从1e-4降低
```

## 📈 训练进度检查清单

### VQVAE训练

- [ ] recon_loss 从初始值（约0.3-0.5）降到 < 0.1
- [ ] quant_loss 稳定在 0.01-0.1 之间
- [ ] TensorBoard重建图像清晰可辨
- [ ] 验证损失不再显著下降
- [ ] 至少训练50+ epochs

### Diffusion训练

- [ ] loss 从初始值（约1.0）降到 < 0.1
- [ ] TensorBoard生成样本逐渐变清晰
- [ ] 生成样本与真实样本相似
- [ ] 验证损失不再显著下降
- [ ] 至少训练300+ epochs

## 🎓 学习路径

如果你是新手，建议按以下顺序学习：

1. **理解VQVAE** (30分钟)
   - 阅读: `README.md` 中的VQVAE结构部分
   - 参考: `GenerativeModels/tutorials/generative/3d_vqvae/3d_vqvae_tutorial.py`

2. **运行快速测试** (10分钟)
   ```bash
   python monai_diffusion/3d_vq_ldm/quick_test.py --stage both
   ```

3. **训练小规模VQVAE** (1-2小时)
   - 使用patch_size=[32, 32, 32]
   - 训练20-30 epochs
   - 观察TensorBoard结果

4. **理解Diffusion** (30分钟)
   - 阅读: `COMPARISON.md` 中的Diffusion部分
   - 理解噪声预测过程

5. **训练小规模Diffusion** (2-3小时)
   - 基于上一步的VQVAE
   - 训练50-100 epochs
   - 观察生成效果

6. **完整训练** (8-12小时)
   - 使用完整配置
   - VQVAE: 100 epochs
   - Diffusion: 500 epochs

## 📚 进阶学习

完成基础训练后，可以探索：

1. **超参数调优**
   - 实验不同的codebook大小
   - 尝试不同的学习率
   - 调整模型深度

2. **条件生成**
   - 添加类别标签
   - 添加文本描述
   - 控制生成过程

3. **性能优化**
   - 使用更高效的采样算法（DDIM）
   - 减少推理步数
   - 模型蒸馏

4. **与LDM对比**
   - 训练对应的LDM模型
   - 对比生成质量
   - 分析性能差异

## 🔗 相关资源

- **项目文档**: `README.md` - 详细的架构说明
- **对比分析**: `COMPARISON.md` - VQ-LDM vs LDM
- **配置文件**: `../config/vq_ldm_config_local.yaml` - 所有配置选项
- **MONAI Generative**: https://github.com/Project-MONAI/GenerativeModels

---

**🎉 祝你训练顺利！如有问题，请参考文档或提Issue。**

