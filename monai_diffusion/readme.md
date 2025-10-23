# MONAI Diffusion 模块

本模块使用 [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) 库实现3D点云的扩散模型生成。

## 快速开始

Clone仓库并进入项目根目录：

```bash
git clone --depth 1 -b master https://kkgithub.com/sylvanding/coarse2fine-pcgen
cd coarse2fine-pcgen/monai_diffusion
```

### 环境设置

#### ShareGPU

```bash
cd /hy-tmp
git clone --depth 1 -b master https://kkgithub.com/sylvanding/coarse2fine-pcgen

python monai_diffusion/3d_ldm/train_autoencoder.py

# tensorboard logs
ln -s /hy-tmp/coarse2fine-pcgen/outputs/logs /tf_logs/
```

#### Ai.Paratera

#### 3060 Ti, CUDA 121

cuda 121 is for XFormers.

```bash
# 创建conda环境
conda create -n monai-diffusion python==3.11
conda activate monai-diffusion

# 降级setuptools -> solve monai warning: `pkg_resources is deprecated as an API`
pip install 'setuptools<69'

# 安装PyTorch (CUDA 121)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# 安装本项目依赖
cd monai_diffusion
pip install -r requirements.txt

# 测试MONAI安装
# 快速验证命令
python -c "import monai; print(f'MONAI {monai.__version__} installed successfully')"
# 完整测试脚本
python test_monai_installation.py
```

完整测试脚本输出(以3060 Ti, CUDA 121为例)：

```text
✓ CUDA version: NVIDIA GeForce RTX 3060 Ti
✓ Xformers version: 0.0.25
✓ MONAI version: 1.3.1

详细配置:
MONAI version: 1.3.1
Numpy version: 1.26.4
Pytorch version: 2.2.1+cu121
MONAI flags: HAS_EXT = False, USE_COMPILED = False, USE_META_DICT = False
MONAI rev id: 96bfda00c6bd290297f5e3514ea227c6be4d08b4
MONAI __file__: /<username>/miniconda3/envs/monai-diffusion-test/lib/python3.11/site-packages/monai/__init__.py

Optional dependencies:
Pytorch Ignite version: 0.5.3
ITK version: 5.4.4
Nibabel version: 5.3.2
scikit-image version: 0.25.2
scipy version: 1.16.2
Pillow version: 11.3.0
Tensorboard version: 2.20.0
gdown version: 5.2.0
TorchVision version: 0.17.1+cu121
tqdm version: 4.67.1
lmdb version: 1.7.5
psutil version: 7.1.1
pandas version: 2.3.3
einops version: 0.8.1
transformers version: 4.28.0
mlflow version: 3.5.0
pynrrd version: 1.1.3
clearml version: NOT INSTALLED or UNKNOWN VERSION.

For details about installing the optional dependencies, please visit:
    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies


✓ 基础变换测试通过

✓ MONAI 安装成功!
```

## 数据集制作与压缩

运行 scripts/h5pc2voxel/convert_h5_to_nifti.py 脚本，将 H5 文件转换为 NIfTI 文件。随后压缩：

```bash
tar -czvf batch_simulation_microtubule_20251017_2048_nifti.tar.gz batch_simulation_microtubule_20251017_2048_nifti
```

## 使用说明

### 导入MONAI Generative库

在本文件夹下的Python脚本中，使用以下标准模式导入：

```python
import sys
from pathlib import Path

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))

# 导入generative模块
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
```

### 核心组件

#### 1. 扩散模型网络 (DiffusionModelUNet)

3D UNet架构，支持注意力机制和条件生成。

```python
unet = DiffusionModelUNet(
    spatial_dims=3,              # 3D数据
    in_channels=1,               # 体素通道数
    out_channels=1,
    num_channels=(64, 128, 256), # 各层通道数
    attention_levels=(False, True, True),
    num_res_blocks=2,
)
```

#### 2. 噪声调度器 (Scheduler)

控制扩散过程的噪声添加策略。

```python
# DDPM: 标准扩散
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    prediction_type="epsilon",
)

# DDIM: 快速采样
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta",
)
```

#### 3. 推理器 (DiffusionInferer)

封装训练和采样逻辑。

```python
inferer = DiffusionInferer(scheduler=scheduler)

# 训练
noise_pred = inferer(
    inputs=voxel_grid,
    diffusion_model=unet,
    noise=noise,
    timesteps=timesteps,
)

# 采样
samples = inferer.sample(
    input_noise=random_noise,
    diffusion_model=unet,
    scheduler=scheduler,
)
```

## 与项目集成

### 完整工作流

```
点云数据 (.h5)
    ↓
[PointCloudH5Loader] (src/data/h5_loader.py)
    ↓
点云数组 (N, num_points, 3)
    ↓
[PointCloudToVoxel] (src/voxel/converter.py)
    ↓
体素网格 (N, 1, 64, 64, 64)
    ↓
[DiffusionModelUNet + DiffusionInferer] (monai_diffusion/)
    ↓
生成的体素 (M, 1, 64, 64, 64)
    ↓
[体素采样] 
    ↓
生成的点云 (M, num_points, 3)
```

### 示例代码

```python
# 完整示例：训练体素扩散模型
from src.data.h5_loader import PointCloudH5Loader
from src.voxel.converter import PointCloudToVoxel

# 1. 加载点云数据
loader = PointCloudH5Loader("data/point_clouds.h5")
point_clouds = loader.load_all()  # (N, 2048, 3)

# 2. 转换为体素
converter = PointCloudToVoxel(voxel_size=64, method='occupancy')
voxels = []
for pc in point_clouds:
    voxel = converter.convert(pc)
    voxels.append(voxel)
voxels = torch.tensor(np.stack(voxels))[:, None, ...]  # (N, 1, 64, 64, 64)

# 3. 训练扩散模型
for epoch in range(num_epochs):
    for voxel_batch in train_loader:
        noise = torch.randn_like(voxel_batch)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        noise_pred = inferer(
            inputs=voxel_batch,
            diffusion_model=unet,
            noise=noise,
            timesteps=timesteps,
        )
        
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

# 4. 生成新样本
scheduler.set_timesteps(50)
with torch.no_grad():
    synthetic_voxels = inferer.sample(
        input_noise=torch.randn((16, 1, 64, 64, 64)),
        diffusion_model=unet,
        scheduler=scheduler,
    )
```

## Cursor Rules 参考

本项目已配置专门的Cursor Rules，可通过以下方式获取帮助：

- **@monai-generative**: MONAI Generative Models库完整API参考和使用指南
- **@monai-integration**: 在monai_diffusion文件夹下集成使用的最佳实践
- **@monai-quick-reference**: 常用代码片段和快速参考

在Cursor中输入 `@monai-generative` 即可查看详细文档。

## 参考资料

- [MONAI Generative文档](https://docs.monai.io/projects/generative/en/latest/)
- [3D DDPM教程](../GenerativeModels/tutorials/generative/3d_ddpm/3d_ddpm_tutorial.ipynb)
- [DDPM论文](https://arxiv.org/abs/2006.11239)
- [DDIM论文](https://arxiv.org/abs/2010.02502)

## 常见问题

### Q: 显存不足怎么办？

A: 
1. 减小 `batch_size`
2. 减小 `voxel_size`（例如从64→32）
3. 使用潜在扩散模型（AutoencoderKL + LatentDiffusionInferer）
4. 减小模型通道数 `num_channels`

### Q: 训练很慢怎么办？

A:
1. 使用混合精度训练 (`torch.cuda.amp`)
2. 增加 `num_workers` 加快数据加载
3. 使用 `pin_memory=True`
4. 使用 PyTorch 2.0 的 `torch.compile()`

### Q: 生成质量不好怎么办？

A:
1. 增加训练步数（至少100k步）
2. 增加模型容量（`num_channels`）
3. 使用 DDIM 增加采样步数（50-100步）
4. 尝试 `v_prediction` 而非 `epsilon`
5. 使用 EMA（指数移动平均）权重

### Q: 如何实现条件生成？

A: 参考 `@monai-quick-reference` 中的"条件生成"示例，使用交叉注意力机制融合条件信息。

## 项目结构

```
monai_diffusion/
├── readme.md                      # 本文件
├── requirements.txt               # Python依赖
├── config/                        # 配置文件
│   └── ldm_config.yaml           # 统一配置文件
├── datasets/                      # 数据集模块
│   ├── __init__.py
│   └── voxel_nifti_dataset.py    # NIfTI数据加载器
├── 3d_ldm/                        # 3D Latent Diffusion Model
│   ├── train_autoencoder.py      # AutoencoderKL训练
│   ├── train_diffusion.py        # Diffusion Model训练
│   ├── generate_samples.py       # 样本生成
│   ├── run_example.py            # 完整流程示例
│   └── README.md                 # 使用文档
```

## 快速开始

### 1. 数据准备

```bash
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_nifti \
    --voxel_size 64
```

### 2. 训练AutoencoderKL

```bash
python monai_diffusion/3d_ldm/train_autoencoder.py \
    --config monai_diffusion/config/ldm_config.yaml
```

### 3. 训练Diffusion Model

```bash
python monai_diffusion/3d_ldm/train_diffusion.py \
    --config monai_diffusion/config/ldm_config.yaml
```

### 4. 生成样本

```bash
python monai_diffusion/3d_ldm/generate_samples.py \
    --config monai_diffusion/config/ldm_config.yaml
```

### 一键运行完整流程

```bash
python monai_diffusion/3d_ldm/run_example.py \
    --h5_file data/point_clouds.h5 \
    --output_base experiments/ldm_test \
    --voxel_size 64 \
    --fast_dev_run  # 快速开发模式
```

详细使用方法请参考: [3d_ldm/README.md](3d_ldm/README.md)

## 开发计划

- [x] 环境配置和依赖安装
- [x] Cursor Rules文档
- [x] 基础训练脚本
- [x] 3D Latent Diffusion Model实现
- [x] 数据转换管道
- [x] NIfTI数据集加载器
- [x] AutoencoderKL训练
- [x] Diffusion Model训练
- [x] 样本生成和可视化
- [x] 训练恢复功能
- [x] 快速开发模式
- [ ] 条件生成实现
- [ ] 评估指标集成
- [ ] 2D版本实现