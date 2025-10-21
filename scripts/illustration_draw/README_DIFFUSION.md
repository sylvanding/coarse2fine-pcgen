# 点云扩散去噪过程模拟

## 概述

`pc_diffusion_process.py` 脚本模拟了扩散模型（Diffusion Models）的去噪过程，从纯高斯噪声逐步收敛到目标点云。这个模拟基于DDPM（Denoising Diffusion Probabilistic Models）的数学原理。

## 数学原理

### DDPM前向过程

给定干净点云 $x_0$，前向扩散过程逐步添加高斯噪声：

$$
x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1-\alpha_t} \cdot \epsilon
$$

其中：
- $x_0$ 是目标点云（干净数据）
- $x_t$ 是第 $t$ 步的加噪点云
- $\alpha_t \in [0, 1]$ 是噪声调度系数
- $\epsilon \sim \mathcal{N}(0, I)$ 是标准高斯噪声

### 去噪过程

反向去噪过程模拟了从噪声到数据的演变：

1. **t=T（初始）**: $x_T = \epsilon$ - 纯高斯噪声（$\alpha_T = 0$）
2. **t=T→0（去噪）**: 逐步增加 $\alpha_t$，噪声比例 $1-\alpha_t$ 逐步减小
3. **t=0（终点）**: $x_0$ - 目标点云（$\alpha_0 = 1$）

### 噪声调度策略

#### 线性调度（Linear）
$$
\alpha_t = \frac{t}{T}
$$

噪声比例均匀递减，适合大多数场景。

#### 余弦调度（Cosine）
$$
\alpha_t = \cos^2\left(\frac{\pi}{2} \cdot \left(1 - \frac{t}{T}\right)\right)
$$

早期去噪较慢，后期较快，更符合真实扩散模型的行为。

## 实现特性

### 1. 大规模点云支持

- **降采样**: 对超过 `--max-points` 的点云自动降采样
- **内存优化**: 使用NumPy向量化操作，避免循环
- **适用场景**: 可处理100万点级别的点云

### 2. 归一化处理

- **可选归一化**: 将点云归一化到单位球，便于跨尺度处理
- **保持原始坐标**: 可选择不归一化，保持原始纳米坐标
- **自动恢复**: 输出结果自动恢复到原始坐标系

### 3. 灵活的调度策略

- **线性调度**: 均匀的去噪过程
- **余弦调度**: 更平滑的去噪轨迹

## 使用方法

### 基本用法

```bash
# 处理单个CSV文件（默认10步，线性调度）
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/pointcloud.csv

# 批量处理文件夹
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/pointclouds/ \
    --recursive
```

### 自定义去噪过程

```bash
# 20步去噪，余弦调度
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/pointcloud.csv \
    --num-steps 20 \
    --schedule cosine

# 处理大规模点云（降采样到10万点）
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/large_pointcloud.csv \
    --max-points 100000
```

### 自定义渲染参数

```bash
# 更高分辨率渲染
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/pointcloud.csv \
    --pixel-size 50 \
    --psf-sigma 100 \
    --intensity-scale 2.0
```

### 不归一化处理

```bash
# 保持原始纳米坐标
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/pointcloud.csv \
    --no-normalize
```

## 输出结果

### 文件命名

每个去噪步骤生成两个文件：

- CSV文件: `{filename}_step_{idx:03d}_alpha_{alpha:.4f}.csv`
- PNG图像: `{filename}_step_{idx:03d}_alpha_{alpha:.4f}.png`

示例：
```
Cell11_inf_30_step_000_alpha_0p0000.csv  # t=T, 纯噪声
Cell11_inf_30_step_000_alpha_0p0000.png
...
Cell11_inf_30_step_009_alpha_1p0000.csv  # t=0, 目标点云
Cell11_inf_30_step_009_alpha_1p0000.png
```

### 目录结构

```
outputs/diffusion_simulation/
└── Cell11_inf_30/
    ├── Cell11_inf_30_step_000_alpha_0p0000.csv
    ├── Cell11_inf_30_step_000_alpha_0p0000.png
    ├── Cell11_inf_30_step_001_alpha_0p1111.csv
    ├── Cell11_inf_30_step_001_alpha_0p1111.png
    ...
    ├── Cell11_inf_30_step_009_alpha_1p0000.csv
    └── Cell11_inf_30_step_009_alpha_1p0000.png
```

## 性能优化

### 内存使用

- **降采样**: 使用 `--max-points` 限制内存占用
- **批量渲染**: 逐步保存，避免积累所有图像

### 计算效率

- **向量化操作**: 使用NumPy高效计算
- **并行潜力**: 每个文件独立处理，可并行运行多个实例

### 建议配置

| 点云规模 | --max-points | 预计内存 | 建议步数 |
|---------|-------------|---------|---------|
| < 10万   | None        | < 1GB   | 10-20   |
| 10万-50万 | 100000      | ~2GB    | 10-15   |
| 50万-100万| 100000      | ~2GB    | 10      |
| > 100万  | 50000-100000| ~2GB    | 5-10    |

## 示例应用

### 1. 制作去噪动画

```bash
# 生成20帧去噪序列
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/sample.csv \
    --num-steps 20 \
    --schedule cosine \
    --output-dir outputs/animation_frames

# 使用ffmpeg制作视频
ffmpeg -framerate 5 -pattern_type glob -i 'outputs/animation_frames/*/*.png' \
    -c:v libx264 -pix_fmt yuv420p denoising_animation.mp4
```

### 2. 比较不同调度策略

```bash
# 线性调度
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/sample.csv \
    --schedule linear \
    --output-dir outputs/linear_schedule

# 余弦调度
python scripts/illustration_draw/pc_diffusion_process.py \
    --input data/sample.csv \
    --schedule cosine \
    --output-dir outputs/cosine_schedule
```

### 3. 批量处理数据集

```bash
# 递归处理整个数据集
python scripts/illustration_draw/pc_diffusion_process.py \
    --input /repos/datasets/pointclouds/ \
    --recursive \
    --num-steps 10 \
    --max-points 100000 \
    --output-dir outputs/batch_results
```

## 技术细节

### 点云坐标系

输入CSV格式：
```csv
x [nm],y [nm],z [nm]
1234.5,2345.6,3456.7
...
```

### 归一化流程

1. **中心化**: 平移到质心为原点
2. **尺度归一化**: 缩放到单位球（可选）
3. **扩散模拟**: 在归一化空间进行
4. **反归一化**: 输出恢复到原始坐标系

### 渲染流程

使用 `src/utils/render.py` 进行2D投影渲染：
- 点扩散函数（PSF）模拟显微镜成像
- Z方向积分投影
- 16位灰度图输出

## 常见问题

### Q1: 如何处理100万点的点云？

使用降采样：
```bash
python scripts/illustration_draw/pc_diffusion_process.py \
    --input large.csv \
    --max-points 100000
```

### Q2: 为什么输出图像看起来模糊？

纯噪声阶段（α≈0）本应该是随机分布的点，渲染后自然模糊。随着α→1，结构会逐渐清晰。

### Q3: 如何加速处理？

1. 减少步数：`--num-steps 5`
2. 降采样：`--max-points 50000`
3. 并行处理多个文件（多个终端）

### Q4: 线性和余弦调度的区别？

- **线性**: 均匀去噪，简单直观
- **余弦**: 早期慢、后期快，更接近真实扩散模型

## 相关资源

- 论文: [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- 项目文档: [README.md](../../readme.md)
- 渲染工具: [src/utils/render.py](../../src/utils/render.py)

