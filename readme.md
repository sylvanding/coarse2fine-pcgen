# Coarse2Fine-PCGen

一个基于体素表示的粗到精点云生成框架，突破传统点云生成网络的点数限制。

## 项目概述

本项目提出了一种创新的点云生成方案，通过体素表示作为中间形式来生成大规模点云：

1. **点云→体素映射**: 将单分子定位点云数据转换为3D体素表示
2. **体素生成网络**: 训练生成式网络（如VAE/GAN/Diffusion）生成3D体素
3. **体素→点云采样**: 将生成的体素重新采样为点云
4. **精细化网络**: 使用PointNet++等网络对粗糙点云进行逐点微调

## 技术优势

- **突破点数限制**: 传统方法生成几万个点，本方案可生成百万级点云
- **显存效率**: 避免一次处理百万个点，通过分步处理降低显存消耗
- **精度平衡**: 粗糙生成+精细微调的两阶段策略

## 项目结构

```
coarse2fine-pcgen/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── h5_loader.py          # H5点云数据加载器
│   ├── voxel/
│   │   ├── __init__.py
│   │   └── converter.py          # 点云-体素转换器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── voxel_generator.py    # 体素生成网络
│   │   └── point_refiner.py      # 点云精细化网络
│   └── utils/
│       ├── __init__.py
│       └── visualization.py     # 可视化工具
├── tests/
│   ├── test_h5_loader.py
│   ├── test_voxel_converter.py
│   └── test_pipeline.py         # 端到端测试
├── scripts/
│   └── test_conversion.py       # 转换效果测试脚本
├── requirements.txt
└── README.md
```

## 环境设置

```bash
# 创建conda环境
conda create -n c2f-pcgen python==3.10
conda activate c2f-pcgen

# 安装依赖
# CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## GPUSHARE

```text
3090
PyTorch 221
Cuda 121
Python 311
```

## 快速开始

### 1. 测试点云到体素转换

```python
from src.data.h5_loader import PointCloudH5Loader
from src.voxel.converter import PointCloudToVoxel

# 加载点云数据
loader = PointCloudH5Loader("path/to/pointcloud.h5")
point_cloud = loader.load_single_cloud(index=0)

# 转换为体素
converter = PointCloudToVoxel(voxel_size=64)
voxel_grid = converter.convert(point_cloud)

# 保存为TIFF查看效果
converter.save_as_tiff(voxel_grid, "output.tiff")
```

### 2. 运行测试脚本

```bash
python scripts/test_conversion.py --input data/sample.h5 --output test_voxel.tiff
```

## 数据格式

输入H5文件应包含shape为`(样本数, 点数, 3)`的点云数据，其中最后一维表示xyz坐标。

## 开发计划

- [x] 项目框架搭建
- [ ] H5数据加载器实现
- [ ] 点云到体素转换器
- [ ] 体素生成网络设计
- [ ] 体素到点云采样
- [ ] PointNet++精细化网络
- [ ] 端到端训练流程

## 引用

如果您使用本项目，请引用：
```
@misc{coarse2fine-pcgen,
  title={Coarse2Fine Point Cloud Generation via Voxel Representation},
  year={2024}
}
```
