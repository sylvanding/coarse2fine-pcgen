# H5点云到NIfTI体素转换

将H5格式的点云数据转换为NIfTI格式的体素数据，用于MONAI训练。

## 使用方法

### 基本用法

```bash
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_nifti \
    --voxel_size 64
```

### 完整参数

```bash
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_nifti \
    --voxel_size 64 \
    --method gaussian \
    --sigma 1.0 \
    --train_ratio 0.8 \
    --data_key point_clouds \
    --volume_dims 20000 20000 2500 \
    --padding 0 0 100
```

## 参数说明

- `--h5_file`: 输入H5文件路径（必需）
- `--output_dir`: 输出目录路径（必需）
- `--voxel_size`: 体素网格分辨率，默认64
- `--method`: 体素化方法，可选 'occupancy', 'density', 'gaussian'，默认 'gaussian'
- `--sigma`: 高斯体素化的标准差，默认1.0
- `--train_ratio`: 训练集比例，默认0.8
- `--data_key`: H5文件中的数据键，默认 'point_clouds'
- `--no_compression`: 不使用压缩（.nii而不是.nii.gz）
- `--volume_dims`: 体积尺寸 [x y z] (nm)
- `--padding`: 体积边界填充 [x y z] (nm)

## 输出结构

```
data/voxels_nifti/
├── train/
│   ├── voxel_00000.nii.gz
│   ├── voxel_00001.nii.gz
│   └── ...
├── val/
│   ├── voxel_00100.nii.gz
│   ├── voxel_00101.nii.gz
│   └── ...
└── dataset_info.json
```

## 文件格式

- **NIfTI格式** (.nii.gz): 医学图像标准格式，支持压缩
- **数据类型**: float32
- **数据范围**: [0, 1]
- **形状**: (1, D, H, W) - 单通道3D图像

## 压缩效果

使用.nii.gz压缩通常可以获得10-50倍的压缩比：

- 64³体素 @ float32: 
  - 未压缩: ~1MB
  - 压缩后: ~50-200KB

## 注意事项

1. 确保H5文件格式正确：shape为(样本数, 点数, 3)
2. 转换过程会自动归一化体素值到[0, 1]
3. 训练/验证集划分使用固定随机种子(42)，保证可重复性
4. dataset_info.json包含数据集元信息，便于追溯

## 示例

### 转换不同分辨率

```bash
# 32x32x32 (更快，显存占用少)
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_32 \
    --voxel_size 32

# 128x128x128 (更精细，需要更多显存)
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_128 \
    --voxel_size 128
```

### 不同体素化方法

```bash
# 占有网格（二值）
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_occupancy \
    --method occupancy

# 密度网格
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_density \
    --method density

# 高斯密度（推荐，更平滑）
python scripts/h5pc2voxel/convert_h5_to_nifti.py \
    --h5_file data/point_clouds.h5 \
    --output_dir data/voxels_gaussian \
    --method gaussian \
    --sigma 1.0
```

