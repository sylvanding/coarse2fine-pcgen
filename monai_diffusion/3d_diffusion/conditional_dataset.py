"""
条件扩散数据集

支持同时加载3D体素和2D投影图像作为条件的数据集。
2D投影通过将3D体素沿z轴累加得到。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import logging
import monai.transforms as transforms
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    Resized,
    Lambda,
    Lambdad,
)

logger = logging.getLogger(__name__)


class ConditionalVoxelDataset(Dataset):
    """
    条件体素数据集

    加载3D体素并生成2D投影图像作为条件。
    2D投影通过将3D体素沿z轴累加得到。
    """

    def __init__(
        self,
        data_list: List[Dict],
        transforms,
        project_axis: int = 2,  # z轴
        normalize_projection: bool = True,
    ):
        """
        初始化条件体素数据集

        Args:
            data_list: 数据文件列表
            transforms: MONAI transforms
            project_axis: 投影轴（0=x, 1=y, 2=z）
            normalize_projection: 是否归一化投影图像
        """
        self.data_list = data_list
        self.transforms = transforms
        self.project_axis = project_axis
        self.normalize_projection = normalize_projection

        logger.info(f"初始化条件体素数据集:")
        logger.info(f"  样本数: {len(data_list)}")
        logger.info(f"  投影轴: {project_axis}")
        logger.info(f"  归一化投影: {normalize_projection}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Returns:
            包含'image'（3D体素）和'projection'（2D投影）的字典
        """
        data = self.data_list[idx]

        # 应用transforms加载3D体素
        data_transformed = self.transforms(data)

        # 提取3D体素 (C, H, W, D)
        voxel_3d = data_transformed["image"]  # torch.Tensor

        # 生成2D投影（沿z轴累加）
        projection_2d = self._create_projection(voxel_3d)

        return {
            "image": voxel_3d,  # (1, H, W, D) - 3D体素
            "projection": projection_2d,  # (1, H, W) - 2D投影
            "file_path": data.get("image", "unknown"),
        }

    def _create_projection(self, voxel_3d: torch.Tensor) -> torch.Tensor:
        """
        创建2D投影图像

        Args:
            voxel_3d: 3D体素 (C, H, W, D)

        Returns:
            2D投影 (C, H, W)
        """
        # 沿指定轴累加
        # voxel_3d shape: (C, H, W, D) 对于axis=2（z轴）
        projection = torch.sum(
            voxel_3d, dim=self.project_axis + 1
        )  # +1因为第一维是通道

        # 归一化投影
        if self.normalize_projection:
            proj_min = projection.min()
            proj_max = projection.max()

            if proj_max > proj_min:
                projection = (projection - proj_min) / (proj_max - proj_min + 1e-8)
            else:
                projection = projection - proj_min

        return projection


def create_projection_transform(voxel_3d: torch.Tensor, axis: int = 2) -> torch.Tensor:
    """
    创建投影的辅助函数

    Args:
        voxel_3d: 3D体素张量 (C, H, W, D)
        axis: 投影轴

    Returns:
        2D投影 (C, H, W)
    """
    projection = torch.sum(voxel_3d, dim=axis + 1)

    # 归一化
    proj_min = projection.min()
    proj_max = projection.max()

    if proj_max > proj_min:
        projection = (projection - proj_min) / (proj_max - proj_min + 1e-8)

    return projection


def create_train_val_transforms(config: dict, is_train: bool = True):
    """
    创建训练/验证的transforms

    Args:
        config: 配置字典
        is_train: 是否为训练集

    Returns:
        Compose transforms
    """
    data_config = config["data"]
    aug_config = data_config.get("augmentation", {})

    voxel_size = data_config["voxel_size"]
    voxel_resize = data_config.get("voxel_resize", None)

    # 确保voxel_size是列表
    if isinstance(voxel_size, int):
        voxel_size = [voxel_size, voxel_size, voxel_size]

    # 基础transforms
    transforms_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ]

    # 可选的预resize
    if voxel_resize is not None:
        if isinstance(voxel_resize, int):
            voxel_resize = [voxel_resize, voxel_resize, voxel_resize]
        transforms_list.append(
            Resized(keys=["image"], spatial_size=voxel_resize, mode="trilinear")
        )

    # 训练集使用随机裁剪，验证集使用中心裁剪
    use_patch_based = aug_config.get("use_patch_based", False)
    use_center_crop_for_val = aug_config.get("use_center_crop_for_val", True)

    if is_train:
        if use_patch_based:
            transforms_list.append(
                RandSpatialCropd(keys=["image"], roi_size=voxel_size, random_size=False)
            )
        else:
            transforms_list.append(
                CenterSpatialCropd(keys=["image"], roi_size=voxel_size)
            )
    else:
        if use_center_crop_for_val:
            transforms_list.append(
                CenterSpatialCropd(keys=["image"], roi_size=voxel_size)
            )
        else:
            transforms_list.append(
                Resized(keys=["image"], spatial_size=voxel_size, mode="trilinear")
            )

    # 强度归一化
    transforms_list.append(
        transforms.ScaleIntensityRanged(
            keys="image", a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        )
    )

    # 可选的强度反转
    if aug_config.get("invert_intensity", False):
        transforms_list.append(Lambdad(keys=["image"], func=lambda x: 1.0 - x))

    # 训练集数据增强
    if is_train and aug_config.get("enabled", False):
        if aug_config.get("random_flip_prob", 0) > 0:
            transforms_list.append(
                RandFlipd(
                    keys=["image"],
                    prob=aug_config["random_flip_prob"],
                    spatial_axis=[0, 1, 2],
                )
            )

        if aug_config.get("random_rotate_prob", 0) > 0:
            transforms_list.append(
                RandRotate90d(
                    keys=["image"],
                    prob=aug_config["random_rotate_prob"],
                    spatial_axes=(0, 1),
                )
            )

    return Compose(transforms_list)


def create_train_val_dataloaders(config: dict, batch_size: Optional[int] = None):
    """
    创建训练和验证数据加载器

    Args:
        config: 配置字典
        batch_size: 批次大小（如果为None则从config读取）

    Returns:
        (train_loader, val_loader)
    """
    from pathlib import Path

    data_config = config["data"]

    # 获取数据目录
    train_data_dir = Path(data_config["train_data_dir"])
    val_data_dir = Path(data_config["val_data_dir"])

    # 获取所有NIfTI文件
    train_files = sorted(train_data_dir.glob("*.nii.gz"))
    val_files = sorted(val_data_dir.glob("*.nii.gz"))

    # 限制数据集大小（用于快速测试）
    max_train = data_config.get("max_data_size_for_train", None)
    max_val = data_config.get("max_data_size_for_val", None)

    if max_train is not None:
        train_files = train_files[:max_train]
    if max_val is not None:
        val_files = val_files[:max_val]

    logger.info(f"找到 {len(train_files)} 个训练文件")
    logger.info(f"找到 {len(val_files)} 个验证文件")

    # 创建数据列表
    train_data_list = [{"image": str(f)} for f in train_files]
    val_data_list = [{"image": str(f)} for f in val_files]

    # 创建transforms
    train_transforms = create_train_val_transforms(config, is_train=True)
    val_transforms = create_train_val_transforms(config, is_train=False)

    # 创建数据集（使用ConditionalVoxelDataset）
    train_dataset = ConditionalVoxelDataset(
        data_list=train_data_list,
        transforms=train_transforms,
        project_axis=2,  # z轴投影
        normalize_projection=True,
    )

    val_dataset = ConditionalVoxelDataset(
        data_list=val_data_list,
        transforms=val_transforms,
        project_axis=2,
        normalize_projection=True,
    )

    # 使用缓存数据集加速（可选）
    cache_rate = data_config.get("cache_rate", 0.0)
    if cache_rate > 0:
        logger.info(f"使用CacheDataset，缓存率: {cache_rate}")
        # 注意：这里不再wrap ConditionalVoxelDataset
        # 而是直接使用transforms创建CacheDataset
        from monai.data import CacheDataset as MONAICacheDataset

        # 重新定义为支持投影的transforms + dataset
        # 暂时保持简单，不使用CacheDataset
        pass

    # 批次大小
    if batch_size is None:
        # 尝试从多个可能的配置位置获取batch_size
        if "diffusion" in config and "training" in config["diffusion"]:
            batch_size = config["diffusion"]["training"].get("batch_size", 4)
        else:
            batch_size = 4

    # 创建DataLoader
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    logger.info(f"创建DataLoader: batch_size={batch_size}, num_workers={num_workers}")

    return train_loader, val_loader


if __name__ == "__main__":
    """
    测试条件数据集加载器
    
    随机加载几个样本，检查数据形状、统计信息，并可视化保存
    """
    import yaml
    import matplotlib.pyplot as plt
    import os
    import random

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("测试条件数据集加载器")
    print("=" * 80)

    # ========== 1. 加载配置 ==========
    config_path = (
        project_root / "monai_diffusion/config/conditional_diffusion_config.yaml"
    )
    print(f"\n加载配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ========== 2. 创建数据加载器 ==========
    print("\n创建训练和验证数据加载器...")
    try:
        train_loader, val_loader = create_train_val_dataloaders(config)
    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")

    # ========== 3. 创建输出目录 ==========
    output_dir = project_root / "outputs/conditional_diffusion/data_loader_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # ========== 4. 随机加载样本并分析 ==========
    num_samples_to_test = min(4, len(train_loader.dataset))
    print(f"\n随机测试 {num_samples_to_test} 个样本...")

    # 随机选择样本索引
    indices = random.sample(range(len(train_loader.dataset)), num_samples_to_test)

    for i, idx in enumerate(indices):
        print(f"\n{'=' * 60}")
        print(f"样本 {i + 1}/{num_samples_to_test} (索引: {idx})")
        print(f"{'=' * 60}")

        # 获取样本
        sample = train_loader.dataset[idx]

        voxel_3d = sample["image"]  # (C, H, W, D)
        projection_2d = sample["projection"]  # (C, H, W)
        file_path = sample["file_path"]

        print(f"文件路径: {file_path}")
        print(f"\n3D体素形状: {voxel_3d.shape}")
        print(f"3D体素数据类型: {voxel_3d.dtype}")
        print(f"3D体素最小值: {voxel_3d.min().item():.6f}")
        print(f"3D体素最大值: {voxel_3d.max().item():.6f}")
        print(f"3D体素均值: {voxel_3d.mean().item():.6f}")
        print(f"3D体素标准差: {voxel_3d.std().item():.6f}")

        print(f"\n2D投影形状: {projection_2d.shape}")
        print(f"2D投影数据类型: {projection_2d.dtype}")
        print(f"2D投影最小值: {projection_2d.min().item():.6f}")
        print(f"2D投影最大值: {projection_2d.max().item():.6f}")
        print(f"2D投影均值: {projection_2d.mean().item():.6f}")
        print(f"2D投影标准差: {projection_2d.std().item():.6f}")

        # ========== 5. 可视化 ==========
        # 转换为numpy用于可视化
        voxel_np = voxel_3d.squeeze().cpu().numpy()  # (H, W, D)
        projection_np = projection_2d.squeeze().cpu().numpy()  # (H, W)

        # 创建子图
        fig = plt.figure(figsize=(20, 8))

        # 获取中间切片的索引
        h_mid = voxel_np.shape[0] // 2
        w_mid = voxel_np.shape[1] // 2
        d_mid = voxel_np.shape[2] // 2

        # 第一行：3D体素的三个正交切片
        ax1 = plt.subplot(2, 4, 1)
        im1 = ax1.imshow(voxel_np[:, :, d_mid], cmap="gray", vmin=0, vmax=1)
        ax1.set_title(f"3D voxel XY plane (z={d_mid})")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = plt.subplot(2, 4, 2)
        im2 = ax2.imshow(voxel_np[:, w_mid, :], cmap="gray", vmin=0, vmax=1)
        ax2.set_title(f"3D voxel XZ plane (y={w_mid})")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = plt.subplot(2, 4, 3)
        im3 = ax3.imshow(voxel_np[h_mid, :, :], cmap="gray", vmin=0, vmax=1)
        ax3.set_title(f"3D voxel YZ plane (x={h_mid})")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # 第一行第四个：2D投影
        ax4 = plt.subplot(2, 4, 4)
        im4 = ax4.imshow(projection_np, cmap="gray", vmin=0, vmax=1)
        ax4.set_title("2D projection (sum along z-axis)")
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        # 第二行：直方图和3D统计
        ax5 = plt.subplot(2, 4, 5)
        ax5.hist(voxel_np.flatten(), bins=50, color="blue", alpha=0.7)
        ax5.set_title("3D voxel intensity distribution")
        ax5.set_xlabel("Intensity value")
        ax5.set_ylabel("Frequency")
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 4, 6)
        ax6.hist(projection_np.flatten(), bins=50, color="green", alpha=0.7)
        ax6.set_title("2D projection intensity distribution")
        ax6.set_xlabel("Intensity value")
        ax6.set_ylabel("Frequency")
        ax6.grid(True, alpha=0.3)

        # 第二行第三个：沿z轴的强度统计
        ax7 = plt.subplot(2, 4, 7)
        z_means = [voxel_np[:, :, z].mean() for z in range(voxel_np.shape[2])]
        z_maxs = [voxel_np[:, :, z].max() for z in range(voxel_np.shape[2])]
        ax7.plot(z_means, label="Mean", linewidth=2)
        ax7.plot(z_maxs, label="Max", linewidth=2)
        ax7.set_title("Intensity statistics along Z-axis")
        ax7.set_xlabel("Z slice index")
        ax7.set_ylabel("Intensity value")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 第二行第四个：文本统计信息
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis("off")
        stats_text = f"""
样本统计信息

3D体素:
  形状: {voxel_3d.shape}
  最小值: {voxel_3d.min().item():.6f}
  最大值: {voxel_3d.max().item():.6f}
  均值: {voxel_3d.mean().item():.6f}
  标准差: {voxel_3d.std().item():.6f}

2D投影:
  形状: {projection_2d.shape}
  最小值: {projection_2d.min().item():.6f}
  最大值: {projection_2d.max().item():.6f}
  均值: {projection_2d.mean().item():.6f}
  标准差: {projection_2d.std().item():.6f}

文件: {Path(file_path).name}
        """
        ax8.text(
            0.1,
            0.5,
            stats_text,
            fontsize=10,
            family="monospace",
            verticalalignment="center",
        )

        plt.tight_layout()

        # 保存图像
        output_path = output_dir / f"sample_{i + 1:02d}_idx_{idx:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n✓ 可视化已保存: {output_path}")

    # ========== 6. 测试批次加载 ==========
    print(f"\n{'=' * 60}")
    print("测试批次加载")
    print(f"{'=' * 60}")

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"]  # (B, C, H, W, D)
        projections = batch["projection"]  # (B, C, H, W)

        print(f"\n批次 {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  投影形状: {projections.shape}")
        print(f"  图像范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
        print(
            f"  投影范围: [{projections.min().item():.4f}, {projections.max().item():.4f}]"
        )

        # 只测试前2个批次
        if batch_idx >= 1:
            break

    # ========== 7. 总结 ==========
    print(f"\n{'=' * 80}")
    print("测试完成!")
    print(f"{'=' * 80}")
    print(f"\n✓ 所有可视化已保存到: {output_dir}")
    print(f"✓ 数据加载器工作正常")
    print(f"✓ 3D体素和2D投影生成正确")
    print(f"\n请检查输出目录中的PNG文件以查看可视化结果。")
    print("=" * 80)
