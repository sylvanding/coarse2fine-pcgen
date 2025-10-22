"""
NIfTI体素数据集

用于加载NIfTI格式的体素数据，支持自适应分辨率调整和数据增强。
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))

import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, CacheDataset
from monai.utils import set_determinism

logger = logging.getLogger(__name__)


class VoxelNiftiDataset:
    """
    NIfTI体素数据集包装器
    
    自动加载NIfTI文件并应用MONAI transforms进行预处理。
    """
    
    def __init__(
        self,
        data_dir: str,
        target_voxel_size: int = 64,
        normalize_to_minus_one_one: bool = True,
        cache_rate: float = 0.0,
        augmentation: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        初始化NIfTI数据集
        
        Args:
            data_dir: NIfTI文件目录
            target_voxel_size: 目标体素分辨率
            normalize_to_minus_one_one: 是否归一化到[-1, 1]
            cache_rate: 缓存比例 (0-1)
            augmentation: 是否启用数据增强
            augmentation_config: 数据增强配置
        """
        self.data_dir = Path(data_dir)
        self.target_voxel_size = target_voxel_size
        self.normalize_to_minus_one_one = normalize_to_minus_one_one
        self.cache_rate = cache_rate
        self.augmentation = augmentation
        
        if augmentation_config is None:
            augmentation_config = {
                'random_flip_prob': 0.5,
                'random_rotate_prob': 0.5
            }
        self.augmentation_config = augmentation_config
        
        # 收集所有NIfTI文件
        self.nifti_files = self._collect_nifti_files()
        
        logger.info(f"初始化NIfTI数据集:")
        logger.info(f"  数据目录: {data_dir}")
        logger.info(f"  样本数量: {len(self.nifti_files)}")
        logger.info(f"  目标体素大小: {target_voxel_size}^3")
        logger.info(f"  缓存比例: {cache_rate}")
        logger.info(f"  数据增强: {augmentation}")
        
        # 创建transforms
        self.transforms = self._create_transforms()
        
        # 创建数据列表
        self.data_list = [{"image": str(f)} for f in self.nifti_files]
        
        # 创建MONAI Dataset
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=4
            )
            logger.info(f"使用缓存数据集 (cache_rate={cache_rate})")
        else:
            self.dataset = Dataset(
                data=self.data_list,
                transform=self.transforms
            )
            logger.info("使用标准数据集 (无缓存)")
    
    def _collect_nifti_files(self) -> List[Path]:
        """收集所有NIfTI文件"""
        nifti_files = []
        
        # 支持.nii和.nii.gz
        for pattern in ['*.nii', '*.nii.gz']:
            nifti_files.extend(self.data_dir.glob(pattern))
        
        nifti_files.sort()  # 确保顺序一致
        
        if len(nifti_files) == 0:
            raise ValueError(f"在目录 {self.data_dir} 中未找到NIfTI文件")
        
        return nifti_files
    
    def _create_transforms(self):
        """创建MONAI transforms管道"""
        transform_list = [
            # 加载NIfTI文件
            transforms.LoadImaged(keys=["image"]),
            
            # 确保通道维度在前
            transforms.EnsureChannelFirstd(keys=["image"]),
            
            # 调整空间分辨率
            transforms.Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear"
            ),
            
            # 调整到目标体素大小
            transforms.Resized(
                keys=["image"],
                spatial_size=(self.target_voxel_size, self.target_voxel_size, self.target_voxel_size),
                mode="trilinear"
            ),
        ]
        
        # 添加数据增强
        if self.augmentation:
            # 随机翻转
            if self.augmentation_config.get('random_flip_prob', 0) > 0:
                transform_list.append(
                    transforms.RandFlipd(
                        keys=["image"],
                        prob=self.augmentation_config['random_flip_prob'],
                        spatial_axis=[0, 1, 2]
                    )
                )
            
            # 随机旋转
            if self.augmentation_config.get('random_rotate_prob', 0) > 0:
                transform_list.append(
                    transforms.RandRotate90d(
                        keys=["image"],
                        prob=self.augmentation_config['random_rotate_prob'],
                        spatial_axes=(0, 1)
                    )
                )
        
        # 归一化
        if self.normalize_to_minus_one_one:
            # 归一化到[-1, 1]
            transform_list.append(
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=0.0,
                    a_max=1.0,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True
                )
            )
        else:
            # 归一化到[0, 1]
            transform_list.append(
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=0.0,
                    a_max=1.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                )
            )
        
        # 确保类型为float32
        transform_list.append(
            transforms.EnsureTyped(keys=["image"], dtype=torch.float32)
        )
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.dataset[idx]


def create_train_val_dataloaders(
    config: Dict[str, Any],
    train_data_dir: Optional[str] = None,
    val_data_dir: Optional[str] = None
):
    """
    创建训练和验证DataLoader
    
    Args:
        config: 配置字典
        train_data_dir: 训练数据目录（覆盖config）
        val_data_dir: 验证数据目录（覆盖config）
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    # 提取配置
    data_config = config['data']
    
    train_dir = train_data_dir or data_config['train_data_dir']
    val_dir = val_data_dir or data_config['val_data_dir']
    
    voxel_size = data_config['voxel_size']
    cache_rate = data_config.get('cache_rate', 0.0)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    augmentation_config = data_config.get('augmentation', {})
    augmentation_enabled = augmentation_config.get('enabled', False)
    
    # 获取batch_size（从autoencoder或diffusion配置中）
    if 'autoencoder' in config and 'training' in config['autoencoder']:
        batch_size = config['autoencoder']['training']['batch_size']
    elif 'diffusion' in config and 'training' in config['diffusion']:
        batch_size = config['diffusion']['training']['batch_size']
    else:
        batch_size = 2  # 默认值
    
    logger.info("创建训练和验证数据集...")
    
    # 创建训练数据集（启用数据增强）
    train_dataset = VoxelNiftiDataset(
        data_dir=train_dir,
        target_voxel_size=voxel_size,
        normalize_to_minus_one_one=True,
        cache_rate=cache_rate,
        augmentation=augmentation_enabled,
        augmentation_config=augmentation_config
    )
    
    # 创建验证数据集（不使用数据增强）
    val_dataset = VoxelNiftiDataset(
        data_dir=val_dir,
        target_voxel_size=voxel_size,
        normalize_to_minus_one_one=True,
        cache_rate=cache_rate,
        augmentation=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"Batch大小: {batch_size}")
    
    return train_loader, val_loader

