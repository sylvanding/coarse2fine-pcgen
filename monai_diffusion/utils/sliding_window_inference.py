"""
滑动窗口推理工具

用于在训练时使用小patch，但推理时能够处理任意大小的完整体积。
这是Patch-Based训练的配套工具。
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Callable
import logging

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
import numpy as np

logger = logging.getLogger(__name__)


class AutoencoderSlidingWindowInferer:
    """
    AutoencoderKL的滑动窗口推理器
    
    用于处理比训练patch更大的体积数据。
    
    Example:
        >>> # 训练时使用96×96×48的patch
        >>> # 推理时处理256×256×64的完整体积
        >>> inferer = AutoencoderSlidingWindowInferer(
        ...     autoencoder=model,
        ...     roi_size=(96, 96, 48),
        ...     sw_batch_size=4,
        ...     overlap=0.25
        ... )
        >>> 
        >>> full_volume = torch.randn(1, 1, 256, 256, 64).cuda()
        >>> reconstruction = inferer.reconstruct(full_volume)
        >>> print(reconstruction.shape)  # (1, 1, 256, 256, 64)
    """
    
    def __init__(
        self,
        autoencoder: nn.Module,
        roi_size: Tuple[int, int, int] = (96, 96, 48),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        mode: str = "gaussian",
        device: Optional[torch.device] = None
    ):
        """
        初始化滑动窗口推理器
        
        Args:
            autoencoder: AutoencoderKL模型
            roi_size: 滑动窗口的大小（应与训练patch一致）
            sw_batch_size: 同时处理的窗口数量
            overlap: 窗口重叠比例（0-1），建议0.25-0.5
            mode: 重叠区域的融合模式
                - "gaussian": 高斯加权融合（推荐，边缘平滑）
                - "constant": 平均融合
            device: 设备，如果为None则自动检测
        """
        self.autoencoder = autoencoder
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        
        if device is None:
            self.device = next(autoencoder.parameters()).device
        else:
            self.device = device
        
        # 创建MONAI的SlidingWindowInferer
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            cache_roi_weight_map=True  # 缓存权重图，加速推理
        )
        
        logger.info(f"初始化滑动窗口推理器:")
        logger.info(f"  ROI大小: {roi_size}")
        logger.info(f"  批次大小: {sw_batch_size}")
        logger.info(f"  重叠比例: {overlap}")
        logger.info(f"  融合模式: {mode}")
    
    def reconstruct(
        self,
        images: torch.Tensor,
        return_latent: bool = False
    ) -> torch.Tensor:
        """
        使用滑动窗口进行重建
        
        Args:
            images: 输入图像 (B, C, H, W, D)，可以是任意大小
            return_latent: 是否返回latent表示
        
        Returns:
            如果return_latent=False: 重建图像 (B, C, H, W, D)
            如果return_latent=True: (重建图像, latent均值, latent标准差)
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            # 定义autoencoder的前向函数
            def _forward_fn(x):
                """AutoencoderKL的前向推理函数"""
                return self.autoencoder(x)[0]  # 只返回重建结果，不返回z_mu和z_sigma
            
            if return_latent:
                # 如果需要返回latent，先重建，然后编码整个重建结果
                logger.warning("滑动窗口推理返回latent时，latent是从完整重建结果中提取的")
                
                # 使用滑动窗口重建
                reconstruction = self.inferer(images, _forward_fn)
                
                # 编码整个重建结果获取latent
                z_mu, z_sigma = self.autoencoder.encode(reconstruction)
                return reconstruction, z_mu, z_sigma
            else:
                # 标准重建
                reconstruction = self.inferer(images, _forward_fn)
                return reconstruction
    
    def encode(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用滑动窗口进行编码
        
        注意：这会产生多个窗口的latent，然后融合。
        结果可能与直接编码整个图像略有不同。
        
        Args:
            images: 输入图像 (B, C, H, W, D)
        
        Returns:
            (z_mu, z_sigma): latent均值和标准差
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            def _encode_fn(x):
                """编码函数，返回concatenated的z_mu和z_sigma"""
                z_mu, z_sigma = self.autoencoder.encode(x)
                # 返回concatenated结果以便滑动窗口推理
                return torch.cat([z_mu, z_sigma], dim=1)
            
            # 使用滑动窗口编码
            result = self.inferer(images, _encode_fn)
            
            # 分离z_mu和z_sigma
            latent_channels = result.shape[1] // 2
            z_mu = result[:, :latent_channels]
            z_sigma = result[:, latent_channels:]
            
            return z_mu, z_sigma
    
    def decode(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        从latent解码（不使用滑动窗口，因为latent通常较小）
        
        Args:
            z: latent表示
        
        Returns:
            重建图像
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            reconstruction = self.autoencoder.decode(z)
            return reconstruction
    
    @staticmethod
    def calculate_optimal_roi_size(
        full_size: Tuple[int, int, int],
        available_memory_gb: float = 8.0,
        model_params_gb: float = 0.5
    ) -> Tuple[int, int, int]:
        """
        根据可用显存估算最优ROI大小
        
        Args:
            full_size: 完整数据大小 (H, W, D)
            available_memory_gb: 可用显存（GB）
            model_params_gb: 模型参数占用显存（GB）
        
        Returns:
            推荐的ROI大小
        """
        # 简化估算：假设每个体素占用约4 bytes (float32)
        # 考虑中间激活值约占10倍
        bytes_per_voxel = 4 * 10
        
        # 可用于数据的显存
        available_for_data_gb = available_memory_gb - model_params_gb
        available_bytes = available_for_data_gb * 1024 ** 3
        
        # 估算可以容纳的体素数
        max_voxels = int(available_bytes / bytes_per_voxel)
        
        # 尝试找到接近立方体的ROI
        side_length = int(max_voxels ** (1/3))
        
        # 确保不超过原始大小
        roi_size = tuple(min(side_length, s) for s in full_size)
        
        logger.info(f"估算的最优ROI大小: {roi_size}")
        logger.info(f"  基于可用显存: {available_memory_gb:.1f}GB")
        logger.info(f"  模型参数占用: {model_params_gb:.1f}GB")
        
        return roi_size


def test_sliding_window_inference():
    """测试滑动窗口推理"""
    import torch
    from generative.networks.nets import AutoencoderKL
    
    # 创建小型AutoencoderKL
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(16, 32),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=8,
        attention_levels=(False, False),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    autoencoder.eval()
    
    # 创建测试数据（模拟256×256×64的大体积）
    full_volume = torch.randn(1, 1, 128, 128, 64).to(device)
    
    print(f"输入形状: {full_volume.shape}")
    
    # 创建滑动窗口推理器（使用64×64×32的ROI）
    inferer = AutoencoderSlidingWindowInferer(
        autoencoder=autoencoder,
        roi_size=(64, 64, 32),
        sw_batch_size=4,
        overlap=0.25
    )
    
    # 测试重建
    with torch.no_grad():
        reconstruction = inferer.reconstruct(full_volume)
    
    print(f"重建形状: {reconstruction.shape}")
    print(f"重建误差: {torch.abs(reconstruction - full_volume).mean().item():.6f}")
    
    # 测试编码
    with torch.no_grad():
        z_mu, z_sigma = inferer.encode(full_volume)
    
    print(f"Latent均值形状: {z_mu.shape}")
    print(f"Latent标准差形状: {z_sigma.shape}")
    
    print("\n✅ 滑动窗口推理测试成功！")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_sliding_window_inference()

