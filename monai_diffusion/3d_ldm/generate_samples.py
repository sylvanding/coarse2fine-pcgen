"""
样本生成脚本

使用训练好的Latent Diffusion Model生成新样本。
"""

import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import nibabel as nib
from monai.utils import set_determinism

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from src.voxel.converter import PointCloudToVoxel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_autoencoder(checkpoint_path: str, device: torch.device, ae_config: dict):
    """加载训练好的AutoencoderKL"""
    autoencoder = AutoencoderKL(
        spatial_dims=ae_config['spatial_dims'],
        in_channels=ae_config['in_channels'],
        out_channels=ae_config['out_channels'],
        num_channels=tuple(ae_config['num_channels']),
        latent_channels=ae_config['latent_channels'],
        num_res_blocks=ae_config['num_res_blocks'],
        norm_num_groups=ae_config.get('norm_num_groups', 16),
        attention_levels=tuple(ae_config['attention_levels'])
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    
    logger.info(f"从 {checkpoint_path} 加载AutoencoderKL")
    return autoencoder


def load_diffusion_model(checkpoint_path: str, device: torch.device, diff_config: dict):
    """加载训练好的Diffusion Model"""
    unet = DiffusionModelUNet(
        spatial_dims=diff_config['spatial_dims'],
        in_channels=diff_config['in_channels'],
        out_channels=diff_config['out_channels'],
        num_channels=tuple(diff_config['num_channels']),
        attention_levels=tuple(diff_config['attention_levels']),
        num_head_channels=tuple(diff_config['num_head_channels']),
        num_res_blocks=diff_config.get('num_res_blocks', 1)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    unet.to(device)
    unet.eval()
    
    scale_factor = checkpoint.get('scale_factor', 1.0)
    
    logger.info(f"从 {checkpoint_path} 加载Diffusion Model")
    logger.info(f"Scale factor: {scale_factor:.4f}")
    
    return unet, scale_factor


def create_affine_matrix(voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """创建NIfTI仿射变换矩阵"""
    affine = np.eye(4)
    affine[0, 0] = voxel_spacing[0]
    affine[1, 1] = voxel_spacing[1]
    affine[2, 2] = voxel_spacing[2]
    return affine


def voxel_to_pointcloud(
    voxel_grid: np.ndarray,
    num_points: int,
    threshold: float,
    method: str,
    volume_dims: list,
    padding: list
):
    """
    将体素网格转换为点云
    
    Args:
        voxel_grid: 体素网格 (D, H, W)
        num_points: 目标点数
        threshold: 体素阈值
        method: 采样方法
        volume_dims: 体积尺寸
        padding: 体积边界填充
        
    Returns:
        点云数组 (N, 3)
    """
    converter = PointCloudToVoxel(
        voxel_size=voxel_grid.shape[0],
        volume_dims=volume_dims,
        padding=padding
    )
    
    # 设置边界信息（用于反归一化）
    converter._last_min_bounds = converter.fixed_min_bounds
    converter._last_max_bounds = converter.fixed_max_bounds
    
    point_cloud = converter.voxel_to_points(
        voxel_grid,
        threshold=threshold,
        num_points=num_points,
        method=method
    )
    
    return point_cloud


def generate_samples(config_path: str, output_dir: str = None):
    """
    生成样本
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录（覆盖配置文件中的设置）
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置随机种子
    set_determinism(config.get('seed', 42))
    
    # 设置设备
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}" if use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 提取配置
    ae_config = config['autoencoder']
    diff_config = config['diffusion']
    sampling_config = config['sampling']
    
    # 输出目录
    if output_dir is None:
        output_dir = sampling_config['output_dir']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"输出目录: {output_path}")
    
    # 加载模型
    autoencoder_path = diff_config['checkpoints']['autoencoder_path']
    diffusion_path = diff_config['checkpoints']['output_dir'] + "/best_model.pt"
    
    if not Path(autoencoder_path).exists():
        raise FileNotFoundError(f"AutoencoderKL checkpoint不存在: {autoencoder_path}")
    if not Path(diffusion_path).exists():
        raise FileNotFoundError(f"Diffusion Model checkpoint不存在: {diffusion_path}")
    
    autoencoder = load_autoencoder(autoencoder_path, device, ae_config)
    unet, scale_factor = load_diffusion_model(diffusion_path, device, diff_config)
    
    # 创建调度器
    scheduler_config = diff_config['scheduler']
    scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        schedule=scheduler_config['schedule'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end']
    )
    
    # 设置采样步数
    num_inference_steps = sampling_config['num_inference_steps']
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    logger.info(f"采样步数: {num_inference_steps}")
    
    # 创建Inferer
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    
    # 生成样本
    num_samples = sampling_config['num_samples']
    save_format = sampling_config.get('save_format', 'nifti')
    
    # 计算潜在空间形状
    voxel_size = config['data']['voxel_size']
    latent_channels = ae_config['latent_channels']
    # 假设autoencoder有3次下采样 (64 -> 32 -> 16 -> 8)
    latent_spatial = voxel_size // 8
    latent_shape = (latent_channels, latent_spatial, latent_spatial, latent_spatial)
    
    logger.info(f"生成 {num_samples} 个样本...")
    logger.info(f"潜在空间形状: {latent_shape}")
    
    # 批量生成
    batch_size = 4  # 每次生成4个样本
    all_samples = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="生成样本"):
        current_batch_size = min(batch_size, num_samples - i)
        
        with torch.no_grad():
            # 生成随机噪声
            noise = torch.randn((current_batch_size, *latent_shape)).to(device)
            
            # 采样
            synthetic_images = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                scheduler=scheduler
            )
            
            all_samples.append(synthetic_images.cpu())
    
    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    logger.info(f"生成完成，样本形状: {all_samples.shape}")
    
    # 创建仿射矩阵
    affine = create_affine_matrix()
    
    # 保存样本
    logger.info("保存样本...")
    
    if save_format == 'nifti' or save_format == 'both':
        # 保存为NIfTI
        nifti_dir = output_path / "nifti"
        nifti_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(num_samples), desc="保存NIfTI"):
            voxel = all_samples[i].numpy()  # (C, D, H, W)
            
            # 反归一化 [-1, 1] -> [0, 1]
            voxel = (voxel + 1.0) / 2.0
            voxel = np.clip(voxel, 0, 1)
            
            # 创建NIfTI图像
            nifti_img = nib.Nifti1Image(voxel, affine)
            nifti_path = nifti_dir / f"sample_{i:04d}.nii.gz"
            nib.save(nifti_img, str(nifti_path))
        
        logger.info(f"NIfTI文件已保存到: {nifti_dir}")
    
    if save_format == 'pointcloud' or save_format == 'both':
        # 转换为点云
        pc_dir = output_path / "pointcloud"
        pc_dir.mkdir(exist_ok=True)
        
        pc_config = sampling_config.get('pointcloud', {})
        num_points = pc_config.get('num_points', 10000)
        threshold = pc_config.get('threshold', 0.1)
        method = pc_config.get('method', 'probabilistic')
        
        # 获取体积参数
        data_config = config.get('data', {})
        volume_dims = [20000, 20000, 2500]  # 默认值
        padding = [0, 0, 100]
        
        logger.info(f"转换为点云: num_points={num_points}, threshold={threshold}, method={method}")
        
        for i in tqdm(range(num_samples), desc="转换点云"):
            voxel = all_samples[i, 0].numpy()  # 移除通道维度
            
            # 反归一化
            voxel = (voxel + 1.0) / 2.0
            voxel = np.clip(voxel, 0, 1)
            
            try:
                point_cloud = voxel_to_pointcloud(
                    voxel, num_points, threshold, method,
                    volume_dims, padding
                )
                
                # 保存为CSV
                pc_path = pc_dir / f"sample_{i:04d}.csv"
                np.savetxt(
                    pc_path, point_cloud, fmt='%.6f', delimiter=',',
                    header='x [nm],y [nm],z [nm]', comments=''
                )
            except Exception as e:
                logger.warning(f"样本 {i} 转换点云失败: {e}")
        
        logger.info(f"点云文件已保存到: {pc_dir}")
    
    logger.info("样本生成完成!")


def main():
    parser = argparse.ArgumentParser(description="生成样本")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/ldm_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    args = parser.parse_args()
    generate_samples(args.config, args.output_dir)


if __name__ == "__main__":
    main()

