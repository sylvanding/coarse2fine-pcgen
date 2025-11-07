"""
LDM验证脚本

加载训练好的Latent Diffusion Model，对验证数据进行推理并可视化保存结果。
支持两种模式：
1. Autoencoder重建验证
2. 完整LDM生成验证
"""

import os
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer

from monai_diffusion.datasets import create_train_val_dataloaders
from monai_diffusion.utils.sliding_window_inference import AutoencoderSlidingWindowInferer

# 用于潜空间可视化
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def load_autoencoder(config: dict, checkpoint_path: str, device: torch.device) -> AutoencoderKL:
    """
    加载AutoencoderKL模型
    
    Args:
        config: 配置字典
        checkpoint_path: checkpoint路径
        device: 设备
        
    Returns:
        加载好权重的AutoencoderKL模型
    """
    ae_config = config['autoencoder']
    
    # 提取下采样配置
    downsample_factors = ae_config.get('downsample_factors', None)
    initial_downsample_factor = ae_config.get('initial_downsample_factor', 1)
    use_conv_downsample = ae_config.get('use_conv_downsample', True)
    use_convtranspose = ae_config.get('use_convtranspose', False)
    
    if downsample_factors is not None:
        downsample_factors = tuple(downsample_factors)
    
    # 创建模型
    autoencoder = AutoencoderKL(
        spatial_dims=ae_config['spatial_dims'],
        in_channels=ae_config['in_channels'],
        out_channels=ae_config['out_channels'],
        num_channels=tuple(ae_config['num_channels']),
        latent_channels=ae_config['latent_channels'],
        num_res_blocks=ae_config['num_res_blocks'],
        norm_num_groups=ae_config.get('norm_num_groups', 16),
        attention_levels=tuple(ae_config['attention_levels']),
        downsample_factors=downsample_factors,
        initial_downsample_factor=initial_downsample_factor,
        use_conv_downsample=use_conv_downsample,
        use_convtranspose=use_convtranspose
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'autoencoder_state_dict' in checkpoint:
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        logger.info(f"从checkpoint加载autoencoder (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        autoencoder.load_state_dict(checkpoint)
        logger.info("加载autoencoder模型")
    
    autoencoder.to(device)
    autoencoder.eval()
    
    return autoencoder


def load_diffusion_model(config: dict, checkpoint_path: str, device: torch.device) -> DiffusionModelUNet:
    """
    加载DiffusionModelUNet
    
    Args:
        config: 配置字典
        checkpoint_path: checkpoint路径
        device: 设备
        
    Returns:
        加载好权重的DiffusionModelUNet模型
    """
    diff_config = config['diffusion']
    
    # 创建模型
    diffusion_model = DiffusionModelUNet(
        spatial_dims=diff_config['spatial_dims'],
        in_channels=diff_config['latent_channels'],
        out_channels=diff_config['latent_channels'],
        num_channels=tuple(diff_config['num_channels']),
        attention_levels=tuple(diff_config['attention_levels']),
        num_res_blocks=diff_config['num_res_blocks'],
        num_head_channels=diff_config['num_head_channels']
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'diffusion_model_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        logger.info(f"从checkpoint加载diffusion model (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        diffusion_model.load_state_dict(checkpoint)
        logger.info("加载diffusion model")
    
    diffusion_model.to(device)
    diffusion_model.eval()
    
    return diffusion_model


def validate_autoencoder_reconstruction(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = None
):
    """
    验证Autoencoder重建效果
    
    对验证集中的所有样本进行重建，并保存可视化结果。
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_samples: 处理的样本数量，None表示全部
    """
    logger.info("=" * 60)
    logger.info("开始验证Autoencoder重建效果")
    logger.info("=" * 60)
    
    # 创建输出目录
    recon_dir = Path(output_dir) / "autoencoder_reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    total_samples = 0
    total_mse = 0.0
    total_mae = 0.0
    
    # 用于收集潜变量统计信息
    all_z_means = []
    all_z_stds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="重建验证集")):
            images = batch["image"].to(device)  # (B, C, H, W, D)
            
            # 重建
            reconstruction, z_mu, z_sigma = autoencoder(images)
            
            # 使用encode_stage_2_inputs获取采样后的潜变量（参考train_diffusion.py）
            z = autoencoder.encode_stage_2_inputs(images)
            
            # 收集潜变量的统计信息
            z_mean = z.mean().item()
            z_std = z.std().item()
            all_z_means.append(z_mean)
            all_z_stds.append(z_std)
            
            # 计算误差
            mse = F.mse_loss(reconstruction, images).item()
            mae = F.l1_loss(reconstruction, images).item()
            
            total_mse += mse * images.shape[0]
            total_mae += mae * images.shape[0]
            
            # 转换到CPU并保存可视化
            images_np = images.cpu().numpy()
            reconstruction_np = reconstruction.cpu().numpy()
            
            for i in range(images_np.shape[0]):
                sample_idx = total_samples + i
                
                # 检查是否达到样本上限
                if num_samples is not None and sample_idx >= num_samples:
                    break
                
                # 取出单个样本 (C, H, W, D)
                input_vol = images_np[i, 0]  # (H, W, D)
                recon_vol = reconstruction_np[i, 0]  # (H, W, D)
                
                # 沿z轴投影（累加）
                input_proj = np.sum(input_vol, axis=2)  # (H, W)
                recon_proj = np.sum(recon_vol, axis=2)  # (H, W)
                
                # 归一化到[0, 1]
                input_proj = (input_proj - input_proj.min()) / (input_proj.max() - input_proj.min() + 1e-8)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                
                # 误差图
                error = np.abs(input_proj - recon_proj)
                
                # 水平堆叠: 输入 | 重建 | 误差
                combined = np.hstack([input_proj, recon_proj, error])  # (H, 3*W)
                
                # 保存为图像
                image = Image.fromarray(np.uint8(combined * 255.0))
                image.save(recon_dir / f"sample_{sample_idx:04d}.png")
            
            total_samples += images.shape[0]
            
            # 检查是否达到样本上限
            if num_samples is not None and total_samples >= num_samples:
                break
    
    # 计算平均指标
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    # 计算潜变量的统计信息
    avg_z_mean = np.mean(all_z_means)
    avg_z_std = np.mean(all_z_stds)
    std_z_mean = np.std(all_z_means)
    std_z_std = np.std(all_z_stds)
    
    logger.info(f"重建验证完成:")
    logger.info(f"  处理样本数: {total_samples}")
    logger.info(f"  平均MSE: {avg_mse:.6f}")
    logger.info(f"  平均MAE: {avg_mae:.6f}")
    logger.info(f"  结果保存到: {recon_dir}")
    logger.info("")
    logger.info("潜变量统计信息 (z = encode_stage_2_inputs):")
    logger.info(f"  潜变量均值的平均: {avg_z_mean:.6f} ± {std_z_mean:.6f}")
    logger.info(f"  潜变量标准差的平均: {avg_z_std:.6f} ± {std_z_std:.6f}")
    logger.info(f"  理想情况: 均值接近0，标准差接近1（标准正态分布）")
    logger.info("=" * 60)
    
    # 保存指标到文件
    metrics_path = recon_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Autoencoder重建验证指标\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"处理样本数: {total_samples}\n")
        f.write(f"平均MSE: {avg_mse:.6f}\n")
        f.write(f"平均MAE: {avg_mae:.6f}\n")
        f.write(f"\n")
        f.write(f"潜变量统计信息 (z = encode_stage_2_inputs):\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"潜变量均值的平均: {avg_z_mean:.6f} ± {std_z_mean:.6f}\n")
        f.write(f"潜变量标准差的平均: {avg_z_std:.6f} ± {std_z_std:.6f}\n")
        f.write(f"理想情况: 均值接近0，标准差接近1（标准正态分布）\n")
    
    logger.info(f"指标保存到: {metrics_path}")


def validate_ldm_generation(
    autoencoder: AutoencoderKL,
    diffusion_model: DiffusionModelUNet,
    scheduler: DDPMScheduler,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = None,
    num_inference_steps: int = 50,
    use_ddim: bool = True
):
    """
    验证LDM生成效果
    
    对验证集中的样本，首先编码到潜在空间，然后添加噪声，
    最后使用扩散模型去噪并解码，验证完整的LDM流程。
    
    Args:
        autoencoder: AutoencoderKL模型
        diffusion_model: DiffusionModelUNet模型
        scheduler: 噪声调度器
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_samples: 处理的样本数量，None表示全部
        num_inference_steps: 推理步数
        use_ddim: 是否使用DDIM调度器
    """
    logger.info("=" * 60)
    logger.info("开始验证LDM生成效果")
    logger.info(f"推理步数: {num_inference_steps}")
    logger.info(f"使用调度器: {'DDIM' if use_ddim else 'DDPM'}")
    logger.info("=" * 60)
    
    # 创建输出目录
    gen_dir = Path(output_dir) / "ldm_generation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    diffusion_model.eval()
    
    # 选择调度器
    if use_ddim:
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler.num_train_timesteps,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195
        )
    else:
        inference_scheduler = scheduler
    
    inference_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # 创建推理器
    inferer = DiffusionInferer(scheduler=inference_scheduler)
    
    total_samples = 0
    
    # 用于收集潜变量统计信息
    all_z_initial_means = []  # 编码后的潜变量
    all_z_initial_stds = []
    all_z_denoised_means = []  # 去噪后的潜变量
    all_z_denoised_stds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="LDM生成验证")):
            images = batch["image"].to(device)  # (B, C, H, W, D)
            
            # 1. 编码到潜在空间（使用encode_stage_2_inputs，参考train_diffusion.py）
            z = autoencoder.encode_stage_2_inputs(images)
            
            # 收集初始潜变量的统计信息
            z_initial_mean = z.mean().item()
            z_initial_std = z.std().item()
            all_z_initial_means.append(z_initial_mean)
            all_z_initial_stds.append(z_initial_std)
            
            # 2. 添加噪声（模拟扩散过程）
            noise = torch.randn_like(z)
            timesteps = torch.full(
                (z.shape[0],), 
                inference_scheduler.num_train_timesteps - 1, 
                device=device,
                dtype=torch.long
            )
            noisy_z = inference_scheduler.add_noise(
                original_samples=z,
                noise=noise,
                timesteps=timesteps
            )
            
            # 3. 使用扩散模型去噪
            denoised_z = inferer.sample(
                input_noise=noisy_z,
                diffusion_model=diffusion_model,
                scheduler=inference_scheduler,
                verbose=False
            )
            
            # 收集去噪后潜变量的统计信息
            z_denoised_mean = denoised_z.mean().item()
            z_denoised_std = denoised_z.std().item()
            all_z_denoised_means.append(z_denoised_mean)
            all_z_denoised_stds.append(z_denoised_std)
            
            # 4. 解码回图像空间
            reconstruction = autoencoder.decode(denoised_z)
            
            # 转换到CPU并保存可视化
            images_np = images.cpu().numpy()
            reconstruction_np = reconstruction.cpu().numpy()
            
            for i in range(images_np.shape[0]):
                sample_idx = total_samples + i
                
                # 检查是否达到样本上限
                if num_samples is not None and sample_idx >= num_samples:
                    break
                
                # 取出单个样本 (C, H, W, D)
                input_vol = images_np[i, 0]  # (H, W, D)
                recon_vol = reconstruction_np[i, 0]  # (H, W, D)
                
                # 沿z轴投影（累加）
                input_proj = np.sum(input_vol, axis=2)  # (H, W)
                recon_proj = np.sum(recon_vol, axis=2)  # (H, W)
                
                # 归一化到[0, 1]
                input_proj = (input_proj - input_proj.min()) / (input_proj.max() - input_proj.min() + 1e-8)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                
                # 误差图
                error = np.abs(input_proj - recon_proj)
                
                # 水平堆叠: 输入 | LDM重建 | 误差
                combined = np.hstack([input_proj, recon_proj, error])  # (H, 3*W)
                
                # 保存为图像
                image = Image.fromarray(np.uint8(combined * 255.0))
                image.save(gen_dir / f"sample_{sample_idx:04d}.png")
            
            total_samples += images.shape[0]
            
            # 检查是否达到样本上限
            if num_samples is not None and total_samples >= num_samples:
                break
    
    # 计算潜变量的统计信息
    avg_z_initial_mean = np.mean(all_z_initial_means)
    avg_z_initial_std = np.mean(all_z_initial_stds)
    std_z_initial_mean = np.std(all_z_initial_means)
    std_z_initial_std = np.std(all_z_initial_stds)
    
    avg_z_denoised_mean = np.mean(all_z_denoised_means)
    avg_z_denoised_std = np.mean(all_z_denoised_stds)
    std_z_denoised_mean = np.std(all_z_denoised_means)
    std_z_denoised_std = np.std(all_z_denoised_stds)
    
    logger.info(f"LDM生成验证完成:")
    logger.info(f"  处理样本数: {total_samples}")
    logger.info(f"  结果保存到: {gen_dir}")
    logger.info("")
    logger.info("潜变量统计信息 (z = encode_stage_2_inputs):")
    logger.info(f"  初始潜变量 (编码后):")
    logger.info(f"    均值的平均: {avg_z_initial_mean:.6f} ± {std_z_initial_mean:.6f}")
    logger.info(f"    标准差的平均: {avg_z_initial_std:.6f} ± {std_z_initial_std:.6f}")
    logger.info(f"  去噪后潜变量:")
    logger.info(f"    均值的平均: {avg_z_denoised_mean:.6f} ± {std_z_denoised_mean:.6f}")
    logger.info(f"    标准差的平均: {avg_z_denoised_std:.6f} ± {std_z_denoised_std:.6f}")
    logger.info(f"  理想情况: 均值接近0，标准差接近1（标准正态分布）")
    logger.info("=" * 60)
    
    # 保存统计信息到文件
    metrics_path = gen_dir / "latent_statistics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"LDM生成验证 - 潜变量统计信息\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"处理样本数: {total_samples}\n")
        f.write(f"\n")
        f.write(f"潜变量统计信息 (z = encode_stage_2_inputs):\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"初始潜变量 (编码后):\n")
        f.write(f"  均值的平均: {avg_z_initial_mean:.6f} ± {std_z_initial_mean:.6f}\n")
        f.write(f"  标准差的平均: {avg_z_initial_std:.6f} ± {std_z_initial_std:.6f}\n")
        f.write(f"\n")
        f.write(f"去噪后潜变量:\n")
        f.write(f"  均值的平均: {avg_z_denoised_mean:.6f} ± {std_z_denoised_mean:.6f}\n")
        f.write(f"  标准差的平均: {avg_z_denoised_std:.6f} ± {std_z_denoised_std:.6f}\n")
        f.write(f"\n")
        f.write(f"理想情况: 均值接近0，标准差接近1（标准正态分布）\n")
    
    logger.info(f"潜变量统计信息保存到: {metrics_path}")


def validate_ldm_unconditional_generation(
    autoencoder: AutoencoderKL,
    diffusion_model: DiffusionModelUNet,
    scheduler: DDPMScheduler,
    device: torch.device,
    output_dir: str,
    latent_shape: tuple,
    num_samples: int = 16,
    num_inference_steps: int = 50,
    use_ddim: bool = True
):
    """
    验证LDM无条件生成
    
    从随机噪声开始，使用扩散模型生成新样本。
    
    Args:
        autoencoder: AutoencoderKL模型
        diffusion_model: DiffusionModelUNet模型
        scheduler: 噪声调度器
        device: 设备
        output_dir: 输出目录
        latent_shape: 潜在空间形状 (C, H, W, D)
        num_samples: 生成样本数量
        num_inference_steps: 推理步数
        use_ddim: 是否使用DDIM调度器
    """
    logger.info("=" * 60)
    logger.info("开始无条件生成验证")
    logger.info(f"生成样本数: {num_samples}")
    logger.info(f"推理步数: {num_inference_steps}")
    logger.info(f"使用调度器: {'DDIM' if use_ddim else 'DDPM'}")
    logger.info("=" * 60)
    
    # 创建输出目录
    uncond_dir = Path(output_dir) / "ldm_unconditional_generation"
    uncond_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    diffusion_model.eval()
    
    # 选择调度器
    if use_ddim:
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler.num_train_timesteps,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195
        )
    else:
        inference_scheduler = scheduler
    
    inference_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # 创建推理器
    inferer = DiffusionInferer(scheduler=inference_scheduler)
    
    with torch.no_grad():
        # 生成随机噪声
        noise = torch.randn(num_samples, *latent_shape).to(device)
        
        logger.info(f"潜在空间噪声形状: {noise.shape}")
        
        # 收集初始噪声的统计信息
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        
        # 使用扩散模型从噪声生成潜在向量
        generated_latents = inferer.sample(
            input_noise=noise,
            diffusion_model=diffusion_model,
            scheduler=inference_scheduler,
            verbose=True
        )
        
        logger.info(f"生成的潜在向量形状: {generated_latents.shape}")
        
        # 收集生成的潜变量统计信息
        generated_latents_mean = generated_latents.mean().item()
        generated_latents_std = generated_latents.std().item()
        
        # 解码到图像空间
        generated_images = autoencoder.decode(generated_latents)
        
        logger.info(f"解码后图像形状: {generated_images.shape}")
        
        # 转换到CPU并保存可视化
        generated_np = generated_images.cpu().numpy()
        
        for i in range(num_samples):
            # 取出单个样本 (C, H, W, D)
            gen_vol = generated_np[i, 0]  # (H, W, D)
            
            # 沿z轴投影（累加）
            gen_proj = np.sum(gen_vol, axis=2)  # (H, W)
            
            # 归一化到[0, 1]
            gen_proj = (gen_proj - gen_proj.min()) / (gen_proj.max() - gen_proj.min() + 1e-8)
            
            # 保存为图像
            image = Image.fromarray(np.uint8(gen_proj * 255.0))
            image.save(uncond_dir / f"generated_{i:04d}.png")
    
    logger.info(f"无条件生成完成:")
    logger.info(f"  生成样本数: {num_samples}")
    logger.info(f"  结果保存到: {uncond_dir}")
    logger.info("")
    logger.info("潜变量统计信息:")
    logger.info(f"  初始随机噪声:")
    logger.info(f"    均值: {noise_mean:.6f}")
    logger.info(f"    标准差: {noise_std:.6f}")
    logger.info(f"  生成的潜变量:")
    logger.info(f"    均值: {generated_latents_mean:.6f}")
    logger.info(f"    标准差: {generated_latents_std:.6f}")
    logger.info(f"  理想情况: 生成的潜变量均值接近0，标准差接近1")
    logger.info("=" * 60)
    
    # 保存统计信息到文件
    metrics_path = uncond_dir / "latent_statistics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"无条件生成 - 潜变量统计信息\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"生成样本数: {num_samples}\n")
        f.write(f"\n")
        f.write(f"潜变量统计信息:\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"初始随机噪声:\n")
        f.write(f"  均值: {noise_mean:.6f}\n")
        f.write(f"  标准差: {noise_std:.6f}\n")
        f.write(f"\n")
        f.write(f"生成的潜变量:\n")
        f.write(f"  均值: {generated_latents_mean:.6f}\n")
        f.write(f"  标准差: {generated_latents_std:.6f}\n")
        f.write(f"\n")
        f.write(f"理想情况: 生成的潜变量均值接近0，标准差接近1（标准正态分布）\n")
    
    logger.info(f"潜变量统计信息保存到: {metrics_path}")


def validate_sample_mixing(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_pairs: int = 10,
    alpha_values: list = [0.0, 0.25, 0.5, 0.75, 1.0]
):
    """
    验证样本混合效果
    
    将两个输入样本按不同比例混合，观察autoencoder如何处理混合输入。
    这可以验证：
    1. Autoencoder是否能够分离混合信号
    2. 编码器的线性响应特性
    3. 解码器的重建能力
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_pairs: 处理的样本对数量
        alpha_values: 混合系数列表，alpha * img1 + (1-alpha) * img2
    """
    logger.info("=" * 60)
    logger.info("开始样本混合测试")
    logger.info(f"混合系数: {alpha_values}")
    logger.info("=" * 60)
    
    # 创建输出目录
    mixing_dir = Path(output_dir) / "sample_mixing"
    mixing_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集一批样本
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            all_samples.append(images)
            if len(all_samples) * images.shape[0] >= num_pairs * 2:
                break
    
    all_samples = torch.cat(all_samples, dim=0)
    
    with torch.no_grad():
        for pair_idx in range(num_pairs):
            # 选择两个样本
            img1 = all_samples[pair_idx * 2:pair_idx * 2 + 1]
            img2 = all_samples[pair_idx * 2 + 1:pair_idx * 2 + 2]
            
            # 对每个alpha值进行混合
            mixed_images = []
            mixed_reconstructions = []
            
            for alpha in alpha_values:
                # 在图像空间混合
                mixed_img = alpha * img1 + (1 - alpha) * img2
                
                # 重建混合图像
                recon, _, _ = autoencoder(mixed_img)
                
                mixed_images.append(mixed_img)
                mixed_reconstructions.append(recon)
            
            # 转换为numpy并可视化
            img1_np = img1[0, 0].cpu().numpy()
            img2_np = img2[0, 0].cpu().numpy()
            
            # 沿z轴投影
            img1_proj = np.sum(img1_np, axis=2)
            img2_proj = np.sum(img2_np, axis=2)
            
            # 归一化
            img1_proj = (img1_proj - img1_proj.min()) / (img1_proj.max() - img1_proj.min() + 1e-8)
            img2_proj = (img2_proj - img2_proj.min()) / (img2_proj.max() - img2_proj.min() + 1e-8)
            
            # 创建可视化网格
            fig, axes = plt.subplots(3, len(alpha_values) + 2, figsize=(3 * (len(alpha_values) + 2), 9))
            
            # 第一行：原始图像1
            axes[0, 0].imshow(img1_proj, cmap='gray')
            axes[0, 0].set_title('Image 1')
            axes[0, 0].axis('off')
            
            # 第二行：原始图像2
            axes[1, 0].imshow(img2_proj, cmap='gray')
            axes[1, 0].set_title('Image 2')
            axes[1, 0].axis('off')
            
            # 隐藏第三行第一列
            axes[2, 0].axis('off')
            
            # 对每个alpha值显示混合和重建结果
            for i, alpha in enumerate(alpha_values):
                mixed_np = mixed_images[i][0, 0].cpu().numpy()
                recon_np = mixed_reconstructions[i][0, 0].cpu().numpy()
                
                # 投影
                mixed_proj = np.sum(mixed_np, axis=2)
                recon_proj = np.sum(recon_np, axis=2)
                
                # 归一化
                mixed_proj = (mixed_proj - mixed_proj.min()) / (mixed_proj.max() - mixed_proj.min() + 1e-8)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                
                # 误差
                error = np.abs(mixed_proj - recon_proj)
                
                # 第一行：混合输入
                axes[0, i + 1].imshow(mixed_proj, cmap='gray')
                axes[0, i + 1].set_title(f'α={alpha:.2f}')
                axes[0, i + 1].axis('off')
                
                # 第二行：重建结果
                axes[1, i + 1].imshow(recon_proj, cmap='gray')
                axes[1, i + 1].set_title('Recon')
                axes[1, i + 1].axis('off')
                
                # 第三行：误差
                axes[2, i + 1].imshow(error, cmap='hot')
                axes[2, i + 1].set_title(f'Error')
                axes[2, i + 1].axis('off')
            
            # 隐藏最后一列
            for row in range(3):
                axes[row, -1].axis('off')
            
            plt.suptitle(f'Sample Mixing Test - Pair {pair_idx + 1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(mixing_dir / f"mixing_pair_{pair_idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"样本混合测试完成，结果保存到: {mixing_dir}")
    logger.info("=" * 60)


def validate_latent_interpolation(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_pairs: int = 10,
    num_steps: int = 10
):
    """
    验证潜空间插值
    
    在两个样本的潜空间表示之间进行线性插值，观察解码后的过渡效果。
    这可以验证：
    1. 潜空间的连续性和平滑性
    2. 解码器对潜空间变化的响应
    3. 潜空间是否形成有意义的流形
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_pairs: 处理的样本对数量
        num_steps: 插值步数
    """
    logger.info("=" * 60)
    logger.info("开始潜空间插值测试")
    logger.info(f"插值步数: {num_steps}")
    logger.info("=" * 60)
    
    # 创建输出目录
    interp_dir = Path(output_dir) / "latent_interpolation"
    interp_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集样本
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            all_samples.append(images)
            if len(all_samples) * images.shape[0] >= num_pairs * 2:
                break
    
    all_samples = torch.cat(all_samples, dim=0)
    
    with torch.no_grad():
        for pair_idx in range(num_pairs):
            # 选择两个样本
            img1 = all_samples[pair_idx * 2:pair_idx * 2 + 1]
            img2 = all_samples[pair_idx * 2 + 1:pair_idx * 2 + 2]
            
            # 编码到潜空间
            z1 = autoencoder.encode_stage_2_inputs(img1)
            z2 = autoencoder.encode_stage_2_inputs(img2)
            
            # 在潜空间进行线性插值
            interpolated_images = []
            interpolated_latents = []
            alphas = np.linspace(0, 1, num_steps)
            
            for alpha in alphas:
                # 确保alpha是float，避免numpy类型问题
                alpha = float(alpha)
                z_interp = alpha * z1 + (1 - alpha) * z2
                img_interp = autoencoder.decode(z_interp)
                interpolated_images.append(img_interp)
                interpolated_latents.append(z_interp)
            
            # 重建原始图像（用于对比）
            recon1 = autoencoder.decode(z1)
            recon2 = autoencoder.decode(z2)
            
            # 可视化
            fig, axes = plt.subplots(4, num_steps, figsize=(2 * num_steps, 8))
            
            for i, (alpha, img_interp, z_interp) in enumerate(zip(alphas, interpolated_images, interpolated_latents)):
                # 原始图像1的投影（第一行）
                img1_np = img1[0, 0].cpu().numpy()
                img1_proj = np.sum(img1_np, axis=2)
                img1_proj = (img1_proj - img1_proj.min()) / (img1_proj.max() - img1_proj.min() + 1e-8)
                axes[0, i].imshow(img1_proj, cmap='gray')
                axes[0, i].set_title(f'Img1' if i == 0 else '')
                axes[0, i].axis('off')
                
                # 插值结果（第二行）
                interp_np = img_interp[0, 0].cpu().numpy()
                interp_proj = np.sum(interp_np, axis=2)
                interp_proj = (interp_proj - interp_proj.min()) / (interp_proj.max() - interp_proj.min() + 1e-8)
                axes[1, i].imshow(interp_proj, cmap='gray')
                axes[1, i].set_title(f'α={alpha:.2f}')
                axes[1, i].axis('off')
                
                # 原始图像2的投影（第三行）
                img2_np = img2[0, 0].cpu().numpy()
                img2_proj = np.sum(img2_np, axis=2)
                img2_proj = (img2_proj - img2_proj.min()) / (img2_proj.max() - img2_proj.min() + 1e-8)
                axes[2, i].imshow(img2_proj, cmap='gray')
                axes[2, i].set_title(f'Img2' if i == 0 else '')
                axes[2, i].axis('off')
                
                # 潜空间距离（第四行）
                z_dist_to_z1 = torch.norm(z_interp - z1).item()
                z_dist_to_z2 = torch.norm(z_interp - z2).item()
                axes[3, i].bar(['z1', 'z2'], [z_dist_to_z1, z_dist_to_z2])
                axes[3, i].set_title(f'L2 Dist')
                axes[3, i].set_ylim([0, max(torch.norm(z1 - z2).item(), 1.0)])
            
            plt.suptitle(f'Latent Space Interpolation - Pair {pair_idx + 1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(interp_dir / f"interpolation_pair_{pair_idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"潜空间插值测试完成，结果保存到: {interp_dir}")
    logger.info("=" * 60)


def validate_latent_arithmetic(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_triplets: int = 10
):
    """
    验证潜空间算术
    
    测试潜空间的向量运算: z_A + (z_B - z_C) 是否产生有意义的结果。
    这可以验证：
    1. 潜空间的线性可加性
    2. 是否能够进行语义编辑
    3. 潜空间结构的质量
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_triplets: 处理的三元组数量
    """
    logger.info("=" * 60)
    logger.info("开始潜空间算术测试")
    logger.info("=" * 60)
    
    # 创建输出目录
    arith_dir = Path(output_dir) / "latent_arithmetic"
    arith_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集样本
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            all_samples.append(images)
            if len(all_samples) * images.shape[0] >= num_triplets * 3:
                break
    
    all_samples = torch.cat(all_samples, dim=0)
    
    with torch.no_grad():
        for triplet_idx in range(num_triplets):
            # 选择三个样本
            img_A = all_samples[triplet_idx * 3:triplet_idx * 3 + 1]
            img_B = all_samples[triplet_idx * 3 + 1:triplet_idx * 3 + 2]
            img_C = all_samples[triplet_idx * 3 + 2:triplet_idx * 3 + 3]
            
            # 编码到潜空间
            z_A = autoencoder.encode_stage_2_inputs(img_A)
            z_B = autoencoder.encode_stage_2_inputs(img_B)
            z_C = autoencoder.encode_stage_2_inputs(img_C)
            
            # 进行潜空间算术: z_result = z_A + (z_B - z_C)
            z_diff = z_B - z_C
            z_result = z_A + z_diff
            
            # 解码
            img_result = autoencoder.decode(z_result)
            recon_A = autoencoder.decode(z_A)
            recon_B = autoencoder.decode(z_B)
            recon_C = autoencoder.decode(z_C)
            
            # 可视化
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # 第一行：原始图像
            images_to_show = [
                (img_A, 'Image A'),
                (img_B, 'Image B'),
                (img_C, 'Image C'),
                (img_result, 'Result: A+(B-C)')
            ]
            
            for i, (img, title) in enumerate(images_to_show):
                img_np = img[0, 0].cpu().numpy()
                img_proj = np.sum(img_np, axis=2)
                img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
                axes[0, i].imshow(img_proj, cmap='gray')
                axes[0, i].set_title(title)
                axes[0, i].axis('off')
            
            # 第二行：重建图像
            recons_to_show = [
                (recon_A, 'Recon A'),
                (recon_B, 'Recon B'),
                (recon_C, 'Recon C'),
                (img_result, 'Result (decoded)')
            ]
            
            for i, (img, title) in enumerate(recons_to_show):
                img_np = img[0, 0].cpu().numpy()
                img_proj = np.sum(img_np, axis=2)
                img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
                axes[1, i].imshow(img_proj, cmap='gray')
                axes[1, i].set_title(title)
                axes[1, i].axis('off')
            
            plt.suptitle(f'Latent Arithmetic: z_A + (z_B - z_C) - Triplet {triplet_idx + 1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(arith_dir / f"arithmetic_triplet_{triplet_idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"潜空间算术测试完成，结果保存到: {arith_dir}")
    logger.info("=" * 60)


def validate_reconstruction_consistency(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 20,
    num_iterations: int = 5
):
    """
    验证重建一致性
    
    对同一样本进行多次编码-解码，观察结果的一致性。
    这可以验证：
    1. 编码器和解码器的稳定性
    2. VAE采样的随机性影响
    3. 重建过程的可重复性
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_samples: 处理的样本数量
        num_iterations: 每个样本重建的次数
    """
    logger.info("=" * 60)
    logger.info("开始重建一致性测试")
    logger.info(f"每个样本重建次数: {num_iterations}")
    logger.info("=" * 60)
    
    # 创建输出目录
    consistency_dir = Path(output_dir) / "reconstruction_consistency"
    consistency_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集样本
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            all_samples.append(images)
            if len(all_samples) * images.shape[0] >= num_samples:
                break
    
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    
    # 用于统计
    all_variances = []
    all_mean_mses = []
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            img = all_samples[sample_idx:sample_idx + 1]
            
            # 多次重建
            reconstructions = []
            for _ in range(num_iterations):
                # 使用完整的forward方法（包含VAE采样）
                recon, z_mu, z_sigma = autoencoder(img)
                reconstructions.append(recon)
            
            # 计算重建之间的方差
            recons_tensor = torch.stack(reconstructions, dim=0)  # (num_iterations, B, C, H, W, D)
            recon_mean = recons_tensor.mean(dim=0)
            recon_variance = recons_tensor.var(dim=0).mean().item()
            
            # 计算每次重建与原图的MSE
            mses = [F.mse_loss(recon, img).item() for recon in reconstructions]
            mean_mse = np.mean(mses)
            std_mse = np.std(mses)
            
            all_variances.append(recon_variance)
            all_mean_mses.append(mean_mse)
            
            # 可视化
            img_np = img[0, 0].cpu().numpy()
            img_proj = np.sum(img_np, axis=2)
            img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
            
            # 创建可视化
            fig, axes = plt.subplots(2, num_iterations + 2, figsize=(2 * (num_iterations + 2), 4))
            
            # 第一行：原始图像
            axes[0, 0].imshow(img_proj, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            # 显示每次重建
            for i, recon in enumerate(reconstructions):
                recon_np = recon[0, 0].cpu().numpy()
                recon_proj = np.sum(recon_np, axis=2)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                
                axes[0, i + 1].imshow(recon_proj, cmap='gray')
                axes[0, i + 1].set_title(f'Recon {i+1}')
                axes[0, i + 1].axis('off')
                
                # 第二行：误差
                error = np.abs(img_proj - recon_proj)
                axes[1, i + 1].imshow(error, cmap='hot')
                axes[1, i + 1].set_title(f'MSE: {mses[i]:.4f}')
                axes[1, i + 1].axis('off')
            
            # 平均重建
            mean_recon_np = recon_mean[0, 0].cpu().numpy()
            mean_proj = np.sum(mean_recon_np, axis=2)
            mean_proj = (mean_proj - mean_proj.min()) / (mean_proj.max() - mean_proj.min() + 1e-8)
            axes[0, -1].imshow(mean_proj, cmap='gray')
            axes[0, -1].set_title('Mean Recon')
            axes[0, -1].axis('off')
            
            # 方差图
            var_np = recons_tensor.var(dim=0)[0, 0].cpu().numpy()
            var_proj = np.sum(var_np, axis=2)
            axes[1, -1].imshow(var_proj, cmap='hot')
            axes[1, -1].set_title(f'Variance: {recon_variance:.6f}')
            axes[1, -1].axis('off')
            
            axes[1, 0].axis('off')
            
            plt.suptitle(f'Reconstruction Consistency - Sample {sample_idx + 1}\nMean MSE: {mean_mse:.6f} ± {std_mse:.6f}', fontsize=14)
            plt.tight_layout()
            plt.savefig(consistency_dir / f"consistency_sample_{sample_idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # 统计摘要
    avg_variance = np.mean(all_variances)
    std_variance = np.std(all_variances)
    avg_mse = np.mean(all_mean_mses)
    std_mse_across_samples = np.std(all_mean_mses)
    
    logger.info(f"重建一致性测试完成:")
    logger.info(f"  平均重建方差: {avg_variance:.6f} ± {std_variance:.6f}")
    logger.info(f"  平均MSE: {avg_mse:.6f} ± {std_mse_across_samples:.6f}")
    logger.info(f"  结果保存到: {consistency_dir}")
    logger.info("=" * 60)
    
    # 保存统计信息
    metrics_path = consistency_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"重建一致性测试统计\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"样本数: {num_samples}\n")
        f.write(f"每个样本重建次数: {num_iterations}\n")
        f.write(f"平均重建方差: {avg_variance:.6f} ± {std_variance:.6f}\n")
        f.write(f"平均MSE: {avg_mse:.6f} ± {std_mse_across_samples:.6f}\n")
        f.write(f"\n理想情况: 方差应该较小，表示重建稳定\n")


def validate_noise_robustness(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 20,
    noise_levels: list = [0.0, 0.05, 0.1, 0.2, 0.5]
):
    """
    验证噪声鲁棒性
    
    向输入添加不同强度的高斯噪声，观察autoencoder的重建能力。
    这可以验证：
    1. Autoencoder的去噪能力
    2. 编码器对噪声的鲁棒性
    3. 潜空间表示的稳定性
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_samples: 处理的样本数量
        noise_levels: 噪声标准差列表
    """
    logger.info("=" * 60)
    logger.info("开始噪声鲁棒性测试")
    logger.info(f"噪声强度: {noise_levels}")
    logger.info("=" * 60)
    
    # 创建输出目录
    noise_dir = Path(output_dir) / "noise_robustness"
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集样本
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            all_samples.append(images)
            if len(all_samples) * images.shape[0] >= num_samples:
                break
    
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    
    # 用于统计
    noise_level_mses = {level: [] for level in noise_levels}
    noise_level_latent_dists = {level: [] for level in noise_levels}
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            img = all_samples[sample_idx:sample_idx + 1]
            
            # 获取原始图像的潜变量
            z_clean = autoencoder.encode_stage_2_inputs(img)
            
            # 对每个噪声级别
            noisy_images = []
            reconstructions = []
            latent_distances = []
            mses = []
            
            for noise_level in noise_levels:
                # 添加高斯噪声
                if noise_level > 0:
                    noise = torch.randn_like(img) * noise_level
                    noisy_img = img + noise
                else:
                    noisy_img = img
                
                # 重建
                recon, _, _ = autoencoder(noisy_img)
                z_noisy = autoencoder.encode_stage_2_inputs(noisy_img)
                
                # 计算指标
                mse = F.mse_loss(recon, img).item()
                latent_dist = torch.norm(z_noisy - z_clean).item()
                
                noisy_images.append(noisy_img)
                reconstructions.append(recon)
                mses.append(mse)
                latent_distances.append(latent_dist)
                
                noise_level_mses[noise_level].append(mse)
                noise_level_latent_dists[noise_level].append(latent_dist)
            
            # 可视化
            fig, axes = plt.subplots(3, len(noise_levels), figsize=(3 * len(noise_levels), 9))
            
            for i, noise_level in enumerate(noise_levels):
                # 有噪声的输入
                noisy_np = noisy_images[i][0, 0].cpu().numpy()
                noisy_proj = np.sum(noisy_np, axis=2)
                noisy_proj = (noisy_proj - noisy_proj.min()) / (noisy_proj.max() - noisy_proj.min() + 1e-8)
                axes[0, i].imshow(noisy_proj, cmap='gray')
                axes[0, i].set_title(f'Noise σ={noise_level:.2f}')
                axes[0, i].axis('off')
                
                # 重建结果
                recon_np = reconstructions[i][0, 0].cpu().numpy()
                recon_proj = np.sum(recon_np, axis=2)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                axes[1, i].imshow(recon_proj, cmap='gray')
                axes[1, i].set_title(f'Reconstruction')
                axes[1, i].axis('off')
                
                # 与原图的误差
                img_np = img[0, 0].cpu().numpy()
                img_proj = np.sum(img_np, axis=2)
                img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min() + 1e-8)
                error = np.abs(recon_proj - img_proj)
                axes[2, i].imshow(error, cmap='hot')
                axes[2, i].set_title(f'MSE: {mses[i]:.4f}\nLatent Δ: {latent_distances[i]:.3f}')
                axes[2, i].axis('off')
            
            plt.suptitle(f'Noise Robustness - Sample {sample_idx + 1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(noise_dir / f"noise_sample_{sample_idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # 统计分析
    logger.info("噪声鲁棒性统计:")
    for noise_level in noise_levels:
        avg_mse = np.mean(noise_level_mses[noise_level])
        std_mse = np.std(noise_level_mses[noise_level])
        avg_latent_dist = np.mean(noise_level_latent_dists[noise_level])
        std_latent_dist = np.std(noise_level_latent_dists[noise_level])
        
        logger.info(f"  噪声σ={noise_level:.2f}: MSE={avg_mse:.6f}±{std_mse:.6f}, 潜空间距离={avg_latent_dist:.6f}±{std_latent_dist:.6f}")
    
    logger.info(f"结果保存到: {noise_dir}")
    logger.info("=" * 60)
    
    # 绘制统计图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE vs 噪声强度
    avg_mses = [np.mean(noise_level_mses[level]) for level in noise_levels]
    std_mses = [np.std(noise_level_mses[level]) for level in noise_levels]
    ax1.errorbar(noise_levels, avg_mses, yerr=std_mses, marker='o', capsize=5)
    ax1.set_xlabel('Noise Level (σ)')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction Error vs Noise Level')
    ax1.grid(True, alpha=0.3)
    
    # 潜空间距离 vs 噪声强度
    avg_dists = [np.mean(noise_level_latent_dists[level]) for level in noise_levels]
    std_dists = [np.std(noise_level_latent_dists[level]) for level in noise_levels]
    ax2.errorbar(noise_levels, avg_dists, yerr=std_dists, marker='s', capsize=5, color='orange')
    ax2.set_xlabel('Noise Level (σ)')
    ax2.set_ylabel('Latent Space Distance')
    ax2.set_title('Latent Perturbation vs Noise Level')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(noise_dir / "noise_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存统计信息
    metrics_path = noise_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"噪声鲁棒性测试统计\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"样本数: {num_samples}\n\n")
        for noise_level in noise_levels:
            avg_mse = np.mean(noise_level_mses[noise_level])
            std_mse = np.std(noise_level_mses[noise_level])
            avg_latent_dist = np.mean(noise_level_latent_dists[noise_level])
            std_latent_dist = np.std(noise_level_latent_dists[noise_level])
            f.write(f"噪声σ={noise_level:.2f}:\n")
            f.write(f"  MSE: {avg_mse:.6f} ± {std_mse:.6f}\n")
            f.write(f"  潜空间距离: {avg_latent_dist:.6f} ± {std_latent_dist:.6f}\n\n")


def validate_latent_clustering(
    autoencoder: AutoencoderKL,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 500,
    use_tsne: bool = True
):
    """
    验证潜空间聚类和可视化
    
    将所有样本编码到潜空间，使用PCA/t-SNE进行降维可视化。
    这可以验证：
    1. 潜空间是否形成有意义的结构
    2. 相似样本是否在潜空间中聚集
    3. 潜空间的维度利用情况
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        num_samples: 处理的样本数量
        use_tsne: 是否使用t-SNE（否则使用PCA）
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn未安装，跳过潜空间聚类可视化")
        return
    
    logger.info("=" * 60)
    logger.info("开始潜空间聚类可视化")
    logger.info(f"降维方法: {'t-SNE' if use_tsne else 'PCA'}")
    logger.info("=" * 60)
    
    # 创建输出目录
    cluster_dir = Path(output_dir) / "latent_clustering"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    
    autoencoder.eval()
    
    # 收集所有潜变量
    all_latents = []
    all_indices = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(val_loader, desc="编码样本"):
            images = batch["image"].to(device)
            z = autoencoder.encode_stage_2_inputs(images)
            
            # 展平潜变量为向量
            z_flat = z.view(z.shape[0], -1)  # (B, C*H*W*D)
            
            all_latents.append(z_flat.cpu().numpy())
            all_indices.extend(range(sample_count, sample_count + z.shape[0]))
            
            sample_count += z.shape[0]
            if sample_count >= num_samples:
                break
    
    all_latents = np.concatenate(all_latents, axis=0)[:num_samples]
    all_indices = all_indices[:num_samples]
    
    logger.info(f"收集了 {len(all_latents)} 个潜变量，形状: {all_latents.shape}")
    
    # 计算潜空间统计
    latent_means = all_latents.mean(axis=0)
    latent_stds = all_latents.std(axis=0)
    
    # PCA分析
    # n_components不能超过min(n_samples, n_features)
    max_components = min(len(all_latents), all_latents.shape[1], 50)
    pca = PCA(n_components=max_components)
    latents_pca = pca.fit_transform(all_latents)
    
    # 安全地访问解释方差比
    n_components_to_show = min(10, len(pca.explained_variance_ratio_))
    logger.info(f"PCA前{n_components_to_show}个主成分解释方差比: {pca.explained_variance_ratio_[:n_components_to_show].sum():.4f}")
    
    # 降维到2D用于可视化
    if use_tsne:
        logger.info("运行t-SNE降维（可能需要几分钟）...")
        # perplexity应该在5到50之间，且小于样本数
        perplexity = min(30, max(5, len(all_latents) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        latents_2d = reducer.fit_transform(all_latents)
    else:
        pca_2d = PCA(n_components=2)
        latents_2d = pca_2d.fit_transform(all_latents)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 潜空间2D散点图
    scatter = axes[0, 0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                                  c=all_indices, cmap='viridis', alpha=0.6, s=10)
    axes[0, 0].set_title(f'Latent Space Visualization ({"t-SNE" if use_tsne else "PCA"})')
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    plt.colorbar(scatter, ax=axes[0, 0], label='Sample Index')
    
    # 2. PCA解释方差
    axes[0, 1].bar(range(len(pca.explained_variance_ratio_)), 
                   pca.explained_variance_ratio_)
    axes[0, 1].set_title('PCA Explained Variance Ratio')
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_xlim([0, min(20, len(pca.explained_variance_ratio_))])
    
    # 3. 潜变量均值分布
    axes[1, 0].hist(latent_means, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', label='Zero')
    axes[1, 0].set_title('Latent Dimensions Mean Distribution')
    axes[1, 0].set_xlabel('Mean Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 4. 潜变量标准差分布
    axes[1, 1].hist(latent_stds, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].axvline(1, color='red', linestyle='--', label='Unit Std')
    axes[1, 1].set_title('Latent Dimensions Std Distribution')
    axes[1, 1].set_xlabel('Std Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(cluster_dir / "latent_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存统计信息
    logger.info(f"潜空间聚类可视化完成，结果保存到: {cluster_dir}")
    logger.info(f"  潜变量均值的均值: {latent_means.mean():.6f}")
    logger.info(f"  潜变量标准差的均值: {latent_stds.mean():.6f}")
    logger.info(f"  前{n_components_to_show}个主成分解释方差: {pca.explained_variance_ratio_[:n_components_to_show].sum():.4f}")
    logger.info("=" * 60)
    
    # 保存统计信息到文件
    metrics_path = cluster_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"潜空间聚类分析\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"样本数: {len(all_latents)}\n")
        f.write(f"潜变量维度: {all_latents.shape[1]}\n")
        f.write(f"潜变量均值的均值: {latent_means.mean():.6f}\n")
        f.write(f"潜变量均值的标准差: {latent_means.std():.6f}\n")
        f.write(f"潜变量标准差的均值: {latent_stds.mean():.6f}\n")
        f.write(f"潜变量标准差的标准差: {latent_stds.std():.6f}\n")
        f.write(f"\nPCA分析:\n")
        # 安全地写入不同数量的主成分
        n_comp = len(pca.explained_variance_ratio_)
        if n_comp >= 10:
            f.write(f"前10个主成分解释方差: {pca.explained_variance_ratio_[:10].sum():.4f}\n")
        if n_comp >= 20:
            f.write(f"前20个主成分解释方差: {pca.explained_variance_ratio_[:20].sum():.4f}\n")
        if n_comp >= 50:
            f.write(f"前50个主成分解释方差: {pca.explained_variance_ratio_[:50].sum():.4f}\n")
        if n_comp < 10:
            f.write(f"前{n_comp}个主成分解释方差: {pca.explained_variance_ratio_.sum():.4f}\n")
        f.write(f"\n理想情况: 均值接近0，标准差接近1，主成分能解释大部分方差\n")


def main():
    parser = argparse.ArgumentParser(description="LDM验证脚本")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/ldm_config_local.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--autoencoder_checkpoint',
        type=str,
        default='/repos/ckpts/monai-diffusion/autoencoder-3channels/latest_checkpoint.pt',
        help='Autoencoder checkpoint路径'
    )
    parser.add_argument(
        '--diffusion_checkpoint',
        type=str,
        default=None,
        help='Diffusion model checkpoint路径（可选，仅用于LDM验证）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/repos/ckpts/monai-diffusion/autoencoder-3channels/ldm_validation',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='验证样本数量，None表示全部'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='clustering',
        choices=[
            'autoencoder',
            'mixing', 'interpolation', 'arithmetic', 'consistency', 
            'noise', 'clustering', 'advanced_all'
        ],
        help='验证模式: '
             'autoencoder（仅AE重建）, '
             'ldm（LDM重建）, '
             'unconditional（无条件生成）, '
             'all（全部基础验证）, '
             'mixing（样本混合）, '
             'interpolation（潜空间插值）, '
             'arithmetic（潜空间算术）, '
             'consistency（重建一致性）, '
             'noise（噪声鲁棒性）, '
             'clustering（潜空间聚类）, '
             'advanced_all（所有高级验证）'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='扩散模型推理步数'
    )
    parser.add_argument(
        '--use_ddim',
        action='store_true',
        default=True,
        help='使用DDIM调度器（默认True）'
    )
    parser.add_argument(
        '--num_unconditional_samples',
        type=int,
        default=16,
        help='无条件生成的样本数量'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 设置设备
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}" if use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 加载Autoencoder
    logger.info(f"加载Autoencoder: {args.autoencoder_checkpoint}")
    autoencoder = load_autoencoder(config, args.autoencoder_checkpoint, device)
    
    # 创建验证数据加载器
    logger.info("创建验证数据加载器...")
    _, val_loader = create_train_val_dataloaders(config)
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 根据模式执行验证
    if args.mode in ['autoencoder', 'all']:
        validate_autoencoder_reconstruction(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_samples=args.num_samples
        )
    
    if args.mode in ['ldm', 'all']:
        if args.diffusion_checkpoint is None:
            logger.warning("未提供diffusion_checkpoint，跳过LDM验证")
        else:
            # 加载Diffusion Model
            logger.info(f"加载Diffusion Model: {args.diffusion_checkpoint}")
            diffusion_model = load_diffusion_model(config, args.diffusion_checkpoint, device)
            
            # 创建调度器
            diff_config = config['diffusion']
            scheduler = DDPMScheduler(
                num_train_timesteps=diff_config['num_train_timesteps'],
                schedule="scaled_linear_beta",
                beta_start=0.0015,
                beta_end=0.0195
            )
            
            validate_ldm_generation(
                autoencoder=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                val_loader=val_loader,
                device=device,
                output_dir=str(output_dir),
                num_samples=args.num_samples,
                num_inference_steps=args.num_inference_steps,
                use_ddim=args.use_ddim
            )
    
    if args.mode in ['unconditional', 'all']:
        if args.diffusion_checkpoint is None:
            logger.warning("未提供diffusion_checkpoint，跳过无条件生成验证")
        else:
            # 加载Diffusion Model（如果还没加载）
            if args.mode == 'unconditional':
                logger.info(f"加载Diffusion Model: {args.diffusion_checkpoint}")
                diffusion_model = load_diffusion_model(config, args.diffusion_checkpoint, device)
                
                diff_config = config['diffusion']
                scheduler = DDPMScheduler(
                    num_train_timesteps=diff_config['num_train_timesteps'],
                    schedule="scaled_linear_beta",
                    beta_start=0.0015,
                    beta_end=0.0195
                )
            
            # 推断潜在空间形状
            ae_config = config['autoencoder']
            data_config = config['data']
            
            # 获取patch大小
            patch_size = data_config['patch_size']
            if isinstance(patch_size, int):
                patch_size = [patch_size] * 3
            
            # 计算下采样倍数
            downsample_factors = ae_config.get('downsample_factors', None)
            initial_downsample_factor = ae_config.get('initial_downsample_factor', 1)
            
            if downsample_factors is not None:
                total_downsample = initial_downsample_factor
                for factor in downsample_factors:
                    total_downsample *= factor
            else:
                total_downsample = initial_downsample_factor * (2 ** (len(ae_config['num_channels']) - 1))
            
            # 计算潜在空间大小
            latent_size = [p // total_downsample for p in patch_size]
            latent_shape = (ae_config['latent_channels'], *latent_size)
            
            logger.info(f"潜在空间形状: {latent_shape}")
            
            validate_ldm_unconditional_generation(
                autoencoder=autoencoder,
                diffusion_model=diffusion_model,
                scheduler=scheduler,
                device=device,
                output_dir=str(output_dir),
                latent_shape=latent_shape,
                num_samples=args.num_unconditional_samples,
                num_inference_steps=args.num_inference_steps,
                use_ddim=args.use_ddim
            )
    
    # 高级验证模式
    if args.mode in ['mixing', 'advanced_all']:
        validate_sample_mixing(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_pairs=min(10, args.num_samples // 2)
        )
    
    if args.mode in ['interpolation', 'advanced_all']:
        validate_latent_interpolation(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_pairs=min(10, args.num_samples // 2),
            num_steps=10
        )
    
    if args.mode in ['arithmetic', 'advanced_all']:
        validate_latent_arithmetic(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_triplets=min(10, args.num_samples // 3)
        )
    
    if args.mode in ['consistency', 'advanced_all']:
        validate_reconstruction_consistency(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_samples=min(20, args.num_samples),
            num_iterations=5
        )
    
    if args.mode in ['noise', 'advanced_all']:
        validate_noise_robustness(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_samples=min(20, args.num_samples),
            noise_levels=[0.0, 0.05, 0.1, 0.2, 0.5]
        )
    
    if args.mode in ['clustering', 'advanced_all']:
        validate_latent_clustering(
            autoencoder=autoencoder,
            val_loader=val_loader,
            device=device,
            output_dir=str(output_dir),
            num_samples=min(500, len(val_loader.dataset)),
            use_tsne=True
        )
    
    logger.info("=" * 60)
    logger.info("验证完成！")
    logger.info(f"所有结果已保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

