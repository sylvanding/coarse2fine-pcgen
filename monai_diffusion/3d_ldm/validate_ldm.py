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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="重建验证集")):
            images = batch["image"].to(device)  # (B, C, H, W, D)
            
            # 重建
            reconstruction, z_mu, z_sigma = autoencoder(images)
            
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
    
    logger.info(f"重建验证完成:")
    logger.info(f"  处理样本数: {total_samples}")
    logger.info(f"  平均MSE: {avg_mse:.6f}")
    logger.info(f"  平均MAE: {avg_mae:.6f}")
    logger.info(f"  结果保存到: {recon_dir}")
    logger.info("=" * 60)
    
    # 保存指标到文件
    metrics_path = recon_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Autoencoder重建验证指标\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"处理样本数: {total_samples}\n")
        f.write(f"平均MSE: {avg_mse:.6f}\n")
        f.write(f"平均MAE: {avg_mae:.6f}\n")
    
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="LDM生成验证")):
            images = batch["image"].to(device)  # (B, C, H, W, D)
            
            # 1. 编码到潜在空间
            z_mu, z_sigma = autoencoder.encode(images)
            z = autoencoder.sampling(z_mu, z_sigma)
            
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
    
    logger.info(f"LDM生成验证完成:")
    logger.info(f"  处理样本数: {total_samples}")
    logger.info(f"  结果保存到: {gen_dir}")
    logger.info("=" * 60)


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
        
        # 使用扩散模型从噪声生成潜在向量
        generated_latents = inferer.sample(
            input_noise=noise,
            diffusion_model=diffusion_model,
            scheduler=inference_scheduler,
            verbose=True
        )
        
        logger.info(f"生成的潜在向量形状: {generated_latents.shape}")
        
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
    logger.info("=" * 60)


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
        default='/repos/ckpts/monai-diffusion/autoencoder-16channels/latest_checkpoint.pt',
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
        default='/repos/ckpts/monai-diffusion/autoencoder-16channels/ldm_validation',
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
        default='autoencoder',
        choices=['autoencoder', 'ldm', 'unconditional', 'all'],
        help='验证模式: autoencoder（仅AE重建）, ldm（LDM重建）, unconditional（无条件生成）, all（全部）'
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
    
    logger.info("=" * 60)
    logger.info("验证完成！")
    logger.info(f"所有结果已保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

