"""
基于VQVAE的Diffusion Model训练脚本

基于MONAI Generative Models的3D Diffusion Model训练，
用于VQ-Latent Diffusion Model的第二阶段训练。

在VQVAE的潜在空间上训练Diffusion模型。
"""

import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml
import shutil

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism, first
from monai.networks.layers import Act
import numpy as np

from generative.inferers import DiffusionInferer
from generative.networks.nets import VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from monai_diffusion.datasets import create_train_val_dataloaders

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


def load_vqvae(checkpoint_path: str, device: torch.device, vqvae_config: dict):
    """
    加载预训练的VQVAE
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备
        vqvae_config: VQVAE配置字典
        
    Returns:
        VQVAE模型（评估模式）
    """
    # 构建采样参数
    downsample_params = vqvae_config.get('downsample_parameters', None)
    upsample_params = vqvae_config.get('upsample_parameters', None)
    
    if downsample_params is not None:
        downsample_parameters = tuple([tuple(p) for p in downsample_params])
        upsample_parameters = tuple([tuple(p) for p in upsample_params])
    else:
        num_channels = tuple(vqvae_config['num_channels'])
        downsample_parameters = tuple([(2, 4, 1, 1)] * (len(num_channels) - 1))
        upsample_parameters = tuple([(2, 4, 1, 1, 0)] * (len(num_channels) - 1))
    
    vqvae = VQVAE(
        spatial_dims=vqvae_config['spatial_dims'],
        in_channels=vqvae_config['in_channels'],
        out_channels=vqvae_config['out_channels'],
        num_channels=tuple(vqvae_config['num_channels']),
        num_res_channels=vqvae_config['num_res_channels'],
        num_res_layers=vqvae_config['num_res_layers'],
        downsample_parameters=downsample_parameters,
        upsample_parameters=upsample_parameters,
        num_embeddings=vqvae_config['num_embeddings'],
        embedding_dim=vqvae_config['embedding_dim'],
        act=Act.RELU,
        output_act=Act.RELU
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
    vqvae.to(device)
    vqvae.eval()
    
    logger.info(f"从 {checkpoint_path} 加载VQVAE")
    return vqvae


def compute_scale_factor(
    vqvae: torch.nn.Module,
    train_loader,
    device: torch.device
) -> float:
    """
    计算潜在空间的缩放因子
    
    根据Rombach et al. (2022)的建议，计算潜在空间的标准差
    作为缩放因子，以确保潜在空间分布接近标准正态分布。
    
    注意：VQVAE的潜在空间是量化的，但我们仍然可以计算缩放因子。
    """
    logger.info("计算潜在空间缩放因子...")
    
    vqvae.eval()
    with torch.no_grad():
        # 获取一个batch的数据
        check_data = first(train_loader)
        images = check_data["image"].to(device)
        
        with autocast(enabled=True):
            # VQVAE编码到潜在空间
            # 使用quantize=True获取量化后的潜在向量
            z = vqvae.encode(images)
            z, _ = vqvae.quantize(z)  # 返回 (quantized, loss)
        
        scale_factor = 1 / torch.std(z)
    
    logger.info(f"缩放因子: {scale_factor:.4f}")
    return scale_factor.item()


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scale_factor: float,
    best_val_loss: float,
    output_dir: str,
    is_best: bool = False
):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scale_factor': scale_factor,
        'best_val_loss': best_val_loss
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存最新checkpoint
    latest_path = output_path / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)
    logger.info(f"保存最新checkpoint到: {latest_path}")
    
    # 如果是最佳模型，保存best checkpoint
    if is_best:
        best_path = output_path / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"保存最佳模型到: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    """加载checkpoint恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    
    unet.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    scale_factor = checkpoint['scale_factor']
    
    logger.info(f"从checkpoint恢复训练: epoch {start_epoch}")
    return start_epoch, best_val_loss, scale_factor


def validate(
    unet: torch.nn.Module,
    vqvae: torch.nn.Module,
    scheduler,
    val_loader,
    device: torch.device,
    scale_factor: float
):
    """
    验证函数
    
    Args:
        unet: Diffusion UNet模型
        vqvae: VQVAE模型
        scheduler: 调度器
        val_loader: 验证数据加载器
        device: 设备
        scale_factor: 缩放因子
    """
    unet.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            
            with autocast(enabled=True):
                # 编码到潜在空间并量化
                z = vqvae.encode(images)
                z_quantized, _ = vqvae.quantize(z)
                
                # 应用缩放因子
                z_quantized = z_quantized * scale_factor
                
                # 生成随机噪声
                noise = torch.randn_like(z_quantized).to(device)
                
                # 随机时间步
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (images.shape[0],), device=device
                ).long()
                
                # 添加噪声
                noisy_latent = scheduler.add_noise(
                    original_samples=z_quantized,
                    noise=noise,
                    timesteps=timesteps
                )
                
                # 预测噪声
                noise_pred = unet(x=noisy_latent, timesteps=timesteps)
                
                loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()
    
    return val_loss / len(val_loader)


def visualize_samples(
    unet: torch.nn.Module,
    vqvae: torch.nn.Module,
    scheduler,
    val_loader,
    latent_shape: tuple,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    scale_factor: float,
    num_samples: int = 4,
    num_inference_steps: int = 1000
):
    """
    可视化生成样本与真实样本的对比
    
    Args:
        unet: Diffusion UNet模型
        vqvae: VQVAE模型
        scheduler: 调度器
        val_loader: 验证数据加载器
        latent_shape: 潜在空间形状
        device: 设备
        writer: TensorBoard writer
        epoch: 当前epoch
        scale_factor: 缩放因子
        num_samples: 可视化的样本数量
        num_inference_steps: 推理步数
    """
    unet.eval()
    vqvae.eval()
    
    with torch.no_grad():
        # 获取真实样本
        batch = next(iter(val_loader))
        real_images = batch["image"].to(device)[:num_samples]
        
        # 生成合成样本
        noise = torch.randn((num_samples, *latent_shape)).to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
        # 采样潜在空间
        synthetic_latent = noise
        for t in scheduler.timesteps:
            with autocast(enabled=True):
                model_output = unet(
                    x=synthetic_latent,
                    timesteps=torch.full((num_samples,), t, device=device, dtype=torch.long)
                )
                synthetic_latent, _ = scheduler.step(model_output, t, synthetic_latent)
        
        # 解码回图像空间（记得除以scale_factor）
        synthetic_latent = synthetic_latent / scale_factor
        synthetic_images = vqvae.decode(synthetic_latent)
        
        # 移到CPU并转换为numpy
        real_images_np = real_images.cpu().numpy()  # (B, C, H, W, D)
        synthetic_images_np = synthetic_images.cpu().numpy()
        
        # 对每个样本进行可视化
        for i in range(min(num_samples, real_images_np.shape[0])):
            # 取出单个样本 (C, H, W, D)
            real_vol = real_images_np[i, 0]  # (H, W, D)
            synthetic_vol = synthetic_images_np[i, 0]  # (H, W, D)
            
            # 将3D体素沿z轴投影成2D图像（累加所有z层）
            real_proj = np.sum(real_vol, axis=2)  # (H, W)
            synthetic_proj = np.sum(synthetic_vol, axis=2)  # (H, W)
            
            # 分别归一化真实样本和生成样本的投影
            real_proj = (real_proj - real_proj.min()) / (real_proj.max() - real_proj.min() + 1e-8)
            synthetic_proj = (synthetic_proj - synthetic_proj.min()) / (synthetic_proj.max() - synthetic_proj.min() + 1e-8)
            
            # 水平堆叠真实样本和生成样本
            combined = np.hstack([real_proj, synthetic_proj])  # (H, 2*W)
            
            # 添加到TensorBoard
            writer.add_image(
                f"vq_ldm_comparison/sample_{i}",
                combined,
                epoch,
                dataformats='HW'
            )
        
        logger.info(f"已保存 {min(num_samples, real_images_np.shape[0])} 个对比可视化结果到TensorBoard")


def train_diffusion(config_path: str):
    """
    训练基于VQVAE的Diffusion Model
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 提取配置参数
    diff_config = config['diffusion']
    checkpoint_config = diff_config['checkpoints']
    log_config = diff_config['logging']
    
    # 清理之前的输出目录
    output_dir = Path(checkpoint_config['output_dir'])
    log_dir = Path(log_config['log_dir'])
    
    if output_dir.exists():
        logger.info(f"删除之前的输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    
    if log_dir.exists():
        logger.info(f"删除之前的日志目录: {log_dir}")
        shutil.rmtree(log_dir)
    
    # 创建新的目录
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    set_determinism(config.get('seed', 42))
    
    # 设置设备
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}" if use_cuda else "cpu")
    mixed_precision = device_config.get('mixed_precision', True)
    logger.info(f"使用设备: {device}")
    logger.info(f"混合精度训练: {mixed_precision}")
    
    # 创建数据加载器
    train_config = diff_config['training']
    train_loader, val_loader = create_train_val_dataloaders(config, batch_size=train_config.get('batch_size', None))
    
    # 提取其他配置参数
    vqvae_config = config['vqvae']
    scheduler_config = diff_config['scheduler']
    
    # 加载预训练的VQVAE
    vqvae_path = checkpoint_config['vqvae_path']
    if not Path(vqvae_path).exists():
        raise FileNotFoundError(
            f"VQVAE checkpoint不存在: {vqvae_path}\n"
            "请先训练VQVAE或指定正确的checkpoint路径"
        )
    
    vqvae = load_vqvae(vqvae_path, device, vqvae_config)
    
    # 创建Diffusion Model UNet
    unet = DiffusionModelUNet(
        spatial_dims=diff_config['spatial_dims'],
        in_channels=diff_config['in_channels'],
        out_channels=diff_config['out_channels'],
        num_channels=tuple(diff_config['num_channels']),
        attention_levels=tuple(diff_config['attention_levels']),
        num_head_channels=tuple(diff_config['num_head_channels']),
        num_res_blocks=diff_config.get('num_res_blocks', 1)
    )
    unet.to(device)
    logger.info("创建DiffusionModelUNet")
    
    # 创建调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        schedule=scheduler_config['schedule'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end']
    )
    logger.info(f"创建DDPMScheduler: {scheduler_config['num_train_timesteps']} timesteps")
    
    # 计算缩放因子
    scale_factor = compute_scale_factor(vqvae, train_loader, device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        params=unet.parameters(),
        lr=train_config['learning_rate']
    )
    
    # 混合精度训练
    scaler = GradScaler() if mixed_precision else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练参数
    n_epochs = train_config['n_epochs']
    val_interval = train_config['val_interval']
    save_interval = train_config['save_interval']
    log_interval = log_config['log_interval']
    visualize_interval = log_config.get('visualize_interval', 10)
    num_visualize_samples = log_config.get('num_visualize_samples', 4)
    num_inference_steps = log_config.get('num_inference_steps', 1000)
    
    # 快速开发模式
    fast_dev_run = train_config.get('fast_dev_run', False)
    fast_dev_run_batches = train_config.get('fast_dev_run_batches', 2)
    
    if fast_dev_run:
        logger.info(f"**快速开发模式**: 每个epoch只运行 {fast_dev_run_batches} 个batch")
        n_epochs = 2
        val_interval = 1
        save_interval = 1
        log_interval = 1
        visualize_interval = 1
        num_visualize_samples = 2
        num_inference_steps = 50  # 快速模式使用更少的推理步数
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    resume_from = checkpoint_config.get('resume_from')
    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_loss, loaded_scale_factor = load_checkpoint(
            resume_from, unet, optimizer
        )
        # 使用加载的scale_factor
        scale_factor = loaded_scale_factor
        logger.info(f"使用加载的scale_factor: {scale_factor:.4f}")
    
    # 获取潜在空间形状（用于生成样本）
    with torch.no_grad():
        check_data = first(train_loader)
        z = vqvae.encode(check_data["image"].to(device))
        z_quantized, _ = vqvae.quantize(z)
        latent_shape = z_quantized.shape[1:]  # (C, D, H, W)
    
    logger.info(f"潜在空间形状: {latent_shape}")
    
    # 训练循环
    logger.info(f"开始训练VQ-Latent Diffusion Model: {n_epochs} epochs")
    
    for epoch in range(start_epoch, n_epochs):
        unet.train()
        vqvae.eval()
        
        epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}/{n_epochs}")
        
        for step, batch in progress_bar:
            # 快速开发模式：只运行指定数量的batch
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=mixed_precision):
                # 编码到潜在空间并量化
                with torch.no_grad():
                    z = vqvae.encode(images)
                    z_quantized, _ = vqvae.quantize(z)
                    
                    # 应用缩放因子
                    z_quantized = z_quantized * scale_factor
                
                # 生成随机噪声
                noise = torch.randn_like(z_quantized).to(device)
                
                # 随机时间步
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (images.shape[0],), device=images.device
                ).long()
                
                # 添加噪声到潜在表示
                noisy_latent = scheduler.add_noise(
                    original_samples=z_quantized,
                    noise=noise,
                    timesteps=timesteps
                )
                
                # 预测噪声
                noise_pred = unet(x=noisy_latent, timesteps=timesteps)
                
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})
            
            # TensorBoard日志
            if step % log_interval == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/step/loss", loss.item(), global_step)
        
        # 记录epoch平均损失
        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        writer.add_scalar("train/epoch/loss", avg_loss, epoch)
        logger.info(f"Epoch {epoch} 训练损失: {avg_loss:.4f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
            val_loss = validate(unet, vqvae, scheduler, val_loader, device, scale_factor)
            writer.add_scalar("val/epoch/loss", val_loss, epoch)
            logger.info(f"Epoch {epoch} 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"新的最佳验证损失: {best_val_loss:.4f}")
        else:
            is_best = False
        
        # 可视化生成样本与真实样本的对比
        if (epoch + 1) % visualize_interval == 0 or epoch == n_epochs - 1:
            logger.info("生成对比可视化结果...")
            visualize_samples(
                unet, vqvae, scheduler, val_loader,
                latent_shape, device, writer, epoch, scale_factor,
                num_visualize_samples, num_inference_steps
            )
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1 or is_best:
            save_checkpoint(
                epoch=epoch,
                unet=unet,
                optimizer=optimizer,
                scale_factor=scale_factor,
                best_val_loss=best_val_loss,
                output_dir=checkpoint_config['output_dir'],
                is_best=is_best
            )
    
    writer.close()
    logger.info("VQ-Latent Diffusion Model训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练基于VQVAE的Diffusion Model")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/vq_ldm_config_local.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    train_diffusion(args.config)


if __name__ == "__main__":
    main()

