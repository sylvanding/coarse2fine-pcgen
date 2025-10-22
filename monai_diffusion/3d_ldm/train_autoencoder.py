"""
AutoencoderKL训练脚本

基于MONAI Generative Models的3D AutoencoderKL训练，
用于Latent Diffusion Model的第一阶段训练。
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
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from monai.utils import set_determinism
import numpy as np

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

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


def KL_loss(z_mu, z_sigma):
    """KL散度损失"""
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=[1, 2, 3, 4]
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def save_checkpoint(
    epoch: int,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    best_val_loss: float,
    output_dir: str,
    is_best: bool = False
):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'autoencoder_state_dict': autoencoder.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
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
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer
):
    """加载checkpoint恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    logger.info(f"从checkpoint恢复训练: epoch {start_epoch}")
    return start_epoch, best_val_loss


def validate(
    autoencoder: torch.nn.Module,
    val_loader,
    device: torch.device,
    kl_weight: float,
    fast_dev_run: bool,
    fast_dev_run_batches: int
):
    """验证函数"""
    autoencoder.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    
    progress_bar = tqdm(val_loader, total=len(val_loader), ncols=120)
    progress_bar.set_description(f"Validating...")
    
    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            
            reconstruction, z_mu, z_sigma = autoencoder(images)
            
            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            kl = KL_loss(z_mu, z_sigma)
            
            loss = recons_loss + kl_weight * kl
            
            val_loss += loss.item()
            val_recon_loss += recons_loss.item()
            val_kl_loss += kl.item()
            
            progress_bar.set_postfix({
                "loss": f"{val_loss / (step + 1):.4f}",
                "recon": f"{val_recon_loss / (step + 1):.4f}",
                "kl": f"{val_kl_loss / (step + 1):.4f}"
            })
    n_batches = len(val_loader)
    return val_loss / n_batches, val_recon_loss / n_batches, val_kl_loss / n_batches


def visualize_reconstruction(
    autoencoder: torch.nn.Module,
    val_loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_samples: int = 4
):
    """
    可视化重建结果
    
    将验证集的前num_samples个样本的输入和重建结果在z方向堆叠，
    传递给TensorBoard进行可视化。
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器
        device: 设备
        writer: TensorBoard writer
        epoch: 当前epoch
        num_samples: 可视化的样本数量
    """
    autoencoder.eval()
    
    with torch.no_grad():
        # 获取第一个batch
        batch = next(iter(val_loader))
        images = batch["image"].to(device)
        
        # 只取前num_samples个样本
        images = images[:num_samples]
        
        # 通过autoencoder获取重建结果
        reconstruction, _, _ = autoencoder(images)
        
        # 移到CPU并转换为numpy
        images_np = images.cpu().numpy()  # (B, C, H, W, D)
        reconstruction_np = reconstruction.cpu().numpy()
        
        # 对每个样本进行可视化
        for i in range(min(num_samples, images_np.shape[0])):
            # 取出单个样本 (C, H, W, D)
            input_vol = images_np[i, 0]  # (H, W, D)
            recon_vol = reconstruction_np[i, 0]  # (H, W, D)
            
            # 将3D体素沿z轴投影成2D图像（累加所有z层）
            input_proj = np.sum(input_vol, axis=2)  # (H, W)
            recon_proj = np.sum(recon_vol, axis=2)  # (H, W)
            
            # 分别归一化输入和重建投影
            input_proj = (input_proj - input_proj.min()) / (input_proj.max() - input_proj.min() + 1e-8)
            recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
            
            # 垂直堆叠输入和重建结果
            combined = np.hstack([input_proj, recon_proj])  # (H, 2*W)
            
            # 添加到TensorBoard
            writer.add_image(
                f"reconstruction/sample_{i}",
                combined,
                epoch,
                dataformats='HW'
            )
        
        logger.info(f"已保存 {min(num_samples, images_np.shape[0])} 个重建可视化结果到TensorBoard")


def train_autoencoder(config_path: str):
    """
    训练AutoencoderKL
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 提取配置参数
    ae_config = config['autoencoder']
    checkpoint_config = ae_config['checkpoints']
    log_config = ae_config['logging']
    
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
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_train_val_dataloaders(config)
    
    # 提取训练配置参数
    train_config = ae_config['training']
    
    # 创建AutoencoderKL
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
    
    # 启用梯度检查点以节省显存
    if train_config.get('use_gradient_checkpointing', False):
        logger.info("启用梯度检查点（Gradient Checkpointing）")
        if hasattr(autoencoder, 'enable_gradient_checkpointing'):
            autoencoder.enable_gradient_checkpointing()
    
    autoencoder.to(device)
    logger.info("创建AutoencoderKL模型")
    
    # 创建判别器
    disc_config = train_config['discriminator']
    discriminator = PatchDiscriminator(
        spatial_dims=ae_config['spatial_dims'],
        num_layers_d=disc_config['num_layers_d'],
        num_channels=disc_config['num_channels'],
        in_channels=ae_config['in_channels'],
        out_channels=ae_config['out_channels']
    )
    discriminator.to(device)
    logger.info("创建PatchDiscriminator")
    
    # 定义损失函数
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    
    # 判断是否使用感知损失（大分辨率时可以禁用以节省显存）
    use_perceptual_loss = train_config.get('use_perceptual_loss', True)
    if use_perceptual_loss:
        loss_perceptual = PerceptualLoss(
            spatial_dims=ae_config['spatial_dims'],
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2
        )
        loss_perceptual.to(device)
        logger.info("启用感知损失（PerceptualLoss）")
    else:
        loss_perceptual = None
        logger.info("禁用感知损失以节省显存")
    
    # 损失权重
    adv_weight = train_config['adv_weight']
    perceptual_weight = train_config.get('perceptual_weight', 0.001) if use_perceptual_loss else 0.0
    kl_weight = train_config['kl_weight']
    autoencoder_warm_up_n_epochs = train_config['autoencoder_warm_up_n_epochs']
    
    # 创建优化器
    optimizer_g = torch.optim.Adam(
        params=autoencoder.parameters(),
        lr=train_config['learning_rate']
    )
    optimizer_d = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=train_config['learning_rate']
    )
    
    # 混合精度训练
    use_amp = device_config.get('mixed_precision', False) and torch.cuda.is_available()
    if use_amp:
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        logger.info("启用混合精度训练（AMP）")
    else:
        scaler_g = None
        scaler_d = None
        logger.info("使用FP32训练")
    
    # 梯度累积步数
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        logger.info(f"使用梯度累积: {gradient_accumulation_steps} 步")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练参数
    n_epochs = train_config['n_epochs']
    val_interval = train_config['val_interval']
    save_interval = train_config['save_interval']
    log_interval = log_config['log_interval']
    visualize_interval = log_config.get('visualize_interval', 10)  # 默认每10个epoch可视化一次
    num_visualize_samples = log_config.get('num_visualize_samples', 4)  # 默认可视化4个样本
    
    # 快速开发模式
    fast_dev_run = train_config.get('fast_dev_run', False)
    fast_dev_run_batches = train_config.get('fast_dev_run_batches', 2)
    
    if fast_dev_run:
        logger.info(f"**快速开发模式**: 每个epoch只运行 {fast_dev_run_batches} 个batch")
        n_epochs = 5  # 快速模式只运行2个epoch
        val_interval = 1
        save_interval = 1
        log_interval = 1
        visualize_interval = 1
        num_visualize_samples = 2
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    resume_from = checkpoint_config.get('resume_from')
    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, autoencoder, discriminator, optimizer_g, optimizer_d
        )
    
    # 训练循环
    logger.info(f"开始训练AutoencoderKL: {n_epochs} epochs")
    
    for epoch in range(start_epoch, n_epochs):
        autoencoder.train()
        discriminator.train()
        
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}/{n_epochs}")
        
        for step, batch in progress_bar:
            # 快速开发模式：只运行指定数量的batch
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            
            # ============ Generator部分 ============
            # 梯度清零（在累积开始时）
            if step % gradient_accumulation_steps == 0:
                optimizer_g.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            with autocast(enabled=use_amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                kl = KL_loss(z_mu, z_sigma)
                
                recons_loss = l1_loss(reconstruction.float(), images.float())
                
                # 感知损失（如果启用）
                if use_perceptual_loss:
                    p_loss = loss_perceptual(reconstruction.float(), images.float())
                    loss_g = recons_loss + kl_weight * kl + perceptual_weight * p_loss
                else:
                    loss_g = recons_loss + kl_weight * kl
                
                # 对抗损失（warm-up后）
                generator_loss_val = 0.0
                if epoch >= autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss
                    generator_loss_val = generator_loss.item()
                    gen_epoch_loss += generator_loss_val
                
                # 梯度累积：除以累积步数
                loss_g = loss_g / gradient_accumulation_steps
            
            # 反向传播
            if use_amp:
                scaler_g.scale(loss_g).backward()
                # 在累积结束时更新参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
            else:
                loss_g.backward()
                # 在累积结束时更新参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer_g.step()
            
            # ============ Discriminator部分 ============
            discriminator_loss_val = 0.0
            if epoch >= autoencoder_warm_up_n_epochs:
                # 梯度清零（在累积开始时）
                if step % gradient_accumulation_steps == 0:
                    optimizer_d.zero_grad(set_to_none=True)
                
                with autocast(enabled=use_amp):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss / gradient_accumulation_steps
                    discriminator_loss_val = discriminator_loss.item()
                
                # 反向传播
                if use_amp:
                    scaler_d.scale(loss_d).backward()
                    # 在累积结束时更新参数
                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                else:
                    loss_d.backward()
                    # 在累积结束时更新参数
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer_d.step()
                
                disc_epoch_loss += discriminator_loss_val
            
            # 记录损失（注意loss_g已经除以了gradient_accumulation_steps）
            epoch_loss += loss_g.item() * gradient_accumulation_steps
            epoch_recon_loss += recons_loss.item()
            epoch_kl_loss += kl.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{epoch_loss / (step + 1):.4f}",
                "recon": f"{epoch_recon_loss / (step + 1):.4f}",
                "kl": f"{epoch_kl_loss / (step + 1):.4f}",
                "gen": f"{gen_epoch_loss / (step + 1):.4f}" if epoch >= autoencoder_warm_up_n_epochs else "N/A",
                "disc": f"{disc_epoch_loss / (step + 1):.4f}" if epoch >= autoencoder_warm_up_n_epochs else "N/A"
            })
            
            # TensorBoard日志
            if step % log_interval == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/step/total_loss", loss_g.item() * gradient_accumulation_steps, global_step)
                writer.add_scalar("train/step/recon_loss", recons_loss.item(), global_step)
                writer.add_scalar("train/step/kl_loss", kl.item(), global_step)
                if epoch >= autoencoder_warm_up_n_epochs:
                    writer.add_scalar("train/step/gen_loss", generator_loss_val, global_step)
                    writer.add_scalar("train/step/disc_loss", discriminator_loss_val, global_step)
        
        # 记录epoch平均损失
        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_recon = epoch_recon_loss / n_steps
        avg_kl = epoch_kl_loss / n_steps
        
        writer.add_scalar("train/epoch/total_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch/recon_loss", avg_recon, epoch)
        writer.add_scalar("train/epoch/kl_loss", avg_kl, epoch)
        
        logger.info(f"Epoch {epoch} 训练损失: total={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
            val_loss, val_recon, val_kl = validate(
                autoencoder, val_loader, device, kl_weight, fast_dev_run, fast_dev_run_batches
            )
            
            writer.add_scalar("val/epoch/total_loss", val_loss, epoch)
            writer.add_scalar("val/epoch/recon_loss", val_recon, epoch)
            writer.add_scalar("val/epoch/kl_loss", val_kl, epoch)
            
            logger.info(f"Epoch {epoch} 验证损失: total={val_loss:.4f}, recon={val_recon:.4f}, kl={val_kl:.4f}")
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"新的最佳验证损失: {best_val_loss:.4f}")
        else:
            is_best = False
        
        # 可视化重建结果
        if (epoch + 1) % visualize_interval == 0 or epoch == n_epochs - 1:
            logger.info("生成重建可视化结果...")
            visualize_reconstruction(
                autoencoder, val_loader, device, writer, epoch, num_visualize_samples
            )
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1 or is_best:
            save_checkpoint(
                epoch=epoch,
                autoencoder=autoencoder,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                best_val_loss=best_val_loss,
                output_dir=checkpoint_config['output_dir'],
                is_best=is_best
            )
    
    writer.close()
    logger.info("AutoencoderKL训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练AutoencoderKL")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/ldm_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    train_autoencoder(args.config)


if __name__ == "__main__":
    main()

