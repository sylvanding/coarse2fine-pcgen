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

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism

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
    kl_weight: float
):
    """验证函数"""
    autoencoder.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            
            reconstruction, z_mu, z_sigma = autoencoder(images)
            
            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            kl = KL_loss(z_mu, z_sigma)
            
            loss = recons_loss + kl_weight * kl
            
            val_loss += loss.item()
            val_recon_loss += recons_loss.item()
            val_kl_loss += kl.item()
    
    n_batches = len(val_loader)
    return val_loss / n_batches, val_recon_loss / n_batches, val_kl_loss / n_batches


def train_autoencoder(config_path: str):
    """
    训练AutoencoderKL
    
    Args:
        config_path: 配置文件路径
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
    
    # 创建数据加载器
    train_loader, val_loader = create_train_val_dataloaders(config)
    
    # 提取配置参数
    ae_config = config['autoencoder']
    train_config = ae_config['training']
    checkpoint_config = ae_config['checkpoints']
    log_config = ae_config['logging']
    
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
    loss_perceptual = PerceptualLoss(
        spatial_dims=ae_config['spatial_dims'],
        network_type="squeeze",
        is_fake_3d=True,
        fake_3d_ratio=0.2
    )
    loss_perceptual.to(device)
    
    # 损失权重
    adv_weight = train_config['adv_weight']
    perceptual_weight = train_config['perceptual_weight']
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
    
    # TensorBoard
    log_dir = Path(log_config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练参数
    n_epochs = train_config['n_epochs']
    val_interval = train_config['val_interval']
    save_interval = train_config['save_interval']
    log_interval = log_config['log_interval']
    
    # 快速开发模式
    fast_dev_run = train_config.get('fast_dev_run', False)
    fast_dev_run_batches = train_config.get('fast_dev_run_batches', 2)
    
    if fast_dev_run:
        logger.info(f"**快速开发模式**: 每个epoch只运行 {fast_dev_run_batches} 个batch")
        n_epochs = 2  # 快速模式只运行2个epoch
    
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
            optimizer_g.zero_grad(set_to_none=True)
            
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl = KL_loss(z_mu, z_sigma)
            
            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl + perceptual_weight * p_loss
            
            # 对抗损失（warm-up后）
            if epoch >= autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss
                gen_epoch_loss += generator_loss.item()
            
            loss_g.backward()
            optimizer_g.step()
            
            # ============ Discriminator部分 ============
            if epoch >= autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss
                
                loss_d.backward()
                optimizer_d.step()
                
                disc_epoch_loss += discriminator_loss.item()
            
            # 记录损失
            epoch_loss += loss_g.item()
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
                writer.add_scalar("train/total_loss", loss_g.item(), global_step)
                writer.add_scalar("train/recon_loss", recons_loss.item(), global_step)
                writer.add_scalar("train/kl_loss", kl.item(), global_step)
                if epoch >= autoencoder_warm_up_n_epochs:
                    writer.add_scalar("train/gen_loss", generator_loss.item(), global_step)
                    writer.add_scalar("train/disc_loss", discriminator_loss.item(), global_step)
        
        # 记录epoch平均损失
        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_recon = epoch_recon_loss / n_steps
        avg_kl = epoch_kl_loss / n_steps
        
        writer.add_scalar("epoch/train_loss", avg_loss, epoch)
        writer.add_scalar("epoch/train_recon", avg_recon, epoch)
        writer.add_scalar("epoch/train_kl", avg_kl, epoch)
        
        logger.info(f"Epoch {epoch} 训练损失: total={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
            val_loss, val_recon, val_kl = validate(
                autoencoder, val_loader, device, kl_weight
            )
            
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_recon", val_recon, epoch)
            writer.add_scalar("epoch/val_kl", val_kl, epoch)
            
            logger.info(f"Epoch {epoch} 验证损失: total={val_loss:.4f}, recon={val_recon:.4f}, kl={val_kl:.4f}")
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"新的最佳验证损失: {best_val_loss:.4f}")
        else:
            is_best = False
        
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

