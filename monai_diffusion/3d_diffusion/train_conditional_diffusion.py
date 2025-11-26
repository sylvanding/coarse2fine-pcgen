"""
条件3D扩散模型训练脚本

使用2D投影图像作为条件来指导3D体素的生成。
基于MONAI Generative Models的3D Diffusion Model。
"""

import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml
import shutil

# 添加GenerativeModels和项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
import numpy as np

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# 导入条件数据集
from conditional_dataset import create_train_val_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConditionalDiffusionUNet(nn.Module):
    """
    条件扩散U-Net
    
    在标准的DiffusionModelUNet基础上，添加2D条件图像的编码器。
    2D条件图像通过卷积编码后，作为cross-attention的context输入到U-Net。
    """
    
    def __init__(
        self,
        # 3D UNet参数
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: tuple = (64, 128, 256),
        attention_levels: tuple = (False, False, True),
        num_res_blocks: int = 2,
        num_head_channels: tuple = (0, 0, 64),
        # 2D条件编码器参数
        condition_channels: int = 1,
        condition_embed_dim: int = 256,
    ):
        """
        初始化条件扩散U-Net
        
        Args:
            spatial_dims: 空间维度（3D）
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_channels: U-Net各层通道数
            attention_levels: 各层是否使用注意力
            num_res_blocks: 残差块数量
            num_head_channels: 注意力头通道数
            condition_channels: 条件图像通道数（2D）
            condition_embed_dim: 条件嵌入维度，也是cross_attention_dim
        """
        super().__init__()
        
        self.condition_embed_dim = condition_embed_dim
        
        # 3D扩散U-Net，启用条件输入
        # 关键：必须指定 cross_attention_dim 参数
        # norm_num_groups: 所有num_channels必须是它的倍数，默认32，这里设为8以支持[16,32,64]
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            norm_num_groups=8,  # 确保所有num_channels都能被8整除
            with_conditioning=True,
            cross_attention_dim=condition_embed_dim,  # 必须指定！
        )
        
        # 2D条件编码器（简单的2D CNN）
        self.condition_encoder = nn.Sequential(
            # (1, H, W) -> (64, H/2, W/2)
            nn.Conv2d(condition_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # (64, H/2, W/2) -> (128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            
            # (128, H/4, W/4) -> (256, H/8, W/8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            
            # 全局平均池化 -> (256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (256,)
        )
        
        # 条件嵌入投影（投影到cross_attention_dim维度）
        self.condition_proj = nn.Sequential(
            nn.Linear(256, condition_embed_dim),
            nn.SiLU(),
            nn.Linear(condition_embed_dim, condition_embed_dim),
        )
        
        logger.info("初始化ConditionalDiffusionUNet:")
        logger.info(f"  3D UNet通道: {num_channels}")
        logger.info(f"  条件嵌入维度 (cross_attention_dim): {condition_embed_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 噪声3D体素 (B, C, H, W, D)
            timesteps: 时间步 (B,)
            condition: 2D条件图像 (B, C, H, W)
            
        Returns:
            预测的噪声 (B, C, H, W, D)
        """
        # 编码条件图像
        condition_feat = self.condition_encoder(condition)  # (B, 256)
        condition_embed = self.condition_proj(condition_feat)  # (B, condition_embed_dim)
        
        # 将条件嵌入作为context传递给UNet的cross-attention层
        # context shape: (B, seq_len, cross_attention_dim)
        # 这里seq_len=1，因为是单个全局条件向量
        context = condition_embed.unsqueeze(1)  # (B, 1, condition_embed_dim)
        
        # 调用UNet，context会被传递到cross-attention层
        output = self.unet(x=x, timesteps=timesteps, context=context)
        
        return output


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    output_dir: str,
    is_best: bool = False
):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer
):
    """加载checkpoint恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    logger.info(f"从checkpoint恢复训练: epoch {start_epoch}")
    return start_epoch, best_val_loss


def validate(
    model: nn.Module,
    scheduler,
    val_loader,
    device: torch.device
):
    """
    验证函数
    
    Args:
        model: 条件扩散模型
        scheduler: 调度器
        val_loader: 验证数据加载器
        device: 设备
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            conditions = batch["projection"].to(device)
            
            with autocast(enabled=True):
                # 生成随机噪声
                noise = torch.randn_like(images).to(device)
                
                # 随机时间步
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (images.shape[0],), device=device
                ).long()
                
                # 添加噪声
                noisy_images = scheduler.add_noise(
                    original_samples=images,
                    noise=noise,
                    timesteps=timesteps
                )
                
                # 预测噪声
                noise_pred = model(x=noisy_images, timesteps=timesteps, condition=conditions)
                
                loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()
    
    return val_loss / len(val_loader)


def visualize_samples(
    model: nn.Module,
    scheduler,
    val_loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_samples: int = 4,
    num_inference_steps: int = 1000
):
    """
    可视化生成样本与真实样本的对比
    
    Args:
        model: 条件扩散模型
        scheduler: 调度器
        val_loader: 验证数据加载器
        device: 设备
        writer: TensorBoard writer
        epoch: 当前epoch
        num_samples: 可视化的样本数量
        num_inference_steps: 推理步数
    """
    model.eval()
    
    with torch.no_grad():
        # 获取真实样本和条件
        batch = next(iter(val_loader))
        real_images = batch["image"].to(device)[:num_samples]
        conditions = batch["projection"].to(device)[:num_samples]
        
        # 获取实际的batch size（可能小于num_samples）
        actual_batch_size = real_images.shape[0]
        logger.info(f"可视化样本数量: {actual_batch_size}")
        
        # 生成合成样本
        noise = torch.randn_like(real_images).to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
        # 采样（使用条件）
        synthetic_images = noise
        for t in scheduler.timesteps:
            # 确保t是int类型（MONAI的scheduler.step需要int）
            t_int = int(t) if isinstance(t, torch.Tensor) else t
            
            with autocast(enabled=True):
                model_output = model(
                    x=synthetic_images,
                    timesteps=torch.full((actual_batch_size,), t_int, device=device, dtype=torch.long),
                    condition=conditions
                )
                synthetic_images, _ = scheduler.step(model_output, t_int, synthetic_images)
        
        # 移到CPU并转换为numpy
        real_images_np = real_images.cpu().numpy()  # (B, C, H, W, D)
        synthetic_images_np = synthetic_images.cpu().numpy()
        conditions_np = conditions.cpu().numpy()  # (B, C, H, W)
        
        # 对每个样本进行可视化
        for i in range(min(num_samples, real_images_np.shape[0])):
            # 取出单个样本 (C, H, W, D)
            real_vol = real_images_np[i, 0]  # (H, W, D)
            synthetic_vol = synthetic_images_np[i, 0]  # (H, W, D)
            condition_img = conditions_np[i, 0]  # (H, W)
            
            # 将3D体素沿z轴投影成2D图像（累加所有z层）
            real_proj = np.sum(real_vol, axis=2)  # (H, W)
            synthetic_proj = np.sum(synthetic_vol, axis=2)  # (H, W)
            
            # 分别归一化
            real_proj = (real_proj - real_proj.min()) / (real_proj.max() - real_proj.min() + 1e-8)
            synthetic_proj = (synthetic_proj - synthetic_proj.min()) / (synthetic_proj.max() - synthetic_proj.min() + 1e-8)
            
            # 水平堆叠：条件图像 | 真实投影 | 生成投影
            combined = np.hstack([condition_img, real_proj, synthetic_proj])  # (H, 3*W)
            
            # 添加到TensorBoard
            writer.add_image(
                f"conditional_diffusion/sample_{i}",
                combined,
                epoch,
                dataformats='HW'
            )
        
        logger.info(f"已保存 {min(num_samples, real_images_np.shape[0])} 个对比可视化结果到TensorBoard")


def train_conditional_diffusion(config_path: str):
    """
    训练条件3D扩散模型
    
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
    train_loader, val_loader = create_train_val_dataloaders(
        config,
        batch_size=train_config.get('batch_size', None)
    )
    
    # 提取其他配置参数
    scheduler_config = diff_config['scheduler']
    
    # 创建条件扩散模型
    model = ConditionalDiffusionUNet(
        spatial_dims=diff_config['spatial_dims'],
        in_channels=diff_config['in_channels'],
        out_channels=diff_config['out_channels'],
        num_channels=tuple(diff_config['num_channels']),
        attention_levels=tuple(diff_config['attention_levels']),
        num_res_blocks=diff_config.get('num_res_blocks', 2),
        num_head_channels=tuple(diff_config['num_head_channels']),
        condition_channels=diff_config.get('condition_channels', 1),
        condition_embed_dim=diff_config.get('condition_embed_dim', 256),
    )
    model.to(device)
    logger.info("创建ConditionalDiffusionUNet")
    
    # 创建调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        schedule=scheduler_config['schedule'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end']
    )
    logger.info(f"创建DDPMScheduler: {scheduler_config['num_train_timesteps']} timesteps")
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        params=model.parameters(),
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
        num_inference_steps = 50
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    resume_from = checkpoint_config.get('resume_from')
    if resume_from and Path(resume_from).exists():
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer)
    
    # 训练循环
    logger.info(f"开始训练条件3D扩散模型: {n_epochs} epochs")
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}/{n_epochs}")
        
        for step, batch in progress_bar:
            # 快速开发模式：只运行指定数量的batch
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            conditions = batch["projection"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=mixed_precision):
                # 生成随机噪声
                noise = torch.randn_like(images).to(device)
                
                # 随机时间步
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (images.shape[0],), device=images.device
                ).long()
                
                # 添加噪声到图像
                noisy_images = scheduler.add_noise(
                    original_samples=images,
                    noise=noise,
                    timesteps=timesteps
                )
                
                # 预测噪声（使用条件）
                noise_pred = model(x=noisy_images, timesteps=timesteps, condition=conditions)
                
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
            val_loss = validate(model, scheduler, val_loader, device)
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
                model, scheduler, val_loader, device, writer, epoch,
                num_visualize_samples, num_inference_steps
            )
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1 or is_best:
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                output_dir=checkpoint_config['output_dir'],
                is_best=is_best
            )
    
    writer.close()
    logger.info("条件3D扩散模型训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练条件3D扩散模型（2D图像指导）")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/conditional_diffusion_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    train_conditional_diffusion(args.config)


if __name__ == "__main__":
    main()

