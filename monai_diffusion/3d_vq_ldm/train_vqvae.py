"""
VQVAE训练脚本

基于MONAI Generative Models的3D VQVAE训练，
用于VQ-Latent Diffusion Model的第一阶段训练。

VQVAE使用离散的量化潜在空间（codebook），通过向量量化学习数据的紧凑表示。
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml
import shutil
from PIL import Image
from torch.utils.data import DataLoader

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from monai.utils import set_determinism
from monai import transforms
from monai.networks.layers import Act
import numpy as np

from generative.networks.nets import VQVAE

from monai_diffusion.datasets import create_train_val_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 损失函数定义 ====================

class DiceLoss(nn.Module):
    """
    Dice Loss
    
    用于分割任务的损失函数，关注前景区域的重叠度。
    适用于稀疏体素数据（如微管）。
    """
    def __init__(self, smooth: float = 1e-5, sigmoid: bool = False):
        """
        Args:
            smooth: 平滑项，避免除零
            sigmoid: 是否对输入应用sigmoid激活
        """
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        
        Args:
            pred: 预测张量 (B, C, H, W, D)
            target: 目标张量 (B, C, H, W, D)
            
        Returns:
            Dice损失值（标量）
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        
        # 展平batch和channel维度
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss = 1 - Dice系数
        return 1.0 - dice.mean()


class WeightedReconstructionLoss(nn.Module):
    """
    加权重建损失（L1或MSE）
    
    对前景像素（微管）给予更高的权重，对背景像素给予更低的权重。
    这样可以迫使模型关注前景，避免预测全黑。
    """
    def __init__(
        self,
        loss_type: str = "l1",  # "l1" or "mse"
        foreground_weight: float = 10.0,
        background_weight: float = 1.0,
        threshold: float = 0.1,  # 用于区分前景和背景的阈值
    ):
        """
        Args:
            loss_type: 损失类型，"l1" 或 "mse"
            foreground_weight: 前景像素权重（应该 >> 1）
            background_weight: 背景像素权重（通常为1.0）
            threshold: 像素值阈值，高于此值认为是前景
        """
        super().__init__()
        self.loss_type = loss_type
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.threshold = threshold
        
        logger.info(f"初始化加权重建损失:")
        logger.info(f"  损失类型: {loss_type}")
        logger.info(f"  前景权重: {foreground_weight}x")
        logger.info(f"  背景权重: {background_weight}x")
        logger.info(f"  前景阈值: {threshold}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权重建损失
        
        Args:
            pred: 预测张量 (B, C, H, W, D)
            target: 目标张量 (B, C, H, W, D)
            
        Returns:
            加权损失值（标量）
        """
        # 计算逐像素误差
        if self.loss_type == "l1":
            pixel_loss = torch.abs(pred - target)
        elif self.loss_type == "mse":
            pixel_loss = (pred - target) ** 2
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        # 创建权重mask：根据目标图像区分前景和背景
        # 注意：这里使用目标图像判断前景/背景
        foreground_mask = (target > self.threshold).float()
        background_mask = 1.0 - foreground_mask
        
        # 应用权重
        weighted_loss = (
            pixel_loss * foreground_mask * self.foreground_weight +
            pixel_loss * background_mask * self.background_weight
        )
        
        # 返回平均损失
        return weighted_loss.mean()


class CombinedReconstructionLoss(nn.Module):
    """
    组合损失：Dice Loss + 加权重建损失
    
    结合两种损失的优势：
    - Dice Loss: 关注前景区域的整体重叠度
    - 加权重建损失: 对前景像素给予更高的逐像素重建权重
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        recon_weight: float = 1.0,
        recon_loss_type: str = "l1",
        foreground_weight: float = 10.0,
        background_weight: float = 1.0,
        threshold: float = 0.1,
        dice_smooth: float = 1e-5,
    ):
        """
        Args:
            dice_weight: Dice Loss的权重
            recon_weight: 重建损失的权重
            recon_loss_type: 重建损失类型 "l1" 或 "mse"
            foreground_weight: 前景像素权重
            background_weight: 背景像素权重
            threshold: 前景阈值
            dice_smooth: Dice平滑项
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.recon_weight = recon_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth, sigmoid=False)
        self.weighted_recon_loss = WeightedReconstructionLoss(
            loss_type=recon_loss_type,
            foreground_weight=foreground_weight,
            background_weight=background_weight,
            threshold=threshold
        )
        
        logger.info(f"初始化组合重建损失:")
        logger.info(f"  Dice Loss权重: {dice_weight}")
        logger.info(f"  重建损失权重: {recon_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测张量 (B, C, H, W, D)
            target: 目标张量 (B, C, H, W, D)
            
        Returns:
            组合损失值（标量）
        """
        dice = self.dice_loss(pred, target)
        recon = self.weighted_recon_loss(pred, target)
        
        return self.dice_weight * dice + self.recon_weight * recon


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_data_loader(data_loader: DataLoader, data_name: str):
    """可视化数据加载器的第一个batch"""
    cpu_device = torch.device("cpu")
    batch = next(iter(data_loader))
    images = batch["image"].to(cpu_device)
    
    logger.info(f"输入patch形状: {images.shape}")
    
    images_np = images.numpy()  # (B, C, H, W, D)
    
    file_path = f"outputs/images_for_data_loader/{data_name}_vqvae"
    os.makedirs(file_path, exist_ok=True)
    
    # 可视化结果
    for i in range(images_np.shape[0]):
        # 取出单个样本 (C, H, W, D)
        input_vol = images_np[i, 0]  # (H, W, D)
        
        # 将3D体素沿z轴投影成2D图像（累加所有z层）
        input_proj = np.sum(input_vol, axis=2)  # (H, W)
        
        # 归一化
        input_proj = (input_proj - input_proj.min()) / (input_proj.max() - input_proj.min() + 1e-8)
        
        # save image
        image = Image.fromarray(np.uint8(input_proj * 255.0))
        image.save(f"{file_path}/sample_{i}.png")
        logger.info(f"保存样本 {i} 到 {file_path}/sample_{i}.png")


def save_checkpoint(
    epoch: int,
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    output_dir: str,
    is_best: bool = False
):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'vqvae_state_dict': vqvae.state_dict(),
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
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    """加载checkpoint恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    
    vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    logger.info(f"从checkpoint恢复训练: epoch {start_epoch}")
    return start_epoch, best_val_loss


def validate(
    vqvae: torch.nn.Module,
    val_loader,
    device: torch.device,
    recon_loss_fn: torch.nn.Module,
    fast_dev_run: bool = False,
    fast_dev_run_batches: int = 2
):
    """
    验证函数
    
    Args:
        vqvae: VQVAE模型
        val_loader: 验证数据加载器
        device: 设备
        recon_loss_fn: 重建损失函数
        fast_dev_run: 是否快速开发模式
        fast_dev_run_batches: 快速开发模式运行的batch数
    """
    vqvae.eval()
    val_loss = 0
    val_recon_loss = 0
    val_quant_loss = 0
    
    progress_bar = tqdm(val_loader, total=len(val_loader), ncols=120)
    progress_bar.set_description(f"Validating...")
    
    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            
            # VQVAE前向传播，返回重建和量化损失
            reconstruction, quantization_loss = vqvae(images=images)
            
            # 重建损失
            recons_loss = recon_loss_fn(reconstruction.float(), images.float())
            
            # 总损失
            loss = recons_loss + quantization_loss
            
            val_loss += loss.item()
            val_recon_loss += recons_loss.item()
            val_quant_loss += quantization_loss.item()
            
            progress_bar.set_postfix({
                "loss": f"{val_loss / (step + 1):.4f}",
                "recon": f"{val_recon_loss / (step + 1):.4f}",
                "quant": f"{val_quant_loss / (step + 1):.4f}"
            })
    
    n_batches = len(val_loader) if not fast_dev_run else fast_dev_run_batches
    return val_loss / n_batches, val_recon_loss / n_batches, val_quant_loss / n_batches


def visualize_reconstruction(
    vqvae: torch.nn.Module,
    val_loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    num_samples: int = 4
):
    """
    可视化VQVAE重建结果
    
    Args:
        vqvae: VQVAE模型
        val_loader: 验证数据加载器
        device: 设备
        writer: TensorBoard writer
        epoch: 当前epoch
        num_samples: 可视化的样本数量
    """
    vqvae.eval()
    
    with torch.no_grad():
        logger.info("=" * 60)
        logger.info("可视化VQVAE重建结果")
        
        # 获取第一个batch
        batch = next(iter(val_loader))
        images = batch["image"].to(device)
        
        # 只取前num_samples个样本
        images = images[:num_samples]
        
        logger.info(f"输入patch形状: {images.shape}")
        
        # VQVAE重建
        reconstruction, _ = vqvae(images=images)
        
        # 移到CPU并转换为numpy
        images_np = images.cpu().numpy()  # (B, C, H, W, D)
        reconstruction_np = reconstruction.cpu().numpy()
        
        # 可视化重建结果
        for i in range(min(num_samples, images_np.shape[0])):
            # 取出单个样本 (C, H, W, D)
            input_vol = images_np[i, 0]  # (H, W, D)
            recon_vol = reconstruction_np[i, 0]  # (H, W, D)
            
            # 将3D体素沿z轴投影成2D图像（累加所有z层）
            input_proj = np.sum(input_vol, axis=2)  # (H, W)
            recon_proj = np.sum(recon_vol, axis=2)  # (H, W)
            
            # 归一化
            input_proj = (input_proj - input_proj.min()) / (input_proj.max() - input_proj.min() + 1e-8)
            recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
            
            # 水平堆叠: 输入 | 重建
            combined = np.hstack([input_proj, recon_proj])  # (H, 2*W)
            
            # 添加到TensorBoard
            writer.add_image(
                f"vqvae_reconstruction/sample_{i}",
                combined,
                epoch,
                dataformats='HW'
            )
            
            # 误差图
            error = np.abs(input_proj - recon_proj)
            writer.add_image(
                f"vqvae_reconstruction/sample_{i}_error",
                error,
                epoch,
                dataformats='HW'
            )
        
        logger.info(f"已保存 {min(num_samples, images.shape[0])} 个重建可视化结果")
        logger.info("=" * 60)


def train_vqvae(config_path: str):
    """
    训练VQVAE
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 提取配置参数
    vqvae_config = config['vqvae']
    checkpoint_config = vqvae_config['checkpoints']
    log_config = vqvae_config['logging']
    
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
    
    # 可视化数据加载器
    visualize_data_loader(train_loader, "train")
    visualize_data_loader(val_loader, "val")
    
    # 提取训练配置参数
    train_config = vqvae_config['training']
    
    # 创建VQVAE模型
    logger.info("=" * 60)
    logger.info("创建VQVAE模型")
    
    # 构建downsample_parameters和upsample_parameters
    num_channels = tuple(vqvae_config['num_channels'])
    num_res_layers = vqvae_config['num_res_layers']
    
    # downsample_parameters: 每层的(stride, kernel_size, padding, output_padding)
    # upsample_parameters: 每层的(stride, kernel_size, padding, output_padding, scale_factor)
    downsample_params = vqvae_config.get('downsample_parameters', None)
    upsample_params = vqvae_config.get('upsample_parameters', None)
    
    if downsample_params is not None:
        downsample_parameters = tuple([tuple(p) for p in downsample_params])
        upsample_parameters = tuple([tuple(p) for p in upsample_params])
        logger.info(f"使用自定义采样参数:")
        logger.info(f"  downsample: {downsample_parameters}")
        logger.info(f"  upsample: {upsample_parameters}")
    else:
        # 使用默认参数（每层2倍下采样）
        # ⚠️ VQVAE要求参数数量等于num_channels的长度
        downsample_parameters = tuple([(2, 4, 1, 1)] * len(num_channels))
        upsample_parameters = tuple([(2, 4, 1, 1, 0)] * len(num_channels))
        logger.info(f"使用默认采样参数（每层2倍）")
    
    vqvae = VQVAE(
        spatial_dims=vqvae_config['spatial_dims'],
        in_channels=vqvae_config['in_channels'],
        out_channels=vqvae_config['out_channels'],
        num_channels=num_channels,
        num_res_channels=vqvae_config['num_res_channels'],
        num_res_layers=num_res_layers,
        downsample_parameters=downsample_parameters,
        upsample_parameters=upsample_parameters,
        num_embeddings=vqvae_config['num_embeddings'],
        embedding_dim=vqvae_config['embedding_dim'],
        act=Act.RELU,
        output_act=Act.RELU
    )
    
    vqvae.to(device)
    logger.info(f"VQVAE模型参数:")
    logger.info(f"  空间维度: {vqvae_config['spatial_dims']}")
    logger.info(f"  通道数: {num_channels}")
    logger.info(f"  残差通道数: {vqvae_config['num_res_channels']}")
    logger.info(f"  残差层数: {num_res_layers}")
    logger.info(f"  Codebook大小: {vqvae_config['num_embeddings']}")
    logger.info(f"  嵌入维度: {vqvae_config['embedding_dim']}")
    logger.info("=" * 60)
    
    # 创建重建损失函数
    loss_config = train_config.get('loss', {})
    recon_loss_type = loss_config.get('reconstruction_loss_type', 'l1')
    
    logger.info("=" * 60)
    logger.info(f"创建重建损失函数: {recon_loss_type}")
    
    if recon_loss_type == 'l1':
        recon_loss_fn = L1Loss()
        logger.info("使用L1Loss作为重建损失")
    
    elif recon_loss_type == 'mse':
        recon_loss_fn = MSELoss()
        logger.info("使用MSELoss作为重建损失")
    
    elif recon_loss_type == 'weighted':
        # 加权重建损失（对前景像素给予更高权重）
        weighted_config = loss_config.get('weighted', {})
        recon_loss_fn = WeightedReconstructionLoss(
            loss_type=weighted_config.get('recon_loss_type', 'l1'),
            foreground_weight=weighted_config.get('foreground_weight', 10.0),
            background_weight=weighted_config.get('background_weight', 1.0),
            threshold=weighted_config.get('threshold', 0.1)
        )
    
    elif recon_loss_type == 'dice':
        # 纯Dice损失
        dice_config = loss_config.get('dice', {})
        recon_loss_fn = DiceLoss(
            smooth=dice_config.get('smooth', 1e-5),
            sigmoid=dice_config.get('sigmoid', False)
        )
        logger.info(f"使用DiceLoss作为重建损失")
    
    elif recon_loss_type == 'combined':
        # 组合损失：Dice Loss + 加权重建损失
        combined_config = loss_config.get('combined', {})
        recon_loss_fn = CombinedReconstructionLoss(
            dice_weight=combined_config.get('dice_weight', 0.5),
            recon_weight=combined_config.get('recon_weight', 1.0),
            recon_loss_type=combined_config.get('recon_loss_type', 'l1'),
            foreground_weight=combined_config.get('foreground_weight', 100.0),
            background_weight=combined_config.get('background_weight', 0.5),
            threshold=combined_config.get('threshold', 0.05),
            dice_smooth=combined_config.get('dice_smooth', 1e-5)
        )
    
    else:
        raise ValueError(f"不支持的重建损失类型: {recon_loss_type}")
    
    logger.info("=" * 60)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        params=vqvae.parameters(),
        lr=train_config['learning_rate']
    )
    
    # 混合精度训练
    use_amp = device_config.get('mixed_precision', False) and torch.cuda.is_available()
    if use_amp:
        scaler = GradScaler()
        logger.info("启用混合精度训练（AMP）")
    else:
        scaler = None
        logger.info("使用FP32训练")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练参数
    n_epochs = train_config['n_epochs']
    val_interval = train_config['val_interval']
    save_interval = train_config['save_interval']
    log_interval = log_config['log_interval']
    visualize_interval = log_config.get('visualize_interval', 10)
    num_visualize_samples = log_config.get('num_visualize_samples', 4)
    
    # 快速开发模式
    fast_dev_run = train_config.get('fast_dev_run', False)
    fast_dev_run_batches = train_config.get('fast_dev_run_batches', 2)
    
    if fast_dev_run:
        logger.info(f"**快速开发模式**: 每个epoch只运行 {fast_dev_run_batches} 个batch")
        n_epochs = 5
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
        start_epoch, best_val_loss = load_checkpoint(resume_from, vqvae, optimizer)
    
    # 训练循环
    logger.info(f"开始训练VQVAE: {n_epochs} epochs")
    
    for epoch in range(start_epoch, n_epochs):
        vqvae.train()
        
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_quant_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}/{n_epochs}")
        
        for step, batch in progress_bar:
            # 快速开发模式：只运行指定数量的batch
            if fast_dev_run and step >= fast_dev_run_batches:
                break
            
            images = batch["image"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            with autocast(enabled=use_amp):
                # VQVAE前向传播，返回重建和量化损失
                reconstruction, quantization_loss = vqvae(images=images)
                
                # 重建损失
                recons_loss = recon_loss_fn(reconstruction.float(), images.float())
                
                # 总损失 = 重建损失 + 量化损失
                loss = recons_loss + quantization_loss
            
            # 反向传播
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            epoch_recon_loss += recons_loss.item()
            epoch_quant_loss += quantization_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{epoch_loss / (step + 1):.4f}",
                "recon": f"{epoch_recon_loss / (step + 1):.4f}",
                "quant": f"{epoch_quant_loss / (step + 1):.4f}"
            })
            
            # TensorBoard日志
            if step % log_interval == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/step/total_loss", loss.item(), global_step)
                writer.add_scalar("train/step/recon_loss", recons_loss.item(), global_step)
                writer.add_scalar("train/step/quant_loss", quantization_loss.item(), global_step)
        
        # 记录epoch平均损失
        n_steps = step + 1
        avg_loss = epoch_loss / n_steps
        avg_recon = epoch_recon_loss / n_steps
        avg_quant = epoch_quant_loss / n_steps
        
        writer.add_scalar("train/epoch/total_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch/recon_loss", avg_recon, epoch)
        writer.add_scalar("train/epoch/quant_loss", avg_quant, epoch)
        
        logger.info(f"Epoch {epoch} 训练损失: total={avg_loss:.4f}, recon={avg_recon:.4f}, quant={avg_quant:.4f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
            val_loss, val_recon, val_quant = validate(
                vqvae, val_loader, device, recon_loss_fn,
                fast_dev_run, fast_dev_run_batches
            )
            
            writer.add_scalar("val/epoch/total_loss", val_loss, epoch)
            writer.add_scalar("val/epoch/recon_loss", val_recon, epoch)
            writer.add_scalar("val/epoch/quant_loss", val_quant, epoch)
            
            logger.info(f"Epoch {epoch} 验证损失: total={val_loss:.4f}, recon={val_recon:.4f}, quant={val_quant:.4f}")
            
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
                vqvae=vqvae,
                val_loader=val_loader,
                device=device,
                writer=writer,
                epoch=epoch,
                num_samples=num_visualize_samples
            )
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == n_epochs - 1 or is_best:
            save_checkpoint(
                epoch=epoch,
                vqvae=vqvae,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                output_dir=checkpoint_config['output_dir'],
                is_best=is_best
            )
    
    writer.close()
    logger.info("VQVAE训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练VQVAE")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/vq_ldm_config_local.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    train_vqvae(args.config)


if __name__ == "__main__":
    main()

