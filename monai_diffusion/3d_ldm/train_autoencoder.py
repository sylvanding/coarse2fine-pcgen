"""
AutoencoderKL训练脚本

基于MONAI Generative Models的3D AutoencoderKL训练，
用于Latent Diffusion Model的第一阶段训练。
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
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from monai.utils import set_determinism
from monai import transforms
import numpy as np

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from monai_diffusion.datasets import create_train_val_dataloaders
from monai_diffusion.utils.sliding_window_inference import AutoencoderSlidingWindowInferer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 去噪自编码器：噪声生成函数 ====================

def add_gaussian_noise(images: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """
    添加高斯噪声
    
    Args:
        images: 输入图像张量 (B, C, H, W, D)，值域范围[0, 1]
        noise_std: 噪声标准差（相对于图像范围）
        
    Returns:
        带噪声的图像张量，值域范围[0, 1]
    """
    noise = torch.randn_like(images) * noise_std
    noisy_images = images + noise
    # 保持与原始图像相同的值域范围 [0, 1]
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images


def add_dropout_noise(images: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
    """
    随机丢弃（置零）一些体素
    
    Args:
        images: 输入图像张量 (B, C, H, W, D)，值域范围[0, 1]
        dropout_prob: 丢弃概率
        
    Returns:
        带噪声的图像张量，值域范围[0, 1]
    """
    mask = torch.rand_like(images) > dropout_prob
    noisy_images = images * mask.float()
    return noisy_images


def add_mixed_noise(
    images: torch.Tensor, 
    noise_std: float = 0.1, 
    dropout_prob: float = 0.05
) -> torch.Tensor:
    """
    混合噪声：同时添加高斯噪声和dropout噪声
    
    Args:
        images: 输入图像张量 (B, C, H, W, D)，值域范围[0, 1]
        noise_std: 高斯噪声标准差
        dropout_prob: 丢弃概率
        
    Returns:
        带噪声的图像张量，值域范围[0, 1]
    """
    # 先添加dropout
    noisy_images = add_dropout_noise(images, dropout_prob)
    # 再添加高斯噪声
    noisy_images = add_gaussian_noise(noisy_images, noise_std)
    return noisy_images


def add_noise(
    images: torch.Tensor,
    noise_type: str = "gaussian",
    noise_std: float = 0.1,
    dropout_prob: float = 0.1
) -> torch.Tensor:
    """
    统一的噪声添加接口
    
    Args:
        images: 输入图像张量 (B, C, H, W, D)，值域范围[0, 1]
        noise_type: 噪声类型，可选 "gaussian", "dropout", "mixed"
        noise_std: 高斯噪声标准差
        dropout_prob: dropout概率
        
    Returns:
        带噪声的图像张量，值域范围[0, 1]
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(images, noise_std)
    elif noise_type == "dropout":
        return add_dropout_noise(images, dropout_prob)
    elif noise_type == "mixed":
        return add_mixed_noise(images, noise_std, dropout_prob)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
        

# ==================== 原有代码继续 ====================


def visualize_data_loader(data_loader: DataLoader, data_name: str):
    # 获取第一个batch
    cpu_device = torch.device("cpu")
    batch = next(iter(data_loader))
    images = batch["image"].to(cpu_device)
    
    logger.info(f"输入patch形状: {images.shape}")
    
    images_np = images.numpy()  # (B, C, H, W, D)
    
    file_path = f"outputs/images_for_data_loader/{data_name}"
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
        image = Image.fromarray(np.uint8(input_proj*255.0))
        image.save(f"{file_path}/sample_{i}.png")
        logger.info(f"保存样本 {i} 到 {file_path}/sample_{i}.png")


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
    fast_dev_run_batches: int,
    # 去噪参数
    use_denoising: bool = False,
    noise_type: str = "gaussian",
    noise_std: float = 0.1,
    dropout_prob: float = 0.1
):
    """
    验证函数（支持去噪自编码器）
    
    Args:
        use_denoising: 是否使用去噪模式
        noise_type: 噪声类型
        noise_std: 高斯噪声标准差
        dropout_prob: dropout概率
    """
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
            
            # 原始干净图像
            clean_images = batch["image"].to(device)
            
            # 如果使用去噪模式，给输入添加噪声
            if use_denoising:
                noisy_images = add_noise(
                    clean_images,
                    noise_type=noise_type,
                    noise_std=noise_std,
                    dropout_prob=dropout_prob
                )
                # 模型输入：带噪声的图像
                # 目标输出：原始干净的图像
                reconstruction, z_mu, z_sigma = autoencoder(noisy_images)
                recons_loss = F.l1_loss(reconstruction.float(), clean_images.float())
            else:
                # 标准自编码器：输入=输出
                reconstruction, z_mu, z_sigma = autoencoder(clean_images)
                recons_loss = F.l1_loss(reconstruction.float(), clean_images.float())
            
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


def create_full_image_dataloader(config: dict, num_samples: int = 4, num_workers: int = 0, pin_memory: bool = False):
    """
    创建用于加载完整图像（不裁剪）的数据加载器
    
    Args:
        config: 配置字典
        num_samples: 加载的样本数量
        num_workers: DataLoader工作进程数（默认0避免共享内存问题）
        pin_memory: 是否使用固定内存（默认False节省内存）
        
    Returns:
        DataLoader: 完整图像数据加载器
    """
    from monai.data import Dataset
    from torch.utils.data import DataLoader
    
    data_config = config['data']
    val_dir = Path(data_config['val_data_dir'])
    
    # 收集NIfTI文件
    nifti_files = []
    for pattern in ['*.nii', '*.nii.gz']:
        nifti_files.extend(val_dir.glob(pattern))
    nifti_files.sort()
    
    # 只取前num_samples个
    nifti_files = nifti_files[:num_samples]
    
    logger.info(f"创建完整图像数据加载器，共 {len(nifti_files)} 个样本")
    
    # 获取voxel_resize配置
    data_config = config['data']
    voxel_resize = data_config.get('voxel_resize', None)
    
    # 处理voxel_resize
    if voxel_resize is not None:
        if isinstance(voxel_resize, (list, tuple)):
            if len(voxel_resize) != 3:
                raise ValueError(f"voxel_resize作为列表时必须包含3个元素[X, Y, Z]，但得到了{len(voxel_resize)}个元素")
            voxel_resize_tuple = tuple(voxel_resize)
        else:
            voxel_resize_tuple = (voxel_resize, voxel_resize, voxel_resize)
        logger.info(f"完整图像将先resize到: {voxel_resize_tuple}")
    else:
        voxel_resize_tuple = None
        logger.info("完整图像不进行预resize")
    
    # 创建transforms（根据是否有voxel_resize调整流程）
    transform_list = [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
    ]
    
    # 如果配置了voxel_resize，添加resize变换
    if voxel_resize_tuple is not None:
        transform_list.append(
            transforms.Resized(
                keys=["image"],
                spatial_size=voxel_resize_tuple,
                mode="trilinear"
            )
        )
    
    # 添加归一化和类型转换
    transform_list.extend([
        # 归一化到[-1, 1]
        transforms.ScaleIntensityRanged(
            keys="image",
            a_min=0.0,
            a_max=255.0,
            b_min=-1.0,
            b_max=1.0,
            clip=True
        ),
        transforms.EnsureTyped(keys=["image"], dtype=torch.float32)
    ])
    
    full_image_transforms = transforms.Compose(transform_list)
    
    # 创建数据列表
    data_list = [{"image": str(f)} for f in nifti_files]
    
    # 创建数据集
    full_image_dataset = Dataset(
        data=data_list,
        transform=full_image_transforms
    )
    
    # 创建数据加载器（batch_size=1，因为完整图像大小可能不一致）
    # 使用num_workers=0避免多进程共享内存问题
    full_image_loader = DataLoader(
        full_image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"DataLoader配置: num_workers={num_workers}, pin_memory={pin_memory}")
    
    return full_image_loader


def visualize_reconstruction(
    autoencoder: torch.nn.Module,
    val_loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: dict,
    num_samples: int = 4,
    use_sliding_window: bool = True,
    roi_size: tuple = None,
    sw_batch_size: int = 4,
    # 去噪参数
    use_denoising: bool = False,
    noise_type: str = "gaussian",
    noise_std: float = 0.1,
    dropout_prob: float = 0.1
):
    """
    可视化重建结果（支持去噪自编码器）
    
    分两部分：
    1. 使用patch-based验证集进行直接重建
    2. 创建完整图像数据加载器，使用滑动窗口推理重建
    
    Args:
        autoencoder: AutoencoderKL模型
        val_loader: 验证数据加载器（patch-based）
        device: 设备
        writer: TensorBoard writer
        epoch: 当前epoch
        config: 配置字典
        num_samples: 可视化的样本数量
        use_sliding_window: 是否使用滑动窗口推理（用于完整图像）
        roi_size: 滑动窗口的ROI大小，如果为None则使用patch大小
        sw_batch_size: 滑动窗口批次大小
        use_denoising: 是否使用去噪模式
        noise_type: 噪声类型
        noise_std: 高斯噪声标准差
        dropout_prob: dropout概率
    """
    autoencoder.eval()
    
    with torch.no_grad():
        # ============ 第一部分：Patch-based直接重建 ============
        logger.info("=" * 60)
        if use_denoising:
            logger.info("第一部分：Patch-based去噪重建")
            logger.info(f"噪声类型: {noise_type}, 高斯噪声std: {noise_std}, dropout概率: {dropout_prob}")
        else:
            logger.info("第一部分：Patch-based直接重建")
        
        # 获取第一个batch
        batch = next(iter(val_loader))
        clean_images = batch["image"].to(device)
        
        # 只取前num_samples个样本
        clean_images = clean_images[:num_samples]
        
        logger.info(f"输入patch形状: {clean_images.shape}")
        
        # 如果使用去噪模式，添加噪声
        if use_denoising:
            noisy_images = add_noise(
                clean_images,
                noise_type=noise_type,
                noise_std=noise_std,
                dropout_prob=dropout_prob
            )
            # 模型输入带噪声的图像，目标是恢复干净图像
            reconstruction_direct, _, _ = autoencoder(noisy_images)
            
            # 移到CPU并转换为numpy
            noisy_images_np = noisy_images.cpu().numpy()  # (B, C, H, W, D)
            clean_images_np = clean_images.cpu().numpy()  # (B, C, H, W, D)
            reconstruction_direct_np = reconstruction_direct.cpu().numpy()
            
            # 可视化去噪重建结果
            for i in range(min(num_samples, clean_images_np.shape[0])):
                # 取出单个样本 (C, H, W, D)
                noisy_vol = noisy_images_np[i, 0]  # (H, W, D)
                clean_vol = clean_images_np[i, 0]  # (H, W, D)
                recon_vol = reconstruction_direct_np[i, 0]  # (H, W, D)
                
                # 将3D体素沿z轴投影成2D图像（累加所有z层）
                noisy_proj = np.sum(noisy_vol, axis=2)  # (H, W)
                clean_proj = np.sum(clean_vol, axis=2)  # (H, W)
                recon_proj = np.sum(recon_vol, axis=2)  # (H, W)
                
                # 归一化
                noisy_proj = (noisy_proj - noisy_proj.min()) / (noisy_proj.max() - noisy_proj.min() + 1e-8)
                clean_proj = (clean_proj - clean_proj.min()) / (clean_proj.max() - clean_proj.min() + 1e-8)
                recon_proj = (recon_proj - recon_proj.min()) / (recon_proj.max() - recon_proj.min() + 1e-8)
                
                # 水平堆叠: 噪声输入 | 模型重建 | 干净目标
                combined = np.hstack([noisy_proj, recon_proj, clean_proj])  # (H, 3*W)
                
                # 添加到TensorBoard
                writer.add_image(
                    f"patch_denoising/sample_{i}",
                    combined,
                    epoch,
                    dataformats='HW'
                )
                
                # 重建误差图：模型输出 vs 干净目标
                error = np.abs(recon_proj - clean_proj)
                writer.add_image(
                    f"patch_denoising/sample_{i}_error",
                    error,
                    epoch,
                    dataformats='HW'
                )
        else:
            # 标准重建模式
            reconstruction_direct, _, _ = autoencoder(clean_images)
            
            # 移到CPU并转换为numpy
            images_np = clean_images.cpu().numpy()  # (B, C, H, W, D)
            reconstruction_direct_np = reconstruction_direct.cpu().numpy()
            
            # 可视化patch重建结果
            for i in range(min(num_samples, images_np.shape[0])):
                # 取出单个样本 (C, H, W, D)
                input_vol = images_np[i, 0]  # (H, W, D)
                recon_vol = reconstruction_direct_np[i, 0]  # (H, W, D)
                
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
                    f"patch_reconstruction/sample_{i}",
                    combined,
                    epoch,
                    dataformats='HW'
                )
                
                # 误差图
                error = np.abs(input_proj - recon_proj)
                writer.add_image(
                    f"patch_reconstruction/sample_{i}_error",
                    error,
                    epoch,
                    dataformats='HW'
                )
        
        logger.info(f"已保存 {min(num_samples, clean_images.shape[0])} 个patch重建可视化结果")
        
        # ============ 第二部分：完整图像滑动窗口推理 ============
        if use_sliding_window:
            logger.info("=" * 60)
            logger.info("第二部分：完整图像滑动窗口推理")
            
            try:
                # 清理GPU缓存，释放内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"清理GPU缓存，当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                
                # 创建完整图像数据加载器（使用内存优化参数）
                full_image_loader = create_full_image_dataloader(
                    config, 
                    num_samples,
                    num_workers=0,  # 避免共享内存问题
                    pin_memory=False  # 节省内存
                )
                
                # 推断ROI大小（使用训练patch大小）
                if roi_size is None:
                    roi_size = clean_images.shape[2:]  # (H, W, D)
                    logger.info(f"使用训练patch大小作为滑动窗口ROI: {roi_size}")
                
                # 创建滑动窗口推理器
                sw_inferer = AutoencoderSlidingWindowInferer(
                    autoencoder=autoencoder,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    overlap=0.25,
                    mode="gaussian",
                    device=device
                )
                
                # 对每个完整图像进行滑动窗口推理
                for idx, batch in enumerate(full_image_loader):
                    if idx >= num_samples:
                        break
                    
                    try:
                        full_image = batch["image"].to(device)  # (1, C, H, W, D)
                        logger.info(f"完整图像 {idx} 形状: {full_image.shape}")
                        
                        # 滑动窗口推理
                        reconstruction_sw = sw_inferer.reconstruct(full_image, return_latent=False)
                        logger.info(f"滑动窗口重建完成，输出形状: {reconstruction_sw.shape}")
                        
                        # 移到CPU并转换为numpy
                        full_image_np = full_image.cpu().numpy()[0, 0]  # (H, W, D)
                        reconstruction_sw_np = reconstruction_sw.cpu().numpy()[0, 0]  # (H, W, D)
                        
                        # 释放GPU内存
                        del full_image, reconstruction_sw
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 将3D体素沿z轴投影成2D图像
                        full_image_proj = np.sum(full_image_np, axis=2)  # (H, W)
                        recon_sw_proj = np.sum(reconstruction_sw_np, axis=2)  # (H, W)
                        
                        # 归一化
                        full_image_proj = (full_image_proj - full_image_proj.min()) / (full_image_proj.max() - full_image_proj.min() + 1e-8)
                        recon_sw_proj = (recon_sw_proj - recon_sw_proj.min()) / (recon_sw_proj.max() - recon_sw_proj.min() + 1e-8)
                        
                        # 水平堆叠: 输入完整图像 | 滑动窗口重建
                        combined = np.hstack([full_image_proj, recon_sw_proj])  # (H, 2*W)
                        
                        # 添加到TensorBoard
                        writer.add_image(
                            f"full_image_sliding_window/sample_{idx}",
                            combined,
                            epoch,
                            dataformats='HW'
                        )
                        
                        # 误差图
                        error_sw = np.abs(full_image_proj - recon_sw_proj)
                        writer.add_image(
                            f"full_image_sliding_window/sample_{idx}_error",
                            error_sw,
                            epoch,
                            dataformats='HW'
                        )
                        
                        logger.info(f"✓ 完成样本 {idx}")
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(f"✗ 样本 {idx} 推理失败：GPU显存不足")
                            logger.error(f"  建议: 1) 减小sw_batch_size (当前={sw_batch_size})")
                            logger.error(f"        2) 减小ROI大小 (当前={roi_size})")
                            logger.error(f"        3) 增加GPU显存或使用更小的图像")
                            # 清理并继续下一个样本
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                
                logger.info(f"已保存 {min(idx + 1, num_samples)} 个完整图像滑动窗口重建可视化结果")
                
            except Exception as e:
                logger.error(f"滑动窗口推理过程出错: {e}")
                logger.error("跳过滑动窗口可视化")
            finally:
                # 最终清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info("=" * 60)


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
    
    # 渲染第一个batch，验证数据加载器的正确性
    visualize_data_loader(train_loader, "train")
    visualize_data_loader(val_loader, "val")
    
    # 提取训练配置参数
    train_config = ae_config['training']
    
    # 创建AutoencoderKL
    # 获取downsample_factors（如果配置中存在）
    downsample_factors = ae_config.get('downsample_factors', None)
    initial_downsample_factor = ae_config.get('initial_downsample_factor', 1)
    use_conv_downsample = ae_config.get('use_conv_downsample', True)
    use_convtranspose = ae_config.get('use_convtranspose', False)
    
    if downsample_factors is not None:
        downsample_factors = tuple(downsample_factors)
        total_downsample = initial_downsample_factor
        for factor in downsample_factors:
            total_downsample *= factor
        logger.info(f"使用自定义下采样因子: initial={initial_downsample_factor}, layers={downsample_factors}")
        logger.info(f"总下采样倍数: {total_downsample}x")
    else:
        # 默认每层2倍下采样
        total_downsample = initial_downsample_factor * (2 ** (len(ae_config['num_channels']) - 1))
        logger.info(f"使用默认下采样配置: initial={initial_downsample_factor}, 总下采样倍数: {total_downsample}x")
    
    # 记录采样方法
    downsample_method = "卷积下采样" if use_conv_downsample else "平均池化下采样"
    upsample_method = "转置卷积上采样" if use_convtranspose else "最近邻插值+卷积上采样"
    logger.info(f"下采样方法: {downsample_method}")
    logger.info(f"上采样方法: {upsample_method}")
    
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
    
    # 滑动窗口推理配置
    use_sliding_window_vis = log_config.get('use_sliding_window', False)  # 默认启用滑动窗口推理
    sw_roi_size = log_config.get('sliding_window_roi_size', None)  # None表示自动推断
    sw_batch_size = log_config.get('sliding_window_batch_size', 4)  # 滑动窗口批次大小
    
    # ==================== 去噪自编码器配置 ====================
    denoising_config = train_config.get('denoising', {})
    use_denoising = denoising_config.get('enabled', False)
    noise_type = denoising_config.get('noise_type', 'gaussian')
    noise_std = denoising_config.get('noise_std', 0.1)
    dropout_prob = denoising_config.get('dropout_prob', 0.1)
    
    if use_denoising:
        logger.info("=" * 60)
        logger.info("🔥 启用去噪自编码器模式 (Denoising Autoencoder)")
        logger.info(f"  噪声类型: {noise_type}")
        logger.info(f"  高斯噪声标准差: {noise_std}")
        logger.info(f"  Dropout概率: {dropout_prob}")
        logger.info("  模型将学习从噪声中恢复干净图像，迫使其学习数据的深层特征！")
        logger.info("=" * 60)
    else:
        logger.info("使用标准自编码器模式")
    
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
            
            # 原始干净图像
            clean_images = batch["image"].to(device)
            
            # ============ 去噪自编码器：添加噪声 ============
            if use_denoising:
                # 给输入添加噪声
                noisy_images = add_noise(
                    clean_images,
                    noise_type=noise_type,
                    noise_std=noise_std,
                    dropout_prob=dropout_prob
                )
                # 模型输入：带噪声的图像
                input_images = noisy_images
                # 目标输出：原始干净的图像
                target_images = clean_images
            else:
                # 标准自编码器：输入=输出
                input_images = clean_images
                target_images = clean_images
            
            # ============ Generator部分 ============
            # 梯度清零（在累积开始时）
            if step % gradient_accumulation_steps == 0:
                optimizer_g.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            with autocast(enabled=use_amp):
                reconstruction, z_mu, z_sigma = autoencoder(input_images)
                kl = KL_loss(z_mu, z_sigma)
                
                # 重建损失：对比重建结果和干净目标
                recons_loss = l1_loss(reconstruction.float(), target_images.float())
                
                # 感知损失（如果启用）
                if use_perceptual_loss:
                    p_loss = loss_perceptual(reconstruction.float(), target_images.float())
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
                    
                    # 判别器判断的是目标（干净）图像的真假
                    logits_real = discriminator(target_images.contiguous().detach())[-1]
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
                autoencoder, val_loader, device, kl_weight, fast_dev_run, fast_dev_run_batches,
                # 去噪参数
                use_denoising=use_denoising,
                noise_type=noise_type,
                noise_std=noise_std,
                dropout_prob=dropout_prob
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
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=device,
                writer=writer,
                epoch=epoch,
                config=config,
                num_samples=num_visualize_samples,
                use_sliding_window=use_sliding_window_vis,
                roi_size=sw_roi_size,
                sw_batch_size=sw_batch_size,
                # 去噪参数
                use_denoising=use_denoising,
                noise_type=noise_type,
                noise_std=noise_std,
                dropout_prob=dropout_prob
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
        default='monai_diffusion/config/ldm_config_local.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    train_autoencoder(args.config)


if __name__ == "__main__":
    main()

