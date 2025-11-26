"""
条件3D扩散模型样本生成脚本

给定2D条件图像，生成对应的3D体素。
"""

import sys
from pathlib import Path
import argparse
import logging
import yaml

# 添加GenerativeModels和项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm

from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

# 导入条件扩散模型
from train_conditional_diffusion import ConditionalDiffusionUNet

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


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device
):
    """
    加载训练好的条件扩散模型
    
    Args:
        checkpoint_path: checkpoint路径
        config: 配置字典
        device: 设备
        
    Returns:
        加载的模型（评估模式）
    """
    diff_config = config['diffusion']
    
    # 创建模型
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"从 {checkpoint_path} 加载模型")
    logger.info(f"训练轮数: {checkpoint['epoch']}")
    
    return model


def load_condition_image(
    image_path: str,
    target_size: tuple,
    normalize: bool = True
) -> torch.Tensor:
    """
    加载2D条件图像
    
    Args:
        image_path: 图像路径（支持png, jpg等）
        target_size: 目标大小 (H, W)
        normalize: 是否归一化到[0, 1]
        
    Returns:
        条件图像张量 (1, 1, H, W)
    """
    # 加载图像
    img = Image.open(image_path).convert('L')  # 转为灰度图
    
    # Resize到目标大小
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # 转为numpy数组
    img_array = np.array(img).astype(np.float32)
    
    # 归一化
    if normalize:
        img_array = img_array / 255.0
    
    # 转为torch tensor并添加batch和channel维度
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    logger.info(f"加载条件图像: {image_path}")
    logger.info(f"图像形状: {img_tensor.shape}")
    
    return img_tensor


def load_condition_from_nifti(
    nifti_path: str,
    project_axis: int = 2,
    normalize: bool = True
) -> torch.Tensor:
    """
    从NIfTI文件加载3D体素并生成2D投影作为条件
    
    Args:
        nifti_path: NIfTI文件路径
        project_axis: 投影轴（0=x, 1=y, 2=z）
        normalize: 是否归一化
        
    Returns:
        条件图像张量 (1, 1, H, W)
    """
    # 加载NIfTI文件
    nifti_img = nib.load(nifti_path)
    voxel_data = nifti_img.get_fdata()  # (H, W, D)
    
    # 生成投影（沿指定轴累加）
    projection = np.sum(voxel_data, axis=project_axis)
    
    # 归一化
    if normalize:
        proj_min = projection.min()
        proj_max = projection.max()
        if proj_max > proj_min:
            projection = (projection - proj_min) / (proj_max - proj_min)
    
    # 转为torch tensor
    projection_tensor = torch.from_numpy(projection).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    logger.info(f"从NIfTI加载条件图像: {nifti_path}")
    logger.info(f"投影形状: {projection_tensor.shape}")
    
    return projection_tensor


def generate_samples(
    model,
    scheduler,
    condition: torch.Tensor,
    voxel_size: tuple,
    device: torch.device,
    num_samples: int = 1,
    num_inference_steps: int = 1000,
    use_ddim: bool = True,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    生成条件3D样本
    
    Args:
        model: 条件扩散模型
        scheduler: 调度器
        condition: 2D条件图像 (1, 1, H, W)
        voxel_size: 3D体素大小 (H, W, D)
        device: 设备
        num_samples: 生成样本数量
        num_inference_steps: 推理步数
        use_ddim: 是否使用DDIM调度器
        guidance_scale: 条件引导强度（>1增强条件影响）
        
    Returns:
        生成的3D体素 (num_samples, 1, H, W, D)
    """
    model.eval()
    
    # 复制条件图像以匹配batch size
    conditions = condition.repeat(num_samples, 1, 1, 1).to(device)
    
    # 设置推理步数
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # 生成随机噪声
    noise = torch.randn((num_samples, 1, *voxel_size)).to(device)
    
    # 采样
    synthetic_voxels = noise
    
    logger.info(f"开始生成 {num_samples} 个样本...")
    logger.info(f"推理步数: {num_inference_steps}")
    logger.info(f"使用{'DDIM' if use_ddim else 'DDPM'}调度器")
    
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="扩散采样"):
            with autocast(enabled=True):
                # 预测噪声
                model_output = model(
                    x=synthetic_voxels,
                    timesteps=torch.full((num_samples,), t, device=device, dtype=torch.long),
                    condition=conditions
                )
                
                # 如果使用guidance_scale，可以实现classifier-free guidance
                # 这需要在训练时支持无条件生成（随机丢弃条件）
                # 暂时保持简单，不实现guidance
                
                # 更新样本
                synthetic_voxels, _ = scheduler.step(model_output, t, synthetic_voxels)
    
    logger.info("样本生成完成")
    
    return synthetic_voxels


def save_voxel_as_nifti(
    voxel: np.ndarray,
    output_path: str,
    affine: np.ndarray = None
):
    """
    保存体素为NIfTI文件
    
    Args:
        voxel: 体素数据 (H, W, D)
        output_path: 输出路径
        affine: 仿射变换矩阵
    """
    if affine is None:
        # 使用单位矩阵
        affine = np.eye(4)
    
    # 创建NIfTI图像
    nifti_img = nib.Nifti1Image(voxel, affine)
    
    # 保存
    nib.save(nifti_img, output_path)
    logger.info(f"保存体素到: {output_path}")


def save_projection_as_image(
    voxel: np.ndarray,
    output_path: str,
    project_axis: int = 2
):
    """
    保存体素的2D投影为图像
    
    Args:
        voxel: 体素数据 (H, W, D)
        output_path: 输出路径
        project_axis: 投影轴
    """
    # 生成投影
    projection = np.sum(voxel, axis=project_axis)
    
    # 归一化到0-255
    proj_min = projection.min()
    proj_max = projection.max()
    if proj_max > proj_min:
        projection = (projection - proj_min) / (proj_max - proj_min) * 255
    else:
        projection = np.zeros_like(projection)
    
    projection = projection.astype(np.uint8)
    
    # 保存为图像
    img = Image.fromarray(projection)
    img.save(output_path)
    logger.info(f"保存投影图像到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成条件3D扩散样本")
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/conditional_diffusion_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        help='条件图像路径（png/jpg）或NIfTI文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/conditional_diffusion/samples/',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='生成样本数量'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=1000,
        help='推理步数'
    )
    parser.add_argument(
        '--use_ddim',
        action='store_true',
        help='使用DDIM调度器（更快）'
    )
    parser.add_argument(
        '--save_projections',
        action='store_true',
        help='保存生成体素的2D投影'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_config.get('gpu_id', 0)}" if use_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, config, device)
    
    # 创建调度器
    diff_config = config['diffusion']
    scheduler_config = diff_config['scheduler']
    
    if args.use_ddim:
        scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_config['num_train_timesteps'],
            schedule=scheduler_config['schedule'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end'],
            clip_sample=False
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=scheduler_config['num_train_timesteps'],
            schedule=scheduler_config['schedule'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end']
        )
    
    # 获取体素大小
    voxel_size = config['data']['voxel_size']
    if isinstance(voxel_size, int):
        voxel_size = [voxel_size, voxel_size, voxel_size]
    voxel_size = tuple(voxel_size)
    
    # 加载条件图像
    condition_path = Path(args.condition)
    if condition_path.suffix in ['.nii', '.gz']:
        # 从NIfTI文件加载
        condition = load_condition_from_nifti(
            str(condition_path),
            project_axis=diff_config.get('projection_axis', 2)
        )
    else:
        # 从普通图像文件加载
        condition = load_condition_image(
            str(condition_path),
            target_size=(voxel_size[0], voxel_size[1])
        )
    
    # 生成样本
    synthetic_voxels = generate_samples(
        model=model,
        scheduler=scheduler,
        condition=condition,
        voxel_size=voxel_size,
        device=device,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        use_ddim=args.use_ddim
    )
    
    # 保存样本
    synthetic_voxels_np = synthetic_voxels.cpu().numpy()
    
    for i in range(args.num_samples):
        voxel = synthetic_voxels_np[i, 0]  # (H, W, D)
        
        # 保存为NIfTI
        nifti_path = output_dir / f"sample_{i:04d}.nii.gz"
        save_voxel_as_nifti(voxel, str(nifti_path))
        
        # 可选：保存投影图像
        if args.save_projections:
            proj_path = output_dir / f"sample_{i:04d}_projection.png"
            save_projection_as_image(
                voxel,
                str(proj_path),
                project_axis=diff_config.get('projection_axis', 2)
            )
    
    # 也保存条件图像作为参考
    condition_ref_path = output_dir / "condition_reference.png"
    condition_np = condition[0, 0].cpu().numpy()  # (H, W)
    condition_img = (condition_np * 255).astype(np.uint8)
    Image.fromarray(condition_img).save(str(condition_ref_path))
    logger.info(f"保存条件参考图像到: {condition_ref_path}")
    
    logger.info(f"所有样本已保存到: {output_dir}")


if __name__ == "__main__":
    main()

