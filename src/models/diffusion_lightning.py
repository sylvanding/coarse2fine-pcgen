"""
3D Diffusion PyTorch Lightning模块

基于PyTorch Lightning的3D体素扩散模型训练模块。
支持自动验证、TIFF保存和模型检查点。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .diffusion_3d import UNet3D, GaussianDiffusion

logger = logging.getLogger(__name__)


class DiffusionLightningModule(pl.LightningModule):
    """
    3D Diffusion PyTorch Lightning模块
    
    包含完整的训练、验证和采样逻辑。
    """
    
    def __init__(
        self,
        # 模型参数
        voxel_size: int = 64,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Optional[list] = None,
        channel_mult: Optional[list] = None,
        dropout: float = 0.1,
        use_attention: bool = True,
        
        # 扩散参数
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        
        # 训练参数
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        use_scheduler: bool = True,
        scheduler_type: str = "cosine",
        
        # 验证参数
        val_sample_interval: int = 10,  # 每10个epoch验证一次
        num_val_samples: int = 4,
        save_val_tiffs: bool = True,
        val_output_dir: str = "validation_outputs",
        
        # 采样参数
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        
        # 其他参数
        compile_model: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 模型参数
        self.voxel_size = voxel_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.val_sample_interval = val_sample_interval
        self.num_val_samples = num_val_samples
        self.save_val_tiffs = save_val_tiffs
        self.val_output_dir = Path(val_output_dir)
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        
        # 默认参数
        if attention_resolutions is None:
            attention_resolutions = [16, 8]
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8]
        
        # 创建UNet模型
        self.model = UNet3D(
            voxel_size=voxel_size,
            in_channels=1,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_attention=use_attention
        )
        
        # 创建扩散过程
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )
        
        # 将扩散过程的缓冲区注册到Lightning模块中，确保设备同步
        for name, tensor in vars(self.diffusion).items():
            if isinstance(tensor, torch.Tensor):
                self.register_buffer(f"diffusion_{name}", tensor.clone())
                # 保持对原始diffusion对象的引用，但使用注册的缓冲区
                # 这样PyTorch Lightning会自动处理设备同步
        
        # 编译模型（PyTorch 2.0+）
        if compile_model and hasattr(torch, 'compile'):
            logger.info("编译模型以提升性能...")
            self.model = torch.compile(self.model)
        
        # 创建输出目录
        self.val_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("初始化3D Diffusion Lightning模块:")
        logger.info(f"  体素大小: {voxel_size}^3")
        logger.info(f"  模型通道: {model_channels}")
        logger.info(f"  扩散步数: {num_timesteps}")
        logger.info(f"  验证采样间隔: {val_sample_interval} epochs")
    
    def on_before_optimizer_step(self, optimizer):
        """优化器步骤前的回调，确保扩散参数与模型在同一设备"""
        self._sync_diffusion_buffers()
    
    def _sync_diffusion_buffers(self):
        """同步扩散缓冲区到当前设备"""
        device = next(self.parameters()).device
        for name, tensor in vars(self.diffusion).items():
            if isinstance(tensor, torch.Tensor):
                buffer_name = f"diffusion_{name}"
                if hasattr(self, buffer_name):
                    # 更新diffusion对象的张量为注册的缓冲区
                    setattr(self.diffusion, name, getattr(self, buffer_name).to(device))
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x, timesteps)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """训练步骤"""
        # 确保扩散参数与模型在同一设备
        self._sync_diffusion_buffers()
        
        voxels = batch['voxel']  # (B, 1, D, H, W)
        batch_size = voxels.shape[0]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.diffusion.num_timesteps, (batch_size,), device=voxels.device
        )
        
        # 添加噪声
        noise = torch.randn_like(voxels)
        noisy_voxels = self.diffusion.q_sample(voxels, timesteps, noise)
        
        # 预测噪声
        noise_pred = self.model(noisy_voxels, timesteps)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        
        # 记录学习率
        optimizer = self.optimizers()
        if hasattr(optimizer, 'param_groups'):
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """验证步骤"""
        # 确保扩散参数与模型在同一设备
        self._sync_diffusion_buffers()
        
        voxels = batch['voxel']  # (B, 1, D, H, W)
        batch_size = voxels.shape[0]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.diffusion.num_timesteps, (batch_size,), device=voxels.device
        )
        
        # 添加噪声
        noise = torch.randn_like(voxels)
        noisy_voxels = self.diffusion.q_sample(voxels, timesteps, noise)
        
        # 预测噪声
        noise_pred = self.model(noisy_voxels, timesteps)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 记录指标
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时的回调"""
        # 检查是否需要生成样本
        if (self.current_epoch + 1) % self.val_sample_interval == 0:
            self.generate_validation_samples()
    
    def generate_validation_samples(self):
        """生成验证样本"""
        logger.info(f"生成验证样本 (Epoch {self.current_epoch + 1})")
        
        # 确保扩散参数与模型在同一设备
        self._sync_diffusion_buffers()
        
        # 设置为评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 生成样本
            shape = (self.num_val_samples, 1, self.voxel_size, self.voxel_size, self.voxel_size)
            
            # 使用DDIM采样
            generated_voxels = self.diffusion.ddim_sample(
                model=self.model,
                shape=shape,
                device=self.device,
                eta=self.ddim_eta,
                num_inference_steps=self.ddim_steps,
                progress=False
            )
            
            # 后处理：将[-1, 1]映射到[0, 1]
            generated_voxels = (generated_voxels + 1.0) / 2.0
            generated_voxels = torch.clamp(generated_voxels, 0.0, 1.0)
            
            if self.save_val_tiffs:
                self.save_validation_tiffs(generated_voxels)
            
            # 计算生成质量指标
            self.compute_generation_metrics(generated_voxels)
        
        # 恢复训练模式
        self.model.train()
    
    def save_validation_tiffs(self, generated_voxels: torch.Tensor):
        """保存验证TIFF文件"""
        epoch_dir = self.val_output_dir / f"epoch_{self.current_epoch + 1:04d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # 导入体素转换器
        from ..voxel.converter import PointCloudToVoxel
        converter = PointCloudToVoxel(voxel_size=self.voxel_size)
        
        for i, voxel in enumerate(generated_voxels):
            # 移除批次和通道维度
            voxel_np = voxel.squeeze().cpu().numpy()
            
            # 保存为TIFF
            output_path = epoch_dir / f"generated_sample_{i:02d}.tiff"
            converter.save_as_tiff(voxel_np, str(output_path))
        
        logger.info(f"已保存 {len(generated_voxels)} 个验证TIFF到: {epoch_dir}")
    
    def compute_generation_metrics(self, generated_voxels: torch.Tensor):
        """计算生成质量指标"""
        with torch.no_grad():
            # 基本统计
            mean_value = torch.mean(generated_voxels)
            std_value = torch.std(generated_voxels)
            max_value = torch.max(generated_voxels)
            min_value = torch.min(generated_voxels)
            
            # 占有率（假设阈值为0.5）
            occupancy_ratio = torch.mean((generated_voxels > 0.5).float())
            
            # 记录指标
            self.log('gen_mean', mean_value, on_epoch=True)
            self.log('gen_std', std_value, on_epoch=True)
            self.log('gen_max', max_value, on_epoch=True)
            self.log('gen_min', min_value, on_epoch=True)
            self.log('gen_occupancy', occupancy_ratio, on_epoch=True)
            
            logger.info(f"生成样本统计: 均值={mean_value:.4f}, 标准差={std_value:.4f}, 占有率={occupancy_ratio:.4f}")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 根据配置决定是否使用学习率调度器
        if not self.use_scheduler:
            # 不使用调度器，保持固定学习率
            return optimizer
        
        # 创建学习率调度器
        if self.scheduler_type == "constant":
            # 固定学习率（在warmup后）
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    return 1.0  # 保持初始学习率
            
        elif self.scheduler_type == "cosine":
            # 余弦退火
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
                    
        elif self.scheduler_type == "linear":
            # 线性衰减
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                    return 1.0 - progress
        else:
            logger.warning(f"未知的调度器类型: {self.scheduler_type}，使用余弦退火")
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def generate_samples(
        self, 
        num_samples: int = 8, 
        ddim_steps: Optional[int] = None,
        ddim_eta: Optional[float] = None,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        生成新样本
        
        Args:
            num_samples (int): 生成样本数量
            ddim_steps (Optional[int]): DDIM采样步数
            ddim_eta (Optional[float]): DDIM随机性参数
            return_intermediate (bool): 是否返回中间步骤
            
        Returns:
            torch.Tensor: 生成的体素样本
        """
        self.eval()
        
        ddim_steps = ddim_steps or self.ddim_steps
        ddim_eta = ddim_eta or self.ddim_eta
        
        with torch.no_grad():
            # 确保扩散参数与模型在同一设备
            self._sync_diffusion_buffers()
            
            shape = (num_samples, 1, self.voxel_size, self.voxel_size, self.voxel_size)
            
            if return_intermediate:
                # TODO: 实现返回中间步骤的逻辑
                logger.warning("return_intermediate 功能尚未实现")
            
            generated = self.diffusion.ddim_sample(
                model=self.model,
                shape=shape,
                device=self.device,
                eta=ddim_eta,
                num_inference_steps=ddim_steps,
                progress=True
            )
            
            # 后处理
            generated = (generated + 1.0) / 2.0
            generated = torch.clamp(generated, 0.0, 1.0)
        
        self.train()
        return generated
    
    def save_checkpoint_with_samples(self, checkpoint_path: str, num_samples: int = 4):
        """
        保存检查点并生成样本
        
        Args:
            checkpoint_path (str): 检查点路径
            num_samples (int): 生成样本数量
        """
        # 保存检查点
        self.trainer.save_checkpoint(checkpoint_path)
        
        # 生成并保存样本
        samples = self.generate_samples(num_samples)
        
        # 保存样本TIFF
        checkpoint_dir = Path(checkpoint_path).parent
        samples_dir = checkpoint_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        from ..voxel.converter import PointCloudToVoxel
        converter = PointCloudToVoxel(voxel_size=self.voxel_size)
        
        for i, sample in enumerate(samples):
            sample_np = sample.squeeze().cpu().numpy()
            output_path = samples_dir / f"checkpoint_sample_{i:02d}.tiff"
            converter.save_as_tiff(sample_np, str(output_path))
        
        logger.info(f"已保存检查点和样本到: {checkpoint_dir}")
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }
    
    def on_train_start(self) -> None:
        """训练开始时的回调"""
        model_info = self.get_model_size()
        logger.info("模型参数统计:")
        logger.info(f"  总参数: {model_info['total_parameters']:,}")
        logger.info(f"  可训练参数: {model_info['trainable_parameters']:,}")
        logger.info(f"  模型大小: {model_info['model_size_mb']:.1f} MB")
        
        # 记录模型信息到tensorboard
        self.logger.log_hyperparams(self.hparams, model_info)
    
    def on_train_epoch_start(self) -> None:
        """训练epoch开始时的回调"""
        if self.current_epoch == 0:
            logger.info("开始训练3D扩散模型...")
        
        if (self.current_epoch + 1) % 10 == 0:
            logger.info(f"开始第 {self.current_epoch + 1} 个epoch")
    
    def on_validation_start(self) -> None:
        """验证开始时的回调"""
        if (self.current_epoch + 1) % self.val_sample_interval == 0:
            logger.info(f"开始验证并生成样本 (Epoch {self.current_epoch + 1})")


class EMA(nn.Module):
    """
    指数移动平均 (Exponential Moving Average)
    
    用于稳定训练和提升生成质量。
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子参数
        self.register()
    
    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.to(self.device) + self.decay * self.shadow[name].to(self.device)
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class DiffusionLightningModuleWithEMA(DiffusionLightningModule):
    """
    带EMA的3D Diffusion Lightning模块
    """
    
    def __init__(self, ema_decay: float = 0.9999, **kwargs):
        super().__init__(**kwargs)
        
        self.ema_decay = ema_decay
        self.ema = EMA(self.model, decay=ema_decay)
        
        logger.info(f"启用EMA，衰减率: {ema_decay}")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """训练步骤（带EMA更新）"""
        loss = super().training_step(batch, batch_idx)
        
        # 更新EMA
        self.ema.update()
        
        return loss
    
    def generate_validation_samples(self):
        """使用EMA模型生成验证样本"""
        logger.info(f"使用EMA模型生成验证样本 (Epoch {self.current_epoch + 1})")
        
        # 应用EMA权重
        self.ema.apply_shadow()
        
        try:
            super().generate_validation_samples()
        finally:
            # 恢复原始权重
            self.ema.restore()
    
    def generate_samples(self, **kwargs) -> torch.Tensor:
        """使用EMA模型生成样本"""
        # 应用EMA权重
        self.ema.apply_shadow()
        
        try:
            return super().generate_samples(**kwargs)
        finally:
            # 恢复原始权重
            self.ema.restore()