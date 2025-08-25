"""
3D Diffusion模型

基于DDPM (Denoising Diffusion Probabilistic Models)的3D体素生成模型。
支持可配置的体素分辨率和UNet架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class TimeEmbedding(nn.Module):
    """时间步嵌入层，用于将时间步编码为特征向量"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: shape (batch_size,) 的时间步张量
            
        Returns:
            time_emb: shape (batch_size, dim) 的时间嵌入
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 创建正弦位置编码
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock3D(nn.Module):
    """3D残差块，包含时间嵌入的3D卷积层"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # GroupNorm的组数不能超过通道数，且通道数必须被组数整除
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        self.dropout = nn.Dropout3d(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, in_channels, D, H, W) 的输入
            time_emb: shape (batch_size, time_emb_dim) 的时间嵌入
            
        Returns:
            output: shape (batch_size, out_channels, D, H, W) 的输出
        """
        # 第一个卷积层
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # 时间嵌入
        time_h = self.time_mlp(time_emb)
        h = h + time_h[..., None, None, None]
        
        # 第二个卷积层
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # 残差连接
        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """3D自注意力块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # GroupNorm的组数不能超过通道数，且通道数必须被组数整除
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, channels, D, H, W) 的输入
            
        Returns:
            output: shape (batch_size, channels, D, H, W) 的输出
        """
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为序列格式
        q = q.view(B, C, D * H * W).transpose(1, 2)  # (B, DHW, C)
        k = k.view(B, C, D * H * W).transpose(1, 2)  # (B, DHW, C)
        v = v.view(B, C, D * H * W).transpose(1, 2)  # (B, DHW, C)
        
        # 计算注意力
        scale = (C ** -0.5)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        h = torch.bmm(attn, v)  # (B, DHW, C)
        h = h.transpose(1, 2).view(B, C, D, H, W)
        
        h = self.proj_out(h)
        
        return x + h


class UNet3D(nn.Module):
    """3D UNet架构，用于噪声预测"""
    
    def __init__(
        self, 
        voxel_size: int = 64,
        in_channels: int = 1,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        channel_mult: List[int] = [1, 2, 4, 8],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 输入卷积
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        input_block_chans = [ch]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = model_channels * mult
                layers = [ResidualBlock3D(ch, out_ch, time_embed_dim, dropout)]
                
                # 添加注意力层
                if use_attention and (voxel_size // (2 ** level)) in attention_resolutions:
                    layers.append(AttentionBlock3D(out_ch))
                
                self.down_blocks.append(nn.Sequential(*layers))
                ch = out_ch
                input_block_chans.append(ch)
            
            # 下采样（除了最后一层）
            if level != len(channel_mult) - 1:
                self.down_samples.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
        
        # 中间层
        self.middle_block = nn.Sequential(
            ResidualBlock3D(ch, ch, time_embed_dim, dropout),
            AttentionBlock3D(ch) if use_attention else nn.Identity(),
            ResidualBlock3D(ch, ch, time_embed_dim, dropout)
        )
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResidualBlock3D(ch + ich, model_channels * mult, time_embed_dim, dropout)]
                ch = model_channels * mult
                
                # 添加注意力层
                if use_attention and (voxel_size // (2 ** level)) in attention_resolutions:
                    layers.append(AttentionBlock3D(ch))
                
                # 上采样（除了最后一层的最后一个块）
                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1))
                
                self.up_blocks.append(nn.Sequential(*layers))
        
        # 输出层
        # GroupNorm的组数不能超过通道数，且通道数必须被组数整除
        output_num_groups = min(8, model_channels)
        while model_channels % output_num_groups != 0:
            output_num_groups -= 1
        
        self.output = nn.Sequential(
            nn.GroupNorm(output_num_groups, model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, in_channels, 3, padding=1)
        )
        
        self.initialize_weights()
        
        logger.info(f"初始化3D UNet: 体素大小={voxel_size}, 模型通道={model_channels}")
        logger.info(f"通道倍数={channel_mult}, 注意力分辨率={attention_resolutions}")
    
    def initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: shape (batch_size, 1, D, H, W) 的噪声体素
            timesteps: shape (batch_size,) 的时间步
            
        Returns:
            noise_pred: shape (batch_size, 1, D, H, W) 的预测噪声
        """
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 输入卷积
        h = self.input_conv(x)
        
        # 存储跳跃连接
        hs = [h]
        
        # 下采样路径
        down_sample_idx = 0
        for level, mult in enumerate(self.channel_mult):
            for block_idx in range(self.num_res_blocks):
                # 计算全局块索引
                global_idx = level * self.num_res_blocks + block_idx
                down_block = self.down_blocks[global_idx]
                
                h = down_block[0](h, time_emb)  # ResidualBlock3D
                if len(down_block) > 1:  # 如果有注意力层
                    h = down_block[1](h)
                hs.append(h)
            
            # 在每个level结束后进行下采样（除了最后一层）
            if level != len(self.channel_mult) - 1:
                h = self.down_samples[down_sample_idx](h)
                hs.append(h)
                down_sample_idx += 1
        
        # 中间层
        h = self.middle_block[0](h, time_emb)  # ResidualBlock3D
        if len(self.middle_block) > 2:  # 有三个层：ResBlock -> Attention -> ResBlock
            h = self.middle_block[1](h)  # AttentionBlock3D 
            h = self.middle_block[2](h, time_emb)  # ResidualBlock3D
        elif len(self.middle_block) > 1:  # 只有两个ResBlock
            h = self.middle_block[1](h, time_emb)  # ResidualBlock3D
        
        # 上采样路径
        for up_block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            
            # 处理上采样块中的每个层
            for layer in up_block:
                if isinstance(layer, ResidualBlock3D):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        # 输出层
        return self.output(h)


class GaussianDiffusion:
    """高斯扩散过程"""
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        # 生成beta调度
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"未知的beta调度: {beta_schedule}")
        
        # 预计算常用的量
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # 用于DDIM采样的量
        self.register_buffer("posterior_variance", 
                           betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        logger.info(f"初始化高斯扩散: {num_timesteps} 步, beta范围=[{beta_start}, {beta_end}]")
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区（模拟nn.Module的行为）"""
        setattr(self, name, tensor)
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """余弦beta调度"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程，向图像添加噪声
        
        Args:
            x_start: 原始无噪声图像
            t: 时间步
            noise: 可选的预定义噪声
            
        Returns:
            x_t: 添加噪声后的图像
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 确保所有张量在同一设备上
        device = x_start.device
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        
        Args:
            x_start: 原始图像
            x_t: 当前时间步的图像
            t: 时间步
            
        Returns:
            后验均值、方差和对数方差
        """
        # 确保所有张量在同一设备上
        device = x_start.device
        
        posterior_mean = (
            self.posterior_mean_coef1[t].to(device).view(-1, 1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].to(device).view(-1, 1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].to(device).view(-1, 1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].to(device).view(-1, 1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        从噪声预测原始图像
        
        Args:
            x_t: 当前噪声图像
            t: 时间步
            noise: 预测的噪声
            
        Returns:
            x_0: 预测的原始图像
        """
        # 确保所有张量在同一设备上
        device = x_t.device
        
        return (
            self.sqrt_recip_alphas_cumprod[t].to(device).view(-1, 1, 1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].to(device).view(-1, 1, 1, 1, 1) * noise
        )
    
    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 p(x_{t-1} | x_t) 的均值和方差
        
        Args:
            model: 噪声预测模型
            x: 当前时间步的图像
            t: 时间步
            
        Returns:
            模型均值、方差、对数方差和预测的x_0
        """
        model_output = model(x, t)
        
        # 预测原始图像
        pred_x_start = self.predict_start_from_noise(x, t, model_output)
        pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # 计算后验均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x, t
        )
        
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        从 p(x_{t-1} | x_t) 采样
        
        Args:
            model: 噪声预测模型
            x: 当前时间步的图像
            t: 时间步
            
        Returns:
            x_{t-1}: 上一时间步的图像
        """
        model_mean, _, model_log_variance, _ = self.p_mean_variance(model, x, t)
        
        noise = torch.randn_like(x)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], device: torch.device, progress: bool = True) -> torch.Tensor:
        """
        完整的采样循环
        
        Args:
            model: 噪声预测模型
            shape: 输出形状
            device: 设备
            progress: 是否显示进度
            
        Returns:
            生成的样本
        """
        img = torch.randn(shape, device=device)
        
        timesteps = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="采样中")
        
        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t_tensor)
        
        return img
    
    def ddim_sample(
        self, 
        model: nn.Module, 
        shape: Tuple[int, ...], 
        device: torch.device,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM采样（确定性采样）
        
        Args:
            model: 噪声预测模型
            shape: 输出形状
            device: 设备
            eta: 随机性参数（0为完全确定性）
            num_inference_steps: 推理步数
            progress: 是否显示进度
            
        Returns:
            生成的样本
        """
        # 创建时间步调度
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        # 初始噪声
        img = torch.randn(shape, device=device)
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="DDIM采样中")
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = model(img, t_tensor)
            
            # 计算alpha值
            alpha_prod_t = self.alphas_cumprod[t].to(device)
            alpha_prod_t_prev = self.alphas_cumprod[timesteps[i + 1]].to(device) if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # 预测原始图像
            pred_original_sample = (img - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            
            # 计算方向
            pred_sample_direction = (1 - alpha_prod_t_prev - eta ** 2 * beta_prod_t_prev) ** (0.5) * noise_pred
            
            # 计算前一个样本
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            
            # 添加噪声（如果eta > 0）
            if eta > 0:
                variance = eta ** 2 * beta_prod_t_prev
                noise = torch.randn_like(img)
                prev_sample = prev_sample + variance ** (0.5) * noise
            
            img = prev_sample
        
        return img