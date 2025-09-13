"""
体素指导模块

实现点云与体素网格之间的映射和插值功能，用于为RefineNet提供体素指导信息。
包含多尺度体素特征提取和点到体素的映射。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class VoxelGuidanceModule(nn.Module):
    """
    体素指导模块
    
    实现从体素网格中提取多尺度特征，并将这些特征映射到点云上。
    包含体素卷积网络、下采样和双线性插值功能。
    """
    
    def __init__(self, 
                 voxel_grid_size: int = 64,
                 conv_channels: List[int] = [1, 16, 32, 64],
                 downsample_factors: List[int] = [2, 2, 2],
                 interpolation_mode: str = 'bilinear'):
        """
        初始化体素指导模块
        
        Args:
            voxel_grid_size: 体素网格尺寸
            conv_channels: 卷积层通道数
            downsample_factors: 下采样因子
            interpolation_mode: 插值方式 ('bilinear' 或 'nearest')
        """
        super(VoxelGuidanceModule, self).__init__()
        
        self.voxel_grid_size = voxel_grid_size
        self.downsample_factors = downsample_factors
        self.interpolation_mode = interpolation_mode
        
        # 构建3D卷积网络
        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(conv_channels[i], conv_channels[i+1], 
                             kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(conv_channels[i+1]),
                    nn.SiLU(inplace=True),
                )
            )
        
        # 多尺度特征输出通道数
        self.feature_channels = conv_channels[1:]  # 去掉输入通道
        self.total_feature_dim = sum(self.feature_channels)
        
        print(f"体素指导模块初始化:")
        print(f"  - 体素网格尺寸: {voxel_grid_size}")
        print(f"  - 卷积通道数: {conv_channels}")
        print(f"  - 下采样因子: {downsample_factors}")
        print(f"  - 插值方式: {interpolation_mode}")
        print(f"  - 总特征维度: {self.total_feature_dim}")
    
    def normalize_points_to_voxel_coords(self, points: torch.Tensor) -> torch.Tensor:
        """
        将归一化的点云坐标转换为体素网格坐标
        
        Args:
            points: 归一化点云坐标，形状为 (batch_size, 3, num_points)，范围 [-1, 1]
            
        Returns:
            torch.Tensor: 体素网格坐标，形状为 (batch_size, 3, num_points)，范围 [0, grid_size-1]
        """
        # 从 [-1, 1] 转换到 [0, grid_size-1]
        voxel_coords = (points + 1) / 2 * (self.voxel_grid_size - 1)
        return voxel_coords
    
    def extract_voxel_features(self, voxel_grid: torch.Tensor) -> List[torch.Tensor]:
        """
        从体素网格中提取多尺度特征
        
        Args:
            voxel_grid: 体素网格，形状为 (batch_size, C, D, H, W)
            
        Returns:
            List[torch.Tensor]: 多尺度特征列表
        """
        features = []
        x = voxel_grid
        features.append(x)
        
        # 通过卷积层提取特征
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            features.append(x)
            
            # 应用下采样（除了最后一层）
            if i < len(self.downsample_factors):
                downsample_factor = self.downsample_factors[i]
                x = F.avg_pool3d(x, kernel_size=downsample_factor, stride=downsample_factor)
        
        return features
    
    def interpolate_features_to_points(self, 
                                     features: torch.Tensor, 
                                     points: torch.Tensor,
                                     feature_grid_size: int) -> torch.Tensor:
        """
        将体素特征通过插值映射到点云上
        
        Args:
            features: 体素特征，形状为 (batch_size, C, D, H, W)
            points: 归一化点云坐标，形状为 (batch_size, 3, num_points)，范围 [-1, 1]
            feature_grid_size: 特征网格的尺寸
            
        Returns:
            torch.Tensor: 插值后的点特征，形状为 (batch_size, C, num_points)
        """
        batch_size, C, D, H, W = features.shape
        _, _, num_points = points.shape
        
        # # 将点坐标转换为体素网格坐标，然后归一化到 [-1, 1] 用于grid_sample
        # voxel_coords = self.normalize_points_to_voxel_coords(points)
        
        # # 归一化到 [-1, 1] 用于grid_sample
        # normalized_coords = voxel_coords / (feature_grid_size - 1) * 2 - 1
        
        # # 调整坐标格式用于grid_sample: (batch_size, num_points, 1, 1, 3)
        # # grid_sample期望的格式是 (N, H_out, W_out, D_out, 3)，我们设置H_out=num_points, W_out=D_out=1
        # grid = normalized_coords.permute(0, 2, 1).unsqueeze(2).unsqueeze(2)  # (batch_size, num_points, 1, 1, 3)
        
        grid = points.permute(0, 2, 1).unsqueeze(2).unsqueeze(2)  # (batch_size, num_points, 1, 1, 3)
        
        # 执行三线性插值
        # features: (batch_size, C, D, H, W)
        # grid: (batch_size, num_points, 1, 1, 3)
        # 输出: (batch_size, C, num_points, 1, 1)
        interpolated = F.grid_sample(
            features, grid, 
            mode=self.interpolation_mode, 
            padding_mode='border', 
            align_corners=False
        )
        
        # 调整输出形状: (batch_size, C, num_points, 1, 1) -> (batch_size, C, num_points)
        interpolated = interpolated.squeeze(-1).squeeze(-1)
        
        return interpolated
    
    def forward(self, voxel_grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播：从体素网格中提取多尺度特征并映射到点云
        
        Args:
            voxel_grid: 体素网格，形状为 (batch_size, C, D, H, W)
            points: 归一化点云坐标，形状为 (batch_size, 3, num_points)，范围 [-1, 1]
            
        Returns:
            torch.Tensor: 点云的体素指导特征，形状为 (batch_size, total_feature_dim, num_points)
        """
        # 提取多尺度体素特征
        multiscale_features = self.extract_voxel_features(voxel_grid)
        
        # 将每个尺度的特征插值到点云上
        point_features = []
        current_grid_size = self.voxel_grid_size
        
        for i, features in enumerate(multiscale_features):
            # 插值到点云
            interpolated_features = self.interpolate_features_to_points(
                features, points, current_grid_size
            )
            point_features.append(interpolated_features)
            
            # 更新下一层的网格尺寸
            if i < len(self.downsample_factors):
                current_grid_size = current_grid_size // self.downsample_factors[i]
        
        # 拼接所有尺度的特征
        combined_features = torch.cat(point_features, dim=1)  # (batch_size, total_feature_dim, num_points)
        
        return combined_features


class VoxelFeatureProjector(nn.Module):
    """
    体素特征投影器
    
    将体素指导特征投影到适合PointNet++的特征空间
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        初始化投影器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
        """
        super(VoxelFeatureProjector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Conv1d(input_dim+1, output_dim, kernel_size=1, bias=False), # 1 for original feature
            nn.BatchNorm1d(output_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征，形状为 (batch_size, input_dim+1, num_points) # 1 for original feature
            
        Returns:
            torch.Tensor: 投影后的特征，形状为 (batch_size, output_dim, num_points)
        """
        return self.projector(features)


def test_voxel_guidance():
    """测试体素指导模块"""
    print("测试体素指导模块...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    guidance_module = VoxelGuidanceModule(
        voxel_grid_size=64,
        conv_channels=[1, 16, 32, 64],
        downsample_factors=[2, 2, 2]
    ).to(device)
    
    # 创建测试数据
    batch_size = 2
    num_points = 2048
    voxel_size = 64
    
    # 体素网格: (batch_size, 1, 64, 64, 64)
    voxel_grid = torch.randn(batch_size, 1, voxel_size, voxel_size, voxel_size).to(device)
    
    # 归一化点云: (batch_size, 3, num_points)
    points = torch.randn(batch_size, 3, num_points).to(device) * 0.8  # 范围大约 [-1, 1]
    
    # 前向传播
    with torch.no_grad():
        voxel_features = guidance_module(voxel_grid, points)
    
    print(f"输入体素网格形状: {voxel_grid.shape}")
    print(f"输入点云形状: {points.shape}")
    print(f"输出体素特征形状: {voxel_features.shape}")
    print(f"总特征维度: {guidance_module.total_feature_dim}")
    
    # 测试投影器
    projector = VoxelFeatureProjector(
        input_dim=guidance_module.total_feature_dim,
        output_dim=64
    ).to(device)
    
    with torch.no_grad():
        projected_features = projector(voxel_features)
    
    print(f"投影后特征形状: {projected_features.shape}")
    print("体素指导模块测试完成！")


if __name__ == "__main__":
    test_voxel_guidance()


