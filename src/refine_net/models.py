import numpy as np
from torch.autograd import Variable
from src.refine_net.pointnet2_module import *
from src.refine_net.voxel_guidance import VoxelGuidanceModule, VoxelFeatureProjector


class PointNet2Generator(torch.nn.Module):
    '''
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    '''
    def __init__(self, device, args):
        super(PointNet2Generator, self).__init__()
        nsamples = 32
        
        # 体素指导相关设置
        self.use_voxel_guidance = getattr(args, 'use_voxel_guidance', False)
        voxel_feature_dim = 0
        
        if self.use_voxel_guidance:
            # 初始化体素指导模块
            voxel_grid_size = getattr(args, 'voxel_grid_size', 64)
            voxel_conv_channels = getattr(args, 'voxel_conv_channels', [1, 16, 32, 64])
            voxel_downsample_factors = getattr(args, 'voxel_downsample_factors', [2, 2, 2])
            voxel_interpolation_mode = getattr(args, 'voxel_interpolation_mode', 'bilinear')
            
            self.voxel_guidance = VoxelGuidanceModule(
                voxel_grid_size=voxel_grid_size,
                conv_channels=voxel_conv_channels,
                downsample_factors=voxel_downsample_factors,
                interpolation_mode=voxel_interpolation_mode
            )
            
            # 体素特征投影器，将体素特征投影到与点云特征相同的维度
            voxel_feature_dim = 64  # 投影后的维度
            self.voxel_projector = VoxelFeatureProjector(
                input_dim=self.voxel_guidance.total_feature_dim,
                output_dim=voxel_feature_dim
            )
            
            print(f"启用体素指导: 特征维度 {self.voxel_guidance.total_feature_dim} -> {voxel_feature_dim}")
        else:
            self.voxel_guidance = None
            self.voxel_projector = None
            print("未启用体素指导")

        # SA1 - 如果使用体素指导，增加输入特征维度
        sa1_sample_ratio = 0.7
        sa1_radius = 0.1 # 0.025
        sa1_max_num_neighbours = nsamples
        # 输入特征维度：3(坐标) + 3(原始特征) + voxel_feature_dim(体素特征)
        # TODO: 目前只在SA1做体素特征嵌入, 后续可以考虑在SA2, SA3, SA4也做体素特征嵌入
        sa1_input_dim = 3 + 3 + voxel_feature_dim
        sa1_mlp = make_mlp(sa1_input_dim, [64, 64, 64])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio,
                                            sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.7
        sa2_radius = 0.3 # 0.05
        sa2_max_num_neighbours = nsamples
        sa2_mlp = make_mlp(64+3, [64, 64, 64, 128])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio,
                                            sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # SA3
        sa3_sample_ratio = 0.7
        sa3_radius = 0.5 # 0.1
        sa3_max_num_neighbours = nsamples
        sa3_mlp = make_mlp(128 + 3, [128, 128, 128, 256])
        self.sa3_module = PointNet2SAModule(sa3_sample_ratio,
                                            sa3_radius, sa3_max_num_neighbours, sa3_mlp)

        # SA4
        sa4_sample_ratio = 0.7
        sa4_radius = 0.7 # 0.2
        sa4_max_num_neighbours = nsamples
        sa4_mlp = make_mlp(256 + 3, [256, 256, 256, 512])
        self.sa4_module = PointNet2SAModule(sa4_sample_ratio,
                                            sa4_radius, sa4_max_num_neighbours, sa4_mlp)

        knn_num = 3

        # FP3, reverse of sa3
        fp4_knn_num = knn_num
        fp4_mlp = make_mlp(512 + 256 + 3, [512, 256, 256, 256])
        self.fp4_module = PointNet2FPModule(fp4_knn_num, fp4_mlp)

        # FP3, reverse of sa3
        fp3_knn_num = knn_num
        fp3_mlp = make_mlp(256+128+3, [256, 256, 256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256+64+3, [256, 256])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        # 计算FP1的输入维度：256(fp2输出) + 3(坐标) + 3(原始特征) + voxel_feature_dim(体素特征)
        fp1_input_dim = 256 + 3 + 3 + voxel_feature_dim
        fp1_mlp = make_mlp(fp1_input_dim, [256, 128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc = torch.nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            torch.nn.SiLU(True),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.Conv1d(32, 16, kernel_size=1, bias=False),
            nn.Conv1d(16, 3, kernel_size=1, bias=False),
        )


    def forward(self, data, voxel_grid=None):
        '''
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input
            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes
                data.pos: (batch_size * ~num_points, 3)
                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        voxel_grid: 体素数据 (batch_size, C, D, H, W), 可选参数，仅在启用体素指导时使用
        '''
        input_points = data.clone()
        
        # 如果启用体素指导且提供了体素数据，提取体素特征
        voxel_features = None
        if self.use_voxel_guidance and voxel_grid is not None:
            # 提取体素指导特征: (batch_size, total_feature_dim, num_points)
            raw_voxel_features = self.voxel_guidance(voxel_grid, data)
            # 投影到目标维度: (batch_size, voxel_feature_dim, num_points)
            voxel_features = self.voxel_projector(raw_voxel_features)

        # Convert to torch_geometric.data.Data type
        data_transposed = data.transpose(1, 2).contiguous()
        batch_size, N, _ = data_transposed.shape  # (batch_size, num_points, 3)
        pos = data_transposed.view(batch_size*N, -1)
        batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
        for i in range(batch_size): batch[i] = i
        batch = batch.view(-1)

        # 构建初始特征
        if self.use_voxel_guidance and voxel_features is not None:
            # 融合点云坐标和体素特征
            # voxel_features: (batch_size, voxel_feature_dim, num_points)
            # 转换为 (batch_size*num_points, voxel_feature_dim)
            voxel_features_flat = voxel_features.transpose(1, 2).contiguous().view(batch_size*N, -1)
            
            # 拼接坐标和体素特征: (batch_size*num_points, 3+voxel_feature_dim)
            initial_features = torch.cat([pos[:, :3], voxel_features_flat], dim=1)
        else:
            # 仅使用坐标特征: (batch_size*num_points, 3)
            initial_features = pos

        data = Data()
        data.x, data.pos, data.batch = initial_features, pos[:, :3].detach(), batch

        if not hasattr(data, 'x'): data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)
        sa4_out = self.sa4_module(sa3_out)

        fp4_out = self.fp4_module(sa4_out, sa3_out)
        fp3_out = self.fp3_module(fp4_out, sa2_out)
        fp2_out = self.fp2_module(fp3_out, sa1_out)
        fp1_out = self.fp1_module(fp2_out, data_in)

        fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out
        x = self.fc(fp1_out_x.transpose(1, 0).unsqueeze(0)).transpose(1, 2).squeeze(0)
        return x.view(batch_size, N, 3).transpose(2, 1) + input_points

    def initialize_params(self, val=0.2):
        self.std = val
        for p in self.parameters():
            torch.nn.init.uniform_(p.data, -val, val)
