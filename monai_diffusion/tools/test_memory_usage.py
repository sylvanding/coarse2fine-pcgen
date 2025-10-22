"""
显存占用测试脚本

测试不同配置下 AutoencoderKL 的显存占用，帮助选择合适的配置。
"""

import sys
from pathlib import Path
import argparse

# 添加路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from generative.losses import PerceptualLoss

def format_memory(bytes_val):
    """格式化显存大小"""
    return f"{bytes_val / 1024**3:.2f} GB"

def test_configuration(
    voxel_size=(256, 256, 64),
    batch_size=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    attention_levels=(False, False, True),
    use_amp=True,
    use_perceptual=True,
    use_discriminator=True
):
    """
    测试指定配置的显存占用
    
    Args:
        voxel_size: 体素分辨率
        batch_size: 批次大小
        num_channels: 通道数配置
        latent_channels: 潜在空间通道数
        attention_levels: 注意力层配置
        use_amp: 是否使用混合精度
        use_perceptual: 是否使用感知损失
        use_discriminator: 是否使用判别器
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，无法测试显存占用")
        return
    
    print("\n" + "="*80)
    print(f"测试配置:")
    print(f"  体素分辨率: {voxel_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  通道数: {num_channels}")
    print(f"  潜在通道数: {latent_channels}")
    print(f"  注意力层: {attention_levels}")
    print(f"  混合精度: {use_amp}")
    print(f"  感知损失: {use_perceptual}")
    print(f"  判别器: {use_discriminator}")
    print("="*80)
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 初始显存
    initial_memory = torch.cuda.memory_allocated()
    print(f"\n初始显存: {format_memory(initial_memory)}")
    
    try:
        # 1. 创建 Autoencoder
        print("\n[1/6] 创建 AutoencoderKL...")
        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=num_channels,
            latent_channels=latent_channels,
            num_res_blocks=1,
            norm_num_groups=min(16, num_channels[0]),
            attention_levels=attention_levels
        ).to(device)
        
        after_model = torch.cuda.memory_allocated()
        print(f"  模型显存: {format_memory(after_model - initial_memory)}")
        
        # 2. 创建判别器
        if use_discriminator:
            print("\n[2/6] 创建 PatchDiscriminator...")
            discriminator = PatchDiscriminator(
                spatial_dims=3,
                num_layers_d=3,
                num_channels=32,
                in_channels=1,
                out_channels=1
            ).to(device)
            after_disc = torch.cuda.memory_allocated()
            print(f"  判别器显存: {format_memory(after_disc - after_model)}")
        else:
            after_disc = after_model
            print("\n[2/6] 跳过判别器")
        
        # 3. 创建感知损失
        if use_perceptual:
            print("\n[3/6] 创建 PerceptualLoss...")
            loss_perceptual = PerceptualLoss(
                spatial_dims=3,
                network_type="squeeze",
                is_fake_3d=True,
                fake_3d_ratio=0.2
            ).to(device)
            after_perceptual = torch.cuda.memory_allocated()
            print(f"  感知损失网络显存: {format_memory(after_perceptual - after_disc)}")
        else:
            after_perceptual = after_disc
            print("\n[3/6] 跳过感知损失")
        
        # 4. 创建输入数据
        print(f"\n[4/6] 创建输入数据 ({batch_size}, 1, {voxel_size[0]}, {voxel_size[1]}, {voxel_size[2]})...")
        images = torch.randn(batch_size, 1, *voxel_size, device=device)
        after_input = torch.cuda.memory_allocated()
        print(f"  输入数据显存: {format_memory(after_input - after_perceptual)}")
        
        # 5. 前向传播
        print("\n[5/6] 执行前向传播（无梯度）...")
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
            else:
                reconstruction, z_mu, z_sigma = autoencoder(images)
        
        after_forward_no_grad = torch.cuda.memory_allocated()
        print(f"  前向传播（无梯度）显存: {format_memory(after_forward_no_grad - after_input)}")
        
        # 6. 前向传播 + 反向传播
        print("\n[6/6] 执行前向+反向传播...")
        autoencoder.zero_grad()
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                reconstruction, z_mu, z_sigma = autoencoder(images)
                loss = torch.nn.functional.mse_loss(reconstruction, images)
            scaler.scale(loss).backward()
        else:
            reconstruction, z_mu, z_sigma = autoencoder(images)
            loss = torch.nn.functional.mse_loss(reconstruction, images)
            loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        print(f"  当前显存: {format_memory(current_memory)}")
        print(f"  峰值显存: {format_memory(peak_memory)}")
        
        # 总结
        print("\n" + "="*80)
        print("显存占用总结:")
        print(f"  模型参数: {format_memory(after_model - initial_memory)}")
        if use_discriminator:
            print(f"  判别器: {format_memory(after_disc - after_model)}")
        if use_perceptual:
            print(f"  感知损失网络: {format_memory(after_perceptual - after_disc)}")
        print(f"  输入数据: {format_memory(after_input - after_perceptual)}")
        print(f"  训练时峰值: {format_memory(peak_memory)}")
        print("="*80)
        
        # 清理
        del autoencoder
        if use_discriminator:
            del discriminator
        if use_perceptual:
            del loss_perceptual
        del images, reconstruction, z_mu, z_sigma, loss
        torch.cuda.empty_cache()
        
        return peak_memory
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "!"*80)
            print("显存不足 (OOM)!")
            print("!"*80)
            torch.cuda.empty_cache()
            return None
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="测试 AutoencoderKL 显存占用")
    parser.add_argument("--resolution", type=str, default="256,256,64", 
                       help="体素分辨率，格式: X,Y,Z (默认: 256,256,64)")
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="批次大小 (默认: 1)")
    parser.add_argument("--channels", type=str, default="32,64,64",
                       help="通道数，格式: C1,C2,C3 (默认: 32,64,64)")
    args = parser.parse_args()
    
    # 解析参数
    voxel_size = tuple(map(int, args.resolution.split(",")))
    channels = tuple(map(int, args.channels.split(",")))
    
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        return
    
    # 显示GPU信息
    print("\nGPU 信息:")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试不同配置
    configs = [
        {
            "name": "基准配置 (FP32, 全部启用)",
            "voxel_size": voxel_size,
            "batch_size": args.batch_size,
            "num_channels": channels,
            "use_amp": False,
            "use_perceptual": True,
            "use_discriminator": True,
        },
        {
            "name": "启用混合精度 (AMP)",
            "voxel_size": voxel_size,
            "batch_size": args.batch_size,
            "num_channels": channels,
            "use_amp": True,
            "use_perceptual": True,
            "use_discriminator": True,
        },
        {
            "name": "AMP + 禁用感知损失",
            "voxel_size": voxel_size,
            "batch_size": args.batch_size,
            "num_channels": channels,
            "use_amp": True,
            "use_perceptual": False,
            "use_discriminator": True,
        },
        {
            "name": "AMP + 减少通道数 [16,32,64]",
            "voxel_size": voxel_size,
            "batch_size": args.batch_size,
            "num_channels": (16, 32, 64),
            "use_amp": True,
            "use_perceptual": False,
            "use_discriminator": True,
        },
    ]
    
    results = {}
    
    for config in configs:
        name = config.pop("name")
        print(f"\n\n{'='*80}")
        print(f"测试: {name}")
        print('='*80)
        
        peak_mem = test_configuration(**config)
        results[name] = peak_mem
    
    # 显示对比
    print("\n\n" + "="*80)
    print("显存占用对比:")
    print("="*80)
    
    for name, peak_mem in results.items():
        if peak_mem is not None:
            print(f"  {name}: {format_memory(peak_mem)}")
        else:
            print(f"  {name}: OOM (显存不足)")
    
    print("="*80)
    
    # 给出建议
    print("\n建议:")
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    
    for name, peak_mem in results.items():
        if peak_mem is not None and peak_mem < gpu_total * 0.9:
            print(f"✓ 推荐使用: {name}")
            print(f"  峰值显存: {format_memory(peak_mem)} / {format_memory(gpu_total)}")
            print(f"  显存利用率: {peak_mem / gpu_total * 100:.1f}%")
            break
    else:
        print("⚠ 警告: 所有配置都可能导致显存不足!")
        print("建议:")
        print("  1. 降低分辨率")
        print("  2. 进一步减少通道数")
        print("  3. 使用梯度检查点")


if __name__ == "__main__":
    main()

