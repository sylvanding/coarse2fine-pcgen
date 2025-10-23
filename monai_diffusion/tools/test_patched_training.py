"""
测试Patch-Based训练配置

验证新的patch-based训练配置是否能正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

import torch
import yaml
from monai.utils import set_determinism
import logging

from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from monai_diffusion.datasets import create_train_val_dataloaders
from monai_diffusion.utils import AutoencoderSlidingWindowInferer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loading(config_path: str):
    """测试数据加载"""
    logger.info("=" * 60)
    logger.info("测试1: 数据加载")
    logger.info("=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # 尝试创建数据加载器
        logger.info("创建训练和验证数据加载器...")
        train_loader, val_loader = create_train_val_dataloaders(config)
        
        logger.info(f"✅ 数据加载器创建成功")
        logger.info(f"   训练集大小: {len(train_loader.dataset)}")
        logger.info(f"   验证集大小: {len(val_loader.dataset)}")
        logger.info(f"   Batch大小: {train_loader.batch_size}")
        
        # 尝试加载一个batch
        logger.info("加载一个训练batch...")
        batch = next(iter(train_loader))
        logger.info(f"✅ Batch加载成功")
        logger.info(f"   图像形状: {batch['image'].shape}")
        logger.info(f"   数据类型: {batch['image'].dtype}")
        logger.info(f"   数值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        
        # 检查是否是期望的patch大小
        expected_size = tuple(config['data']['voxel_size'])
        actual_size = tuple(batch['image'].shape[2:])  # (B, C, H, W, D)
        
        if actual_size == expected_size:
            logger.info(f"✅ Patch大小正确: {actual_size}")
        else:
            logger.warning(f"⚠️ Patch大小不匹配: 期望{expected_size}, 实际{actual_size}")
        
        return True, train_loader, val_loader
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_model_creation(config_path: str):
    """测试模型创建"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 模型创建")
    logger.info("=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ae_config = config['autoencoder']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建AutoencoderKL
        logger.info("创建AutoencoderKL...")
        autoencoder = AutoencoderKL(
            spatial_dims=ae_config['spatial_dims'],
            in_channels=ae_config['in_channels'],
            out_channels=ae_config['out_channels'],
            num_channels=tuple(ae_config['num_channels']),
            latent_channels=ae_config['latent_channels'],
            num_res_blocks=ae_config['num_res_blocks'],
            norm_num_groups=ae_config.get('norm_num_groups', 16),
            attention_levels=tuple(ae_config['attention_levels']),
            downsample_factors=tuple(ae_config.get('downsample_factors', [2, 2])),
            initial_downsample_factor=ae_config.get('initial_downsample_factor', 1)
        )
        autoencoder.to(device)
        logger.info(f"✅ AutoencoderKL创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in autoencoder.parameters())
        trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        logger.info(f"   总参数量: {total_params:,}")
        logger.info(f"   可训练参数: {trainable_params:,}")
        
        # 创建判别器
        logger.info("创建PatchDiscriminator...")
        disc_config = ae_config['training']['discriminator']
        discriminator = PatchDiscriminator(
            spatial_dims=ae_config['spatial_dims'],
            num_layers_d=disc_config['num_layers_d'],
            num_channels=disc_config['num_channels'],
            in_channels=ae_config['in_channels'],
            out_channels=ae_config['out_channels']
        )
        discriminator.to(device)
        logger.info(f"✅ PatchDiscriminator创建成功")
        
        disc_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(f"   判别器参数量: {disc_params:,}")
        
        return True, autoencoder, discriminator, device
        
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_forward_pass(autoencoder, discriminator, train_loader, device):
    """测试前向传播"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 前向传播")
    logger.info("=" * 60)
    
    try:
        # 获取一个batch
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        
        logger.info(f"输入图像形状: {images.shape}")
        
        # 测试Autoencoder前向传播
        logger.info("测试Autoencoder前向传播...")
        autoencoder.eval()
        with torch.no_grad():
            reconstruction, z_mu, z_sigma = autoencoder(images)
        
        logger.info(f"✅ Autoencoder前向传播成功")
        logger.info(f"   重建图像形状: {reconstruction.shape}")
        logger.info(f"   Latent均值形状: {z_mu.shape}")
        logger.info(f"   Latent标准差形状: {z_sigma.shape}")
        
        # 计算重建误差
        recon_error = torch.abs(reconstruction - images).mean().item()
        logger.info(f"   重建误差 (随机初始化): {recon_error:.4f}")
        
        # 测试Discriminator前向传播
        logger.info("测试Discriminator前向传播...")
        discriminator.eval()
        with torch.no_grad():
            disc_out = discriminator(images)
        
        logger.info(f"✅ Discriminator前向传播成功")
        logger.info(f"   判别器输出层数: {len(disc_out)}")
        logger.info(f"   最终输出形状: {disc_out[-1].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sliding_window_inference(autoencoder, device, config):
    """测试滑动窗口推理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 滑动窗口推理")
    logger.info("=" * 60)
    
    try:
        # 创建一个大的测试体积（模拟256×256×64）
        large_volume = torch.randn(1, 1, 192, 192, 96).to(device)
        logger.info(f"创建大体积用于测试: {large_volume.shape}")
        
        # 创建滑动窗口推理器
        roi_size = tuple(config['data']['voxel_size'])
        logger.info(f"ROI大小 (训练patch大小): {roi_size}")
        
        inferer = AutoencoderSlidingWindowInferer(
            autoencoder=autoencoder,
            roi_size=roi_size,
            sw_batch_size=4,
            overlap=0.25
        )
        
        # 测试重建
        logger.info("使用滑动窗口进行重建...")
        with torch.no_grad():
            reconstruction = inferer.reconstruct(large_volume)
        
        logger.info(f"✅ 滑动窗口推理成功")
        logger.info(f"   输出形状: {reconstruction.shape}")
        logger.info(f"   形状是否匹配: {reconstruction.shape == large_volume.shape}")
        
        # 测试编码
        logger.info("测试滑动窗口编码...")
        with torch.no_grad():
            z_mu, z_sigma = inferer.encode(large_volume)
        
        logger.info(f"✅ 滑动窗口编码成功")
        logger.info(f"   Latent形状: {z_mu.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 滑动窗口推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(autoencoder, discriminator, train_loader, device):
    """测试显存占用"""
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 显存占用")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.info("⚠️ CUDA不可用，跳过显存测试")
        return True
    
    try:
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 获取一个batch
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        
        # 测试前向传播的显存占用
        autoencoder.train()
        discriminator.train()
        
        # Generator前向传播
        reconstruction, z_mu, z_sigma = autoencoder(images)
        loss_g = torch.nn.functional.mse_loss(reconstruction, images)
        
        # Discriminator前向传播
        disc_out_fake = discriminator(reconstruction.detach())
        disc_out_real = discriminator(images)
        
        # 记录峰值显存
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        logger.info(f"✅ 显存占用测试完成")
        logger.info(f"   峰值显存: {peak_memory_mb:.1f} MB")
        logger.info(f"   当前显存: {current_memory_mb:.1f} MB")
        logger.info(f"   Batch大小: {images.shape[0]}")
        logger.info(f"   每个样本显存: {peak_memory_mb / images.shape[0]:.1f} MB")
        
        # 清理
        del reconstruction, z_mu, z_sigma, loss_g, disc_out_fake, disc_out_real
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 显存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    logger.info("🚀 开始测试Patch-Based训练配置")
    logger.info("=" * 60)
    
    # 配置文件路径
    config_path = "monai_diffusion/config/ldm_config_patched.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return
    
    logger.info(f"配置文件: {config_path}\n")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_determinism(config.get('seed', 42))
    
    # 运行所有测试
    results = {}
    
    # 测试1: 数据加载
    success, train_loader, val_loader = test_data_loading(config_path)
    results['data_loading'] = success
    
    if not success:
        logger.error("\n❌ 数据加载测试失败，无法继续后续测试")
        logger.error("请检查数据路径是否正确，或者跳过此测试运行其他测试")
        # 不要完全退出，继续测试其他功能
        logger.info("\n继续测试模型创建（不依赖数据）...")
    
    # 测试2: 模型创建
    success, autoencoder, discriminator, device = test_model_creation(config_path)
    results['model_creation'] = success
    
    if not success:
        logger.error("\n❌ 模型创建测试失败，无法继续后续测试")
        return
    
    # 如果数据加载成功，继续测试
    if results['data_loading']:
        # 测试3: 前向传播
        success = test_forward_pass(autoencoder, discriminator, train_loader, device)
        results['forward_pass'] = success
        
        # 测试5: 显存占用
        success = test_memory_usage(autoencoder, discriminator, train_loader, device)
        results['memory_usage'] = success
    else:
        logger.warning("⚠️ 跳过前向传播和显存测试（数据加载失败）")
        results['forward_pass'] = None
        results['memory_usage'] = None
    
    # 测试4: 滑动窗口推理（不依赖真实数据）
    success = test_sliding_window_inference(autoencoder, device, config)
    results['sliding_window'] = success
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试总结")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️ 跳过"
        logger.info(f"{test_name:.<30} {status}")
    
    all_passed = all(r in [True, None] for r in results.values())
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！Patch-Based训练配置可以正常使用！")
        logger.info("\n下一步：运行训练")
        logger.info("python monai_diffusion/3d_ldm/train_autoencoder.py --config monai_diffusion/config/ldm_config_patched.yaml")
    else:
        logger.warning("\n⚠️ 部分测试失败，请检查上面的错误信息")


if __name__ == "__main__":
    main()

