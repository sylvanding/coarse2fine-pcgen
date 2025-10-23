"""
测试AutoencoderKL的downsample_factors功能

验证不同下采样因子配置下的模型创建和前向传播。
"""

import sys
from pathlib import Path

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))

import torch
from generative.networks.nets import AutoencoderKL
import numpy as np


def test_downsample_factor_config(
    input_size=(64, 64, 32),
    num_channels=(32, 64, 64),
    downsample_factors=None,
    initial_downsample_factor=1,
    batch_size=2
):
    """
    测试指定下采样因子配置
    
    Args:
        input_size: 输入体素大小 (H, W, D)
        num_channels: 每层通道数
        downsample_factors: 下采样因子列表
        initial_downsample_factor: 初始下采样因子
        batch_size: 批次大小
    """
    print(f"\n{'='*60}")
    print(f"测试配置:")
    print(f"  输入大小: {input_size}")
    print(f"  num_channels: {num_channels}")
    print(f"  initial_downsample_factor: {initial_downsample_factor}")
    print(f"  downsample_factors: {downsample_factors}")
    print(f"  batch_size: {batch_size}")
    print(f"{'='*60}")
    
    # 计算总下采样倍数
    if downsample_factors is None:
        total_downsample = initial_downsample_factor * (2 ** (len(num_channels) - 1))
        print(f"✓ 使用默认配置: 总下采样 {total_downsample}x")
    else:
        total_downsample = initial_downsample_factor * np.prod(downsample_factors)
        print(f"✓ 使用自定义配置: 总下采样 {total_downsample}x")
    
    # 计算预期latent大小
    expected_latent_size = tuple(int(s / total_downsample) for s in input_size)
    print(f"✓ 预期Latent大小: {expected_latent_size}")
    
    try:
        # 创建模型
        model = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=num_channels,
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=tuple([False] * (len(num_channels) - 1) + [True]),
            downsample_factors=downsample_factors,
            initial_downsample_factor=initial_downsample_factor
        )
        print(f"✓ 模型创建成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 参数量: {total_params:,}")
        
        # 创建随机输入
        x = torch.randn(batch_size, 1, *input_size)
        print(f"✓ 创建输入张量: {x.shape}")
        
        # 前向传播
        with torch.no_grad():
            # Encoder
            z = model.encode(x)
            z_mu, z_sigma = z
            print(f"✓ Encoder输出: z_mu={z_mu.shape}, z_sigma={z_sigma.shape}")
            
            # 采样
            z_sample = model.sampling(z_mu, z_sigma)
            print(f"✓ 采样输出: {z_sample.shape}")
            
            # Decoder
            reconstruction = model.decode(z_sample)
            print(f"✓ Decoder输出: {reconstruction.shape}")
            
            # 完整前向传播
            output, z_mu_full, z_sigma_full = model(x)
            print(f"✓ 完整前向传播: {output.shape}")
        
        # 验证输出形状
        assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
        assert z_mu.shape[2:] == expected_latent_size, \
            f"Latent大小不匹配: {z_mu.shape[2:]} vs {expected_latent_size}"
        
        print(f"\n{'✅ 测试通过!'}")
        
        # 估算显存占用（粗略）
        input_memory = np.prod(x.shape) * 4 / 1024 / 1024  # MB
        latent_memory = np.prod(z_mu.shape) * 4 / 1024 / 1024  # MB
        compression_ratio = input_memory / latent_memory
        
        print(f"\n显存分析:")
        print(f"  输入数据: {input_memory:.2f} MB")
        print(f"  Latent空间: {latent_memory:.2f} MB")
        print(f"  压缩比: {compression_ratio:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("AutoencoderKL下采样因子功能测试")
    print("="*60)
    
    results = []
    
    # 测试1: 默认配置（向后兼容）
    print("\n【测试1】默认配置 - 向后兼容性")
    results.append(test_downsample_factor_config(
        input_size=(64, 64, 32),
        num_channels=(32, 64, 64),
        downsample_factors=None,  # 默认 [2, 2]
        batch_size=2
    ))
    
    # 测试2: 8倍下采样 [4, 2]
    print("\n【测试2】8倍下采样 - 推荐配置")
    results.append(test_downsample_factor_config(
        input_size=(256, 256, 64),
        num_channels=(16, 32, 64),
        downsample_factors=[4, 2],
        batch_size=1
    ))
    
    # 测试3: 16倍下采样 [8, 2]
    print("\n【测试3】16倍下采样 - 超高分辨率")
    results.append(test_downsample_factor_config(
        input_size=(512, 512, 128),
        num_channels=(16, 32, 64),
        downsample_factors=[8, 2],
        batch_size=1
    ))
    
    # 测试4: 16倍下采样 [4, 4]
    print("\n【测试4】16倍下采样 - 均衡配置")
    results.append(test_downsample_factor_config(
        input_size=(256, 256, 64),
        num_channels=(32, 64, 64),
        downsample_factors=[4, 4],
        batch_size=1
    ))
    
    # 测试5: 单层大下采样 [16]
    print("\n【测试5】单层16倍下采样 - 极简配置")
    results.append(test_downsample_factor_config(
        input_size=(256, 256, 64),
        num_channels=(32, 64),  # 只有2层
        downsample_factors=[16],
        batch_size=1
    ))
    
    # 测试6: 4层网络
    print("\n【测试6】4层网络 - 32倍下采样")
    results.append(test_downsample_factor_config(
        input_size=(512, 512, 128),
        num_channels=(16, 32, 64, 128),
        downsample_factors=[4, 4, 2],
        batch_size=1
    ))
    
    # 测试7: 使用Initial Downsample - 16倍下采样
    print("\n【测试7】Initial Downsample=2 - 16倍下采样")
    results.append(test_downsample_factor_config(
        input_size=(256, 256, 64),
        num_channels=(32, 64, 64),
        downsample_factors=[4, 2],
        initial_downsample_factor=2,  # ⭐ 新功能
        batch_size=1
    ))
    
    # 测试8: 使用Initial Downsample - 32倍下采样
    print("\n【测试8】Initial Downsample=4 - 32倍下采样")
    results.append(test_downsample_factor_config(
        input_size=(512, 512, 128),
        num_channels=(16, 32, 64),
        downsample_factors=[4, 2],
        initial_downsample_factor=4,  # ⭐ 更激进
        batch_size=1
    ))
    
    # 测试9: 极致Initial Downsample - 64倍下采样
    print("\n【测试9】Initial Downsample=8 - 64倍下采样")
    results.append(test_downsample_factor_config(
        input_size=(512, 512, 128),
        num_channels=(16, 32, 64),
        downsample_factors=[4, 2],
        initial_downsample_factor=8,  # ⭐ 极致压缩
        batch_size=1
    ))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"总测试数: {total}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 所有测试通过！功能正常工作。")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查配置。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

