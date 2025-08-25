#!/usr/bin/env python
"""
3D Diffusion运行示例

演示如何使用3D扩散模型进行训练和生成的完整示例。
"""

import sys
from pathlib import Path
import logging
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header():
    """打印欢迎信息"""
    print("\n" + "="*60)
    print("🚀 Coarse2Fine-PCGen 3D Diffusion 示例")
    print("="*60)
    print()
    print("这个示例演示了如何使用3D扩散模型：")
    print("1. 从点云数据生成体素")
    print("2. 训练3D扩散模型")
    print("3. 生成新的体素样本")
    print("4. 将体素转换回点云")
    print()


def print_usage():
    """打印使用说明"""
    print("📋 使用说明:")
    print()
    print("1. 系统测试:")
    print("   python scripts/test_diffusion_system.py")
    print()
    print("2. 训练模型:")
    print("   python scripts/train_diffusion.py --config configs/diffusion_config.yaml")
    print()
    print("3. 生成样本:")
    print("   python scripts/generate_samples.py checkpoints/best_model.ckpt --num-samples 8")
    print()
    print("4. 自定义配置训练:")
    print("   python scripts/train_diffusion.py \\")
    print("     --data-path /path/to/your/data.h5 \\")
    print("     --voxel-size 64 \\")
    print("     --batch-size 4 \\")
    print("     --max-epochs 100")
    print()


def print_config_info():
    """打印配置文件信息"""
    print("⚙️ 配置文件说明:")
    print()
    print("主配置文件: configs/diffusion_config.yaml")
    print()
    print("主要配置项:")
    print("- data.voxel_size: 体素网格分辨率 (32, 64, 128)")
    print("- data.voxelization_method: 体素化方法 (gaussian, density, occupancy)")
    print("- model.model_channels: 模型基础通道数")
    print("- training.batch_size: 训练批次大小")
    print("- training.learning_rate: 学习率")
    print("- validation.sample_interval: 验证采样间隔 (每N个epoch)")
    print()
    print("环境变量覆盖示例:")
    print("export DIFFUSION_DATA_VOXEL_SIZE=128")
    print("export DIFFUSION_TRAINING_BATCH_SIZE=8")
    print()


def print_data_format():
    """打印数据格式说明"""
    print("📊 数据格式要求:")
    print()
    print("输入数据格式: HDF5 (.h5)")
    print("数据结构:")
    print("  - 键名: 'point_clouds' (可配置)")
    print("  - 形状: (样本数, 点数, 3)")
    print("  - 数据类型: float32")
    print("  - 坐标: (x, y, z) 在纳米单位")
    print()
    print("示例数据创建:")
    print("  python scripts/test_conversion/create_sample_data.py")
    print()


def print_model_info():
    """打印模型架构信息"""
    print("🧠 模型架构:")
    print()
    print("3D UNet + DDPM 扩散模型")
    print("- 基于3D卷积的UNet架构")
    print("- 支持自注意力机制")
    print("- 时间步嵌入")
    print("- 残差连接和组归一化")
    print("- DDIM快速采样支持")
    print()
    print("关键特性:")
    print("- 可配置体素分辨率")
    print("- PyTorch Lightning训练框架")
    print("- 自动验证和TIFF保存")
    print("- 指数移动平均 (EMA) 支持")
    print("- 分布式训练支持")
    print()


def print_training_tips():
    """打印训练建议"""
    print("💡 训练建议:")
    print()
    print("1. 硬件需求:")
    print("   - GPU: 8GB+ VRAM (体素大小64^3)")
    print("   - CPU: 8+ 核心")
    print("   - 内存: 16GB+")
    print()
    print("2. 参数调优:")
    print("   - 从小的体素大小开始 (32^3)")
    print("   - 使用gaussian体素化方法")
    print("   - 批次大小根据GPU内存调整")
    print("   - 学习率通常在1e-4到1e-3之间")
    print()
    print("3. 监控指标:")
    print("   - val_loss: 验证损失")
    print("   - gen_occupancy: 生成体素占有率")
    print("   - learning_rate: 学习率变化")
    print()
    print("4. 调试选项:")
    print("   - --fast-dev-run: 快速测试")
    print("   - --overfit-batches: 过拟合少量数据")
    print("   - --limit-train-batches: 限制训练数据")
    print()


def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    print()
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('h5py', 'HDF5 for Python'),
        ('tifffile', 'TIFF file handling'),
        ('scipy', 'SciPy'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'Progress bars')
    ]
    
    missing = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} (缺失)")
            missing.append(package)
    
    print()
    
    if missing:
        print("❌ 缺少依赖项，请安装:")
        print(f"pip install {' '.join(missing)}")
        print()
        return False
    else:
        print("✅ 所有依赖项已安装")
        print()
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="3D Diffusion运行示例和说明",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查依赖项"
    )
    parser.add_argument(
        "--test-system",
        action="store_true", 
        help="运行系统测试"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    if args.check_deps:
        deps_ok = check_dependencies()
        if not deps_ok:
            sys.exit(1)
    
    if args.test_system:
        print("🧪 运行系统测试...")
        print()
        try:
            from scripts.test_diffusion_system import run_all_tests
            success = run_all_tests()
            if success:
                print("\n✅ 系统测试通过！可以开始使用。")
            else:
                print("\n❌ 系统测试失败，请检查安装。")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ 测试运行失败: {e}")
            sys.exit(1)
        return
    
    # 显示所有信息
    print_usage()
    print_config_info()
    print_data_format()
    print_model_info()
    print_training_tips()
    
    print("🎯 快速开始:")
    print()
    print("1. 检查依赖: python scripts/run_example.py --check-deps")
    print("2. 系统测试: python scripts/run_example.py --test-system")
    print("3. 准备数据: 确保H5文件格式正确")
    print("4. 修改配置: 编辑 configs/diffusion_config.yaml")
    print("5. 开始训练: python scripts/train_diffusion.py")
    print()
    print("📚 更多信息请查看 README.md")
    print()


if __name__ == "__main__":
    main()
