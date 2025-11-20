"""
快速测试脚本

用于验证VQ-LDM代码是否能正常运行。
使用fast_dev_run模式，只运行几个batch来检查代码正确性。

使用方法:
    python monai_diffusion/3d_vq_ldm/quick_test.py --stage vqvae
    python monai_diffusion/3d_vq_ldm/quick_test.py --stage diffusion
    python monai_diffusion/3d_vq_ldm/quick_test.py --stage both
"""

import sys
from pathlib import Path
import argparse
import logging
import yaml
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))
sys.path.insert(0, str(project_root))

# 由于模块名包含数字，需要特殊方式导入
import importlib.util

# 导入train_vqvae
train_vqvae_path = project_root / "monai_diffusion" / "3d_vq_ldm" / "train_vqvae.py"
spec = importlib.util.spec_from_file_location("train_vqvae", train_vqvae_path)
train_vqvae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_vqvae_module)
train_vqvae = train_vqvae_module.train_vqvae

# 导入train_diffusion
train_diffusion_path = project_root / "monai_diffusion" / "3d_vq_ldm" / "train_diffusion.py"
spec = importlib.util.spec_from_file_location("train_diffusion", train_diffusion_path)
train_diffusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_diffusion_module)
train_diffusion = train_diffusion_module.train_diffusion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config(base_config_path: str, temp_dir: str) -> str:
    """
    创建测试配置，启用fast_dev_run模式
    
    Args:
        base_config_path: 基础配置文件路径
        temp_dir: 临时目录
        
    Returns:
        测试配置文件路径
    """
    # 加载基础配置
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改为快速开发模式
    if 'vqvae' in config:
        config['vqvae']['training']['fast_dev_run'] = True
        config['vqvae']['training']['fast_dev_run_batches'] = 2
        config['vqvae']['training']['n_epochs'] = 2
        config['vqvae']['checkpoints']['output_dir'] = f"{temp_dir}/vqvae_checkpoints"
        config['vqvae']['logging']['log_dir'] = f"{temp_dir}/vqvae_logs"
    
    if 'diffusion' in config:
        config['diffusion']['training']['fast_dev_run'] = True
        config['diffusion']['training']['fast_dev_run_batches'] = 2
        config['diffusion']['training']['n_epochs'] = 2
        config['diffusion']['checkpoints']['output_dir'] = f"{temp_dir}/diffusion_checkpoints"
        config['diffusion']['logging']['log_dir'] = f"{temp_dir}/diffusion_logs"
        # 使用临时VQVAE checkpoint路径
        config['diffusion']['checkpoints']['vqvae_path'] = f"{temp_dir}/vqvae_checkpoints/best_model.pt"
    
    # 保存测试配置
    test_config_path = f"{temp_dir}/test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"创建测试配置: {test_config_path}")
    return test_config_path


def test_vqvae(config_path: str):
    """测试VQVAE训练"""
    logger.info("=" * 60)
    logger.info("开始测试VQVAE训练")
    logger.info("=" * 60)
    
    try:
        train_vqvae(config_path)
        logger.info("✅ VQVAE测试成功!")
        return True
    except Exception as e:
        logger.error(f"❌ VQVAE测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diffusion(config_path: str):
    """测试Diffusion训练"""
    logger.info("=" * 60)
    logger.info("开始测试Diffusion训练")
    logger.info("=" * 60)
    
    try:
        train_diffusion(config_path)
        logger.info("✅ Diffusion测试成功!")
        return True
    except Exception as e:
        logger.error(f"❌ Diffusion测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="VQ-LDM快速测试")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['vqvae', 'diffusion', 'both'],
        default='both',
        help='测试哪个阶段: vqvae, diffusion, 或 both'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='monai_diffusion/config/vq_ldm_config_local.yaml',
        help='基础配置文件路径'
    )
    parser.add_argument(
        '--keep-output',
        action='store_true',
        help='保留测试输出（默认会删除）'
    )
    
    args = parser.parse_args()
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="vq_ldm_test_")
    logger.info(f"临时目录: {temp_dir}")
    
    try:
        # 创建测试配置
        test_config_path = create_test_config(args.config, temp_dir)
        
        # 运行测试
        if args.stage in ['vqvae', 'both']:
            vqvae_success = test_vqvae(test_config_path)
        else:
            vqvae_success = True
        
        if args.stage in ['diffusion', 'both']:
            if args.stage == 'both':
                # 如果测试完整流程，需要先训练VQVAE
                if not vqvae_success:
                    logger.error("VQVAE测试失败，跳过Diffusion测试")
                    diffusion_success = False
                else:
                    diffusion_success = test_diffusion(test_config_path)
            else:
                # 如果只测试Diffusion，假设VQVAE checkpoint已存在
                logger.warning("仅测试Diffusion，需要确保VQVAE checkpoint存在")
                diffusion_success = test_diffusion(test_config_path)
        else:
            diffusion_success = True
        
        # 打印总结
        logger.info("=" * 60)
        logger.info("测试总结:")
        if args.stage in ['vqvae', 'both']:
            logger.info(f"  VQVAE: {'✅ 成功' if vqvae_success else '❌ 失败'}")
        if args.stage in ['diffusion', 'both']:
            logger.info(f"  Diffusion: {'✅ 成功' if diffusion_success else '❌ 失败'}")
        logger.info("=" * 60)
        
    finally:
        # 清理临时目录
        if not args.keep_output:
            logger.info(f"清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            logger.info(f"保留测试输出: {temp_dir}")


if __name__ == "__main__":
    main()

