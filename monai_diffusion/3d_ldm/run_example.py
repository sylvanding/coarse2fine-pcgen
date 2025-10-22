"""
完整训练流程示例脚本

演示从数据转换到模型训练的完整流程。
"""

import sys
from pathlib import Path
import argparse
import logging
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str):
    """运行命令并记录输出"""
    logger.info(f"=" * 80)
    logger.info(f"步骤: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    logger.info(f"=" * 80)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"{description} 失败!")
        sys.exit(1)
    
    logger.info(f"{description} 完成!")


def main():
    parser = argparse.ArgumentParser(
        description="运行完整的3D Latent Diffusion Model训练流程"
    )
    
    parser.add_argument(
        '--h5_file',
        type=str,
        required=True,
        help='输入H5点云文件路径'
    )
    
    parser.add_argument(
        '--output_base',
        type=str,
        default='experiments/ldm_example',
        help='输出基础目录'
    )
    
    parser.add_argument(
        '--voxel_size',
        type=int,
        default=64,
        help='体素分辨率'
    )
    
    parser.add_argument(
        '--skip_conversion',
        action='store_true',
        help='跳过数据转换步骤'
    )
    
    parser.add_argument(
        '--skip_autoencoder',
        action='store_true',
        help='跳过AutoencoderKL训练'
    )
    
    parser.add_argument(
        '--fast_dev_run',
        action='store_true',
        help='快速开发模式（只运行少量batch）'
    )
    
    args = parser.parse_args()
    
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 路径配置
    voxel_dir = output_base / "voxels_nifti"
    config_file = output_base / "ldm_config.yaml"
    
    logger.info(f"开始3D Latent Diffusion Model训练流程")
    logger.info(f"输出目录: {output_base}")
    
    # ========== 步骤1: 数据转换 ==========
    if not args.skip_conversion:
        if not Path(args.h5_file).exists():
            logger.error(f"H5文件不存在: {args.h5_file}")
            sys.exit(1)
        
        cmd = [
            'python', 'scripts/h5pc2voxel/convert_h5_to_nifti.py',
            '--h5_file', args.h5_file,
            '--output_dir', str(voxel_dir),
            '--voxel_size', str(args.voxel_size),
            '--method', 'gaussian',
            '--sigma', '1.0'
        ]
        
        run_command(cmd, "步骤1: H5点云 → NIfTI体素转换")
    else:
        logger.info("跳过数据转换步骤")
        if not voxel_dir.exists():
            logger.error(f"体素数据目录不存在: {voxel_dir}")
            sys.exit(1)
    
    # ========== 步骤2: 创建配置文件 ==========
    logger.info("步骤2: 创建配置文件")
    
    # 复制默认配置文件并修改路径
    import yaml
    
    default_config = Path("monai_diffusion/config/ldm_config.yaml")
    with open(default_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新路径
    config['data']['train_data_dir'] = str(voxel_dir / "train")
    config['data']['val_data_dir'] = str(voxel_dir / "val")
    config['data']['voxel_size'] = args.voxel_size
    
    config['autoencoder']['checkpoints']['output_dir'] = str(output_base / "outputs" / "autoencoder")
    config['autoencoder']['logging']['log_dir'] = str(output_base / "logs" / "autoencoder")
    
    config['diffusion']['checkpoints']['output_dir'] = str(output_base / "outputs" / "diffusion")
    config['diffusion']['checkpoints']['autoencoder_path'] = str(output_base / "outputs" / "autoencoder" / "best_model.pt")
    config['diffusion']['logging']['log_dir'] = str(output_base / "logs" / "diffusion")
    
    config['sampling']['output_dir'] = str(output_base / "outputs" / "samples")
    
    # 快速开发模式
    if args.fast_dev_run:
        logger.info("启用快速开发模式")
        config['autoencoder']['training']['fast_dev_run'] = True
        config['autoencoder']['training']['n_epochs'] = 2
        config['diffusion']['training']['fast_dev_run'] = True
        config['diffusion']['training']['n_epochs'] = 2
    
    # 保存配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"配置文件已创建: {config_file}")
    
    # ========== 步骤3: 训练AutoencoderKL ==========
    if not args.skip_autoencoder:
        cmd = [
            'python', 'monai_diffusion/3d_ldm/train_autoencoder.py',
            '--config', str(config_file)
        ]
        
        run_command(cmd, "步骤3: 训练AutoencoderKL")
    else:
        logger.info("跳过AutoencoderKL训练")
        ae_checkpoint = output_base / "outputs" / "autoencoder" / "best_model.pt"
        if not ae_checkpoint.exists():
            logger.error(f"AutoencoderKL checkpoint不存在: {ae_checkpoint}")
            sys.exit(1)
    
    # ========== 步骤4: 训练Diffusion Model ==========
    cmd = [
        'python', 'monai_diffusion/3d_ldm/train_diffusion.py',
        '--config', str(config_file)
    ]
    
    run_command(cmd, "步骤4: 训练Diffusion Model")
    
    # ========== 步骤5: 生成样本 ==========
    cmd = [
        'python', 'monai_diffusion/3d_ldm/generate_samples.py',
        '--config', str(config_file),
        '--output_dir', str(output_base / "outputs" / "samples")
    ]
    
    run_command(cmd, "步骤5: 生成样本")
    
    # ========== 完成 ==========
    logger.info("=" * 80)
    logger.info("训练流程完成!")
    logger.info(f"输出目录: {output_base}")
    logger.info(f"- 配置文件: {config_file}")
    logger.info(f"- AutoencoderKL: {output_base / 'outputs' / 'autoencoder'}")
    logger.info(f"- Diffusion Model: {output_base / 'outputs' / 'diffusion'}")
    logger.info(f"- 生成样本: {output_base / 'outputs' / 'samples'}")
    logger.info(f"- 训练日志: {output_base / 'logs'}")
    logger.info("")
    logger.info("查看训练日志:")
    logger.info(f"  tensorboard --logdir {output_base / 'logs'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

