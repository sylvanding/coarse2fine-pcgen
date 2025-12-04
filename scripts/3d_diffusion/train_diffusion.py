#!/usr/bin/env python
"""
3D Diffusion模型训练脚本

使用PyTorch Lightning训练基于体素的3D扩散模型。
支持配置文件、命令行参数覆盖、自动验证和TIFF生成。
"""

import argparse
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# 导入项目模块
from src.models.diffusion_lightning import DiffusionLightningModule, DiffusionLightningModuleWithEMA
from src.data.voxel_dataset import (
    create_train_val_datasets, 
    create_prerendered_train_val_datasets,
    VoxelCollator
)
from src.utils.config_loader import load_config_with_overrides, create_default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练3D Diffusion模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/diffusion_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="H5数据文件路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="实验输出目录"
    )
    
    # 模型参数覆盖
    parser.add_argument(
        "--voxel-size",
        type=int,
        help="体素网格大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="学习率（覆盖配置文件）"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="最大训练轮数（覆盖配置文件）"
    )
    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        help="是否使用学习率调度器"
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true", 
        help="禁用学习率调度器（固定学习率）"
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        choices=["constant", "cosine", "linear"],
        help="学习率调度器类型（覆盖配置文件）"
    )
    
    # 硬件设置
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["auto", "gpu", "cpu"],
        help="加速器类型（覆盖配置文件）"
    )
    parser.add_argument(
        "--devices",
        type=str,
        help="设备数量或设备ID列表（覆盖配置文件）"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="训练精度（覆盖配置文件）"
    )
    
    # 调试选项
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速开发运行模式（只运行1个batch）"
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="过拟合指定数量的批次（调试用）"
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="限制训练批次比例"
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="限制验证批次比例"
    )
    
    # 其他选项
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="DataLoader工作进程数（覆盖配置文件）"
    )
    
    return parser.parse_args()


def setup_experiment_dir(config: dict, output_dir: str) -> Path:
    """设置实验目录"""
    experiment_name = config['logging']['experiment_name']
    exp_dir = Path(output_dir) / experiment_name
    
    # 创建唯一的版本目录
    version = 0
    while (exp_dir / f"version_{version}").exists():
        version += 1
    
    exp_dir = exp_dir / f"version_{version}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "validation_outputs").mkdir(exist_ok=True)
    
    logger.info(f"实验目录: {exp_dir}")
    return exp_dir


def create_data_loaders(config: dict) -> tuple:
    """创建数据加载器"""
    data_config = config['data']
    training_config = config['training']
    system_config = config['system']
    
    logger.info("创建数据集...")
    
    # 检查是否使用预渲染数据
    use_prerendered = data_config.get('use_prerendered', False)
    
    if use_prerendered:
        # 使用预渲染体素数据
        prerendered_dir = data_config.get('prerendered_dir')
        if not prerendered_dir or not Path(prerendered_dir).exists():
            raise ValueError(f"预渲染数据目录不存在或未指定: {prerendered_dir}")
        
        logger.info(f"使用预渲染数据: {prerendered_dir}")
        
        train_dataset, val_dataset = create_prerendered_train_val_datasets(
            prerendered_dir=prerendered_dir,
            train_ratio=data_config['train_ratio'],
            normalize=data_config.get('normalize', True),
            augment=data_config.get('augment', True)
        )
    else:
        # 动态渲染点云到体素
        logger.info("使用动态点云到体素转换")
        
        train_dataset, val_dataset = create_train_val_datasets(
            h5_file_path=data_config['h5_file_path'],
            train_ratio=data_config['train_ratio'],
            voxel_size=data_config['voxel_size'],
            voxelization_method=data_config['voxelization_method'],
            data_key=data_config['data_key'],
            sigma=data_config.get('sigma', 1.0),
            volume_dims=data_config.get('volume_dims'),
            padding=data_config.get('padding'),
            normalize=data_config.get('normalize', True),
            cache_voxels=data_config.get('cache_voxels', True),
            max_cache_size=data_config.get('max_cache_size', 1000)
        )
    
    # 创建数据整理器
    collator = VoxelCollator()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=system_config.get('num_workers', 4),
        pin_memory=system_config.get('pin_memory', True),
        persistent_workers=system_config.get('persistent_workers', True),
        collate_fn=collator,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=system_config.get('num_workers', 4),
        pin_memory=system_config.get('pin_memory', True),
        persistent_workers=system_config.get('persistent_workers', True),
        collate_fn=collator,
        drop_last=False
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    logger.info(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    
    return train_loader, val_loader, train_dataset


def create_model(config: dict, exp_dir: Path) -> pl.LightningModule:
    """创建模型"""
    model_config = config['model']
    validation_config = config['validation']
    
    # 合并所有参数
    model_params = {
        # 基础参数
        'voxel_size': config['data']['voxel_size'],
        
        # 模型架构参数
        'model_channels': model_config['model_channels'],
        'num_res_blocks': model_config['num_res_blocks'],
        'attention_resolutions': model_config['attention_resolutions'],
        'channel_mult': model_config['channel_mult'],
        'dropout': model_config['dropout'],
        'use_attention': model_config['use_attention'],
        
        # 扩散过程参数
        'num_timesteps': model_config['num_timesteps'],
        'beta_start': model_config['beta_start'],
        'beta_end': model_config['beta_end'],
        'beta_schedule': model_config['beta_schedule'],
        
        # 训练参数
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_steps': config['training']['warmup_steps'],
        'use_scheduler': config['training'].get('use_scheduler', True),
        'scheduler_type': config['training'].get('scheduler_type', 'cosine'),
        
        # 验证参数
        'val_sample_interval': validation_config['sample_interval'],
        'num_val_samples': validation_config['num_samples'],
        'save_val_tiffs': validation_config['save_tiffs'],
        'val_output_dir': str(exp_dir / "validation_outputs"),
        'ddim_steps': validation_config['ddim_steps'],
        'ddim_eta': validation_config['ddim_eta'],
        
        # 高级选项
        'compile_model': model_config.get('compile_model', False)
    }
    
    # 选择是否使用EMA
    if model_config.get('use_ema', False):
        model_params['ema_decay'] = model_config.get('ema_decay', 0.9999)
        model = DiffusionLightningModuleWithEMA(**model_params)
        logger.info("使用EMA模型")
    else:
        model = DiffusionLightningModule(**model_params)
        logger.info("使用标准模型")
    
    return model


def create_callbacks(config: dict, exp_dir: Path) -> list:
    """创建回调函数"""
    training_config = config['training']
    
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor=training_config['monitor'],
        mode=training_config['mode'],
        save_top_k=training_config['save_top_k'],
        save_last=training_config.get('save_last', True),
        auto_insert_metric_name=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    callbacks.append(lr_monitor)
    
    # 早停（可选）
    if 'early_stopping' in training_config:
        early_stop_config = training_config['early_stopping']
        early_stopping = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_loss'),
            mode=early_stop_config.get('mode', 'min'),
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            verbose=True
        )
        callbacks.append(early_stopping)
        logger.info(f"启用早停: 监控={early_stop_config.get('monitor')}, 耐心={early_stop_config.get('patience')}")
    
    return callbacks


def create_trainer(config: dict, exp_dir: Path, callbacks: list, resume_checkpoint: str = None) -> pl.Trainer:
    """创建训练器"""
    training_config = config['training']
    system_config = config['system']
    logging_config = config['logging']
    
    # TensorBoard日志
    logger_tb = TensorBoardLogger(
        save_dir=exp_dir / "logs",
        name="",
        version="",
        log_graph=False
    )
    
    # 训练器参数
    trainer_params = {
        'max_epochs': training_config['max_epochs'],
        'accelerator': system_config['accelerator'],
        'devices': system_config['devices'],
        'precision': system_config['precision'],
        'gradient_clip_val': training_config.get('gradient_clip_val'),
        'accumulate_grad_batches': training_config.get('accumulate_grad_batches', 1),
        'val_check_interval': training_config.get('val_check_interval', 1.0),
        'limit_val_batches': training_config.get('limit_val_batches', 1.0),
        'log_every_n_steps': logging_config.get('log_every_n_steps', 50),
        'callbacks': callbacks,
        'logger': logger_tb,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'enable_checkpointing': True
    }
    
    # 调试选项
    if system_config.get('fast_dev_run', False):
        trainer_params['fast_dev_run'] = True
    
    if system_config.get('overfit_batches', 0) > 0:
        trainer_params['overfit_batches'] = system_config['overfit_batches']
    
    if system_config.get('limit_train_batches', 1.0) < 1.0:
        trainer_params['limit_train_batches'] = system_config['limit_train_batches']
    
    # 分布式训练设置
    if 'strategy' in system_config:
        trainer_params['strategy'] = system_config['strategy']
    
    # 创建训练器
    trainer = pl.Trainer(**trainer_params)
    
    return trainer


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed, workers=True)
    
    logger.info("=== 3D Diffusion模型训练 ===")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"随机种子: {args.seed}")
    
    # 加载配置
    try:
        config = load_config_with_overrides(args.config)
        logger.info("配置文件加载成功")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        logger.info("使用默认配置")
        config = create_default_config()
    
    # 应用命令行参数覆盖
    overrides = {}
    if args.data_path:
        overrides['data.h5_file_path'] = args.data_path
    if args.voxel_size:
        overrides['data.voxel_size'] = args.voxel_size
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.max_epochs:
        overrides['training.max_epochs'] = args.max_epochs
    if args.accelerator:
        overrides['system.accelerator'] = args.accelerator
    if args.devices:
        overrides['system.devices'] = args.devices
    if args.precision:
        overrides['system.precision'] = args.precision
    if args.num_workers:
        overrides['system.num_workers'] = args.num_workers
    
    # 学习率调度器覆盖
    if args.use_scheduler:
        overrides['training.use_scheduler'] = True
    if args.no_scheduler:
        overrides['training.use_scheduler'] = False
    if args.scheduler_type:
        overrides['training.scheduler_type'] = args.scheduler_type
    
    # 调试选项覆盖
    if args.fast_dev_run:
        overrides['system.fast_dev_run'] = True
    if args.overfit_batches > 0:
        overrides['system.overfit_batches'] = args.overfit_batches
    if args.limit_train_batches < 1.0:
        overrides['system.limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches < 1.0:
        overrides['system.limit_val_batches'] = args.limit_val_batches
    
    # 应用覆盖
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
        logger.info(f"覆盖配置: {key_path} = {value}")
    
    # 验证数据文件
    h5_file_path = config['data']['h5_file_path']
    if not Path(h5_file_path).exists():
        logger.error(f"数据文件不存在: {h5_file_path}")
        sys.exit(1)
    
    # 设置实验目录
    exp_dir = setup_experiment_dir(config, args.output_dir)
    
    # 保存配置到实验目录
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    config_loader.config = config
    config_loader.save_config(exp_dir / "config.yaml")
    
    try:
        # 创建数据加载器
        train_loader, val_loader, train_dataset = create_data_loaders(config)
        
        # 创建模型
        model = create_model(config, exp_dir)
        
        # 创建回调函数
        callbacks = create_callbacks(config, exp_dir)
        
        # 创建训练器
        trainer = create_trainer(config, exp_dir, callbacks, args.resume_from_checkpoint)
        
        # 打印训练信息
        logger.info("=== 训练配置摘要 ===")
        logger.info(f"体素大小: {config['data']['voxel_size']}^3")
        logger.info(f"批次大小: {config['training']['batch_size']}")
        logger.info(f"学习率: {config['training']['learning_rate']}")
        logger.info(f"最大轮数: {config['training']['max_epochs']}")
        logger.info(f"加速器: {config['system']['accelerator']}")
        logger.info(f"设备: {config['system']['devices']}")
        logger.info(f"精度: {config['system']['precision']}")
        
        # 开始训练
        logger.info("开始训练...")
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=args.resume_from_checkpoint
        )
        
        # 训练完成
        logger.info("训练完成!")
        
        # 保存最终检查点和样本
        if not config['system'].get('fast_dev_run', False):
            final_checkpoint = exp_dir / "checkpoints" / "final_model.ckpt"
            model.save_checkpoint_with_samples(str(final_checkpoint), num_samples=8)
        
        logger.info(f"实验结果保存在: {exp_dir}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        # 清理资源
        if 'train_dataset' in locals():
            train_dataset.clear_cache()


if __name__ == "__main__":
    main()
