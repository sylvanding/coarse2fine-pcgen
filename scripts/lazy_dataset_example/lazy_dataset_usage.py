#!/usr/bin/env python3
"""
惰性加载数据集使用示例

演示如何使用新的惰性加载H5数据集，包括：
1. 基本的惰性加载数据集
2. 带缓存的高性能数据集
3. 多进程数据加载
4. 性能监控
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
import argparse
import logging
from src.refine_net.data_handler import get_dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockArgs:
    """模拟参数配置"""
    def __init__(self):
        # 文件路径（需要根据实际情况修改）
        self.gt_h5_path = "/path/to/gt_data.h5"
        self.gt_data_key = "gt_clouds"
        self.noisy_h5_path = "/path/to/noisy_data.h5"
        self.noisy_data_key = "noisy_clouds"
        
        # 体素指导（可选）
        self.use_voxel_guidance = False
        self.voxel_h5_path = "/path/to/voxel_data.h5"
        self.voxel_data_key = "voxel_grids"
        
        # 数据集分割
        self.train_ratio = 0.8
        
        # 采样参数
        self.sample_points = 2048
        self.iterations = 1000
        
        # 归一化参数
        self.volume_dims = [20000, 20000, 2500]
        self.padding = [0, 0, 100]
        
        # 缓存参数（仅用于LazyH5Dataset）
        self.sample_cache_size = 50
        self.enable_sample_cache = True


def test_basic_lazy_dataset():
    """测试基本的惰性加载数据集"""
    logger.info("=== 测试基本惰性加载数据集 ===")
    
    args = MockArgs()
    
    # 创建数据集
    dataset_class = get_dataset('h5_paired')
    train_dataset = dataset_class(None, torch.device('cpu'), args, split='train')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"样本索引数量: {len(train_dataset.sample_indices)}")
    
    # 获取一个样本
    try:
        sample = train_dataset[0]
        if len(sample) == 2:
            gt_pc, noisy_pc = sample
            logger.info(f"GT点云形状: {gt_pc.shape}")
            logger.info(f"噪声点云形状: {noisy_pc.shape}")
        else:
            gt_pc, noisy_pc, voxel_data = sample
            logger.info(f"GT点云形状: {gt_pc.shape}")
            logger.info(f"噪声点云形状: {noisy_pc.shape}")
            logger.info(f"体素数据形状: {voxel_data.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")


def test_cached_lazy_dataset():
    """测试带缓存的惰性加载数据集"""
    logger.info("=== 测试带缓存的惰性加载数据集 ===")
    
    args = MockArgs()
    
    # 创建带缓存的数据集
    dataset_class = get_dataset('h5_lazy_cached')
    train_dataset = dataset_class(None, torch.device('cpu'), args, split='train')
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    
    # 多次访问相同样本以测试缓存
    try:
        logger.info("首次访问样本...")
        sample1 = train_dataset[0]
        
        logger.info("再次访问相同样本（应该命中缓存）...")
        sample2 = train_dataset[0]
        
        logger.info("访问不同样本...")
        sample3 = train_dataset[1]
        
        # 打印缓存统计
        stats = train_dataset.get_cache_stats()
        logger.info(f"缓存统计: {stats}")
        
    except Exception as e:
        logger.error(f"缓存测试失败: {e}")


def test_multiprocess_dataloader():
    """测试多进程数据加载"""
    logger.info("=== 测试多进程数据加载 ===")
    
    args = MockArgs()
    
    # 创建数据集
    dataset_class = get_dataset('h5_lazy_cached')
    train_dataset = dataset_class(None, torch.device('cpu'), args, split='train')
    
    # 创建多进程DataLoader
    try:
        dataloader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=2,  # 使用2个worker进程
            pin_memory=True
        )
        
        logger.info("创建多进程DataLoader成功")
        
        # 测试几个批次
        for i, batch in enumerate(dataloader):
            if i >= 3:  # 只测试3个批次
                break
                
            if len(batch) == 2:
                gt_batch, noisy_batch = batch
                logger.info(f"批次 {i}: GT={gt_batch.shape}, 噪声={noisy_batch.shape}")
            else:
                gt_batch, noisy_batch, voxel_batch = batch
                logger.info(f"批次 {i}: GT={gt_batch.shape}, 噪声={noisy_batch.shape}, 体素={voxel_batch.shape}")
        
        logger.info("多进程数据加载测试完成")
        
    except Exception as e:
        logger.error(f"多进程测试失败: {e}")


def test_inference_dataset():
    """测试推理数据集"""
    logger.info("=== 测试推理数据集 ===")
    
    args = MockArgs()
    
    # 创建推理数据集
    dataset_class = get_dataset('h5_inference')
    inference_dataset = dataset_class(None, torch.device('cpu'), args)
    
    logger.info(f"推理数据集大小: {len(inference_dataset)}")
    
    try:
        # 获取推理样本
        sample = inference_dataset[0]
        
        logger.info("推理样本包含的键:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
            else:
                logger.info(f"  {key}: {type(value)} - {value}")
                
    except Exception as e:
        logger.error(f"推理数据集测试失败: {e}")


def performance_benchmark():
    """性能基准测试"""
    logger.info("=== 性能基准测试 ===")
    
    args = MockArgs()
    
    # 比较不同数据集的性能
    datasets = [
        ('基本惰性加载', get_dataset('h5_paired')),
        ('带缓存惰性加载', get_dataset('h5_lazy_cached')),
    ]
    
    for name, dataset_class in datasets:
        logger.info(f"\n测试 {name}...")
        
        try:
            dataset = dataset_class(None, torch.device('cpu'), args, split='train')
            
            import time
            start_time = time.time()
            
            # 访问10个样本
            for i in range(min(10, len(dataset.sample_indices))):
                sample = dataset[i]
            
            elapsed_time = time.time() - start_time
            logger.info(f"{name} - 10个样本加载时间: {elapsed_time:.4f}秒")
            
            # 如果是缓存数据集，打印统计信息
            if hasattr(dataset, 'get_cache_stats'):
                stats = dataset.get_cache_stats()
                logger.info(f"缓存统计: {stats}")
                
        except Exception as e:
            logger.error(f"{name} 性能测试失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='惰性加载数据集使用示例')
    parser.add_argument('--test', choices=['basic', 'cached', 'multiprocess', 'inference', 'benchmark', 'all'],
                       default='all', help='要运行的测试类型')
    
    args = parser.parse_args()
    
    logger.info("开始惰性加载数据集测试...")
    logger.info("注意: 请确保修改MockArgs中的文件路径指向实际的H5数据文件")
    
    if args.test in ['basic', 'all']:
        test_basic_lazy_dataset()
        
    if args.test in ['cached', 'all']:
        test_cached_lazy_dataset()
        
    if args.test in ['multiprocess', 'all']:
        test_multiprocess_dataloader()
        
    if args.test in ['inference', 'all']:
        test_inference_dataset()
        
    if args.test in ['benchmark', 'all']:
        performance_benchmark()
    
    logger.info("测试完成!")


if __name__ == "__main__":
    main()
