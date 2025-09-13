#!/usr/bin/env python3
"""
惰性加载数据集配置示例

展示如何配置不同类型的惰性加载数据集参数
"""

class LazyDatasetConfig:
    """惰性加载数据集配置类"""
    
    def __init__(self):
        # === 基本文件路径配置 ===
        # GT点云数据文件
        self.gt_h5_path = "/path/to/your/gt_pointclouds.h5"
        self.gt_data_key = "gt_clouds"  # H5文件中的数据键名
        
        # 噪声点云数据文件
        self.noisy_h5_path = "/path/to/your/noisy_pointclouds.h5" 
        self.noisy_data_key = "noisy_clouds"
        
        # === 体素指导配置（可选）===
        self.use_voxel_guidance = False  # 是否使用体素指导
        self.voxel_h5_path = "/path/to/your/voxel_grids.h5"
        self.voxel_data_key = "voxel_grids"
        
        # === 数据集分割配置 ===
        self.train_ratio = 0.8  # 训练集占比
        
        # === 点云处理配置 ===
        self.sample_points = 2048  # 采样点数
        self.iterations = 10000    # 训练迭代次数
        
        # === 点云归一化配置 ===
        # 实际空间尺寸 (x, y, z)
        self.volume_dims = [20000, 20000, 2500]  # 单位：毫米
        # 边界padding (x, y, z)
        self.padding = [0, 0, 100]  # 单位：毫米
        
        # === 缓存配置（仅用于LazyH5Dataset）===
        self.enable_sample_cache = True  # 是否启用样本缓存
        self.sample_cache_size = 100     # 缓存的样本数量
        

class PerformanceConfig(LazyDatasetConfig):
    """高性能配置：适用于大型数据集和多GPU训练"""
    
    def __init__(self):
        super().__init__()
        
        # 更大的缓存以减少磁盘I/O
        self.sample_cache_size = 500
        
        # 更多采样点以充分利用GPU
        self.sample_points = 4096
        
        # 更高的训练迭代次数
        self.iterations = 50000


class MemoryEfficientConfig(LazyDatasetConfig):
    """内存高效配置：适用于内存受限的环境"""
    
    def __init__(self):
        super().__init__()
        
        # 较小的缓存以节省内存
        self.sample_cache_size = 20
        
        # 较少的采样点以节省内存
        self.sample_points = 1024


class DevelopmentConfig(LazyDatasetConfig):
    """开发配置：适用于快速原型开发和调试"""
    
    def __init__(self):
        super().__init__()
        
        # 小数据集用于快速测试
        self.train_ratio = 0.9  # 更多训练数据
        self.iterations = 1000  # 较少迭代
        
        # 中等缓存大小
        self.sample_cache_size = 50
        
        # 中等采样点数
        self.sample_points = 2048


class InferenceConfig(LazyDatasetConfig):
    """推理配置：适用于模型推理和评估"""
    
    def __init__(self):
        super().__init__()
        
        # 推理时通常不需要缓存
        self.enable_sample_cache = False
        
        # 推理时保持原始点数，不进行采样
        # sample_points 在推理模式下会被忽略


# 使用示例
def get_config(config_type='default'):
    """
    获取指定类型的配置
    
    Args:
        config_type (str): 配置类型
            - 'default': 默认配置
            - 'performance': 高性能配置
            - 'memory_efficient': 内存高效配置
            - 'development': 开发配置
            - 'inference': 推理配置
    
    Returns:
        配置对象
    """
    config_map = {
        'default': LazyDatasetConfig,
        'performance': PerformanceConfig,
        'memory_efficient': MemoryEfficientConfig,
        'development': DevelopmentConfig,
        'inference': InferenceConfig,
    }
    
    if config_type not in config_map:
        raise ValueError(f"未知的配置类型: {config_type}")
    
    return config_map[config_type]()


# 配置文件使用示例
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    
    from src.refine_net.data_handler import get_dataset
    import torch
    
    # 示例1: 使用默认配置
    print("=== 默认配置示例 ===")
    config = get_config('default')
    
    # 创建数据集
    dataset_class = get_dataset('h5_lazy_cached')
    try:
        dataset = dataset_class(None, torch.device('cpu'), config, split='train')
        print(f"数据集创建成功，配置类型: 默认")
    except Exception as e:
        print(f"数据集创建失败: {e}")
    
    # 示例2: 使用高性能配置
    print("\n=== 高性能配置示例 ===")
    perf_config = get_config('performance')
    print(f"缓存大小: {perf_config.sample_cache_size}")
    print(f"采样点数: {perf_config.sample_points}")
    print(f"训练迭代: {perf_config.iterations}")
    
    # 示例3: 使用内存高效配置
    print("\n=== 内存高效配置示例 ===")
    mem_config = get_config('memory_efficient')
    print(f"缓存大小: {mem_config.sample_cache_size}")
    print(f"采样点数: {mem_config.sample_points}")
    
    # 示例4: 自定义配置
    print("\n=== 自定义配置示例 ===")
    custom_config = LazyDatasetConfig()
    
    # 根据具体需求修改配置
    custom_config.gt_h5_path = "/custom/path/to/gt_data.h5"
    custom_config.sample_cache_size = 200
    custom_config.sample_points = 8192
    custom_config.use_voxel_guidance = True
    
    print(f"自定义GT路径: {custom_config.gt_h5_path}")
    print(f"自定义缓存大小: {custom_config.sample_cache_size}")
    print(f"启用体素指导: {custom_config.use_voxel_guidance}")
