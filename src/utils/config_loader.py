"""
配置文件加载器

用于加载和验证YAML配置文件，支持配置合并和环境变量替换。
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    配置文件加载器
    
    支持YAML配置文件的加载、验证、合并和环境变量替换。
    """
    
    def __init__(self):
        self.config = {}
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path (Union[str, Path]): 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        logger.info(f"加载配置文件: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 环境变量替换
            config = self._replace_env_vars(config)
            
            # 验证配置
            self._validate_config(config)
            
            self.config = config
            logger.info("配置文件加载成功")
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise
    
    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归替换配置中的环境变量
        
        支持格式: ${ENV_VAR_NAME} 或 ${ENV_VAR_NAME:default_value}
        
        Args:
            config (Dict[str, Any]): 原始配置
            
        Returns:
            Dict[str, Any]: 替换后的配置
        """
        import re
        
        def replace_value(value):
            if isinstance(value, str):
                # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replacer(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: replace_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_value(item) for item in value]
            else:
                return value
        
        return replace_value(config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置文件的基本结构
        
        Args:
            config (Dict[str, Any]): 配置字典
            
        Raises:
            ValueError: 配置验证失败
        """
        required_sections = ['data', 'model', 'training']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证数据配置
        data_config = config['data']
        if 'h5_file_path' not in data_config:
            raise ValueError("data节缺少h5_file_path配置")
        
        if 'voxel_size' not in data_config:
            raise ValueError("data节缺少voxel_size配置")
        
        # 验证体素大小
        voxel_size = data_config['voxel_size']
        if not isinstance(voxel_size, int) or voxel_size <= 0:
            raise ValueError(f"voxel_size必须是正整数，得到: {voxel_size}")
        
        # 检查是否是2的幂
        if not (voxel_size & (voxel_size - 1) == 0):
            logger.warning(f"voxel_size ({voxel_size}) 不是2的幂，可能影响性能")
        
        # 验证模型配置
        model_config = config['model']
        if 'model_channels' not in model_config:
            raise ValueError("model节缺少model_channels配置")
        
        # 验证训练配置
        training_config = config['training']
        if 'batch_size' not in training_config:
            raise ValueError("training节缺少batch_size配置")
        
        if 'learning_rate' not in training_config:
            raise ValueError("training节缺少learning_rate配置")
        
        logger.info("配置验证通过")
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个配置字典
        
        Args:
            base_config (Dict[str, Any]): 基础配置
            override_config (Dict[str, Any]): 覆盖配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = deepcopy(base)
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value)
            
            return result
        
        merged = deep_merge(base_config, override_config)
        logger.info("配置合并完成")
        return merged
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """获取验证配置"""
        return self.config.get('validation', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.config.get('system', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.config.get('inference', {})
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        保存当前配置到文件
        
        Args:
            output_path (Union[str, Path]): 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        logger.info(f"配置已保存到: {output_path}")
    
    def print_config(self, section: Optional[str] = None) -> None:
        """
        打印配置信息
        
        Args:
            section (Optional[str]): 指定打印的配置节，None表示打印全部
        """
        if section:
            if section in self.config:
                print(f"\n=== {section.upper()} 配置 ===")
                print(yaml.dump({section: self.config[section]}, default_flow_style=False, indent=2))
            else:
                print(f"配置节 '{section}' 不存在")
        else:
            print("\n=== 完整配置 ===")
            print(yaml.dump(self.config, default_flow_style=False, indent=2))
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            key_path (str): 配置键路径，用点分隔，如 "data.batch_size"
            value (Any): 新值
        """
        keys = key_path.split('.')
        current = self.config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置值
        current[keys[-1]] = value
        logger.info(f"更新配置: {key_path} = {value}")
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path (str): 配置键路径，用点分隔
            default (Any): 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "DIFFUSION_"
) -> Dict[str, Any]:
    """
    加载配置文件并应用覆盖
    
    Args:
        config_path (Union[str, Path]): 配置文件路径
        overrides (Optional[Dict[str, Any]]): 覆盖配置
        env_prefix (str): 环境变量前缀
        
    Returns:
        Dict[str, Any]: 最终配置
    """
    loader = ConfigLoader()
    config = loader.load_config(config_path)
    
    # 应用环境变量覆盖
    env_overrides = {}
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_key = key[len(env_prefix):].lower().replace('_', '.')
            
            # 尝试转换类型
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
            except:
                pass  # 保持字符串类型
            
            # 设置到env_overrides中
            keys = config_key.split('.')
            current = env_overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    # 合并覆盖配置
    if env_overrides:
        config = loader.merge_configs(config, env_overrides)
        logger.info(f"应用了 {len(env_overrides)} 个环境变量覆盖")
    
    if overrides:
        config = loader.merge_configs(config, overrides)
        logger.info(f"应用了 {len(overrides)} 个手动覆盖")
    
    return config


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        'data': {
            'h5_file_path': 'data/point_clouds.h5',
            'data_key': 'point_clouds',
            'train_ratio': 0.8,
            'voxel_size': 64,
            'voxelization_method': 'gaussian',
            'sigma': 1.0,
            'volume_dims': [20000, 20000, 2500],
            'padding': [0, 0, 100],
            'normalize': True,
            'augment': True,
            'cache_voxels': True,
            'max_cache_size': 1000
        },
        'model': {
            'model_channels': 128,
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'channel_mult': [1, 2, 4, 8],
            'dropout': 0.1,
            'use_attention': True,
            'num_timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'linear',
            'compile_model': False,
            'use_ema': True,
            'ema_decay': 0.9999
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'max_epochs': 200,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1.0,
            'limit_val_batches': 100,
            'save_top_k': 3,
            'monitor': 'val_loss',
            'mode': 'min',
            'save_last': True
        },
        'validation': {
            'sample_interval': 10,
            'num_samples': 4,
            'save_tiffs': True,
            'output_dir': 'validation_outputs',
            'ddim_steps': 50,
            'ddim_eta': 0.0
        },
        'system': {
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': '16-mixed',
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'fast_dev_run': False,
            'overfit_batches': 0,
            'limit_train_batches': 1.0
        },
        'logging': {
            'experiment_name': '3d_diffusion_voxels',
            'project_name': 'coarse2fine-pcgen',
            'log_dir': 'logs',
            'log_every_n_steps': 50,
            'save_dir': 'checkpoints'
        }
    }