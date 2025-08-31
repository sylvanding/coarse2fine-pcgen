#!/usr/bin/env python3
"""
RefineNet训练脚本

用于训练点云修正网络，支持H5数据加载、数据集分割、
TensorBoard记录、验证和推理等功能。
"""

import shutil
import sys
import os
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入RefineNet模块
from src.refine_net import options, main

def create_output_directories(save_path, resume_training=False):
    """创建输出目录"""
    save_path = Path(save_path)
    
    # 如果不是恢复训练且目录已存在,则删除
    if save_path.exists() and not resume_training:
        print(f"删除已存在的输出目录: {save_path}")
        shutil.rmtree(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    subdirs = [
        'validation_results',
        'inference_results', 
        'checkpoints',
        'tensorboard'
    ]
    
    for subdir in subdirs:
        (save_path / subdir).mkdir(exist_ok=True)
    
    if resume_training:
        print(f"保留现有输出目录: {save_path}")
    else:
        print(f"输出目录已创建: {save_path}")

def main_train():
    """主训练函数"""
    # 获取参数解析器
    parser = options.get_parser('RefineNet点云修正网络训练')
    
    # 添加训练脚本特定参数
    # parser.add_argument('--output-dir', type=str, default='/repos/coarse2fine-pcgen/refine-exp',
    #                     help='训练输出目录')
    
    # 解析参数
    args = parser.parse_args()
    
    # # 设置保存路径
    # args.save_path = Path(args.output_dir)
    
    # 判断是否为恢复训练
    resume_training = bool(args.resume_from)
    
    # 创建输出目录
    create_output_directories(args.save_path, resume_training)
    
    # 检查输入文件
    if not os.path.exists(args.gt_h5_path):
        raise FileNotFoundError(f"GT H5文件不存在: {args.gt_h5_path}")
    
    if not os.path.exists(args.noisy_h5_path):
        raise FileNotFoundError(f"噪声H5文件不存在: {args.noisy_h5_path}")
    
    print("=== RefineNet训练配置 ===")
    print(f"GT H5文件: {args.gt_h5_path}")
    print(f"噪声H5文件: {args.noisy_h5_path}")
    print(f"GT数据键: {args.gt_data_key}")
    print(f"噪声数据键: {args.noisy_data_key}")
    print(f"采样点数: {args.sample_points}")
    print(f"训练数据占比: {args.train_ratio}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"最大迭代次数: {args.iterations}")
    print(f"验证间隔: {args.val_interval} iterations")
    print(f"推理开始iteration: {args.inference_start_iteration}")
    print(f"推理间隔: {args.inference_interval} iterations")
    print(f"检查点保存间隔: {args.checkpoint_interval} iterations")
    print(f"输出目录: {args.save_path}")
    print(f"使用TensorBoard: {args.use_tensorboard}")
    print(f"体积维度: {args.volume_dims}")
    print(f"边界填充: {args.padding}")
    
    # 恢复训练信息
    if resume_training:
        print("\n--- 恢复训练配置 ---")
        print(f"恢复方式: {args.resume_from}")
        print(f"恢复优化器: {args.resume_optimizer}")
        if args.resume_start_iteration > 0:
            print(f"强制起始iteration: {args.resume_start_iteration}")
    else:
        print("\n--- 从头开始训练 ---")
    
    print("=" * 30)
    
    # 保存训练配置
    config_path = args.save_path / 'training_config.txt'
    with open(config_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"训练配置已保存: {config_path}")
    
    # 开始训练
    try:
        main.train(args)
        print("\n🎉 训练成功完成！")
        
        print("\n=== 训练结果 ===")
        print(f"最终模型: {args.save_path / 'generators/final_model.pt'}")
        print(f"最终检查点: {args.save_path / 'generators/final.pt'}")
        print(f"最佳模型: {args.save_path / 'generators/best.pt'}")
        print(f"验证结果: {args.save_path / 'validation_results'}")
        print(f"推理结果: {args.save_path / 'inference_results'}")
        print(f"检查点目录: {args.save_path / 'generators'}")
        if args.use_tensorboard:
            print(f"TensorBoard日志: {args.save_path / 'tensorboard'}")
            print("启动TensorBoard: tensorboard --logdir " + str(args.save_path / 'tensorboard'))
        
        print("\n=== 恢复训练示例 ===")
        print("从最新检查点恢复:")
        print(f"  python {__file__} --resume-from latest --output-dir {args.output_dir}")
        print("从最佳检查点恢复:")
        print(f"  python {__file__} --resume-from best --output-dir {args.output_dir}")
        print("从指定iteration恢复:")
        print(f"  python {__file__} --resume-from iteration_010000 --output-dir {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main_train()
    sys.exit(exit_code)
