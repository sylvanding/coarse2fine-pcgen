import argparse
from pathlib import Path
import os
import src.refine_net.util as util
import warnings


def get_parser(name='Self-Sampling') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--iterations', default=20000000, type=int)
    parser.add_argument('--export-interval', default=1000, type=int)
    parser.add_argument('--D1', default=5000, type=int)
    parser.add_argument('--D2', default=5000, type=int)
    parser.add_argument('--max-points', default=-1, type=int)
    parser.add_argument('--save-path', type=Path, default='/repos/coarse2fine-pcgen/refine-exp')
    parser.add_argument('--pc', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--stn', action='store_true')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--init-var', default=0.2, type=float)
    parser.add_argument('--sampling-mode', default='h5_paired', type=str)
    parser.add_argument('--p1', default=0.9, type=float)
    parser.add_argument('--p2', default=-1.0, type=float)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--percentile', type=float, default=-1.0)
    parser.add_argument('--ang-wt', type=float, default=0.1)
    parser.add_argument('--force-normal-estimation', action='store_true')
    parser.add_argument('--kmeans', action='store_true')
    parser.add_argument('--mse', default=False, action='store_true')
    parser.add_argument('--curvature-cache', type=str, default='')
    
    # H5配对数据集参数
    parser.add_argument('--gt-h5-path', type=str, default='/hy-tmp/exp_pc_mt/mt_exp_pointcloud_samples.h5',
                        help='GT点云H5文件路径')
    parser.add_argument('--noisy-h5-path', type=str, default='/hy-tmp/exp_pc_mt/mt_exp_pointcloud_samples_noised.h5',
                        help='噪声点云H5文件路径')
    parser.add_argument('--gt-data-key', type=str, default='point_clouds',
                        help='GT数据在H5文件中的键名')
    parser.add_argument('--noisy-data-key', type=str, default='point_clouds',
                        help='噪声数据在H5文件中的键名')
    parser.add_argument('--sample-points', type=int, default=2048,
                        help='采样点数，用于统一GT和噪声点云的点数')
    
    # 验证和保存参数
    parser.add_argument('--val-interval', type=int, default=100,
                        help='验证间隔（iteration数）')
    parser.add_argument('--val-save-csv', action='store_true', default=False,
                        help='验证时保存点云到CSV文件')
    
    # 批处理推理参数
    parser.add_argument('--inference-batch-size', type=int, default=2048,
                        help='推理时的批处理点数(和训练时一致)')
    
    # 数据集分割参数
    parser.add_argument('--train-ratio', type=float, default=0.95,
                        help='训练数据占比')
    
    # 归一化参数
    parser.add_argument('--volume-dims', type=int, nargs=3, default=[8000, 8000, 300],
                        help='体积维度 [x, y, z]')
    parser.add_argument('--padding', type=int, nargs=3, default=[0, 0, 0],
                        help='边界填充 [x, y, z]')
    
    # TensorBoard和推理参数
    parser.add_argument('--use-tensorboard', action='store_true', default=True,
                        help='使用TensorBoard记录训练过程')
    parser.add_argument('--inference-start-iteration', type=int, default=10000,
                        help='开始推理的iteration数')
    parser.add_argument('--inference-interval', type=int, default=10000,
                        help='推理间隔（iteration数）')
    
    # 检查点保存参数
    parser.add_argument('--checkpoint-interval', type=int, default=20000,
                        help='检查点保存间隔（iteration数）')
    
    # 噪声增强参数
    parser.add_argument('--add-training-noise', action='store_true', default=False,
                        help='训练时是否对噪声点云添加额外随机漂移')
    parser.add_argument('--noise-ratio', type=float, default=0.2,
                        help='添加噪声的点数比例 (0.0-1.0)')
    parser.add_argument('--noise-scale', type=float, default=0.2,
                        help='噪声漂移的最大幅度 (相对于归一化坐标范围)')
    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['gaussian', 'uniform'],
                        help='噪声类型：gaussian（高斯噪声）或uniform（均匀噪声）')
    
    # 继续训练参数
    parser.add_argument('--resume-from', type=str, default='',
                        help='从指定检查点恢复训练。支持格式：\n'
                             '- latest: 自动寻找最新检查点\n'
                             '- best: 寻找最佳检查点\n'
                             '- final: 从最终检查点恢复\n'
                             '- iteration_XXXXXX: 从指定iteration恢复\n'
                             '- /path/to/checkpoint.pt: 指定检查点文件路径')
    parser.add_argument('--resume-optimizer', action='store_true', default=True,
                        help='恢复训练时是否同时恢复优化器状态')
    parser.add_argument('--resume-start-iteration', type=int, default=-1,
                        help='强制指定恢复训练的起始iteration，-1表示使用检查点中的iteration')

    return parser


def parse_args(parser: argparse.ArgumentParser, inference=False):
    args = parser.parse_args()

    if args.p2 == -1.0:
        args.p2 = 1 - args.p1

    if not os.path.exists(args.save_path):
        Path.mkdir(args.save_path, exist_ok=True, parents=True)
    if not inference:
        Path.mkdir(args.save_path / 'exports', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'targets', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'sources', exist_ok=True, parents=True)
        Path.mkdir(args.save_path / 'generators', exist_ok=True, parents=True)

    with open(args.save_path / ('inference_args.txt' if inference else 'args.txt'), 'w+') as file:
        file.write(util.args_to_str(args))

    return args
