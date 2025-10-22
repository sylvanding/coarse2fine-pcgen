"""
可视化生成样本

加载NIfTI格式的生成样本并进行可视化。
"""

import sys
from pathlib import Path
import argparse
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_nifti(nifti_path: str) -> np.ndarray:
    """加载NIfTI文件"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # 移除通道维度
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    
    return data


def visualize_3d_slices(
    voxel: np.ndarray,
    title: str = "3D Voxel Visualization",
    figsize: tuple = (15, 5),
    save_path: str = None
):
    """
    可视化3D体素的三个正交切片
    
    Args:
        voxel: 3D体素数组 (D, H, W)
        title: 图像标题
        figsize: 图像大小
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig)
    
    # 取中间切片
    mid_d = voxel.shape[0] // 2
    mid_h = voxel.shape[1] // 2
    mid_w = voxel.shape[2] // 2
    
    # 轴向切片 (横断面)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(voxel[mid_d, :, :], cmap='gray', origin='lower')
    ax1.set_title('Axial (横断面)')
    ax1.axis('off')
    
    # 冠状切片
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(voxel[:, mid_h, :], cmap='gray', origin='lower')
    ax2.set_title('Coronal (冠状面)')
    ax2.axis('off')
    
    # 矢状切片
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(voxel[:, :, mid_w], cmap='gray', origin='lower')
    ax3.set_title('Sagittal (矢状面)')
    ax3.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"可视化已保存到: {save_path}")
    
    return fig


def visualize_multiple_samples(
    sample_paths: list,
    output_path: str = None,
    max_samples: int = 4
):
    """
    可视化多个样本
    
    Args:
        sample_paths: 样本文件路径列表
        output_path: 输出路径
        max_samples: 最大显示样本数
    """
    n_samples = min(len(sample_paths), max_samples)
    
    fig = plt.figure(figsize=(15, 4 * n_samples))
    gs = GridSpec(n_samples, 3, figure=fig)
    
    for i, sample_path in enumerate(sample_paths[:n_samples]):
        logger.info(f"加载样本: {sample_path}")
        voxel = load_nifti(sample_path)
        
        # 取中间切片
        mid_d = voxel.shape[0] // 2
        mid_h = voxel.shape[1] // 2
        mid_w = voxel.shape[2] // 2
        
        # 轴向切片
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(voxel[mid_d, :, :], cmap='gray', origin='lower')
        if i == 0:
            ax1.set_title('Axial (横断面)', fontsize=12)
        ax1.set_ylabel(f'Sample {i+1}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 冠状切片
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(voxel[:, mid_h, :], cmap='gray', origin='lower')
        if i == 0:
            ax2.set_title('Coronal (冠状面)', fontsize=12)
        ax2.axis('off')
        
        # 矢状切片
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(voxel[:, :, mid_w], cmap='gray', origin='lower')
        if i == 0:
            ax3.set_title('Sagittal (矢状面)', fontsize=12)
        ax3.axis('off')
    
    fig.suptitle('Generated Samples Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"可视化已保存到: {output_path}")
    
    plt.show()
    return fig


def compute_statistics(voxel: np.ndarray) -> dict:
    """计算体素统计信息"""
    return {
        'shape': voxel.shape,
        'dtype': voxel.dtype,
        'min': np.min(voxel),
        'max': np.max(voxel),
        'mean': np.mean(voxel),
        'std': np.std(voxel),
        'non_zero_ratio': np.sum(voxel > 0) / voxel.size
    }


def main():
    parser = argparse.ArgumentParser(description="可视化生成样本")
    
    parser.add_argument(
        '--sample_dir',
        type=str,
        required=True,
        help='样本目录路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='可视化输出目录'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=4,
        help='最大显示样本数'
    )
    
    parser.add_argument(
        '--show_stats',
        action='store_true',
        help='显示统计信息'
    )
    
    args = parser.parse_args()
    
    # 收集样本文件
    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        logger.error(f"样本目录不存在: {sample_dir}")
        return
    
    sample_files = sorted(list(sample_dir.glob("*.nii.gz")) + list(sample_dir.glob("*.nii")))
    
    if len(sample_files) == 0:
        logger.error(f"在 {sample_dir} 中未找到NIfTI文件")
        return
    
    logger.info(f"找到 {len(sample_files)} 个样本")
    
    # 输出目录
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = sample_dir / "visualizations"
        output_path.mkdir(exist_ok=True)
    
    # 可视化样本
    if args.show_stats:
        # 显示每个样本的统计信息
        for i, sample_file in enumerate(sample_files[:args.max_samples]):
            logger.info(f"\n样本 {i+1}: {sample_file.name}")
            voxel = load_nifti(str(sample_file))
            stats = compute_statistics(voxel)
            
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # 单独可视化
            vis_path = output_path / f"{sample_file.stem}_visualization.png"
            visualize_3d_slices(
                voxel,
                title=f"Sample {i+1}: {sample_file.name}",
                save_path=str(vis_path)
            )
            plt.close()
    
    # 批量可视化
    output_file = output_path / "samples_overview.png"
    visualize_multiple_samples(
        [str(f) for f in sample_files],
        output_path=str(output_file),
        max_samples=args.max_samples
    )
    
    logger.info(f"\n可视化完成! 输出目录: {output_path}")


if __name__ == "__main__":
    main()

