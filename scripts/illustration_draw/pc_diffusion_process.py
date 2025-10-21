"""
点云扩散去噪过程模拟脚本

模拟diffusion model的真实去噪过程，从各向同性高斯噪声逐步收敛到目标点云。
采用DDPM (Denoising Diffusion Probabilistic Models)的反向过程，
通过插值模拟每一步的去噪状态。

核心原理:
    1. t=T (初始): x_T ~ N(0, I) - 纯高斯噪声
    2. t=T→0 (去噪): x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
    3. t=0 (终点): x_0 - 目标点云

    其中α_t是噪声调度系数，从0→1线性或余弦变化

Usage:
    # 处理单个文件
    python scripts/illustration_draw/pc_diffusion_process.py --input path/to/pointcloud.csv

    # 批量处理文件夹
    python scripts/illustration_draw/pc_diffusion_process.py --input path/to/folder --recursive

    # 自定义时间步数和采样策略
    python scripts/illustration_draw/pc_diffusion_process.py --input file.csv --num-steps 10 --schedule cosine

    # 对大规模点云进行降采样
    python scripts/illustration_draw/pc_diffusion_process.py --input file.csv --max-points 100000

Example:
    python scripts/illustration_draw/pc_diffusion_process.py --input data/pointclouds/sample.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.render import render_to_2d_image, save_image


class PointCloudDiffusionSimulator:
    """点云扩散去噪过程模拟器

    模拟DDPM的反向去噪过程，从高斯噪声逐步收敛到目标点云。
    """

    def __init__(
        self,
        num_steps: int = 10,
        schedule: str = "linear",
        output_steps: Optional[List[int]] = None,
        max_points: Optional[int] = None,
        output_dir: str = "outputs/diffusion_simulation",
        pixel_size: float = 100.0,
        psf_sigma: float = 150.0,
        intensity_scale: float = 1.0,
        normalize: bool = False,
        crop_to_original: bool = True,
    ):
        """
        初始化扩散模拟器

        Args:
            num_steps: 去噪步数（包括t=T和t=0）
            schedule: 噪声调度策略，'linear'或'cosine'
            output_steps: 需要输出的步骤列表（索引从0开始），None表示输出所有步骤
            max_points: 最大点数，超过则降采样，None表示不限制
            output_dir: 输出目录
            pixel_size: 渲染图像的像素尺寸（纳米）
            psf_sigma: 点扩散函数的标准差（纳米）
            intensity_scale: 渲染强度缩放因子
            normalize: 是否将点云归一化到单位球
            crop_to_original: 是否裁剪点云到原始点云的边界
        """
        self.num_steps = num_steps
        self.schedule = schedule
        self.max_points = max_points
        self.output_dir = Path(output_dir)
        self.pixel_size = pixel_size
        self.psf_sigma = psf_sigma
        self.intensity_scale = intensity_scale
        self.normalize = normalize
        self.crop_to_original = crop_to_original

        # 设置需要输出的步骤
        if output_steps is None:
            # 默认输出所有步骤
            self.output_steps = list(range(num_steps))
        else:
            # 验证输出步骤的有效性
            self.output_steps = sorted(set(output_steps))  # 去重并排序
            invalid_steps = [s for s in self.output_steps if s < 0 or s >= num_steps]
            if invalid_steps:
                raise ValueError(
                    f"无效的输出步骤: {invalid_steps}，步骤索引必须在[0, {num_steps - 1}]范围内"
                )

        # 计算噪声调度系数 α_t
        self.alpha_schedule = self._compute_alpha_schedule()

    def _compute_alpha_schedule(self) -> np.ndarray:
        """
        计算噪声调度系数 α_t

        Returns:
            np.ndarray: 形状为(num_steps,)的α_t数组，从0到1
        """
        if self.schedule == "linear":
            # 线性调度: α_t从0线性增长到1
            alpha_schedule = np.linspace(0.0, 1.0, self.num_steps)

        elif self.schedule == "cosine":
            # 余弦调度: α_t = cos^2(π/2 * t/T)
            # 这种调度在早期去噪更慢，后期更快
            t = np.linspace(0, 1, self.num_steps)
            alpha_schedule = np.cos(np.pi / 2 * (1 - t)) ** 2

        else:
            raise ValueError(f"不支持的调度策略: {self.schedule}")

        return alpha_schedule

    def downsample_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """
        对点云进行降采样

        Args:
            points: 形状为(N, 3)的点云数据

        Returns:
            np.ndarray: 降采样后的点云
        """
        if self.max_points is None or len(points) <= self.max_points:
            return points

        # 随机采样
        indices = np.random.choice(len(points), self.max_points, replace=False)
        downsampled = points[indices]

        print(f"  降采样: {len(points)} → {len(downsampled)} 个点")
        return downsampled

    def normalize_pointcloud(self, points: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        将点云归一化到单位球或保持原始坐标

        Args:
            points: 形状为(N, 3)的点云数据

        Returns:
            Tuple[np.ndarray, dict]: (归一化后的点云, 归一化参数字典)
        """
        # 计算中心和范围
        center = np.mean(points, axis=0)
        centered_points = points - center

        if self.normalize:
            # 归一化到单位球
            max_radius = np.max(np.linalg.norm(centered_points, axis=1))
            normalized_points = (
                centered_points / max_radius if max_radius > 0 else centered_points
            )

            norm_params = {
                "center": center,
                "max_radius": max_radius,
                "normalized": True,
            }
        else:
            # 保持原始坐标
            normalized_points = centered_points
            norm_params = {"center": center, "max_radius": 1.0, "normalized": False}

        return normalized_points, norm_params

    def denormalize_pointcloud(
        self, points: np.ndarray, norm_params: dict
    ) -> np.ndarray:
        """
        反归一化点云到原始坐标系

        Args:
            points: 归一化后的点云
            norm_params: 归一化参数

        Returns:
            np.ndarray: 反归一化后的点云
        """
        if norm_params["normalized"]:
            points = points * norm_params["max_radius"]

        points = points + norm_params["center"]
        return points

    def generate_gaussian_noise(
        self, num_points: int, scale: float = 1.0
    ) -> np.ndarray:
        """
        生成各向同性高斯噪声点云

        Args:
            num_points: 点数
            scale: 噪声尺度（标准差）

        Returns:
            np.ndarray: 形状为(num_points, 3)的噪声点云
        """
        # 生成标准正态分布噪声
        noise = np.random.randn(num_points, 3) * scale
        return noise

    def diffusion_forward_step(
        self, x0: np.ndarray, alpha_t: float, noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        扩散前向过程的某一步: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε

        Args:
            x0: 目标点云（干净数据）
            alpha_t: 当前步的噪声系数
            noise: 噪声项，如果为None则生成新的噪声

        Returns:
            np.ndarray: 加噪后的点云 x_t
        """
        if noise is None:
            noise = self.generate_gaussian_noise(len(x0), scale=1.0)

        # DDPM前向过程公式
        xt = np.sqrt(alpha_t) * x0 + np.sqrt(1 - alpha_t) * noise
        return xt

    def load_pointcloud_from_csv(self, csv_path: str) -> np.ndarray:
        """
        从CSV文件加载点云数据

        Args:
            csv_path: CSV文件路径

        Returns:
            np.ndarray: 形状为(N, 3)的点云数据，单位为纳米

        Raises:
            ValueError: 当CSV文件格式不正确时
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查列名
        expected_columns = ["x [nm]", "y [nm]", "z [nm]"]
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(
                f"CSV文件必须包含列: {expected_columns}, 实际列: {df.columns.tolist()}"
            )

        # 提取xyz坐标
        points = df[expected_columns].values

        print(f"从 {csv_path.name} 加载了 {len(points)} 个点")
        print(f"  X范围: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] nm")
        print(f"  Y范围: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] nm")
        print(f"  Z范围: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] nm")

        return points

    def simulate_denoising_trajectory(
        self, x0: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> dict:
        """
        模拟去噪轨迹，只计算需要输出的步骤

        Args:
            x0: 目标点云（干净数据）
            noise: 初始噪声，如果为None则生成新的噪声

        Returns:
            dict: 字典，键为步骤索引，值为该步骤的点云数组
        """
        if noise is None:
            noise = self.generate_gaussian_noise(len(x0), scale=1.0)

        trajectory = {}

        # 只计算需要输出的步骤
        for step_idx in self.output_steps:
            alpha_t = self.alpha_schedule[step_idx]

            # 计算当前步的点云
            xt = self.diffusion_forward_step(x0, alpha_t, noise)
            trajectory[step_idx] = xt

            # 打印进度
            noise_ratio = 1.0 - alpha_t
            print(
                f"  步骤 {step_idx + 1}/{self.num_steps}: α_t={alpha_t:.4f}, 噪声比例={noise_ratio:.4f}"
            )

        return trajectory

    def save_pointcloud_to_csv(
        self, points: np.ndarray, output_path: str, decimal_places: int = 1
    ):
        """
        保存点云到CSV文件

        Args:
            points: 形状为(N, 3)的点云数据
            output_path: 输出CSV文件路径
            decimal_places: 保留的小数位数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建DataFrame
        df = pd.DataFrame(points, columns=["x [nm]", "y [nm]", "z [nm]"])

        # 保存为CSV，控制小数位数
        df.to_csv(output_path, index=False, float_format=f"%.{decimal_places}f")
        print(f"  保存CSV: {output_path.name} ({len(points)} 个点)")

    def get_volume_dims(self, points: np.ndarray) -> List[float]:
        """
        计算点云的体积维度

        Args:
            points: 形状为(N, 3)的点云数据

        Returns:
            List[float]: [x_size, y_size, z_size]，单位纳米
        """
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        volume_dims = (max_coords - min_coords).tolist()
        return volume_dims

    def render_and_save_image(
        self,
        points: np.ndarray,
        output_path: str,
        original_points: np.ndarray,
        z_range: Optional[Tuple[float, float]] = None,
    ):
        """
        渲染点云为2D图像并保存

        Args:
            points: 形状为(N, 3)的点云数据（可能是归一化或原始坐标）
            output_path: 输出PNG文件路径
            original_points: 原始点云（用于计算体积维度）
            z_range: Z方向的深度范围，如果为None则包含所有点
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用当前点云的尺度计算体积维度
        volume_dims = self.get_volume_dims(points)

        # 渲染点云为2D图像
        image = render_to_2d_image(
            points=points,
            pixel_size=self.pixel_size,
            psf_sigma=self.psf_sigma,
            volume_dims=volume_dims,
            z_range=z_range,
            intensity_scale=self.intensity_scale,
            add_background_noise=False,
            output_dtype=np.uint16,
            normalize=True,
        )

        # 保存图像
        save_image(image, str(output_path))
        print(f"  保存图像: {output_path.name}")

    def crop_to_original_points(
        self, points: np.ndarray, original_points: np.ndarray
    ) -> np.ndarray:
        """
        裁剪点云到原始点云的边界
        """
        min_coords = np.min(original_points, axis=0)
        max_coords = np.max(original_points, axis=0)
        points = points[
            (points[:, 0] >= min_coords[0])
            & (points[:, 0] <= max_coords[0])
            & (points[:, 1] >= min_coords[1])
            & (points[:, 1] <= max_coords[1])
            & (points[:, 2] >= min_coords[2])
            & (points[:, 2] <= max_coords[2])
        ]
        return points

    def process_single_pointcloud(
        self, csv_path: str, output_subdir: Optional[str] = None
    ):
        """
        处理单个点云文件，模拟从高斯噪声到目标点云的去噪过程

        Args:
            csv_path: 输入CSV文件路径
            output_subdir: 输出子目录名称，如果为None则使用CSV文件名
        """
        csv_path = Path(csv_path)

        print(f"\n{'=' * 80}")
        print(f"处理点云: {csv_path.name}")
        print(f"{'=' * 80}")

        # 1. 加载原始点云
        original_points = self.load_pointcloud_from_csv(csv_path)

        # 2. 降采样（如果需要）
        points = self.downsample_pointcloud(original_points)

        # 3. 归一化点云
        print(f"\n归一化点云...")
        x0_normalized, norm_params = self.normalize_pointcloud(points)
        print(
            f"  归一化参数: center={norm_params['center']}, "
            f"max_radius={norm_params['max_radius']:.2f}"
        )

        # 4. 模拟去噪轨迹
        print(f"\n模拟去噪轨迹 (共{self.num_steps}步, 调度策略={self.schedule})...")
        print(f"  只计算步骤: {self.output_steps} (共{len(self.output_steps)}个)")
        trajectory = self.simulate_denoising_trajectory(x0_normalized)

        # 5. 设置输出目录
        if output_subdir is None:
            output_subdir = csv_path.stem
        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n保存结果到: {output_dir}")

        # 6. 保存所有计算的步骤结果
        output_count = 0
        for step_idx in sorted(trajectory.keys()):
            xt_normalized = trajectory[step_idx]

            alpha_t = self.alpha_schedule[step_idx]
            noise_ratio = 1.0 - alpha_t

            print(
                f"\n保存步骤 {step_idx + 1}/{self.num_steps} (α_t={alpha_t:.4f}, 噪声={noise_ratio:.4f})..."
            )

            # 反归一化到原始坐标系
            xt_denormalized = self.denormalize_pointcloud(xt_normalized, norm_params)

            # 生成文件名（使用步骤索引和alpha值）
            step_str = f"step_{step_idx:03d}_alpha_{alpha_t:.4f}".replace(".", "p")
            csv_output = output_dir / f"{csv_path.stem}_{step_str}.csv"
            png_output = output_dir / f"{csv_path.stem}_{step_str}.png"

            # crop to original points
            if self.crop_to_original:
                xt_denormalized = self.crop_to_original_points(
                    xt_denormalized, original_points
                )

            # 保存CSV
            self.save_pointcloud_to_csv(xt_denormalized, csv_output, decimal_places=1)

            # 渲染并保存图像
            self.render_and_save_image(
                points=xt_denormalized,
                output_path=png_output,
                original_points=original_points,
                z_range=None,
            )

            output_count += 1

        print(f"\n{'=' * 80}")
        print(f"✓ 完成处理: {csv_path.name}")
        print(f"  输出文件数: {output_count} CSV + {output_count} PNG")
        print(f"  输出目录: {output_dir}")
        print(f"{'=' * 80}")

    def find_csv_files(self, input_path: str, recursive: bool = False) -> List[Path]:
        """
        查找CSV文件

        Args:
            input_path: 输入路径（文件或文件夹）
            recursive: 是否递归搜索子文件夹

        Returns:
            List[Path]: CSV文件路径列表
        """
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() == ".csv":
                return [input_path]
            else:
                raise ValueError(f"输入文件不是CSV格式: {input_path}")

        elif input_path.is_dir():
            if recursive:
                csv_files = list(input_path.rglob("*.csv"))
            else:
                csv_files = list(input_path.glob("*.csv"))

            if not csv_files:
                raise ValueError(f"在 {input_path} 中未找到CSV文件")

            return sorted(csv_files)

        else:
            raise FileNotFoundError(f"路径不存在: {input_path}")

    def process_batch(self, input_path: str, recursive: bool = False):
        """
        批量处理CSV文件

        Args:
            input_path: 输入路径（文件或文件夹）
            recursive: 是否递归搜索子文件夹
        """
        csv_files = self.find_csv_files(input_path, recursive)

        print(f"找到 {len(csv_files)} 个CSV文件")

        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'=' * 80}")
            print(f"进度: {i}/{len(csv_files)}")
            print(f"{'=' * 80}")

            try:
                # 对于批量处理，使用相对路径作为子目录
                input_path_obj = Path(input_path)
                if input_path_obj.is_dir():
                    rel_path = csv_file.relative_to(input_path_obj)
                    output_subdir = (
                        str(rel_path.parent)
                        if rel_path.parent != Path(".")
                        else rel_path.stem
                    )
                else:
                    output_subdir = csv_file.stem

                self.process_single_pointcloud(csv_file, output_subdir)

            except Exception as e:
                print(f"✗ 处理失败: {csv_file.name}")
                print(f"  错误: {e}")
                continue

        print(f"\n{'=' * 80}")
        print(f"批量处理完成！共处理 {len(csv_files)} 个文件")
        print(f"输出目录: {self.output_dir}")
        print(f"{'=' * 80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="点云扩散去噪过程模拟 - 从高斯噪声逐步收敛到目标点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/repos/datasets/illustration_diffusion_process",
        help="输入CSV文件或包含CSV文件的文件夹路径",
    )

    parser.add_argument(
        "--recursive",
        default=True,
        action="store_true",
        help="递归搜索子文件夹中的CSV文件",
    )

    parser.add_argument(
        "--num-steps", type=int, default=1000, help="去噪步数（默认: 10）"
    )

    parser.add_argument(
        "--output-steps",
        type=str,
        default="600,850,950,980,990,996,999",
        help='需要输出的步骤，用逗号分隔（例如: "0,5,9"）或指定范围（例如: "0-9"）。不指定则输出所有步骤',
    )

    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="噪声调度策略: linear或cosine（默认: linear）",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="最大点数限制，超过则降采样（默认: None，不限制）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/repos/datasets/illustration_diffusion_process1",
        help="输出目录（默认: outputs/diffusion_simulation）",
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=50.0,
        help="渲染图像的像素尺寸（纳米）（默认: 100.0）",
    )

    parser.add_argument(
        "--psf-sigma",
        type=float,
        default=100.0,
        help="点扩散函数的标准差（纳米）（默认: 100.0）",
    )

    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=1.0,
        help="渲染强度缩放因子（默认: 1.0）",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="不归一化点云到单位球（默认: False，即默认归一化）",
    )

    parser.add_argument(
        "--crop-to-original",
        action="store_true",
        default=False,
        help="裁剪点云到原始点云的边界（默认: True）",
    )

    args = parser.parse_args()

    # 验证参数
    if args.num_steps < 2:
        parser.error("去噪步数必须 >= 2")

    # 解析输出步骤参数
    output_steps = None
    if args.output_steps is not None:
        try:
            # 支持逗号分隔的列表: "0,5,9"
            if "," in args.output_steps:
                output_steps = [int(s.strip()) for s in args.output_steps.split(",")]
            # 支持范围: "0-9"
            elif "-" in args.output_steps:
                start, end = args.output_steps.split("-")
                output_steps = list(range(int(start.strip()), int(end.strip()) + 1))
            # 支持单个数字: "5"
            else:
                output_steps = [int(args.output_steps.strip())]
        except ValueError:
            parser.error(
                f"无效的输出步骤格式: {args.output_steps}，请使用逗号分隔（例如: '0,5,9'）或范围（例如: '0-9'）"
            )

    # 创建模拟器
    simulator = PointCloudDiffusionSimulator(
        num_steps=args.num_steps,
        schedule=args.schedule,
        output_steps=output_steps,
        max_points=args.max_points,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        psf_sigma=args.psf_sigma,
        intensity_scale=args.intensity_scale,
        normalize=not args.no_normalize,  # 默认True，除非指定--no-normalize
        crop_to_original=args.crop_to_original,
    )

    # 打印配置
    print("=" * 80)
    print("点云扩散去噪过程模拟")
    print("=" * 80)
    print(f"配置:")
    print(f"  去噪步数: {args.num_steps}")
    print(f"  输出步骤: {output_steps if output_steps else '全部'}")
    print(f"  调度策略: {args.schedule}")
    print(f"  最大点数: {args.max_points if args.max_points else '无限制'}")
    print(f"  归一化: {not args.no_normalize}")
    print(f"  裁剪到原始点云: {args.crop_to_original}")
    print(f"  渲染参数: pixel_size={args.pixel_size}nm, psf_sigma={args.psf_sigma}nm")
    print("=" * 80)

    # 处理输入
    try:
        simulator.process_batch(args.input, recursive=args.recursive)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
