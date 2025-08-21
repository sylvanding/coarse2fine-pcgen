"""
体素转换器单元测试

测试点云到体素转换功能的各种场景。
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from src.voxel.converter import PointCloudToVoxel


class TestPointCloudToVoxel(unittest.TestCase):
    """体素转换器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试点云数据
        np.random.seed(42)
        self.simple_cloud = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        # 创建较大的随机点云
        self.large_cloud = np.random.rand(1000, 3).astype(np.float32)
        
        # 创建规则排列的点云（用于验证转换）
        x, y, z = np.meshgrid(
            np.linspace(0, 1, 8),
            np.linspace(0, 1, 8), 
            np.linspace(0, 1, 8)
        )
        self.grid_cloud = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """测试转换器初始化"""
        # 默认参数
        converter = PointCloudToVoxel()
        self.assertEqual(converter.voxel_size, 64)
        self.assertEqual(converter.method, 'occupancy')
        
        # 自定义参数
        converter = PointCloudToVoxel(voxel_size=32, method='density')
        self.assertEqual(converter.voxel_size, 32)
        self.assertEqual(converter.method, 'density')
    
    def test_invalid_initialization(self):
        """测试无效初始化参数"""
        with self.assertRaises(ValueError):
            PointCloudToVoxel(voxel_size=0)
        
        with self.assertRaises(ValueError):
            PointCloudToVoxel(method='invalid_method')
        
        with self.assertRaises(ValueError):
            PointCloudToVoxel(padding_ratio=1.5)
    
    def test_occupancy_conversion(self):
        """测试占有网格转换"""
        converter = PointCloudToVoxel(voxel_size=16, method='occupancy')
        voxel_grid = converter.convert(self.simple_cloud)
        
        # 验证输出形状
        self.assertEqual(voxel_grid.shape, (16, 16, 16))
        
        # 验证数据类型
        self.assertEqual(voxel_grid.dtype, np.uint8)
        
        # 验证至少有一些体素被占用
        self.assertGreater(np.sum(voxel_grid), 0)
        
        # 验证所有值都是0或1
        unique_values = np.unique(voxel_grid)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))
    
    def test_density_conversion(self):
        """测试密度网格转换"""
        converter = PointCloudToVoxel(voxel_size=16, method='density')
        voxel_grid = converter.convert(self.large_cloud)
        
        # 验证输出形状
        self.assertEqual(voxel_grid.shape, (16, 16, 16))
        
        # 验证数据类型
        self.assertEqual(voxel_grid.dtype, np.float32)
        
        # 验证至少有一些体素被占用
        self.assertGreater(np.sum(voxel_grid), 0)
        
        # 验证所有值都是非负的
        self.assertTrue(np.all(voxel_grid >= 0))
    
    def test_gaussian_conversion(self):
        """测试高斯密度转换"""
        converter = PointCloudToVoxel(voxel_size=16, method='gaussian')
        voxel_grid = converter.convert(self.simple_cloud, sigma=1.0)
        
        # 验证输出形状
        self.assertEqual(voxel_grid.shape, (16, 16, 16))
        
        # 验证数据类型
        self.assertEqual(voxel_grid.dtype, np.float32)
        
        # 验证高斯滤波效果（应该有平滑的密度分布）
        self.assertGreater(np.sum(voxel_grid), 0)
        
        # 验证连续性（相邻体素应该有相似的值）
        # 这里简单检查是否存在非零的梯度
        grad_x = np.diff(voxel_grid, axis=0)
        grad_y = np.diff(voxel_grid, axis=1)
        grad_z = np.diff(voxel_grid, axis=2)
        total_gradient = np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y)) + np.sum(np.abs(grad_z))
        self.assertGreater(total_gradient, 0)
    
    def test_empty_point_cloud(self):
        """测试空点云处理"""
        converter = PointCloudToVoxel(voxel_size=8)
        empty_cloud = np.empty((0, 3))
        
        voxel_grid = converter.convert(empty_cloud)
        
        # 应该返回全零网格
        self.assertEqual(voxel_grid.shape, (8, 8, 8))
        self.assertEqual(np.sum(voxel_grid), 0)
    
    def test_invalid_point_cloud_shape(self):
        """测试无效点云形状"""
        converter = PointCloudToVoxel()
        
        # 2D点云（缺少z坐标）
        invalid_cloud_2d = np.random.rand(100, 2)
        with self.assertRaises(ValueError):
            converter.convert(invalid_cloud_2d)
        
        # 4D点云（多余坐标）
        invalid_cloud_4d = np.random.rand(100, 4)
        with self.assertRaises(ValueError):
            converter.convert(invalid_cloud_4d)
        
        # 1D数组
        invalid_cloud_1d = np.random.rand(100)
        with self.assertRaises(ValueError):
            converter.convert(invalid_cloud_1d)
    
    def test_batch_conversion(self):
        """测试批量转换"""
        converter = PointCloudToVoxel(voxel_size=8, method='occupancy')
        
        # 创建批次数据
        batch_size = 3
        batch_clouds = np.array([self.simple_cloud] * batch_size)
        
        batch_voxels = converter.convert_batch(batch_clouds)
        
        # 验证输出形状
        self.assertEqual(batch_voxels.shape, (batch_size, 8, 8, 8))
        
        # 验证每个样本都被正确转换
        for i in range(batch_size):
            self.assertGreater(np.sum(batch_voxels[i]), 0)
    
    def test_voxel_to_points_conversion(self):
        """测试体素到点云的反向转换"""
        converter = PointCloudToVoxel(voxel_size=16, method='occupancy')
        
        # 先转换为体素
        voxel_grid = converter.convert(self.simple_cloud)
        
        # 再转换回点云
        reconstructed_points = converter.voxel_to_points(voxel_grid, threshold=0.5)
        
        # 验证输出格式
        self.assertEqual(reconstructed_points.shape[1], 3)
        self.assertGreaterEqual(len(reconstructed_points), 0)
        
        # 验证点云在合理范围内
        if len(reconstructed_points) > 0:
            self.assertTrue(np.all(np.isfinite(reconstructed_points)))
    
    def test_save_as_tiff(self):
        """测试TIFF保存功能"""
        converter = PointCloudToVoxel(voxel_size=8, method='occupancy')
        voxel_grid = converter.convert(self.simple_cloud)
        
        output_path = os.path.join(self.temp_dir, 'test_voxel.tiff')
        converter.save_as_tiff(voxel_grid, output_path)
        
        # 验证文件已创建
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_conversion_info(self):
        """测试转换信息获取"""
        converter = PointCloudToVoxel(voxel_size=16, method='density', padding_ratio=0.2)
        
        # 执行转换
        converter.convert(self.simple_cloud)
        
        # 获取转换信息
        info = converter.get_conversion_info()
        
        # 验证信息完整性
        self.assertIn('voxel_size', info)
        self.assertIn('method', info)
        self.assertIn('padding_ratio', info)
        self.assertIn('last_min_bounds', info)
        self.assertIn('last_max_bounds', info)
        
        self.assertEqual(info['voxel_size'], 16)
        self.assertEqual(info['method'], 'density')
        self.assertEqual(info['padding_ratio'], 0.2)
    
    def test_custom_bounds(self):
        """测试自定义边界"""
        custom_bounds = ((-1, -1, -1), (2, 2, 2))
        converter = PointCloudToVoxel(
            voxel_size=8, 
            method='occupancy',
            bounds=custom_bounds
        )
        
        voxel_grid = converter.convert(self.simple_cloud)
        
        # 验证转换成功
        self.assertEqual(voxel_grid.shape, (8, 8, 8))
        self.assertGreater(np.sum(voxel_grid), 0)
    
    def test_different_voxel_sizes(self):
        """测试不同体素分辨率"""
        sizes = [4, 8, 16, 32]
        
        for size in sizes:
            with self.subTest(voxel_size=size):
                converter = PointCloudToVoxel(voxel_size=size)
                voxel_grid = converter.convert(self.simple_cloud)
                
                self.assertEqual(voxel_grid.shape, (size, size, size))
                self.assertGreater(np.sum(voxel_grid), 0)


if __name__ == '__main__':
    unittest.main()
