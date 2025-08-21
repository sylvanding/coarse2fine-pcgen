"""
H5数据加载器单元测试

测试点云H5文件读取功能的各种场景。
"""

import unittest
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

from src.data.h5_loader import PointCloudH5Loader, BatchIterator


class TestPointCloudH5Loader(unittest.TestCase):
    """H5加载器测试类"""
    
    def setUp(self):
        """
        测试前准备：创建临时测试数据
        """
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_pointcloud.h5')
        
        # 创建测试数据
        self.num_samples = 5
        self.num_points = 1000
        
        # 生成随机点云数据
        np.random.seed(42)  # 保证可重复性
        self.test_data = np.random.rand(self.num_samples, self.num_points, 3).astype(np.float32)
        
        # 保存到H5文件
        with h5py.File(self.test_file, 'w') as f:
            f.create_dataset('data', data=self.test_data)
    
    def tearDown(self):
        """
        测试后清理：删除临时文件
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """测试加载器初始化"""
        loader = PointCloudH5Loader(self.test_file)
        
        self.assertEqual(loader.num_samples, self.num_samples)
        self.assertEqual(loader.num_points, self.num_points)
        self.assertEqual(loader.data_shape, (self.num_samples, self.num_points, 3))
    
    def test_invalid_file(self):
        """测试无效文件处理"""
        with self.assertRaises(FileNotFoundError):
            PointCloudH5Loader('nonexistent_file.h5')
    
    def test_invalid_data_key(self):
        """测试无效数据键处理"""
        with self.assertRaises(ValueError):
            PointCloudH5Loader(self.test_file, data_key='nonexistent_key')
    
    def test_load_single_cloud(self):
        """测试单个点云加载"""
        loader = PointCloudH5Loader(self.test_file)
        
        # 加载第一个样本
        point_cloud = loader.load_single_cloud(0)
        
        self.assertEqual(point_cloud.shape, (self.num_points, 3))
        self.assertEqual(point_cloud.dtype, np.float32)
        np.testing.assert_array_equal(point_cloud, self.test_data[0])
    
    def test_load_single_cloud_invalid_index(self):
        """测试无效索引处理"""
        loader = PointCloudH5Loader(self.test_file)
        
        with self.assertRaises(IndexError):
            loader.load_single_cloud(self.num_samples)  # 超出范围
        
        with self.assertRaises(IndexError):
            loader.load_single_cloud(-1)  # 负索引
    
    def test_load_multiple_clouds(self):
        """测试多个点云加载"""
        loader = PointCloudH5Loader(self.test_file)
        
        indices = [0, 2, 4]
        point_clouds = loader.load_multiple_clouds(indices)
        
        self.assertEqual(point_clouds.shape, (len(indices), self.num_points, 3))
        np.testing.assert_array_equal(point_clouds, self.test_data[indices])
    
    def test_load_all_clouds(self):
        """测试全部点云加载"""
        loader = PointCloudH5Loader(self.test_file)
        
        all_clouds = loader.load_all_clouds()
        
        self.assertEqual(all_clouds.shape, self.test_data.shape)
        np.testing.assert_array_equal(all_clouds, self.test_data)
    
    def test_get_point_cloud_bounds(self):
        """测试边界计算"""
        loader = PointCloudH5Loader(self.test_file)
        
        min_bounds, max_bounds = loader.get_point_cloud_bounds()
        
        expected_min = np.min(self.test_data, axis=(0, 1))
        expected_max = np.max(self.test_data, axis=(0, 1))
        
        np.testing.assert_array_almost_equal(min_bounds, expected_min)
        np.testing.assert_array_almost_equal(max_bounds, expected_max)
    
    def test_batch_iterator(self):
        """测试批次迭代器"""
        loader = PointCloudH5Loader(self.test_file)
        batch_size = 2
        
        iterator = loader.get_batch_iterator(batch_size)
        
        batches = list(iterator)
        
        # 验证批次数量
        expected_num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.assertEqual(len(batches), expected_num_batches)
        
        # 验证批次大小
        for i, batch in enumerate(batches):
            if i < len(batches) - 1:  # 不是最后一批
                self.assertEqual(batch.shape[0], batch_size)
            else:  # 最后一批可能较小
                expected_last_batch_size = self.num_samples - i * batch_size
                self.assertEqual(batch.shape[0], expected_last_batch_size)


class TestBatchIterator(unittest.TestCase):
    """批次迭代器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_batch.h5')
        
        self.num_samples = 10
        self.num_points = 500
        
        np.random.seed(42)
        test_data = np.random.rand(self.num_samples, self.num_points, 3).astype(np.float32)
        
        with h5py.File(self.test_file, 'w') as f:
            f.create_dataset('data', data=test_data)
        
        self.loader = PointCloudH5Loader(self.test_file)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_iterator_length(self):
        """测试迭代器长度计算"""
        batch_size = 3
        iterator = self.loader.get_batch_iterator(batch_size)
        
        expected_length = (self.num_samples + batch_size - 1) // batch_size
        self.assertEqual(len(iterator), expected_length)
    
    def test_iterator_without_shuffle(self):
        """测试无打乱的迭代"""
        batch_size = 4
        iterator = self.loader.get_batch_iterator(batch_size, shuffle=False)
        
        batches = list(iterator)
        
        # 重新组合所有批次
        reconstructed = np.concatenate(batches, axis=0)
        original = self.loader.load_all_clouds()
        
        # 验证数据完整性（顺序应该保持）
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_iterator_with_shuffle(self):
        """测试打乱的迭代"""
        batch_size = 3
        iterator = self.loader.get_batch_iterator(batch_size, shuffle=True)
        
        batches = list(iterator)
        
        # 验证总样本数保持不变
        total_samples = sum(batch.shape[0] for batch in batches)
        self.assertEqual(total_samples, self.num_samples)


if __name__ == '__main__':
    unittest.main()
