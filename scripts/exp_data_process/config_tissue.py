"""
配置文件 for mt
"""

# ==============================================================================
# 数据输入配置
# ==============================================================================

# CSV文件所在的根目录（会递归搜索所有子目录）
INPUT_DIR = "/repos/datasets/tissue-datasets/pointclouds-clean-check"

# CSV文件表头格式（默认）
CSV_COLUMNS = ["x [nm]", "y [nm]", "z [nm]"]

# 是否递归查找符合要求的文件
RECURSIVE = False

# ==============================================================================
# 样本生成配置
# ==============================================================================

# 需要生成的样本数量
NUM_SAMPLES = 1024

# 提取区域的大小（单位：nm）
# 格式: (x_size, y_size, z_size)
REGION_SIZE = (28000.0, 28000.0, 1600.0)

# 期望的点数（每个样本的目标点数）
# 如果实际点数不足，会随机复制点；如果过多，会随机删除点
TARGET_POINTS = 100000

# 每个样本的最小点数要求
# 如果提取的区域点数少于此值，会自动重试
MIN_POINTS = 1000

# 当点数不足 TARGET_POINTS 时的处理方式
# 可选: "duplicate" (随机复制), "interpolate" (插值)
UPSAMPLE_METHOD = "interpolate"

# 插值时使用的邻居数量（仅当 UPSAMPLE_METHOD="interpolate" 时有效）
UPSAMPLE_K = 3

# 每个CSV文件的最大尝试次数
# 如果连续尝试都失败，则放弃该CSV文件
MAX_ATTEMPTS = 20

# 坐标精度（保留的小数位数）
DECIMALS = 1

# 随机种子（用于结果可重复性）
# 设置为 None 则每次运行结果不同
RANDOM_SEED = 42

# ==============================================================================
# 输出配置
# ==============================================================================

# 输出H5文件路径
OUTPUT_H5 = "/repos/datasets/tissue-datasets/pointclouds-clean-check.h5"

# H5文件中数据集的名称
DATASET_NAME = "pointclouds"

# H5文件压缩方法
# 可选: "gzip", "lzf", None
COMPRESSION = "gzip"

# ==============================================================================
# 测试输出配置（用于测试脚本）
# ==============================================================================

# 测试输出目录
TEST_OUTPUT_DIR = "/repos/datasets/tissue-datasets/pointclouds-clean-check-test"

# 测试样本数量（建议使用较小的值）
TEST_NUM_SAMPLES = 20

# 可视化参数
VISUALIZATION = {
    "figsize": (8, 8),
    "dpi": 150,
    "cmap": "viridis",
    "point_size": 1.0,
    "show_colorbar": True,
    "show_stats": True
}

# ==============================================================================
# 数据增强配置（高级）
# ==============================================================================

# 是否启用随机旋转（绕Z轴）
ENABLE_ROTATION = True

# 旋转角度范围（弧度）
# None 表示 [0, 2π] 的均匀分布
ROTATION_ANGLE_RANGE = None

# 是否启用沿Z轴翻转
ENABLE_Z_FLIP = True

# Z轴翻转概率（0.0-1.0）
# 0.0 表示不翻转，1.0 表示总是翻转，0.5 表示50%概率翻转
Z_FLIP_PROBABILITY = 0.5

# 是否对区域提取位置进行约束
# 例如: 只从点云中心区域提取
USE_CENTER_BIAS = True

# 中心偏移系数（0.0-1.0）
# 0.0 表示完全随机，1.0 表示只从中心提取
CENTER_BIAS_FACTOR = 1.0

# 是否忽略区域尺寸不足的情况
# False: 当点云尺寸小于指定区域大小时，跳过该文件
# True: 当点云尺寸小于指定区域大小时，使用点云的实际尺寸
IGNORE_INSUFFICIENT_REGION_SIZE = True

# ==============================================================================
# 日志配置
# ==============================================================================

# 日志级别
# 可选: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# 是否将日志保存到文件
SAVE_LOG_TO_FILE = True

# 日志文件路径
LOG_FILE = "/repos/datasets/tissue-datasets/pointclouds-clean-check-log.log"
