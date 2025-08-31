#!/bin/bash

# RefineNet推理脚本
# 用于使用训练好的模型进行点云修正推理

set -e  # 遇到错误时退出

# 默认配置参数
MODEL_PATH=""
GT_H5_PATH="/repos/datasets/batch_simulation_mitochondria.h5"
NOISY_H5_PATH="/repos/datasets/batch_simulation_mitochondria_noised.h5"
OUTPUT_DIR="/repos/coarse2fine-pcgen/outputs/refine_net_inference_$(date +%Y%m%d_%H%M%S)"
SAMPLE_IDX=-1  # -1表示处理所有样本

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gt-h5-path)
            GT_H5_PATH="$2"
            shift 2
            ;;
        --noisy-h5-path)
            NOISY_H5_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sample-idx)
            SAMPLE_IDX="$2"
            shift 2
            ;;
        -h|--help)
            echo "RefineNet推理脚本"
            echo ""
            echo "用法: $0 --model-path MODEL_PATH [选项]"
            echo ""
            echo "必需参数:"
            echo "  --model-path PATH     训练好的模型路径"
            echo ""
            echo "可选参数:"
            echo "  --gt-h5-path PATH     GT点云H5文件路径 (默认: $GT_H5_PATH)"
            echo "  --noisy-h5-path PATH  噪声点云H5文件路径 (默认: $NOISY_H5_PATH)"
            echo "  --output-dir DIR      输出目录 (默认: 自动生成)"
            echo "  --sample-idx IDX      要处理的样本索引，-1表示所有样本 (默认: -1)"
            echo "  -h, --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --model-path /path/to/model.pt"
            echo "  $0 --model-path /path/to/model.pt --sample-idx 0"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 必须指定模型路径"
    echo "使用 --model-path 指定训练好的模型路径"
    echo "使用 -h 或 --help 查看完整帮助信息"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$GT_H5_PATH" ]; then
    echo "错误: GT H5文件不存在: $GT_H5_PATH"
    exit 1
fi

if [ ! -f "$NOISY_H5_PATH" ]; then
    echo "错误: 噪声H5文件不存在: $NOISY_H5_PATH"
    exit 1
fi

echo "=== RefineNet推理脚本 ==="
echo "模型路径: $MODEL_PATH"
echo "GT数据文件: $GT_H5_PATH"
echo "噪声数据文件: $NOISY_H5_PATH"
echo "输出目录: $OUTPUT_DIR"
if [ "$SAMPLE_IDX" -eq -1 ]; then
    echo "处理样本: 所有推理样本"
else
    echo "处理样本: 索引 $SAMPLE_IDX"
fi
echo "=========================="

# 切换到项目根目录
cd /repos/coarse2fine-pcgen

# 运行推理
python scripts/refine_net/inference_refine_net.py \
    --model-path "$MODEL_PATH" \
    --gt-h5-path "$GT_H5_PATH" \
    --noisy-h5-path "$NOISY_H5_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gt-data-key "point_clouds" \
    --noisy-data-key "point_clouds" \
    --sample-points 8192 \
    --train-ratio 0.8 \
    --sample-idx "$SAMPLE_IDX" \
    --volume-dims 20000 20000 2500 \
    --padding 0 0 100

echo ""
echo "🎉 推理脚本执行完成！"
echo "结果保存到: $OUTPUT_DIR"
echo ""
echo "查看推理结果:"
echo "  ls $OUTPUT_DIR"
echo "  cat $OUTPUT_DIR/inference_summary.csv"
