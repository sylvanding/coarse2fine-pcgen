#!/bin/bash

# RefineNet训练脚本
# 用于训练点云修正网络

set -e  # 遇到错误时退出

# 配置参数
GT_H5_PATH="/repos/datasets/batch_simulation_mitochondria.h5"
NOISY_H5_PATH="/repos/datasets/batch_simulation_mitochondria_noised.h5"
OUTPUT_DIR="/repos/coarse2fine-pcgen/outputs/refine_net_$(date +%Y%m%d_%H%M%S)"

# 检查数据文件是否存在
if [ ! -f "$GT_H5_PATH" ]; then
    echo "错误: GT H5文件不存在: $GT_H5_PATH"
    exit 1
fi

if [ ! -f "$NOISY_H5_PATH" ]; then
    echo "错误: 噪声H5文件不存在: $NOISY_H5_PATH"
    exit 1
fi

echo "=== RefineNet训练脚本 ==="
echo "GT数据文件: $GT_H5_PATH"
echo "噪声数据文件: $NOISY_H5_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "=========================="

# 切换到项目根目录
cd /repos/coarse2fine-pcgen

# 运行训练
python scripts/refine_net/train_refine_net.py \
    --output-dir "$OUTPUT_DIR" \
    --gt-h5-path "$GT_H5_PATH" \
    --noisy-h5-path "$NOISY_H5_PATH" \
    --gt-data-key "point_clouds" \
    --noisy-data-key "point_clouds" \
    --sample-points 8192 \
    --train-ratio 0.8 \
    --batch-size 2 \
    --lr 0.001 \
    --iterations 10000 \
    --val-interval 10 \
    --val-samples 4 \
    --inference-start-epoch 50 \
    --inference-interval 20 \
    --use-tensorboard \
    --volume-dims 20000 20000 2500 \
    --padding 0 0 100 \
    --export-interval 500

echo ""
echo "🎉 训练脚本执行完成！"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "查看训练结果:"
echo "  最终模型: $OUTPUT_DIR/final_model.pt"
echo "  检查点: $OUTPUT_DIR/checkpoints/"
echo "  验证结果: $OUTPUT_DIR/validation_results/"
echo "  推理结果: $OUTPUT_DIR/inference_results/"
echo ""
echo "启动TensorBoard查看训练过程:"
echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard"
