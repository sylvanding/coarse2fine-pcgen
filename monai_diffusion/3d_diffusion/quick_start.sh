#!/bin/bash
# 条件3D扩散模型快速启动脚本

set -e

echo "======================================"
echo "条件3D扩散模型 - 快速启动"
echo "======================================"
echo ""

# 检查配置文件是否存在
CONFIG_FILE="monai_diffusion/config/conditional_diffusion_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显示菜单
echo "请选择操作:"
echo "1) 训练模型（完整训练）"
echo "2) 快速测试（fast_dev_run）"
echo "3) 生成样本"
echo "4) 查看TensorBoard日志"
echo "5) 退出"
echo ""

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "开始训练条件3D扩散模型..."
        echo "配置文件: $CONFIG_FILE"
        echo ""
        python monai_diffusion/3d_diffusion/train_conditional_diffusion.py \
            --config $CONFIG_FILE
        ;;
    
    2)
        echo ""
        echo "快速测试模式（只运行2个batch）..."
        echo ""
        # 临时创建一个测试配置
        TEST_CONFIG="monai_diffusion/config/conditional_diffusion_config_test.yaml"
        cp $CONFIG_FILE $TEST_CONFIG
        
        # 修改配置为fast_dev_run
        python -c "
import yaml
with open('$TEST_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['diffusion']['training']['fast_dev_run'] = True
config['diffusion']['training']['fast_dev_run_batches'] = 2
config['diffusion']['training']['n_epochs'] = 2
with open('$TEST_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
        
        python monai_diffusion/3d_diffusion/train_conditional_diffusion.py \
            --config $TEST_CONFIG
        
        echo ""
        echo "测试完成！如果没有错误，可以进行完整训练。"
        ;;
    
    3)
        echo ""
        read -p "请输入checkpoint路径: " checkpoint_path
        read -p "请输入条件图像路径: " condition_path
        read -p "请输入输出目录 (默认: outputs/conditional_diffusion/samples/): " output_dir
        
        if [ -z "$output_dir" ]; then
            output_dir="outputs/conditional_diffusion/samples/"
        fi
        
        read -p "生成样本数量 (默认: 1): " num_samples
        if [ -z "$num_samples" ]; then
            num_samples=1
        fi
        
        read -p "推理步数 (默认: 1000): " num_steps
        if [ -z "$num_steps" ]; then
            num_steps=1000
        fi
        
        read -p "使用DDIM加速? (y/n, 默认: y): " use_ddim
        if [ -z "$use_ddim" ] || [ "$use_ddim" = "y" ]; then
            ddim_flag="--use_ddim"
        else
            ddim_flag=""
        fi
        
        echo ""
        echo "开始生成样本..."
        python monai_diffusion/3d_diffusion/generate_conditional_samples.py \
            --config $CONFIG_FILE \
            --checkpoint $checkpoint_path \
            --condition $condition_path \
            --output $output_dir \
            --num_samples $num_samples \
            --num_inference_steps $num_steps \
            $ddim_flag \
            --save_projections
        
        echo ""
        echo "样本生成完成！查看输出目录: $output_dir"
        ;;
    
    4)
        echo ""
        LOG_DIR="outputs/conditional_diffusion/logs"
        
        if [ ! -d "$LOG_DIR" ]; then
            echo "错误: 日志目录不存在: $LOG_DIR"
            echo "请先训练模型以生成日志。"
            exit 1
        fi
        
        echo "启动TensorBoard..."
        echo "访问 http://localhost:6006 查看训练日志"
        echo ""
        tensorboard --logdir $LOG_DIR
        ;;
    
    5)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "操作完成！"
echo "======================================"

