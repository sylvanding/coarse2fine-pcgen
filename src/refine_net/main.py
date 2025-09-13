try:
    import open3d
except:
    pass
import torch
import src.refine_net.options as options
import src.refine_net.util as util
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.refine_net.losses import chamfer_distance
from src.refine_net.data_handler import get_dataset, H5InferenceDataset
from torch.utils.data import DataLoader
from src.refine_net.models import PointNet2Generator
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import glob
import re


def find_checkpoint_file(save_path, resume_from):
    """
    根据resume_from参数查找对应的检查点文件
    
    Args:
        save_path: 保存目录路径
        resume_from: 恢复参数，支持latest、best、final、iteration_XXXXXX或文件路径
    
    Returns:
        str: 检查点文件路径
    """
    if os.path.isfile(resume_from):
        # 直接指定了文件路径
        return resume_from
    
    checkpoint_dir = save_path / 'generators'
    
    if resume_from == 'latest':
        # 查找最新的检查点
        pattern = str(checkpoint_dir / 'iteration_*.pt')
        checkpoint_files = glob.glob(pattern)
        if not checkpoint_files:
            raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到任何检查点文件")
        
        # 按iteration数排序，取最新的
        iterations = []
        for f in checkpoint_files:
            match = re.search(r'iteration_(\d+)\.pt', f)
            if match:
                iterations.append((int(match.group(1)), f))
        
        if not iterations:
            raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到有效的检查点文件")
        
        iterations.sort(key=lambda x: x[0])
        return iterations[-1][1]
    
    elif resume_from == 'best':
        # 查找最佳检查点
        best_file = checkpoint_dir / 'best.pt'
        if best_file.exists():
            return str(best_file)
        else:
            raise FileNotFoundError(f"未找到最佳检查点文件: {best_file}")
    
    elif resume_from == 'final':
        # 查找最终检查点
        final_file = checkpoint_dir / 'final.pt'
        if final_file.exists():
            return str(final_file)
        else:
            raise FileNotFoundError(f"未找到最终检查点文件: {final_file}")
    
    elif resume_from.startswith('iteration_'):
        # 指定iteration的检查点
        checkpoint_file = checkpoint_dir / f'{resume_from}.pt'
        if checkpoint_file.exists():
            return str(checkpoint_file)
        else:
            raise FileNotFoundError(f"未找到指定的检查点文件: {checkpoint_file}")
    
    else:
        raise ValueError(f"不支持的resume_from参数: {resume_from}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    加载检查点文件
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型对象
        optimizer: 优化器对象（可选）
        device: 设备
    
    Returns:
        dict: 包含iteration、loss等信息的字典
    """
    print(f"正在加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型状态已加载")
    
    # 加载优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"优化器状态已加载")
    
    # 提取训练信息
    start_iteration = checkpoint.get('iteration', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    print(f"检查点信息:")
    print(f"  - 训练迭代数: {start_iteration}")
    print(f"  - 最佳损失: {best_loss:.6f}")
    
    if 'timestamp' in checkpoint:
        print(f"  - 保存时间: {checkpoint['timestamp']}")
    
    return {
        'start_iteration': start_iteration,
        'best_loss': best_loss,
        'checkpoint_info': checkpoint
    }


def validate(model, val_dataset, device, args, iteration, writer=None):
    """
    验证函数，评估模型在验证集上的表现
    
    Args:
        model: 训练好的模型
        val_dataset: 验证数据集
        device: 计算设备
        args: 参数配置
        iteration: 当前训练迭代次数
        writer: TensorBoard writer
    """
    model.eval()
    val_losses = []
    val_results = []
    
    print(f"\n=== 开始第 {iteration} 次迭代验证 ===")
    
    with torch.no_grad():
        for i in range(len(val_dataset.sample_indices)):
            batch_data = val_dataset[i]
            
            # 处理数据（支持体素指导）
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                # 使用体素指导: (gt_pc, noisy_pc, voxel_grid)
                gt_pc, noisy_pc, voxel_grid = batch_data
                gt_pc, noisy_pc, voxel_grid = gt_pc.to(device), noisy_pc.to(device), voxel_grid.to(device)
                
                # 添加batch维度
                noisy_batch = noisy_pc.unsqueeze(0)
                gt_batch = gt_pc.unsqueeze(0)
                voxel_batch = voxel_grid.unsqueeze(0)
                
                # 模型预测
                refined_pc = model(noisy_batch, voxel_batch)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                # 不使用体素指导: (gt_pc, noisy_pc)
                gt_pc, noisy_pc = batch_data
                gt_pc, noisy_pc = gt_pc.to(device), noisy_pc.to(device)
                
                # 添加batch维度
                noisy_batch = noisy_pc.unsqueeze(0)
                gt_batch = gt_pc.unsqueeze(0)
                
                # 模型预测
                refined_pc = model(noisy_batch)
            else:
                raise ValueError(f"不支持的验证数据格式")
            
            # 计算损失
            val_loss = chamfer_distance(refined_pc, gt_batch, mse=args.mse)
            val_losses.append(val_loss.item())
            
            # 保存验证结果到CSV
            if args.val_save_csv:
                save_validation_to_csv(
                    gt_pc.cpu().numpy(),
                    noisy_pc.cpu().numpy(), 
                    refined_pc[0].cpu().numpy(),
                    args.save_path,
                    iteration,
                    i
                )
            
            # 记录结果
            val_results.append({
                'iteration': iteration,
                'sample': i,
                'loss': val_loss.item()
            })
    
    avg_val_loss = np.mean(val_losses)
    print(f"验证平均损失: {avg_val_loss:.6f}")
    
    # TensorBoard记录
    if writer:
        writer.add_scalar('Loss/Validation', avg_val_loss, iteration)
    
    print(f"=== 第 {iteration} 次迭代验证完成 ===\n")
    
    # 保存验证日志
    save_validation_log(val_results, args.save_path, iteration)
    
    return avg_val_loss


def save_validation_to_csv(gt_pc, noisy_pc, refined_pc, save_path, iteration, sample_idx):
    """
    保存验证结果点云到CSV文件
    
    Args:
        gt_pc: GT点云 (3, N)
        noisy_pc: 噪声点云 (3, N)  
        refined_pc: 修正后点云 (3, N)
        save_path: 保存路径
        iteration: 训练迭代次数
        sample_idx: 样本索引
    """
    val_dir = save_path / 'validation_results'
    val_dir.mkdir(exist_ok=True)
    
    iteration_dir = val_dir / f'iteration_{iteration:06d}'
    iteration_dir.mkdir(exist_ok=True)
    
    # 转换为 (N, 3) 格式
    gt_points = gt_pc.T  # (N, 3)
    noisy_points = noisy_pc.T
    refined_points = refined_pc.T
    
    # 保存到CSV
    pd.DataFrame(gt_points, columns=['x', 'y', 'z']).to_csv(
        iteration_dir / f'gt_sample_{sample_idx:03d}.csv', index=False)
    pd.DataFrame(noisy_points, columns=['x', 'y', 'z']).to_csv(
        iteration_dir / f'noisy_sample_{sample_idx:03d}.csv', index=False)
    pd.DataFrame(refined_points, columns=['x', 'y', 'z']).to_csv(
        iteration_dir / f'refined_sample_{sample_idx:03d}.csv', index=False)


def save_validation_log(val_results, save_path, iteration):
    """保存验证日志"""
    log_path = save_path / 'validation_log.csv'
    
    df = pd.DataFrame(val_results)
    
    # 如果文件存在，则追加；否则创建新文件
    if log_path.exists():
        existing_df = pd.read_csv(log_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(log_path, index=False)


def inference_iteration(model, inference_dataset, device, args, iteration, writer=None):
    """
    推理函数，对验证集前4个样本进行完整推理
    
    Args:
        model: 训练好的模型
        inference_dataset: 推理数据集
        device: 计算设备
        args: 参数配置
        iteration: 当前训练迭代次数
        writer: TensorBoard writer
    """
    model.eval()
    
    print(f"\n=== 开始第 {iteration} 次迭代推理 ===")
    
    inference_dir = args.save_path / 'inference_results' / f'iteration_{iteration:06d}'
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i in range(len(inference_dataset)):
            data = inference_dataset[i]
            
            # 获取数据
            gt_normalized = data['gt_normalized'].to(device)  # (3, N)
            noisy_normalized = data['noisy_normalized'].to(device)  # (3, N)
            gt_original = data['gt_original']  # (N, 3)
            noisy_original = data['noisy_original']  # (N, 3)
            sample_idx = data['sample_idx']
            num_points = data['num_points']
            
            # 体素数据（如果启用体素指导）
            voxel_grid = data.get('voxel_grid', None)
            if voxel_grid is not None:
                voxel_grid = voxel_grid.to(device)
            
            use_voxel_grid_str = "是" if voxel_grid is not None else "否"
            print(f"处理样本 {sample_idx}, 点数: {num_points}, 使用体素指导: {use_voxel_grid_str}")
            
            # 分批处理大规模点云
            if num_points > args.sample_points:
                refined_normalized = process_large_pointcloud(
                    model, noisy_normalized, args.sample_points, device, voxel_grid
                )
            else:
                # 直接处理
                if model.use_voxel_guidance and voxel_grid is not None:
                    refined_normalized = model(noisy_normalized.unsqueeze(0), voxel_grid.unsqueeze(0))[0]  # (3, N)
                else:
                    refined_normalized = model(noisy_normalized.unsqueeze(0))[0]  # (3, N)
            
            # 反归一化: (3, N) -> (N, 3)
            refined_normalized_np = refined_normalized.transpose(0, 1).cpu().numpy()
            refined_original = inference_dataset.denormalize_pointcloud(refined_normalized_np)
            
            # 保存结果到CSV
            pd.DataFrame(gt_original, columns=['x', 'y', 'z']).to_csv(
                inference_dir / f'gt_sample_{sample_idx:03d}.csv', index=False)
            pd.DataFrame(noisy_original, columns=['x', 'y', 'z']).to_csv(
                inference_dir / f'noisy_sample_{sample_idx:03d}.csv', index=False)
            pd.DataFrame(refined_original, columns=['x', 'y', 'z']).to_csv(
                inference_dir / f'refined_sample_{sample_idx:03d}.csv', index=False)
            
            print(f"样本 {sample_idx} 推理完成")
    
    print(f"=== 第 {iteration} 次迭代推理完成 ===\n")


def process_large_pointcloud(model, noisy_pc, batch_size, device, voxel_grid=None):
    """
    分批处理大规模点云
    
    Args:
        model: 模型
        noisy_pc: 噪声点云 (3, N)
        batch_size: 批处理大小
        device: 设备
        voxel_grid: 体素网格 (C, D, H, W)，可选
        
    Returns:
        torch.Tensor: 修正后的点云 (3, N)
    """
    num_points = noisy_pc.shape[1]
    refined_batches = []
    
    for start_idx in range(0, num_points, batch_size):
        end_idx = min(start_idx + batch_size, num_points)
        
        # 获取批次
        batch = noisy_pc[:, start_idx:end_idx]  # (3, batch_size)
        
        # 如果批次太小，进行padding
        if batch.shape[1] < batch_size:
            padding_size = batch_size - batch.shape[1]
            # 重复最后几个点进行padding
            repeat_indices = torch.randint(0, batch.shape[1], (padding_size,))
            padding = batch[:, repeat_indices]
            batch = torch.cat([batch, padding], dim=1)
        
        # 模型推理
        if model.use_voxel_guidance and voxel_grid is not None:
            refined_batch = model(batch.unsqueeze(0), voxel_grid.unsqueeze(0))[0]  # (3, batch_size)
        else:
            refined_batch = model(batch.unsqueeze(0))[0]  # (3, batch_size)
        
        # 移除padding
        if end_idx - start_idx < batch_size:
            refined_batch = refined_batch[:, :end_idx - start_idx]
        
        refined_batches.append(refined_batch)
    
    # 拼接所有批次
    refined_pc = torch.cat(refined_batches, dim=1)
    return refined_pc


def train(args):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    print(f'device: {device}')

    # 创建训练和验证数据集
    dummy_pc = torch.zeros((100, 3))  # 占位符
    train_dataset = get_dataset(args.sampling_mode)(dummy_pc, device, args, split='train')
    val_dataset = get_dataset(args.sampling_mode)(dummy_pc, device, args, split='val')
    inference_dataset = H5InferenceDataset(dummy_pc, device, args)
    
    print(f"使用数据集模式: {args.sampling_mode}")
    print(f"GT H5文件: {args.gt_h5_path}")
    print(f"噪声H5文件: {args.noisy_h5_path}")
    print(f"采样点数: {args.sample_points}")

    # 初始化模型
    model = PointNet2Generator(device, args)
    print(f'模型参数数量: {util.n_params(model)}')
    model.initialize_params(args.init_var)
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard
    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.save_path / 'tensorboard')
        print(f"TensorBoard日志目录: {args.save_path / 'tensorboard'}")
    
    # 检查点恢复
    start_iteration = 0
    best_loss = float('inf')
    
    if args.resume_from:
        try:
            # 查找检查点文件
            checkpoint_path = find_checkpoint_file(args.save_path, args.resume_from)
            
            # 加载检查点
            resume_optimizer = args.resume_optimizer if hasattr(args, 'resume_optimizer') else True
            checkpoint_info = load_checkpoint(
                checkpoint_path, 
                model, 
                optimizer if resume_optimizer else None, 
                device
            )
            
            # 设置起始迭代
            if args.resume_start_iteration > 0:
                start_iteration = args.resume_start_iteration
                print(f"强制设置起始iteration为: {start_iteration}")
            else:
                start_iteration = checkpoint_info['start_iteration']
            
            best_loss = checkpoint_info['best_loss']
            
            print(f"✅ 成功从检查点恢复训练")
            print(f"   - 起始iteration: {start_iteration}")
            print(f"   - 历史最佳损失: {best_loss:.6f}")
            
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            print(f"   将从头开始训练")
            start_iteration = 0
            best_loss = float('inf')
    
    # 训练数据加载器
    train_loader = DataLoader(train_dataset, num_workers=0,
                          batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # 训练循环
    model.train()
    total_iterations = int(len(train_dataset) / args.batch_size)
    global_step = start_iteration
    
    print(f"开始训练，总迭代次数: {total_iterations}")
    if start_iteration > 0:
        print(f"从第 {start_iteration} 次迭代恢复训练")
    
    start_time = time.time()
    current_best_loss = best_loss
    
    for i, batch_data in enumerate(tqdm(train_loader, desc="训练进度")):
        # 调整实际迭代计数
        actual_iteration = i + start_iteration
        
        # # 如果是恢复训练且当前迭代小于起始迭代，跳过
        # if i < start_iteration and start_iteration > 0:
        #     continue
        
        # 处理数据批次（支持体素指导）
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            # 使用体素指导: (gt_pc, noisy_pc, voxel_grid)
            gt_pc, noisy_pc, voxel_grid = batch_data
            gt_pc, noisy_pc, voxel_grid = gt_pc.to(device), noisy_pc.to(device), voxel_grid.to(device)
        elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            # 不使用体素指导: (gt_pc, noisy_pc)
            gt_pc, noisy_pc = batch_data
            gt_pc, noisy_pc = gt_pc.to(device), noisy_pc.to(device)
            voxel_grid = None
        else:
            raise ValueError(f"不支持的批次数据格式: {type(batch_data)}, 长度: {len(batch_data) if hasattr(batch_data, '__len__') else 'N/A'}")
        
        model.train()
        optimizer.zero_grad()
        
        # 前向传播：在噪声点云上预测修正
        if model.use_voxel_guidance and voxel_grid is not None:
            refined_pc = model(noisy_pc, voxel_grid)
        else:
            refined_pc = model(noisy_pc)
        
        # 计算损失：修正后点云与GT点云的差异
        loss = chamfer_distance(refined_pc, gt_pc, mse=args.mse)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        global_step = actual_iteration

        # TensorBoard记录训练损失
        if writer and global_step % 10 == 0:
            writer.add_scalar('Loss/Train', loss.item(), global_step)
        
        # 打印训练进度
        if actual_iteration % 1 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iteration: {actual_iteration}; Total: {actual_iteration} / {total_iterations + start_iteration}; '
                  f'Loss: {util.show_truncated(loss.item(), 6)}; '
                  f'Time: {elapsed_time:.2f}s')
        
        # 验证
        if i > 0 and i % args.val_interval == 0:
            val_loss = validate(model, val_dataset, device, args, actual_iteration, writer)
            
            # 推理
            if i >= args.inference_start_iteration and i % args.inference_interval == 0:
                inference_iteration(model, inference_dataset, device, args, actual_iteration, writer)
        
        # 保存检查点
        if actual_iteration > 0 and actual_iteration % args.checkpoint_interval == 0:
            checkpoint_path = args.save_path / f'checkpoints/iteration_{actual_iteration:06d}.pt'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'iteration': actual_iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_loss': current_best_loss,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    # 训练结束，保存最终模型
    final_model_path = args.save_path / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存: {final_model_path}")
    
    # 保存完整检查点
    final_checkpoint_path = args.save_path / 'final_checkpoint.pt'
    torch.save({
        'iteration': total_iterations + start_iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': current_best_loss,
        'args': vars(args),
        'total_iterations': total_iterations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, final_checkpoint_path)
    print(f"🏁 最终检查点已保存: {final_checkpoint_path}")
    
    # 关闭TensorBoard
    if writer:
        writer.close()
    
    print("训练完成！")


if __name__ == "__main__":
    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)
    train(args)
