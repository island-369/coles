#!/usr/bin/env python3
"""
多卡训练测试脚本
用于验证分布式训练是否正常工作

使用方法:
1. 单卡测试: python test_multi_gpu.py
2. 多卡测试: torchrun --nproc_per_node=2 test_multi_gpu.py
3. 多机多卡测试: 
   # 主节点
   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=主节点IP --master_port=29500 test_multi_gpu.py
   # 从节点
   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=主节点IP --master_port=29500 test_multi_gpu.py
"""

import os
import torch
import torch.distributed as dist
from datetime import datetime

def print_distributed_info():
    """打印分布式训练信息"""
    print(f"\n=== 分布式训练信息 [{datetime.now().strftime('%H:%M:%S')}] ===")
    
    # 基本信息
    print(f"进程ID: {os.getpid()}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    # 环境变量
    env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # 分布式状态
    if torch.distributed.is_available():
        print(f"分布式后端可用: True")
        if torch.distributed.is_initialized():
            print(f"分布式已初始化: True")
            print(f"当前进程rank: {torch.distributed.get_rank()}")
            print(f"总进程数world_size: {torch.distributed.get_world_size()}")
            print(f"后端: {torch.distributed.get_backend()}")
        else:
            print(f"分布式已初始化: False")
    else:
        print(f"分布式后端可用: False")
    
    # GPU信息
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前CUDA设备: {current_device}")
        print(f"设备名称: {torch.cuda.get_device_name(current_device)}")
        print(f"设备内存: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB")
    
    print(f"=== 分布式训练信息结束 ===\n")

def test_data_distribution():
    """测试数据是否在不同进程间正确分布"""
    if not torch.distributed.is_initialized():
        print("分布式未初始化，跳过数据分布测试")
        return
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    print(f"\n=== 数据分布测试 (Rank {rank}) ===")
    
    # 模拟数据集
    total_data_size = 1000
    data_per_rank = total_data_size // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank
    
    print(f"总数据量: {total_data_size}")
    print(f"每个进程数据量: {data_per_rank}")
    print(f"当前进程数据范围: [{start_idx}, {end_idx})")
    
    # 创建测试张量
    local_data = torch.arange(start_idx, end_idx, dtype=torch.float32)
    if torch.cuda.is_available():
        local_data = local_data.cuda()
    
    print(f"本地数据前5个: {local_data[:5].tolist()}")
    print(f"本地数据后5个: {local_data[-5:].tolist()}")
    
    # 收集所有进程的数据来验证
    gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_data, local_data)
    
    if rank == 0:
        print(f"\n=== 所有进程数据汇总 (仅Rank 0显示) ===")
        for i, data in enumerate(gathered_data):
            print(f"Rank {i} 数据范围: [{data[0].item():.0f}, {data[-1].item():.0f}]")
        
        # 检查是否有重叠
        all_data = torch.cat(gathered_data)
        unique_data = torch.unique(all_data)
        print(f"总数据点: {len(all_data)}, 唯一数据点: {len(unique_data)}")
        if len(all_data) == len(unique_data):
            print("✅ 数据分布正确，无重叠")
        else:
            print("❌ 数据分布有问题，存在重叠")
    
    print(f"=== 数据分布测试结束 ===\n")

def main():
    """主函数"""
    
    # 初始化分布式训练
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    
    # 设置当前进程使用的GPU设备
    if torch.distributed.is_initialized() and torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"进程 {torch.distributed.get_rank()} 设置使用GPU {local_rank}")
    
    print(f"开始多卡训练测试 - {datetime.now()}")
    
    # 打印分布式信息
    print_distributed_info()
    
    # 如果是分布式环境，测试数据分布
    if torch.distributed.is_initialized():
        test_data_distribution()
    
    # 简单的计算测试
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        x = torch.randn(1000, 1000, device=device)
        y = torch.mm(x, x.t())
        print(f"GPU计算测试完成，结果形状: {y.shape}")
    
    print(f"测试完成 - {datetime.now()}")

if __name__ == "__main__":
    main()