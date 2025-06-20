#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比MultiFileColesIterableDataset和StreamingUserColesIterableDataset的性能和内存使用

这个脚本用于比较两种数据集实现的差异：
1. 内存使用情况
2. 处理速度
3. 数据一致性
"""

import os
import sys
import time
import psutil
import torch
from data_load_xqy import MultiFileColesIterableDataset, StreamingUserColesIterableDataset
from config import init_config
from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames.coles import ColesDataset
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles.split_strategy import SampleSlices

def get_memory_usage():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_simple_preprocessor():
    """创建简单的预处理器"""
    return PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='unix_timestamp',
        event_time_transformation='none',
        cols_category=['发卡机构地址', '发卡机构银行', '卡等级'],
        cols_numerical=['交易金额'],
        return_records=True
    )

def build_simple_dataset(processed_df):
    """构建简单的数据集"""
    return ColesDataset(
        MemoryMapDataset(
            data=processed_df,
            i_filters=[SeqLenFilter(min_seq_len=1)],
        ),
        splitter=SampleSlices(
            split_count=3,
            cnt_min=1,
            cnt_max=3,
        ),
    )

def test_dataset(dataset_class, dataset_name, test_files, max_samples=50):
    """测试指定的数据集类"""
    print(f"\n=== 测试 {dataset_name} ===")
    
    # 记录开始时的内存使用
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # 创建预处理器和数据集
    preprocessor = create_simple_preprocessor()
    
    dataset = dataset_class(
        file_paths=test_files,
        preprocessor=preprocessor,
        dataset_builder=build_simple_dataset,
        debug_print_func=lambda *args, **kwargs: None  # 静默模式
    )
    
    # 记录数据集创建后的内存使用
    dataset_created_memory = get_memory_usage()
    
    print(f"数据集创建完成:")
    print(f"  - 初始内存: {start_memory:.2f} MB")
    print(f"  - 创建后内存: {dataset_created_memory:.2f} MB")
    print(f"  - 内存增长: {dataset_created_memory - start_memory:.2f} MB")
    
    # 测试数据迭代
    sample_count = 0
    max_memory = dataset_created_memory
    
    iteration_start_time = time.time()
    
    try:
        for sample in dataset:
            sample_count += 1
            
            # 每10个样本检查一次内存使用
            if sample_count % 10 == 0:
                current_memory = get_memory_usage()
                max_memory = max(max_memory, current_memory)
            
            if sample_count >= max_samples:
                break
                
    except Exception as e:
        print(f"迭代过程中出现错误: {str(e)}")
        return None
    
    iteration_end_time = time.time()
    final_memory = get_memory_usage()
    
    # 计算统计信息
    total_time = iteration_end_time - start_time
    iteration_time = iteration_end_time - iteration_start_time
    samples_per_second = sample_count / iteration_time if iteration_time > 0 else 0
    
    results = {
        'dataset_name': dataset_name,
        'sample_count': sample_count,
        'total_time': total_time,
        'iteration_time': iteration_time,
        'samples_per_second': samples_per_second,
        'start_memory': start_memory,
        'dataset_created_memory': dataset_created_memory,
        'max_memory': max_memory,
        'final_memory': final_memory,
        'memory_overhead': dataset_created_memory - start_memory,
        'max_memory_usage': max_memory - start_memory,
        'final_memory_usage': final_memory - start_memory
    }
    
    print(f"\n性能统计:")
    print(f"  - 处理样本数: {sample_count}")
    print(f"  - 总耗时: {total_time:.3f} 秒")
    print(f"  - 迭代耗时: {iteration_time:.3f} 秒")
    print(f"  - 处理速度: {samples_per_second:.2f} 样本/秒")
    
    print(f"\n内存统计:")
    print(f"  - 数据集创建开销: {results['memory_overhead']:.2f} MB")
    print(f"  - 最大内存使用: {results['max_memory_usage']:.2f} MB")
    print(f"  - 最终内存使用: {results['final_memory_usage']:.2f} MB")
    
    return results

def compare_datasets():
    """对比两种数据集的性能"""
    print("=== 数据集性能对比测试 ===")
    
    # 设置测试数据路径
    test_files = [
        'test_directory/train/train copy.jsonl',
        'test_directory/train/train copy 2.jsonl'
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file_path in test_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"警告: 测试文件不存在: {file_path}")
    
    if not existing_files:
        print("错误: 没有找到任何测试文件")
        return
    
    print(f"使用测试文件: {[os.path.basename(f) for f in existing_files]}")
    
    max_samples = 100  # 测试样本数量
    
    # 测试文件级数据集
    file_results = test_dataset(
        MultiFileColesIterableDataset,
        "MultiFileColesIterableDataset (文件级读取)",
        existing_files,
        max_samples
    )
    
    # 强制垃圾回收
    import gc
    gc.collect()
    time.sleep(1)
    
    # 测试流式用户级数据集
    streaming_results = test_dataset(
        StreamingUserColesIterableDataset,
        "StreamingUserColesIterableDataset (流式用户级读取)",
        existing_files,
        max_samples
    )
    
    # 对比结果
    if file_results and streaming_results:
        print(f"\n\n=== 对比结果 ===")
        print(f"{'指标':<25} {'文件级读取':<20} {'流式用户级读取':<20} {'差异':<15}")
        print("-" * 80)
        
        # 性能对比
        speed_diff = streaming_results['samples_per_second'] - file_results['samples_per_second']
        speed_pct = (speed_diff / file_results['samples_per_second']) * 100 if file_results['samples_per_second'] > 0 else 0
        print(f"{'处理速度 (样本/秒)':<25} {file_results['samples_per_second']:<20.2f} {streaming_results['samples_per_second']:<20.2f} {speed_pct:+.1f}%")
        
        # 内存对比
        memory_diff = streaming_results['memory_overhead'] - file_results['memory_overhead']
        memory_pct = (memory_diff / file_results['memory_overhead']) * 100 if file_results['memory_overhead'] > 0 else 0
        print(f"{'数据集创建开销 (MB)':<25} {file_results['memory_overhead']:<20.2f} {streaming_results['memory_overhead']:<20.2f} {memory_pct:+.1f}%")
        
        max_memory_diff = streaming_results['max_memory_usage'] - file_results['max_memory_usage']
        max_memory_pct = (max_memory_diff / file_results['max_memory_usage']) * 100 if file_results['max_memory_usage'] > 0 else 0
        print(f"{'最大内存使用 (MB)':<25} {file_results['max_memory_usage']:<20.2f} {streaming_results['max_memory_usage']:<20.2f} {max_memory_pct:+.1f}%")
        
        print(f"\n=== 结论 ===")
        if streaming_results['max_memory_usage'] < file_results['max_memory_usage']:
            print("✓ 流式用户级读取在内存使用方面更优")
        else:
            print("✗ 文件级读取在内存使用方面更优")
            
        if streaming_results['samples_per_second'] > file_results['samples_per_second']:
            print("✓ 流式用户级读取在处理速度方面更优")
        else:
            print("✗ 文件级读取在处理速度方面更优")

if __name__ == "__main__":
    compare_datasets()