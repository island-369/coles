#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试StreamingUserColesIterableDataset的简单脚本

这个脚本用于验证新的流式用户级数据集是否能正常工作
"""

import os
import sys
import torch
from data_load_xqy import StreamingUserColesIterableDataset
from config import init_config
from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames.coles import ColesDataset
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles.split_strategy import SampleSlices

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

def test_streaming_dataset():
    """测试流式数据集"""
    print("=== 测试StreamingUserColesIterableDataset ===")
    
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
            print(f"找到测试文件: {file_path}")
        else:
            print(f"警告: 测试文件不存在: {file_path}")
    
    if not existing_files:
        print("错误: 没有找到任何测试文件")
        return
    
    # 创建预处理器和数据集构建器
    preprocessor = create_simple_preprocessor()
    
    # 创建流式数据集
    streaming_dataset = StreamingUserColesIterableDataset(
        file_paths=existing_files,
        preprocessor=preprocessor,
        dataset_builder=build_simple_dataset,
        debug_print_func=print
    )
    
    print("\n=== 开始测试数据迭代 ===")
    
    # 测试迭代
    sample_count = 0
    max_samples = 10  # 只测试前10个样本
    
    try:
        for sample in streaming_dataset:
            sample_count += 1
            print(f"样本 {sample_count}: {type(sample)}")
            
            # 打印样本的基本信息
            if hasattr(sample, 'payload'):
                print(f"  - payload keys: {list(sample.payload.keys())}")
                if 'event_time' in sample.payload:
                    print(f"  - sequence length: {len(sample.payload['event_time'])}")
            
            if sample_count >= max_samples:
                print(f"\n测试完成，成功处理了 {sample_count} 个样本")
                break
                
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== 测试结束，总共处理了 {sample_count} 个样本 ===")

if __name__ == "__main__":
    test_streaming_dataset()