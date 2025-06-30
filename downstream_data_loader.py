import json
import os
import glob
import pandas as pd
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import itertools
import random
import time
from torch.utils.data import IterableDataset, get_worker_info
import torch


class DownstreamBinaryClassificationDataset(IterableDataset):
    """用于下游二分类任务的流式数据集
    
    数据集结构：
    - 根目录下直接包含多个jsonl文件
    - 每行数据通过label字段标识类别
    - jsonl文件内的数据格式：{"label": 0, "trans": [[...]]}
    
    特点：
    - 支持二分类任务的标签读取
    - 流式处理，内存占用低
    - 支持分布式训练和多进程数据加载
    - 兼容现有的预处理器和数据集构建器
    """
    
    def __init__(self, data_root, preprocessor, dataset_builder, debug_print_func=None, 
                 num_classes=2, shuffle_files=True):
        """
        Args:
            data_root: 数据根目录路径
            preprocessor: 数据预处理器
            dataset_builder: 数据集构建函数，接收处理后的DataFrame，返回Dataset
            debug_print_func: 调试打印函数
            num_classes: 类别数量，默认为2（二分类）
            shuffle_files: 是否随机打乱文件顺序
        """
        self.data_root = data_root
        self.preprocessor = preprocessor
        self.dataset_builder = dataset_builder
        self.debug_print = debug_print_func if debug_print_func else print
        self.shuffle_files = shuffle_files
        self.num_classes = num_classes
        
        # 定义交易字段名称和顺序（与训练时保持一致）
        self.trx_field_names = [
            '发卡机构地址', '发卡机构银行', '卡等级', 'year', 'month', 'day',
            'hour', 'minutes', 'seconds', 'unix_timestamp', '收单机构地址',
            '收单机构银行', 'cups_交易代码', '交易渠道', 'cups_服务点输入方式',
            'cups_应答码', 'cups_商户类型', 'cups_连接方式', 'cups_受卡方名称地址',
            '交易金额'
        ]
        
        # 扫描数据目录，构建文件列表
        self.file_list = []
        # 扫描data_root下的所有jsonl数据文件
        self._scan_data_directory()    
        
        # 记录数据集信息
        self._log_dataset_info()
    
    def _scan_data_directory(self):
        """扫描数据目录，构建文件列表"""
        if not os.path.exists(self.data_root):
            raise ValueError(f"数据根目录不存在: {self.data_root}")
        
        # 直接扫描根目录下的所有jsonl文件
        for filename in os.listdir(self.data_root):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(self.data_root, filename)
                self.file_list.append(file_path)
        
        if len(self.file_list) == 0:
            raise ValueError(f"在数据目录中未找到任何jsonl文件: {self.data_root}")
        
        # 可选的文件随机打乱
        if self.shuffle_files:
            random.shuffle(self.file_list)
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数，适配二分类任务"""
        from ptls.data_load.utils import collate_feature_dict
        import torch
        
        # 分离样本和标签，并处理ColesDataset返回的splits列表
        all_samples = []
        all_labels = []
        
        for item in batch:
            splits, label = item[0], item[1]
            # ColesDataset返回的是splits列表，需要展平
            if isinstance(splits, list):
                for split in splits:
                    all_samples.append(split)
                    all_labels.append(label)
            else:
                all_samples.append(splits)
                all_labels.append(label)
        
        # 处理特征数据
        padded_batch = collate_feature_dict(all_samples)
        
        # 返回特征和标签
        return padded_batch, torch.LongTensor(all_labels)
    
    @rank_zero_only
    def _log_dataset_info(self):
        """记录数据集基本信息"""
        self.debug_print(f"\n=== DownstreamBinaryClassificationDataset 初始化 ===")
        self.debug_print(f"数据根目录: {self.data_root}")
        self.debug_print(f"类别数量: {self.num_classes}")
        self.debug_print(f"总文件数: {len(self.file_list)}")
        self.debug_print(f"流式处理模式: 用户级逐行读取 + 从label字段获取标签")
        self.debug_print(f"=== 数据集初始化完成 ===\n")
    
    def __iter__(self):
        """迭代器实现，支持分布式训练的流式用户级读取"""
        # 获取分布式训练信息
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # 处理文件数量少于进程数的情况：循环分配文件
        if len(self.file_list) < world_size:
            # 通过循环索引确保每个进程都能分配到文件
            extended_files = []
            for i in range(world_size):
                file_idx = i % len(self.file_list)
                extended_files.append(self.file_list[file_idx])
            file_list = extended_files
        else:
            file_list = self.file_list
        
        # 获取worker信息（用于DataLoader多进程）
        worker_info = get_worker_info()
        if worker_info is not None:
            # 在DataLoader多进程环境中
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # 先按DDP进程分片，再按worker分片
            files_per_rank = file_list[rank::world_size]
            assigned_files = files_per_rank[worker_id::num_workers]
            
            self.debug_print(f"Worker {worker_id}/{num_workers} on rank {rank}/{world_size} processing {len(assigned_files)} files")
        else:
            # 单进程环境，只按DDP分片
            assigned_files = file_list[rank::world_size]
            self.debug_print(f"Rank {rank}/{world_size} processing {len(assigned_files)} files")
        
        # 如果当前进程仍然没有分配到文件，记录警告但不阻塞
        if len(assigned_files) == 0:
            self.debug_print(f"警告: Rank {rank} 没有分配到文件，将跳过训练")
            return
        
        # 使用itertools.cycle无限循环assigned_files，避免分布式训练死锁
        self.debug_print(f"开始无限循环处理文件，避免分布式训练死锁")
        
        # 无限循环处理文件
        for file_idx, file_path in enumerate(itertools.cycle(assigned_files)):
            try:
                # 由于使用无限循环，计算实际文件索引
                actual_file_idx = file_idx % len(assigned_files)
                self.debug_print(f"开始流式处理文件 (循环第{file_idx + 1}次, 文件{actual_file_idx + 1}/{len(assigned_files)}): {os.path.basename(file_path)}")
                
                user_count = 0
                total_samples = 0
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        try:
                            # 解析用户数据
                            user_data = json.loads(line)
                            
                            # 从label字段获取标签
                            if 'label' not in user_data:
                                self.debug_print(f"Warning: Missing label field in line {line_idx}, skipping")
                                continue
                            label_id = int(user_data['label'])
                            
                            if 'trans' not in user_data:
                                continue
                            
                            records = user_data['trans']
                            if not records:
                                continue
                            
                            # 检查并处理列数不匹配的情况
                            processed_records = []
                            for trx_array in records:
                                if isinstance(trx_array, list) and len(trx_array) >= len(self.trx_field_names):
                                    # 只取前20个字段，避免列数不匹配
                                    trx_dict = {}
                                    for i, field_name in enumerate(self.trx_field_names):
                                        trx_dict[field_name] = trx_array[i]
                                    processed_records.append(trx_dict)
                                else:
                                    self.debug_print(f"Warning: Invalid transaction format, expected >= {len(self.trx_field_names)} columns, got {len(trx_array) if isinstance(trx_array, list) else 'non-list'}")
                                    continue
                            
                            if not processed_records:
                                continue
                            
                            # 转换为DataFrame
                            df = pd.DataFrame(processed_records)
                            
                            # 设置client_id（生成全局唯一ID）
                            worker_id = worker_info.id if worker_info is not None else 0
                            if 'user_id' in user_data and user_data['user_id'] is not None:
                                # 如果有原始user_id，添加前缀确保唯一性
                                client_id = f"r{rank}_w{worker_id}_f{actual_file_idx}_{user_data['user_id']}"
                            else:
                                # 生成全局唯一的client_id
                                client_id = f"r{rank}_w{worker_id}_f{actual_file_idx}_l{line_idx}"
                            df['client_id'] = client_id
                            
                            # 将字符串格式的时间字段转换为数值类型
                            time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
                            for col in time_cols:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # 预处理单个用户的数据
                            processed = self.preprocessor.fit_transform(df)
                            
                            # 构建数据集并生成样本
                            dataset = self.dataset_builder(processed)
                            
                            user_sample_count = 0
                            for sample in dataset:
                                # 返回样本和对应的标签
                                yield (sample, label_id)
                                user_sample_count += 1
                            
                            user_count += 1
                            total_samples += user_sample_count
                            
                            # 每处理1000个用户打印一次进度
                            if user_count % 1000 == 0:
                                self.debug_print(f"文件 {os.path.basename(file_path)}: 已处理 {user_count} 个用户，生成 {total_samples} 个样本")
                                
                        except json.JSONDecodeError as e:
                            self.debug_print(f"JSON解析错误，跳过行 {line_idx}: {str(e)}")
                            continue
                        except Exception as e:
                            self.debug_print(f"处理用户数据时出错，跳过行 {line_idx}: {str(e)}")
                            continue
                
                self.debug_print(f"文件 {os.path.basename(file_path)} 处理完成，处理了 {user_count} 个用户，生成 {total_samples} 个样本")
                
            except Exception as e:
                self.debug_print(f"处理文件 {file_path} 时出错: {str(e)}")
                # 继续处理下一个文件，不中断整个训练过程
                continue
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes
    
    def get_file_count(self):
        """获取文件数量"""
        return len(self.file_list)


class DownstreamTestDataset(DownstreamBinaryClassificationDataset):
    """用于下游任务测试的数据集
    
    继承自DownstreamBinaryClassificationDataset，但不使用无限循环，
    适合测试和验证阶段使用。
    """
    
    def __iter__(self):
        """迭代器实现，不使用无限循环，适合测试阶段"""
        # 获取分布式训练信息
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # 获取worker信息（用于DataLoader多进程）
        worker_info = get_worker_info()
        if worker_info is not None:
            # 在DataLoader多进程环境中
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # 先按DDP进程分片，再按worker分片
            files_per_rank = self.file_list[rank::world_size]
            assigned_files = files_per_rank[worker_id::num_workers]
            
            self.debug_print(f"Test Worker {worker_id}/{num_workers} on rank {rank}/{world_size} processing {len(assigned_files)} files")
        else:
            # 单进程环境，只按DDP分片
            assigned_files = self.file_list[rank::world_size]
            self.debug_print(f"Test Rank {rank}/{world_size} processing {len(assigned_files)} files")
        
        # 如果当前进程没有分配到文件，直接返回
        if len(assigned_files) == 0:
            self.debug_print(f"警告: Test Rank {rank} 没有分配到文件")
            return
        
        # 处理分配到的文件（只遍历一次，不循环）
        for file_idx, file_path in enumerate(assigned_files):
            try:
                self.debug_print(f"开始测试文件 ({file_idx + 1}/{len(assigned_files)}): {os.path.basename(file_path)}")
                
                user_count = 0
                total_samples = 0
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        try:
                            # 解析用户数据
                            user_data = json.loads(line)
                            if 'trans' not in user_data or 'label' not in user_data:
                                continue
                            
                            # 从每行数据的label字段获取标签
                            label_id = int(user_data['label'])
                            
                            records = user_data['trans']
                            if not records:
                                continue
                            
                            # 检查并处理列数不匹配的情况
                            processed_records = []
                            for trx_array in records:
                                if isinstance(trx_array, list) and len(trx_array) >= len(self.trx_field_names):
                                    # 只取前20个字段，避免列数不匹配
                                    trx_dict = {}
                                    for i, field_name in enumerate(self.trx_field_names):
                                        trx_dict[field_name] = trx_array[i]
                                    processed_records.append(trx_dict)
                                else:
                                    continue
                            
                            if not processed_records:
                                continue
                            
                            # 转换为DataFrame
                            df = pd.DataFrame(processed_records)
                            
                            # 设置client_id
                            worker_id = worker_info.id if worker_info is not None else 0
                            if 'user_id' in user_data and user_data['user_id'] is not None:
                                client_id = f"test_r{rank}_w{worker_id}_f{file_idx}_{user_data['user_id']}"
                            else:
                                client_id = f"test_r{rank}_w{worker_id}_f{file_idx}_l{line_idx}"
                            df['client_id'] = client_id
                            
                            # 将字符串格式的时间字段转换为数值类型
                            time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
                            for col in time_cols:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # 预处理单个用户的数据
                            processed = self.preprocessor.fit_transform(df)
                            
                            # 构建数据集并生成样本
                            dataset = self.dataset_builder(processed)
                            
                            user_sample_count = 0
                            for sample in dataset:
                                # 返回样本和对应的标签
                                yield (sample, label_id)
                                user_sample_count += 1
                            
                            user_count += 1
                            total_samples += user_sample_count
                            
                            # 每处理1000个用户打印一次进度
                            if user_count % 1000 == 0:
                                self.debug_print(f"测试文件 {os.path.basename(file_path)}: 已处理 {user_count} 个用户，生成 {total_samples} 个样本")
                                
                        except json.JSONDecodeError as e:
                            self.debug_print(f"JSON解析错误，跳过行 {line_idx}: {str(e)}")
                            continue
                        except Exception as e:
                            self.debug_print(f"处理用户数据时出错，跳过行 {line_idx}: {str(e)}")
                            continue
                
                self.debug_print(f"测试文件 {os.path.basename(file_path)} 处理完成，处理了 {user_count} 个用户，生成 {total_samples} 个样本")
                
            except Exception as e:
                self.debug_print(f"处理测试文件 {file_path} 时出错: {str(e)}")
                continue