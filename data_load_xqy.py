import json
import os
import glob
import json
import pandas as pd
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import ptls
import itertools
import random
import time


from torch.utils.data import IterableDataset, get_worker_info
import torch
import os
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class StreamingUserColesIterableDataset(IterableDataset):
    """流式用户级数据集，逐行读取用户数据并进行预处理
    
    相比MultiFileColesIterableDataset的文件级读取，这个类实现用户级流式读取：
    - 逐行读取jsonl文件，每行是一个用户的数据
    - 将每个用户的交易数据转换为DataFrame
    - 对单个用户数据进行预处理
    - 生成该用户的所有样本后再处理下一个用户
    
    优势：
    - 内存占用更低，不需要一次性加载整个文件
    - 更好的流式处理，适合大文件
    - 保持用户级别的数据完整性
    """
    
    def __init__(self, file_paths, preprocessor, dataset_builder, debug_print_func=None):
        """
        Args:
            file_paths: 训练文件路径列表
            preprocessor: 数据预处理器
            dataset_builder: 数据集构建函数，接收处理后的DataFrame，返回Dataset
            debug_print_func: 调试打印函数
        """
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        self.dataset_builder = dataset_builder
        self.debug_print = debug_print_func if debug_print_func else print
        
        # 定义交易字段名称和顺序
        self.trx_field_names = [
            '发卡机构地址', '发卡机构银行', '卡等级', 'year', 'month', 'day',
            'hour', 'minutes', 'seconds', 'unix_timestamp', '收单机构地址',
            '收单机构银行', 'cups_交易代码', '交易渠道', 'cups_服务点输入方式',
            'cups_应答码', 'cups_商户类型', 'cups_连接方式', 'cups_受卡方名称地址',
            '交易金额'
        ]
        
        # 记录数据集信息
        self._log_dataset_info()
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数，与ColesDataset保持一致"""
        from operator import iadd
        from functools import reduce
        from ptls.data_load.utils import collate_feature_dict
        import torch
        
        class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples]
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, torch.LongTensor(class_labels)
    
    @rank_zero_only
    def _log_dataset_info(self):
        """记录数据集基本信息"""
        self.debug_print(f"\n=== StreamingUserColesIterableDataset 初始化 ===")
        self.debug_print(f"总文件数: {len(self.file_paths)}")
        self.debug_print(f"文件列表: {[os.path.basename(f) for f in self.file_paths[:5]]}{'...' if len(self.file_paths) > 5 else ''}")
        self.debug_print(f"流式处理模式: 用户级逐行读取")
        self.debug_print(f"=== 数据集初始化完成 ===\n")
    
    def __iter__(self):
        """迭代器实现，支持分布式训练的流式用户级读取"""
        # 获取分布式训练信息
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # 处理文件数量少于进程数的情况：循环分配文件
        if len(self.file_paths) < world_size:
            # 通过循环索引确保每个进程都能分配到文件
            file_list = [self.file_paths[i % len(self.file_paths)] for i in range(world_size)]
        else:
            file_list = self.file_paths
        
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
        # 每次进入循环前对文件列表进行随机shuffle，避免过拟合
        # self.debug_print(f"开始无限循环处理文件，避免分布式训练死锁")
        
        # 初始shuffle
        # cycle_count = 0
        # random.seed(int(time.time()) + rank + cycle_count)
        # random.shuffle(assigned_files)
        # self.debug_print(f"第{cycle_count + 1}轮循环开始，已进行初始shuffle")
        
        # 无限循环处理文件
        for file_idx, file_path in enumerate(itertools.cycle(assigned_files)):
            # 每完成一轮所有文件后，重新shuffle文件顺序
            # if file_idx > 0 and file_idx % len(assigned_files) == 0:
            #     cycle_count += 1
            #     # 使用时间戳和循环次数作为随机种子，确保每次shuffle都不同
            #     random.seed(int(time.time()) + rank + cycle_count)
            #     random.shuffle(assigned_files)
            #     self.debug_print(f"第{cycle_count + 1}轮循环开始，已重新shuffle文件顺序")
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
                            if 'trans' not in user_data:
                                continue
                            
                            records = user_data['trans']
                            if not records:
                                continue
                            
                            # 检查并处理列数不匹配的情况，采用之前的处理逻辑
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
                            
                            # 设置client_id（使用原始user_id或行号）
                            # client_id = user_data.get('user_id', f"{file_idx}_{line_idx}")
                            
                            # 设置client_id（使用原始user_id或生成全局唯一ID）
                            # 为避免client_id冲突，使用rank、worker_id、file_idx、line_idx组合生成唯一ID
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
                                yield sample
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
                
                self.debug_print(f"文件 {os.path.basename(file_path)} 流式处理完成，处理了 {user_count} 个用户，生成 {total_samples} 个样本")
                
            except Exception as e:
                self.debug_print(f"处理文件 {file_path} 时出错: {str(e)}")
                # 继续处理下一个文件，不中断整个训练过程
                continue


class MultiFileIterableDataset(IterableDataset):
    """保持向后兼容的原始实现"""
    def __init__(self, file_paths, preprocessor, dataset_builder):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        self.dataset_builder = dataset_builder

    def __iter__(self):
        # 获取当前DDP进程rank和总数
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # 分片文件列表
        assigned_files = self.file_paths[rank::world_size]

        for path in assigned_files:
            df = load_jsonl_as_dataframe_new_format(path)
            processed = self.preprocessor.fit_transform(df)
            dataset = self.dataset_builder(processed)
            for sample in dataset:
                yield sample


@rank_zero_only
def print_ptls_info():
    print(f"\n=== PTLS库信息 ===")
    print(f"PTLS路径: {ptls.__file__}")
    print(f"=== PTLS库信息结束 ===\n")


def load_jsonl_as_dataframe_new_format(jsonl_path):
    """从单个jsonl文件加载新格式数据为DataFrame
    新格式：每行是一个字典，字典中的'trans'键存储用户交易序列
    每个交易序列中的元素是列表，按固定顺序存储字段值
    """
    # 定义字段名称和顺序
    field_names = [
        '发卡机构地址', '发卡机构银行', '卡等级', 'year', 'month', 'day', 
        'hour', 'minutes', 'seconds', 'unix_timestamp', '收单机构地址', 
        '收单机构银行', 'cups_交易代码', '交易渠道', 'cups_服务点输入方式', 
        'cups_应答码', 'cups_商户类型', 'cups_连接方式', 'cups_受卡方名称地址', 
        '交易金额', 
    ]
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            user_data = json.loads(line)  # 每行是一个字典
            # 从字典中提取'trans'键对应的交易序列
            if 'trans' in user_data:
                user_trx = user_data['trans']
                # 将列表格式转换为字典格式
                converted_trx_list = []
                for trx_array in user_trx:
                    if isinstance(trx_array, list) and len(trx_array) >= len(field_names):
                        # 将列表转换为字典
                        trx_dict = {}
                        for i, field_name in enumerate(field_names):
                            trx_dict[field_name] = trx_array[i]
                        converted_trx_list.append(trx_dict)
                    else:
                        print(f"Warning: Invalid transaction format in line: {line.strip()}")
                        continue
                data.append(converted_trx_list)
            else:
                # 如果没有'trans'键，跳过这行或者抛出警告
                print(f"Warning: 'trans' key not found in line: {line.strip()}")
                continue

    # 展平成 DataFrame，同时添加 client_id
    all_records = []
    for client_id, trx_list in enumerate(data):       
        for trx in trx_list:
            trx['client_id'] = client_id
            all_records.append(trx)
    df = pd.DataFrame(all_records)

    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_jsonl_as_dataframe(jsonl_path):
    """从单个jsonl文件加载数据为DataFrame"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            user_trx = json.loads(line)  # 每行是一个用户的交易序列
            data.append(user_trx)

    # 展平成 DataFrame，同时添加 client_id
    all_records = []
    for client_id, trx_list in enumerate(data):       
        for trx in trx_list:
            trx['client_id'] = client_id
            all_records.append(trx)
    df = pd.DataFrame(all_records)

    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 构造标准化时间字段 'trans_date' , df[trans_date][0]:  2025-04-01 19:27:40
    # df['trans_date'] = pd.to_datetime(
    #     df[['year', 'month', 'day', 'hour', 'minutes', 'seconds']],
    #     errors='coerce'  # 避免不合法时间导致 crash
    # )
    return df


class MultiFileColesIterableDataset(IterableDataset):
    """改进的多文件可迭代数据集，支持分布式训练和连续数据流
    
    这个数据集将多个训练文件整合成一个连续的数据流，避免了重复创建Trainer的开销。
    支持分布式训练，每个进程处理不同的文件子集。
    """
    
    def __init__(self, file_paths, preprocessor, dataset_builder, debug_print_func=None, max_files=None):
        """
        Args:
            file_paths: 训练文件路径列表
            preprocessor: 数据预处理器
            dataset_builder: 数据集构建函数，接收处理后的DataFrame，返回Dataset
            debug_print_func: 调试打印函数
            max_files: 最大处理文件数量，None表示处理所有文件
        """
        # 限制文件数量
        if max_files is not None and max_files > 0:
            self.file_paths = file_paths[:max_files]
        else:
            self.file_paths = file_paths
            
        self.preprocessor = preprocessor
        self.dataset_builder = dataset_builder
        self.debug_print = debug_print_func if debug_print_func else print
        self.max_files = max_files
        
        # 记录数据集信息
        self._log_dataset_info()
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数，与ColesDataset保持一致"""
        from operator import iadd
        from functools import reduce
        from ptls.data_load.utils import collate_feature_dict
        import torch
        
        class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples]
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, torch.LongTensor(class_labels)
    
    @rank_zero_only
    def _log_dataset_info(self):
        """记录数据集基本信息"""
        self.debug_print(f"\n=== MultiFileColesIterableDataset 初始化 ===")
        self.debug_print(f"总文件数: {len(self.file_paths)}")
        self.debug_print(f"文件列表: {[os.path.basename(f) for f in self.file_paths[:5]]}{'...' if len(self.file_paths) > 5 else ''}")
        self.debug_print(f"=== 数据集初始化完成 ===\n")
    
    def __iter__(self):
        """迭代器实现，支持分布式训练"""
        # 获取分布式训练信息
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # 处理文件数量少于进程数的情况：循环分配文件
        if len(self.file_paths) < world_size:
            # 通过循环索引确保每个进程都能分配到文件
            extended_files = []
            for i in range(world_size):
                file_idx = i % len(self.file_paths)
                extended_files.append(self.file_paths[file_idx])
            file_list = extended_files
        else:
            file_list = self.file_paths
        
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
        
        # 逐文件处理并生成样本
        for file_idx, file_path in enumerate(assigned_files):
            try:
                self.debug_print(f"开始处理文件 {file_idx + 1}/{len(assigned_files)}: {os.path.basename(file_path)}")
                
                # 加载和预处理数据
                df = load_jsonl_as_dataframe_new_format(file_path)
                processed_data = self.preprocessor.fit_transform(df)
                
                # 构建数据集
                dataset = self.dataset_builder(processed_data)
                
                # 生成样本
                sample_count = 0
                for sample in dataset:
                    yield sample
                    sample_count += 1
                
                self.debug_print(f"文件 {os.path.basename(file_path)} 处理完成，生成 {sample_count} 个样本")
                
            except Exception as e:
                self.debug_print(f"处理文件 {file_path} 时出错: {str(e)}")
                # 继续处理下一个文件，不中断整个训练过程
                continue



def load_jsonl_from_directory(directory_path):
    """从目录下的多个jsonl文件加载数据为DataFrame
    
    Args:
        directory_path (str): 包含jsonl文件的目录路径
        
    Returns:
        pd.DataFrame: 合并后的DataFrame，包含所有文件的数据
    """
    # 获取目录下所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"在目录 {directory_path} 中没有找到任何jsonl文件")
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件: {[os.path.basename(f) for f in jsonl_files]}")
    
    all_data = []
    global_client_id = 0  # 全局客户ID计数器，确保不同文件间的client_id不重复
    
    for file_path in sorted(jsonl_files):  # 排序确保处理顺序一致
        print(f"正在处理文件: {os.path.basename(file_path)}")
        
        # 读取单个文件的数据
        file_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                user_trx = json.loads(line)  # 每行是一个用户的交易序列
                file_data.append(user_trx)
        
        # 展平成记录，同时分配全局唯一的client_id
        for trx_list in file_data:
            for trx in trx_list:
                trx['client_id'] = global_client_id
                all_data.append(trx)
            global_client_id += 1  # 每个用户序列分配一个唯一ID
        
        print(f"文件 {os.path.basename(file_path)} 处理完成，包含 {len(file_data)} 个用户序列")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    print(f"总共加载了 {len(df)} 条交易记录，来自 {global_client_id} 个用户")
    
    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df