import os
import sys
import time
import glob
import json
from datetime import datetime
from functools import partial
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import ptls

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ptls.nn import TransformerSeqEncoder
from ptls.frames import PtlsDataModule
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.padded_batch import PaddedBatch
from ptls.preprocessing import (
    PandasDataPreprocessor, 
    extract_predefined_mappings_from_feature_config, 
    get_categorical_columns_from_feature_config
)

from encode_3 import UniversalFeatureEncoder
from config import init_config
# from train_coles import load_jsonl_as_dataframe_new_format
from data_load_xqy import load_jsonl_as_dataframe_new_format, MultiFileColesIterableDataset, StreamingUserColesIterableDataset
from pytorch_lightning.profilers import PyTorchProfiler
import torch.profiler





os.environ['OMP_NUM_THREADS'] = '1'

# 设置日志文件，将调试信息重定向到文件（只在主进程中执行）
@rank_zero_only
def setup_logging():
    log_dir = 'logs_xqy'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    return log_file, log_filename

# 初始化日志（只在主进程中执行）
log_file, log_filename = setup_logging() if rank_zero_only(lambda: True)() else (None, None)

@rank_zero_only
def debug_print(*args, **kwargs):
    """专门用于调试信息的打印函数，只输出到日志文件"""
    if log_file is not None:
        print(*args, file=log_file, **kwargs)
        log_file.flush()

# 时间记录工具函数
@rank_zero_only
def print_time_point(message, start_time=None):
    """打印时间点信息，包括当前时间和从开始时间的耗时"""
    current_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if start_time is not None:
        elapsed = current_time - start_time
        debug_print(f"[{timestamp}] {message} (耗时: {elapsed:.2f}秒)")
    else:
        debug_print(f"[{timestamp}] {message}")
    
    return current_time

# 记录程序开始时间
program_start_time = print_time_point("=== CoLES训练程序开始 ===")



# 创建智能输出类，区分进度信息和调试信息（只在主进程中使用）
class SmartLogger:
    def __init__(self, log_file, console_output):
        self.log_file = log_file
        self.console_output = console_output
        self.progress_keywords = ['Epoch', 'it/s', 'v_num', 'recall_top_k', 'Validation', 'Training', 'Train Loss', 'Val Recall', '开始训练', '训练完成', '保存模型', '程序执行完成', '=== Epoch', 'Train End', 'Validation End', '时间点', '耗时', 'CoLES训练程序开始', '训练开始', '验证开始', '训练结束', '验证结束', '分布式训练调试', 'CUDA可用', 'WORLD_SIZE', 'RANK', '当前进程rank', '总进程数', '数据加载调试', '训练数据大小', '批次大小', '采样器类型']
        
    def write(self, data):
        # 只在主进程中写入日志文件
        if self.log_file is not None:
            self.log_file.write(data)
            self.log_file.flush()
            
            # 只有包含进度关键词的信息才输出到控制台，并且只在主进程中输出
            # 排除空行，避免终端显示大段空白
            if data.strip() and any(keyword in data for keyword in self.progress_keywords):
                self.console_output.write(data)
                self.console_output.flush()
    
    def flush(self):
        if self.log_file is not None:
            self.log_file.flush()
        # 只在主进程中刷新控制台输出
        if self.log_file is not None:
            self.console_output.flush()

# 重定向stdout到日志（只在主进程中执行）
original_stdout = sys.stdout
if log_file is not None:
    sys.stdout = SmartLogger(log_file, original_stdout)

# 初始化信息只输出到日志文件（只在主进程中执行）
if log_file is not None:
    debug_print(f"=== 调试日志开始 - {datetime.now()} ===")
    debug_print(f"日志文件: {log_filename}")
    debug_print(f"PTLS路径: {ptls.__file__}")
    debug_print(f"=== 基础信息结束 ===\n")

# 在控制台显示简洁的启动信息（只在主进程中执行）
@rank_zero_only
def print_startup_info():
    debug_print(f"\n=== 训练启动信息 ===")
    if log_filename is not None:
        debug_print(f"训练开始 - 日志文件: {log_filename}")
    else:
        debug_print("训练开始")
    debug_print(f"=== 训练启动信息结束 ===\n")

print_startup_info()

class MetricsTracker(Callback):
    """记录训练损失和验证指标的回调函数"""
    
    def __init__(self):
        super().__init__()
        self.train_losses = []   # 训练过程损失
        self.val_metrics = []    # 验证指标（recall@top_k）
        self.epochs = []
        self.epoch_start_time = None
        self.validation_start_time = None
    
    def on_train_epoch_start(self, trainer, pl_module):
        """训练epoch开始时记录时间"""
        self.epoch_start_time = print_time_point(f"Epoch {trainer.current_epoch} 训练开始")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """训练epoch结束时记录损失和验证指标"""
        # 记录训练epoch结束时间
        if self.epoch_start_time:
            print_time_point(f"Epoch {trainer.current_epoch} 训练结束", self.epoch_start_time)
        
        # 调试：打印所有可用的指标键
        debug_print(f"\n=== Epoch {trainer.current_epoch} Train End ===")
        debug_print(f"Available callback_metrics keys: {list(trainer.callback_metrics.keys())}")
        debug_print(f"All callback_metrics: {trainer.callback_metrics}")
        
        # 获取训练损失
        train_loss = None
        if 'loss' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['loss'].item()
            debug_print(f"Found aggregated train loss with key 'loss' from callback_metrics: {train_loss:.4f}")
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
            debug_print(f"Epoch {trainer.current_epoch}: Train Loss = {train_loss:.4f}")
        else:
            debug_print(f"Warning: No train loss found in callback_metrics")
            
        # 在训练 epoch 结束时也尝试记录验证指标
        val_metric = None
        if 'valid/recall_top_k' in trainer.callback_metrics:
            val_metric = trainer.callback_metrics['valid/recall_top_k'].item()
            debug_print(f"Found aggregated validation metric with key 'valid/recall_top_k' from callback_metrics: {val_metric:.4f}")
        
        if val_metric is not None:
            # 检查是否已经记录过当前 epoch 的验证指标，避免重复记录
            if trainer.current_epoch not in self.epochs:
                self.val_metrics.append(val_metric)
                self.epochs.append(trainer.current_epoch)
                debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK recorded at Train End = {val_metric:.4f}")
            else:
                # 如果已经记录过，检查值是否一致
                epoch_index = self.epochs.index(trainer.current_epoch)
                previous_val_metric = self.val_metrics[epoch_index]
                if abs(previous_val_metric - val_metric) > 1e-6:  # 使用小的阈值比较浮点数
                    debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK value changed from {previous_val_metric:.4f} to {val_metric:.4f}, updating record.")
                    self.val_metrics[epoch_index] = val_metric
                else:
                    debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK already recorded with same value {val_metric:.4f}.")
        else:
            debug_print(f"Warning: No validation metric found in callback_metrics at Train End")
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """验证epoch开始时记录时间"""
        self.validation_start_time = print_time_point(f"Epoch {trainer.current_epoch} 验证开始")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时记录验证指标"""
        # 记录验证epoch结束时间
        if self.validation_start_time:
            print_time_point(f"Epoch {trainer.current_epoch} 验证结束", self.validation_start_time)
        
        # 调试：打印所有可用的指标键
        debug_print(f"\n=== Epoch {trainer.current_epoch} Validation End ===")
        debug_print(f"Available callback_metrics keys: {list(trainer.callback_metrics.keys())}")
        debug_print(f"All callback_metrics: {trainer.callback_metrics}")
        
        # 获取验证指标（recall@top_k）
        val_metric = None
        if 'valid/recall_top_k' in trainer.callback_metrics:
            val_metric = trainer.callback_metrics['valid/recall_top_k'].item()
            debug_print(f"Found aggregated validation metric with key 'valid/recall_top_k' from callback_metrics: {val_metric:.4f}")
        
        if val_metric is not None:
             # 检查是否已经记录过当前 epoch 的验证指标，避免重复记录
            if trainer.current_epoch not in self.epochs:
                self.val_metrics.append(val_metric)
                self.epochs.append(trainer.current_epoch)
                debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK recorded at Validation End = {val_metric:.4f}")
            else:
                # 如果已经记录过，检查值是否一致
                epoch_index = self.epochs.index(trainer.current_epoch)
                previous_val_metric = self.val_metrics[epoch_index]
                if abs(previous_val_metric - val_metric) > 1e-6:  # 使用小的阈值比较浮点数
                    debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK value changed from {previous_val_metric:.4f} to {val_metric:.4f}, updating record.")
                    self.val_metrics[epoch_index] = val_metric
                else:
                    debug_print(f"Epoch {trainer.current_epoch}: Val Recall@TopK already recorded with same value {val_metric:.4f}.")
        else:
            debug_print(f"Warning: No validation metric found in callback_metrics at Validation End")
    
    def plot_metrics(self, save_path=None):
        """绘制训练损失和验证指标曲线"""
        plt.figure(figsize=(12, 8))
        
        # 创建两个子图
        plt.subplot(2, 1, 1)
        if self.train_losses:
            epochs_range = range(len(self.train_losses))
            plt.plot(epochs_range, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
            
            # 为每个训练损失数据点添加数值标注
            for i, (epoch, loss) in enumerate(zip(epochs_range, self.train_losses)):
                plt.annotate(f'{loss:.4f}', 
                            xy=(epoch, loss), 
                            xytext=(0, 10),  # 向上偏移10个像素
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                            
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 第二个子图：显示验证指标（recall@top_k）
        plt.subplot(2, 1, 2)
        if self.val_metrics:
            # 确保epochs和val_metrics长度一致且排序
            sorted_indices = sorted(range(len(self.epochs)), key=lambda i: self.epochs[i])
            sorted_epochs = [self.epochs[i] for i in sorted_indices]
            sorted_val_metrics = [self.val_metrics[i] for i in sorted_indices]

            plt.plot(sorted_epochs, sorted_val_metrics, 'g-', label='Validation Recall@TopK', linewidth=2, marker='o')
            
            # 为每个验证指标数据点添加数值标注
            for epoch, metric in zip(sorted_epochs, sorted_val_metrics):
                plt.annotate(f'{metric:.4f}', 
                            xy=(epoch, metric), 
                            xytext=(0, 10),  # 向上偏移10个像素
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
                            
            plt.title('Validation Recall@TopK', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Recall@TopK')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加最大值标注（recall@top_k越高越好）
            if sorted_val_metrics:
                max_val_metric = max(sorted_val_metrics)
                max_epoch = sorted_epochs[sorted_val_metrics.index(max_val_metric)]
                plt.annotate(f'Max: {max_val_metric:.4f}\nEpoch: {max_epoch}', 
                            xy=(max_epoch, max_val_metric), 
                            xytext=(max_epoch + len(sorted_epochs)*0.1, max_val_metric - (max(sorted_val_metrics) - min(sorted_val_metrics))*0.1),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                            fontsize=10, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            debug_print(f"指标图表已保存到: {save_path}")
        
        plt.show()
        
        # 打印统计信息
        if self.train_losses:
            debug_print(f"\n=== 训练损失统计 ===")
            debug_print(f"最终训练损失: {self.train_losses[-1]:.4f}")
            debug_print(f"最小训练损失: {min(self.train_losses):.4f}")
            debug_print(f"平均训练损失: {np.mean(self.train_losses):.4f}")
        
        if self.val_metrics:
            debug_print(f"\n=== 验证指标统计 ===")
            debug_print(f"最终验证Recall@TopK: {self.val_metrics[-1]:.4f}")
            debug_print(f"最高验证Recall@TopK: {max(self.val_metrics):.4f} (Epoch {self.epochs[self.val_metrics.index(max(self.val_metrics))]})")
            debug_print(f"平均验证Recall@TopK: {np.mean(self.val_metrics):.4f}")


class UniversalTrxEncoder(nn.Module):
    """包装UniversalFeatureEncoder以兼容ptls的TrxEncoder接口
    
    1. 将ptls框架的PaddedBatch数据格式转换为UniversalFeatureEncoder期望的字典格式
    2. 处理特征编码，包括类别特征、时间特征和数值特征
    3. 添加可选的噪声和线性投影层
    4. 将编码结果重新包装为PaddedBatch格式，以便后续的序列编码器使用
    """
    
    def __init__(self, feature_config, emb_dim_cfg=None, num_fbr_cfg=None, 
                 feature_fusion='concat', field_transformer_args=None, 
                 embeddings_noise=0.0, linear_projection_size=None):
        super().__init__()
        
        # 保存特征配置，用于数据验证和处理
        self.feature_config = feature_config
        self.embeddings_noise = embeddings_noise
        
        # 创建UniversalFeatureEncoder - 核心的特征编码器
        # feature_config: 特征类型配置（类别型、时间型、数值型等）
        # emb_dim_cfg: 每个特征的嵌入维度配置
        # num_fbr_cfg: 数值特征的频率表示配置
        # feature_fusion: 特征融合方式（concat或field_transformer）
        self.universal_encoder = UniversalFeatureEncoder(
            feature_config=feature_config,
            emb_dim_cfg=emb_dim_cfg,
            num_fbr_cfg=num_fbr_cfg,
            feature_fusion=feature_fusion,
            # field_transformer_args=field_transformer_args
        )
        
        # 可选的线性投影层 - 将特征维度投影到指定大小
        # 这对于控制模型复杂度和与下游模块的兼容性很有用
        if linear_projection_size is not None:
            self.projection = nn.Linear(self.universal_encoder.out_dim, linear_projection_size)
            self.output_size = linear_projection_size
        else:
            self.projection = None
            self.output_size = self.universal_encoder.out_dim
            
        # 可选的噪声层 - 用于正则化，防止过拟合
        if embeddings_noise > 0:
            self.noise_layer = nn.Dropout(embeddings_noise)
        else:
            self.noise_layer = None
    
    @property
    def category_names(self):
        """返回所有特征名称的集合，兼容ptls框架"""
        return set(self.feature_config.keys())
    
    @property
    def category_max_size(self):
        """返回类别特征的字典大小映射，兼容ptls框架"""
        result = {}
        for name, config in self.feature_config.items():
            if config['type'] == 'categorical':
                result[name] = len(config['choices'])
            elif config['type'] == 'time':
                result[name] = config['range']
        return result
    
    def forward(self, x):
        """前向传播方法 - 将PaddedBatch数据转换并编码
        
        数据流程：
        1. PaddedBatch输入 -> 提取payload字典
        2. 特征验证和范围检查 -> 防止索引越界
        3. UniversalFeatureEncoder编码 -> 生成特征嵌入
        4. 可选的噪声和投影处理
        5. 重新包装为PaddedBatch -> 保持与ptls框架的兼容性
        
        Args:
            x (PaddedBatch): ptls格式的批次数据，包含payload和seq_lens
            
        Returns:
            PaddedBatch: 编码后的特征数据，保持原有的序列长度信息
        """
        debug_print(f"\n=== UniversalTrxEncoder.forward ===")
        debug_print(f"Input x type: {type(x)}")
        debug_print(f"x attributes: {dir(x)}")
        if hasattr(x, 'payload'):
            debug_print(f"x.payload type: {type(x.payload)}")
            debug_print(f"x.payload keys: {list(x.payload.keys())}")
        if hasattr(x, 'seq_lens'):
            debug_print(f"x.seq_lens: {x.seq_lens}")
        
        # 1: 从PaddedBatch中提取实际数据
        # x.payload是一个字典，包含所有特征的张量数据
        # 每个特征的形状为 [batch_size, seq_len, feature_dim]
        payload = x.payload
        batch_size, seq_len = payload['event_time'].shape
        debug_print(f"Batch size: {batch_size}, Seq len: {seq_len}")
        
        # 2: 构建特征字典，进行数据验证和预处理
        feature_dict = {}
        
        for feature_name, feature_info in self.feature_config.items():
            if feature_name in payload:
                feature_values = payload[feature_name]
                
                # 调试信息：打印特征的统计信息
                debug_print(f"Feature {feature_name}: min={feature_values.min()}, max={feature_values.max()}, shape={feature_values.shape}")
                
                # 类别特征的范围检查和截断
                # 防止embedding层索引越界导致CUDA错误
                if feature_info['type'] == 'categorical':
                    n_choices = len(feature_info['choices'])  # 类别数量
                    if feature_values.max() >= n_choices:
                        debug_print(f"WARNING: Feature {feature_name} has max value {feature_values.max()} >= n_choices ({n_choices}), max value: {feature_values.max()}")
                        # 使用torch.clamp将超出范围的值截断到有效范围[0, n_choices-1]
                        feature_values = torch.clamp(feature_values, 0, n_choices - 1)
                        
                # 时间特征的范围检查和截断
                elif feature_info['type'] == 'time':
                    time_range = feature_info['range']  # 时间特征的范围
                    if feature_values.max() >= time_range:
                        debug_print(f"WARNING: Feature {feature_name} has max value {feature_values.max()}  > range ({time_range}), max value: {feature_values.max()}")
                        # 将超出范围的时间值截断到有效范围[0, time_range-1]
                        feature_values = torch.clamp(feature_values, 0, time_range - 1)
                
                # 将处理后的特征添加到字典中
                feature_dict[feature_name] = feature_values
        
        # 3: 使用UniversalFeatureEncoder进行特征编码
        # 这里会将所有特征转换为嵌入向量并进行融合
        encoded = self.universal_encoder(feature_dict)
        
        # 4: 可选的后处理
        # 添加噪声进行正则化（训练时）
        if self.noise_layer is not None:
            encoded = self.noise_layer(encoded)
        
        # 线性投影到目标维度
        if self.projection is not None:
            encoded = self.projection(encoded)
        
        # 5: 重新包装为PaddedBatch格式
        # 保持原有的序列长度信息，确保与后续的序列编码器兼容
        new_x = PaddedBatch(encoded, x.seq_lens)
        
        debug_print(f"Encoded tensor shape: {encoded.shape}")
        debug_print(f"Output PaddedBatch type: {type(new_x)}")
        debug_print(f"Output seq_lens: {new_x.seq_lens}")
        if hasattr(new_x, 'payload'):
            debug_print(f"Output payload shape: {new_x.payload.shape}")
        debug_print("=== End UniversalTrxEncoder.forward ===\n")
        
        return new_x


def load_feature_config_from_json(config_path):
    """从JSON配置文件加载特征配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
        return full_config.get('feature_config', {})


def create_feature_config_for_universal_encoder(config_path='feature_config.json'):
    """将JSON配置转换为UniversalFeatureEncoder需要的格式
    
    Args:
        config_path: JSON配置文件路径
    
    Returns:
        dict: UniversalFeatureEncoder需要的特征配置格式
    """
    # 从JSON文件加载真实的特征配置
    json_config = load_feature_config_from_json(config_path)
    
    feature_config = {}
    
    for feature_name, feature_info in json_config.items():
        if feature_info['type'] == 'categorical':
            # 对于类别特征，使用choices字段
            choices = feature_info.get('choices', [])
            feature_config[feature_name] = {
                'type': 'categorical',
                'choices': list(range(len(choices)))  # 转换为索引列表
            }
        elif feature_info['type'] == 'time':
            # 对于时间特征，使用range字段
            time_range = feature_info.get('range', 100)
            feature_config[feature_name] = {
                'type': 'time',
                'range': time_range
            }
        elif feature_info['type'] == 'numerical':
            # 对于数值特征，保持原有格式
            feature_config[feature_name] = {
                'type': 'numerical'
            }
    
    # 添加调试信息
    debug_print(f"\n=== 从配置文件加载的特征配置 ===\n")
    for name, config in feature_config.items():
        if config['type'] == 'categorical':
            debug_print(f"{name}: {config['type']}, choices数量: {len(config['choices'])}")
        elif config['type'] == 'time':
            debug_print(f"{name}: {config['type']}, range: {config['range']}")
        else:
            debug_print(f"{name}: {config['type']}")
    debug_print("=== 特征配置加载完成 ===\n")
    
    # print("create_feature_config_for_universal_encoder:",feature_config)
    return feature_config


def train_continuous_coles_universal(train_dir, val_dir, config):
    """使用StreamingUserColesIterableDataset进行多个文件连续训练的CoLES模型
    
    这将多个训练文件整合成一个连续的文件流，只创建一次Trainer进行训练。
    相比逐文件训练，这种方式可以保持参数状态的连续性，避免重复创建Trainer的开销。
    
    Args:
        train_dir (str): 训练文件目录
        val_dir (str): 验证文件目录  
        config (dict): 配置字典
    
    Returns:
        tuple: (model, metrics_tracker)
    """
    # 记录函数开始时间
    function_start_time = print_time_point("=== 训练函数开始 ===")
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析配置
    model_config = config['model_config']
    data_config = config['data_config']
    train_config = config['train_config']
    feature_config = config['feature_config']
    
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    model_dim = model_config['model_dim']
    n_heads = model_config['n_heads']
    n_layers = model_config['n_layers']
    feature_fusion = model_config.get('feature_fusion', 'concat')
    
    # 从配置文件中读取嵌入维度配置和数值特征配置
    emb_dim_cfg = config.get('universal_encoder_config', {}).get('emb_dim_cfg', {})
    num_fbr_cfg = config.get('universal_encoder_config', {}).get('num_fbr_cfg', {})
    
    if not emb_dim_cfg:
        debug_print("no emb_dim_cfg")
    else:
        debug_print("emb_dim_cfg", emb_dim_cfg)
        
    if not num_fbr_cfg:
        debug_print("no num_fbr_cfg")
    else:
        debug_print("num_fbr_cfg", num_fbr_cfg)
    
    # 记录文件扫描开始时间
    file_scan_start_time = print_time_point("开始扫描训练和验证文件")
    
    # 获取训练和验证文件列表
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.jsonl")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.jsonl")))
    
    if not train_files:
        raise ValueError(f"在目录 {train_dir} 中没有找到任何jsonl文件")
    else:
        debug_print(f"找到 {len(train_files)} 个训练文件: {[os.path.basename(f) for f in train_files]}")
    if not val_files:
        raise ValueError(f"在目录 {val_dir} 中没有找到任何jsonl文件")
    else:
        debug_print(f"找到 {len(val_files)} 个验证文件: {[os.path.basename(f) for f in val_files]}")
    
    print_time_point("文件扫描完成", file_scan_start_time)
    
    # 记录预处理器创建开始时间
    preprocessor_start_time = print_time_point("开始创建数据预处理器")
    
    # 从feature_config中提取预定义映射
    predefined_mappings = extract_predefined_mappings_from_feature_config(feature_config)
    categorical_columns = get_categorical_columns_from_feature_config(feature_config)
    
    # 创建数据预处理器
    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='unix_timestamp',
        event_time_transformation='none',
        cols_category=categorical_columns,
        cols_category_with_mapping=predefined_mappings,
        cols_numerical=['交易金额'],
        return_records=True
    )
    
    print_time_point("数据预处理器创建完成", preprocessor_start_time)
    
    # 创建验证数据的流式加载（与训练数据保持一致）
    val_load_start_time = print_time_point("开始创建验证数据流")
    
    debug_print(f"验证文件列表: {[os.path.basename(f) for f in val_files]}")
    print_time_point("验证数据流创建完成", val_load_start_time)
    
    # 创建模型
    model_creation_start_time = print_time_point("开始创建模型")
    
    # 创建交易编码器
    trx_encoder = UniversalTrxEncoder(
        feature_config=feature_config,
        emb_dim_cfg=emb_dim_cfg,
        num_fbr_cfg=num_fbr_cfg,
        feature_fusion=feature_fusion,
        embeddings_noise=0.003,
        linear_projection_size=model_dim
    )
    
    # 创建序列编码器
    seq_encoder = TransformerSeqEncoder(
        trx_encoder=trx_encoder,
        input_size=None,
        is_reduce_sequence=True,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1,
        dim_hidden=model_dim*4
    )
    
    # 创建CoLES模型
    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=learning_rate),
        lr_scheduler_partial=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9
        )
    )
    
    print_time_point("模型创建完成", model_creation_start_time)
    
    # 定义数据集构建函数
    def build_dataset_from_df(processed_df):
        """从预处理后的DataFrame构建ColesDataset"""
        return ColesDataset(
            MemoryMapDataset(
                data=processed_df,
                i_filters=[SeqLenFilter(min_seq_len=1)],
            ),
            splitter=SampleSlices(
                split_count=5,
                cnt_min=1,
                cnt_max=5,
            ),
        )
    
    # 使用流式用户级读取数据集
    debug_print("\n=== 创建训练数据StreamingUserColesIterableDataset（流式用户级读取）===")
    train_iterable_dataset = StreamingUserColesIterableDataset(
        file_paths=train_files,
        preprocessor=preprocessor,
        dataset_builder=build_dataset_from_df,
        debug_print_func=debug_print
    )
    
    debug_print("\n=== 创建验证数据StreamingUserColesIterableDataset（流式用户级读取）===")
    valid_iterable_dataset = StreamingUserColesIterableDataset(
        file_paths=val_files,
        preprocessor=preprocessor,
        dataset_builder=build_dataset_from_df,
        debug_print_func=debug_print
    )
    
    # 创建数据模块（训练和验证都使用流式数据集）
    datamodule = PtlsDataModule(
        train_data=train_iterable_dataset,
        valid_data=valid_iterable_dataset,
        train_batch_size=batch_size,
        train_num_workers=2,
        valid_batch_size=batch_size,
        valid_num_workers=2,
    )
    
    # 创建指标跟踪器
    metrics_tracker = MetricsTracker()
    
    # 配置性能监控器
    debug_print("\n=== 配置性能监控器 ===")
    is_dist = torch.distributed.is_initialized()
    should_profile = (not is_dist) or (torch.distributed.get_rank() == 0)
    
    if should_profile:
        debug_print("启用性能监控 - 当前进程将进行性能分析")
        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/torch_prof_lightning'),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            schedule=None,
        )
    else:
        debug_print("跳过性能监控 - 非主进程或非分布式环境")
        profiler = None
    
    # 创建Trainer（只创建一次）
    debug_print("\n=== 创建Trainer（只创建一次）===")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy='ddp',
        devices=2 if torch.cuda.is_available() else 'auto',
        enable_progress_bar=True,
        callbacks=[metrics_tracker],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        logger=False,
        enable_model_summary=True,
        profiler=profiler,
    )
    
    # 开始训练（一次性训练所有epochs）
    training_start_time = print_time_point("=== 开始连续训练 ===")
    debug_print(f"训练配置: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    debug_print(f"训练文件数: {len(train_files)}")
    debug_print(f"验证文件数: {len(val_files)}")

    
    trainer.fit(model, datamodule)
    
    print_time_point("连续训练完成", training_start_time)
    
    # 保存训练结果
    debug_print("\n=== 保存训练结果 ===")
    
    # 保存模型
    model_save_path = os.path.join(output_dir, 'continuous_coles_model.ckpt')
    trainer.save_checkpoint(model_save_path)
    debug_print(f"模型已保存到: {model_save_path}")
    
    # 绘制和保存指标图表
    plot_save_path = os.path.join(output_dir, 'continuous_training_metrics.png')
    metrics_tracker.plot_metrics(save_path=plot_save_path)
    
    # 保存指标数据
    metrics_data = {
        'train_losses': metrics_tracker.train_losses,
        'val_metrics': metrics_tracker.val_metrics,
        'epochs': metrics_tracker.epochs,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_dim': model_dim,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'train_files_count': len(train_files),
            'val_files_count': len(val_files)
        }
    }
    
    metrics_save_path = os.path.join(output_dir, 'continuous_training_metrics.json')
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    debug_print(f"训练指标已保存到: {metrics_save_path}")
    
    # 性能监控结果说明
    if should_profile and profiler is not None:
        debug_print("\n=== 性能监控结果 ===")
        debug_print("性能分析数据已保存到: ./logs/torch_prof_lightning/")
        debug_print("查看性能分析结果的方法:")
        debug_print("1. 启动TensorBoard: tensorboard --logdir=./logs/torch_prof_lightning")
        debug_print("2. 在浏览器中打开: http://localhost:6006")
        debug_print("3. 点击 'PROFILE' 标签页查看性能分析")
        debug_print("4. 可以查看CPU/GPU使用率、内存使用、算子耗时等详细信息")
    
    print_time_point("连续训练函数完成", function_start_time)
    
    return model, metrics_tracker


if __name__ == '__main__':
    try:
        # 记录主程序开始时间
        main_start_time = print_time_point("=== 主程序开始执行 ===")
        
        # 记录配置加载开始时间
        config_start_time = print_time_point("开始加载配置")
        config = init_config()
        # debug_print(config)
        print_time_point("配置加载完成", config_start_time)
        
        # 训练和验证数据路径
        train_path = 'test_directory/train'  # 训练数据目录
        val_path = 'test_directory/val'      # 验证数据目录
        
        # 使用连续训练模式
        if True:
            debug_print("\n=== 启用连续训练模式 ===")
            debug_print("使用StreamingUserColesIterableDataset进行流式用户级读取，逐行处理用户数据")
            
            # 记录连续训练模式开始时间
            continuous_mode_start_time = print_time_point("连续训练模式开始")
            
            model, metrics_tracker = train_continuous_coles_universal(
                train_dir=train_path,
                val_dir=val_path,
                config=config
            )
            
            # 记录训练完成时间
            print_time_point("训练完成", continuous_mode_start_time)
            
            debug_print(f"\n=== 训练总结 ===")
            debug_print(f"训练完成时间: {datetime.now()}")
            debug_print(f"总训练epoch数: {config['train_config']['epochs']}")
            if metrics_tracker.val_metrics:
                best_val_metric = max(metrics_tracker.val_metrics)
                debug_print(f"最佳验证指标: {best_val_metric:.4f}")
            debug_print(f"所有输出文件保存在: ./output/ 目录")
        
        # 记录主程序完成时间
        print_time_point("=== 主程序执行完成 ===", main_start_time)
        
    finally:
        # 恢复原始stdout并关闭日志文件（只在主进程中执行）
        if log_file is not None:
            debug_print(f"\n=== 调试日志结束 - {datetime.now()} ===")
            debug_print(f"所有调试信息已保存到: {log_filename}")
            debug_print(f"\n程序执行完成，调试信息已保存到文件: {log_filename}")
            if log_filename is not None:
                debug_print("你可以使用以下命令查看完整的调试信息:")
                debug_print(f"cat {log_filename}")
                debug_print(f"或者使用: less {log_filename}")
            debug_print("\n损失图表和数据文件已保存到 ./output/ 目录中")
            debug_print("你可以查看生成的PNG图片来分析训练过程")
            sys.stdout = original_stdout
            log_file.close()
        else:
            debug_print(f"\n程序执行完成")
            debug_print("\n损失图表和数据文件已保存到 ./output/ 目录中")
            debug_print("你可以查看生成的PNG图片来分析训练过程")
