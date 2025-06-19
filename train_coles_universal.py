import json
import os
import sys
import time
import glob
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
from data_load_xqy import load_jsonl_as_dataframe_new_format



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

# 重定向stdout到智能日志系统（只在主进程中执行）
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
        self.train_losses = []
        self.val_metrics = []  # 改为记录验证指标（recall@top_k）
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
        
        # 尝试多种可能的训练损失键名
        train_loss = None
        for key in ['train_loss', 'loss', 'train/loss']:
            if key in trainer.callback_metrics:
                train_loss = trainer.callback_metrics[key].item()
                debug_print(f"Found aggregated train loss with key '{key}' from callback_metrics: {train_loss:.4f}")
                break
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
            debug_print(f"Epoch {trainer.current_epoch}: Train Loss = {train_loss:.4f}")
        else:
            debug_print(f"Warning: No train loss found in callback_metrics")
            
        # 在训练 epoch 结束时也尝试记录验证指标
        val_metric = None
        for key in ['valid/recall_top_k', 'valid/BatchRecallTopK', 'val_recall_top_k', 'recall_top_k']:
            if key in trainer.callback_metrics:
                val_metric = trainer.callback_metrics[key].item()
                debug_print(f"Found aggregated validation metric with key '{key}' from callback_metrics: {val_metric:.4f}")
                break
        
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
        
        # 尝试多种可能的验证指标键名（recall@top_k）
        val_metric = None
        for key in ['valid/recall_top_k']:
            if key in trainer.callback_metrics:
                val_metric = trainer.callback_metrics[key].item()
                debug_print(f"Found aggregated validation metric with key '{key}' from callback_metrics: {val_metric:.4f}")
                break
        
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


def train_incremental_coles_universal(train_dir, val_dir, config, checkpoint_dir='./checkpoints'):
    """按epoch遍历文件训练CoLES模型
    Args:
        train_dir (str): 训练文件目录
        val_dir (str): 验证文件目录  
        config (dict): 配置字典
        checkpoint_dir (str): 检查点保存目录
    """
    # 记录函数开始时间
    function_start_time = print_time_point("=== 增量训练函数开始 ===")
    

    output_dir = './output'                   #保存图像结果
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_config = config['model_config']     #模型参数
    data_config = config['data_config']       #数据参数
    train_config = config['train_config']     #训练参数
    feature_config=config['feature_config']
    
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

    # 如果配置文件中没有相应配置，则使用默认值
    if not emb_dim_cfg:
        debug_print("no emb_dim_cfg")
    else:
        debug_print("emb_dim_cfg",emb_dim_cfg)
        
    if not num_fbr_cfg:
        debug_print("no num_fbr_cfg")
    else:
        debug_print("num_fbr_cfg",num_fbr_cfg)    

    # 记录文件扫描开始时间
    file_scan_start_time = print_time_point("开始扫描训练和验证文件")
    
    # 获取训练文件列表（可能需要shuffle？）
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
    
    # 记录文件扫描完成时间
    print_time_point("文件扫描完成", file_scan_start_time)
    

    # 记录预处理器创建开始时间
    preprocessor_start_time = print_time_point("开始创建数据预处理器")
    
    # 从feature_config中提取预定义映射
    predefined_mappings = extract_predefined_mappings_from_feature_config(feature_config)
    categorical_columns = get_categorical_columns_from_feature_config(feature_config)
    
    # 数据预处理器
    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='unix_timestamp',
        event_time_transformation='none',
        cols_category=categorical_columns,
        cols_category_with_mapping=predefined_mappings,
        cols_numerical=['交易金额'],
        return_records=True
    )
    
    # 记录预处理器创建完成时间
    print_time_point("数据预处理器创建完成", preprocessor_start_time)
    
    # 记录验证数据加载开始时间
    val_data_start_time = print_time_point("开始加载验证数据")
    
    # 加载验证数据，目前加载单个jsonl文件,后续可能需要改成多文件加载
    debug_print("\n=== 加载验证数据 ===")
    source_data_val = load_jsonl_as_dataframe_new_format("test_directory/val/train copy.jsonl")
    # print("source_data_val",source_data_val)
    # 通过preprocessor.fit_transform已经对原数据进行了预处理
    valid_data = preprocessor.fit_transform(source_data_val)
    debug_print("\n=== 验证数据预处理完成 ===")
    
    # 记录验证数据加载完成时间
    print_time_point("验证数据加载和预处理完成", val_data_start_time)

    # 记录模型创建开始时间
    model_creation_start_time = print_time_point("开始创建模型")
    
    # 创建模型（只创建一次）
    universal_feature_config = feature_config
    
    trx_encoder = UniversalTrxEncoder(
        feature_config=universal_feature_config,
        emb_dim_cfg=emb_dim_cfg,
        num_fbr_cfg=num_fbr_cfg,
        feature_fusion=feature_fusion,
        embeddings_noise=0.003,
        linear_projection_size=model_dim
    )
    
    seq_encoder = TransformerSeqEncoder(
        trx_encoder=trx_encoder,
        input_size=None,
        is_reduce_sequence=True,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1,
        dim_hidden=model_dim*4
    )
    
    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=learning_rate),
        lr_scheduler_partial=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9
        )
    )
    
    # 记录模型创建完成时间
    print_time_point("模型创建完成", model_creation_start_time)
    
    # 全局指标跟踪
    all_train_losses = []
    all_val_metrics = []
    all_epochs = []
    
    # 创建全局指标跟踪器
    global_metrics_tracker = MetricsTracker()
    
    # 记录整体训练开始时间
    overall_training_start_time = print_time_point("=== 开始整体训练过程 ===")
    

    
    # 按epoch遍历训练
    for epoch in range(epochs):
        # 记录当前epoch开始时间
        epoch_start_time = print_time_point(f"=== 开始第 {epoch + 1}/{epochs} 个Epoch ===")
        
        epoch_train_losses = []
        epoch_val_metrics = []
        
        # 在每个epoch中遍历所有训练文件
        # 可以通过设置max_files_per_epoch来限制每个epoch处理的文件数量
        max_files_per_epoch = 50  # 设置最大处理文件数，比如只处理前50个文件
        files_to_process = min(len(train_files), max_files_per_epoch)
        
        # 使用tqdm显示epoch级别的进度条
        epoch_desc = f"Epoch {epoch + 1}/{epochs}"
        file_pbar = tqdm(train_files[:files_to_process], 
                        desc=epoch_desc,
                        unit="file",
                        ncols=120,
                        file=sys.stdout,
                        leave=True)
        
        for file_idx, train_file in enumerate(file_pbar):
            # 更新进度条描述，显示当前文件名和指标
            current_file = os.path.basename(train_file)
            avg_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0.0
            avg_val = sum(epoch_val_metrics) / len(epoch_val_metrics) if epoch_val_metrics else 0.0
            
            file_pbar.set_postfix({
                'file': current_file[:20] + '...' if len(current_file) > 20 else current_file,
                'avg_loss': f'{avg_loss:.4f}' if avg_loss > 0 else 'N/A',
                'avg_val': f'{avg_val:.4f}' if avg_val > 0 else 'N/A'
            })
            
            file_start_time = print_time_point(f"Epoch {epoch + 1}, 开始处理文件 {file_idx + 1}/{files_to_process}: {os.path.basename(train_file)}")
            
            # 加载和预处理当前训练文件
            data_load_start_time = print_time_point(f"开始加载文件 {os.path.basename(train_file)}")
            
            # 加载当前训练文件
            source_data_train = load_jsonl_as_dataframe_new_format(train_file)
            train_data = preprocessor.fit_transform(source_data_train)
            
            
            # 记录数据加载完成时间
            print_time_point(f"文件 {os.path.basename(train_file)} 加载和预处理完成", data_load_start_time)

            # 创建数据模块
            datamodule = PtlsDataModule(
                train_data=ColesDataset(
                    MemoryMapDataset(
                    # MemoryIterableDataset(
                        data=train_data,
                        i_filters=[SeqLenFilter(min_seq_len=1)],
                    ),
                    splitter=SampleSlices(
                        split_count=5,
                        cnt_min=1,
                        cnt_max=5,
                    ),
                ),
                valid_data=ColesDataset(
                    MemoryMapDataset(
                    # MemoryIterableDataset(
                        data=valid_data,
                        i_filters=[SeqLenFilter(min_seq_len=1)],
                    ),
                    splitter=SampleSlices(
                        split_count=5,
                        cnt_min=1,
                        cnt_max=5,
                    ),
                ),
                train_batch_size=batch_size,
                train_num_workers=2,
                valid_batch_size=batch_size,
                valid_num_workers=2,
            )
            
            # 创建文件级指标跟踪器
            file_metrics_tracker = MetricsTracker()
            
            # 训练器配置 - 每个文件只训练1个epoch
            
            # 添加分布式训练调试信息
            _log_distributed_info()
            
            # 动态控制模型摘要显示：只在第一次创建Trainer时显示
            global _first_trainer_created
            enable_summary = not _first_trainer_created
            if not _first_trainer_created:
                _first_trainer_created = True
                debug_print("首次创建Trainer，启用模型摘要显示")
            else:
                debug_print("非首次创建Trainer，禁用模型摘要显示")
            
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                strategy='ddp',
                devices=2 if torch.cuda.is_available() else 'auto',
                enable_progress_bar=False,             # 禁用trainer内置进度条，使用我们自定义的tqdm进度条
                callbacks=[file_metrics_tracker],
                enable_checkpointing=False,
                check_val_every_n_epoch=1,
                val_check_interval=1.0,
                logger=False,                         # 禁用默认logger输出
                enable_model_summary=enable_summary,  # 动态控制模型参数统计信息的自动输出
            )
            
            # 添加优化器调试信息
            _log_optimizer_info(model, train_file)
            
            # 添加数据加载调试信息
            _log_data_info(datamodule, train_data, valid_data, batch_size)
            
            # 记录训练开始时间
            training_start_time = print_time_point(f"开始训练文件 {os.path.basename(train_file)}")
            
            # 训练当前文件
            debug_print(f"开始训练文件 {os.path.basename(train_file)}...")
            trainer.fit(model, datamodule)
            debug_print(f"文件 {os.path.basename(train_file)} 训练完成")
            
            # 训练后检查分布式状态（此时分布式已正确初始化）
            debug_print(f"\n=== 训练后分布式状态检查 ===\n")
            if torch.distributed.is_available():
                debug_print(f"分布式后端可用: True")
                if torch.distributed.is_initialized():
                    debug_print(f"分布式已初始化: True")
                    debug_print(f"当前进程rank: {torch.distributed.get_rank()}")
                    debug_print(f"总进程数world_size: {torch.distributed.get_world_size()}")
                    debug_print(f"分布式后端: {torch.distributed.get_backend()}")
                else:
                    debug_print(f"分布式已初始化: False")
            else:
                debug_print(f"分布式后端可用: False")
            debug_print(f"=== 分布式状态检查结束 ===\n")
            
            # 记录训练完成时间
            print_time_point(f"文件 {os.path.basename(train_file)} 训练完成", training_start_time)
            debug_print(f"file_metrics_tracker.train_losses: {file_metrics_tracker.train_losses}")
            debug_print(f"file_metrics_tracker.val_metrics: {file_metrics_tracker.val_metrics}")
            debug_print(f"file_metrics_tracker.epochs: {file_metrics_tracker.epochs}")
            # 训练后的优化器状态
            debug_print(f"\n=== 训练后优化器状态 (文件: {os.path.basename(train_file)}) ===")
            optimizers_after = model.configure_optimizers()
            if isinstance(optimizers_after, tuple) and len(optimizers_after) >= 2:
                optimizer_list_after, scheduler_list_after = optimizers_after
                if optimizer_list_after:
                    optimizer_after = optimizer_list_after[0]
                    debug_print(f"训练后学习率: {optimizer_after.param_groups[0]['lr']:.6f}")
                    
                    if hasattr(optimizer_after, 'state') and optimizer_after.state:
                        debug_print(f"训练后优化器状态字典大小: {len(optimizer_after.state)}")
                        first_param_after = next(iter(optimizer_after.state.keys()), None)
                        if first_param_after is not None:
                            state_after = optimizer_after.state[first_param_after]
                            if 'step' in state_after:
                                debug_print(f"训练后优化器步数: {state_after['step']}")
            debug_print("=== 训练后优化器状态结束 ===\n")
            
            # 收集当前文件的指标
            if file_metrics_tracker.train_losses:
                epoch_train_losses.extend(file_metrics_tracker.train_losses)
            if file_metrics_tracker.val_metrics:
                epoch_val_metrics.extend(file_metrics_tracker.val_metrics)
            
            # 更新tqdm进度条显示最新指标
            current_file = os.path.basename(train_file)
            avg_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0.0
            avg_val = sum(epoch_val_metrics) / len(epoch_val_metrics) if epoch_val_metrics else 0.0
            
            file_pbar.set_postfix({
                'file': current_file[:20] + '...' if len(current_file) > 20 else current_file,
                'avg_loss': f'{avg_loss:.4f}' if avg_loss > 0 else 'N/A',
                'avg_val': f'{avg_val:.4f}' if avg_val > 0 else 'N/A',
                'status': 'completed'
            })
            
            # 释放内存
            del source_data_train, train_data, datamodule
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 记录单个文件处理完成时间
            print_time_point(f"文件 {os.path.basename(train_file)} 处理完成", file_start_time)
        
        # 计算当前epoch的平均指标
        if epoch_train_losses:
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            all_train_losses.append(avg_train_loss)
            global_metrics_tracker.train_losses.append(avg_train_loss)
            debug_print(f"Epoch {epoch + 1} 平均训练损失: {avg_train_loss:.4f}")
        
        if epoch_val_metrics:
            avg_val_metric = sum(epoch_val_metrics) / len(epoch_val_metrics)
            all_val_metrics.append(avg_val_metric)
            global_metrics_tracker.val_metrics.append(avg_val_metric)
            global_metrics_tracker.epochs.append(epoch)
            debug_print(f"Epoch {epoch + 1} 平均验证指标: {avg_val_metric:.4f}")
        
        all_epochs.append(epoch)
        
        # 关闭当前epoch的进度条
        file_pbar.close()
        
        # 显示epoch完成总结
        final_avg_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0.0
        final_avg_val = sum(epoch_val_metrics) / len(epoch_val_metrics) if epoch_val_metrics else 0.0
        print(f"\n✓ Epoch {epoch + 1}/{epochs} 完成 - 处理了 {files_to_process} 个文件")
        print(f"  最终平均损失: {final_avg_loss:.4f}")
        print(f"  最终平均验证指标: {final_avg_val:.4f}")
        print("-" * 80)
        
        # 记录当前epoch完成时间
        print_time_point(f"第 {epoch + 1}/{epochs} 个Epoch 完成", epoch_start_time)
        
        # 每5个epoch保存一次检查点
        # if (epoch + 1) % 5 == 0:
        #     checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        #     torch.save({
        #         'model_state_dict': model.seq_encoder.state_dict(),
        #         'epoch': epoch + 1,
        #         'train_losses': all_train_losses,
        #         'val_metrics': all_val_metrics,
        #         'epochs': all_epochs
        #     }, checkpoint_path)
        #     print(f"检查点已保存到: {checkpoint_path}")
    
    # 记录整体训练完成时间
    print_time_point("=== 整体训练过程完成 ===", overall_training_start_time)
    
    # 记录后处理开始时间
    post_processing_start_time = print_time_point("开始生成最终报告和保存结果")
    
    # 训练完成后的处理
    debug_print("\n=== 所有文件训练完成，正在生成最终报告 ===")
    
    # 绘制指标图表
    metrics_plot_path = os.path.join(output_dir, f'incremental_metrics_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    global_metrics_tracker.plot_metrics(save_path=metrics_plot_path)
    
    # 保存指标数据
    metrics_data = {
        'epoch': all_epochs,
        'train_loss': all_train_losses,
    }
    
    if all_val_metrics:
        val_df = pd.DataFrame({
            'epoch': list(range(len(all_val_metrics))),
            'val_recall_top_k': all_val_metrics
        })
        
        train_df = pd.DataFrame(metrics_data)
        merged_df = train_df.merge(val_df, on='epoch', how='left')
        
        metrics_data_path = os.path.join(output_dir, f'incremental_metrics_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        merged_df.to_csv(metrics_data_path, index=False)
        debug_print(f"增量训练指标数据已保存到: {metrics_data_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_incremental_model_universal.pth')
    torch.save(model.seq_encoder.state_dict(), final_model_path)
    debug_print(f"最终模型已保存到: {final_model_path}")
    
    # 记录后处理完成时间
    print_time_point("最终报告生成和结果保存完成", post_processing_start_time)
    
    # 记录函数完成时间
    print_time_point("=== 增量训练函数完成 ===", function_start_time)
    
    return model, global_metrics_tracker


@rank_zero_only
def _log_distributed_info():
    """记录分布式训练调试信息"""
    debug_print(f"\n=== 分布式训练调试信息 ===")
    debug_print(f"CUDA可用: {torch.cuda.is_available()}")
    debug_print(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    # 检查分布式环境变量
    debug_print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    debug_print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    debug_print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    debug_print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    debug_print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    
    # 显示当前使用的设备
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        debug_print(f"当前CUDA设备: {current_device}")
        debug_print(f"设备名称: {torch.cuda.get_device_name(current_device)}")
    debug_print(f"=== 训练前环境信息结束 ===\n")

# 全局标志，用于控制PyTorch Lightning模型摘要只在第一次显示
_first_trainer_created = False

def _log_optimizer_info(model, train_file):
    """记录优化器状态调试信息"""
    debug_print(f"\n=== 优化器状态调试信息 (文件: {os.path.basename(train_file)}) ===")
    
    # 获取当前优化器状态
    optimizers = model.configure_optimizers()
    if isinstance(optimizers, tuple) and len(optimizers) >= 2:
        optimizer_list, scheduler_list = optimizers
        if optimizer_list:
            optimizer = optimizer_list[0]
            debug_print(f"优化器类型: {type(optimizer).__name__}")
            debug_print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            debug_print(f"优化器参数组数量: {len(optimizer.param_groups)}")
            
            # 检查优化器状态
            if hasattr(optimizer, 'state') and optimizer.state:
                debug_print(f"优化器状态字典大小: {len(optimizer.state)}")
                # 显示第一个参数的状态信息（如果存在）
                first_param = next(iter(optimizer.state.keys()), None)
                if first_param is not None:
                    state = optimizer.state[first_param]
                    debug_print(f"第一个参数状态键: {list(state.keys())}")
                    if 'step' in state:
                        debug_print(f"优化器步数: {state['step']}")
            else:
                debug_print("优化器状态为空 (新初始化)")
                
            # 显示调度器信息
            if scheduler_list:
                scheduler = scheduler_list[0]
                if isinstance(scheduler, dict):
                    scheduler = scheduler['scheduler']
                debug_print(f"学习率调度器类型: {type(scheduler).__name__}")
                if hasattr(scheduler, 'last_epoch'):
                    debug_print(f"调度器当前epoch: {scheduler.last_epoch}")
                if hasattr(scheduler, 'step_size'):
                    debug_print(f"调度器步长: {scheduler.step_size}")
                if hasattr(scheduler, 'gamma'):
                    debug_print(f"调度器衰减因子: {scheduler.gamma}")
    
    # 显示模型参数的一些统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    debug_print(f"模型总参数数: {total_params:,}")
    debug_print(f"可训练参数数: {trainable_params:,}")
    
    # 显示模型参数的梯度统计（如果有的话）
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    if grad_norms:
        debug_print(f"参数梯度范数统计: 平均={sum(grad_norms)/len(grad_norms):.6f}, 最大={max(grad_norms):.6f}, 最小={min(grad_norms):.6f}")
    else:
        debug_print("当前无梯度信息")
    
    debug_print("=== 优化器状态调试信息结束 ===\n")

def _log_data_info(datamodule, train_data, valid_data, batch_size):
    """记录数据加载调试信息"""
    debug_print(f"\n=== 数据加载调试信息 ===")
    debug_print(f"训练数据大小: {len(train_data):,}")
    debug_print(f"验证数据大小: {len(valid_data):,}")
    debug_print(f"批次大小: {batch_size}")
    debug_print(f"")
    
    # 检查数据加载器的分布式设置
    train_dataloader = datamodule.train_dataloader()
    debug_print(f"训练数据加载器类型: {type(train_dataloader)}")
    if hasattr(train_dataloader, 'sampler'):
        debug_print(f"训练数据采样器类型: {type(train_dataloader.sampler)}")
    if hasattr(train_dataloader, 'batch_sampler'):
        debug_print(f"训练批次采样器类型: {type(train_dataloader.batch_sampler)}")
    
    # 尝试获取一个批次来检查数据
    try:
        sample_batch = next(iter(train_dataloader))
        print_main_only(f"\n=== 数据批次信息 ===")
        print_main_only(f"sample_batch: {sample_batch}")
        print_main_only(f"sample_batch类型: {type(sample_batch)}")
        
        # CoLES数据加载器返回的是元组 (padded_batch, class_labels)
        # 统一处理逻辑：提取padded_batch和class_labels
        padded_batch = None
        class_labels = None
        
        if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
            # CoLES数据加载器返回的是元组 (padded_batch, class_labels)
            padded_batch, class_labels = sample_batch
            print_main_only(f"padded_batch类型: {type(padded_batch)}")
            print_main_only(f"class_labels类型: {type(class_labels)}")
        elif hasattr(sample_batch, 'payload'):
            # 直接是PaddedBatch对象的情况
            padded_batch = sample_batch
            class_labels = None
        else:
            debug_print(f"未知的sample_batch格式: {type(sample_batch)}")
            return
        
        # 统一处理padded_batch
        if padded_batch and hasattr(padded_batch, 'payload'):
            batch_size_actual = padded_batch.payload['event_time'].shape[0]
            debug_print(f"实际批次大小: {batch_size_actual}")
            debug_print(f"payload字段: {list(padded_batch.payload.keys())}")
            
            # 获取进程信息
            rank = int(os.environ.get('RANK', 0))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            pid = os.getpid()
            
            # 显示样本标识（取前5个样本）
            print_with_rank(f"当前批次大小: {batch_size_actual}")
            sample_count = min(20, batch_size_actual)
            
            if 'event_time' in padded_batch.payload:
                sample_times = padded_batch.payload['event_time'][:sample_count].tolist()
                print_with_rank(f"样本时间戳 (前{sample_count}个): {sample_times}")
                
                if class_labels is not None:
                    sample_labels = class_labels[:sample_count].tolist()
                    print_with_rank(f"对应标签 (前{sample_count}个): {sample_labels}")
            else:
                # 使用批次索引作为标识
                sample_indices = list(range(sample_count))
                print_with_rank(f"样本索引 (前{sample_count}个): {sample_indices}")
                
                if class_labels is not None:
                    sample_labels = class_labels[:sample_count].tolist()
                    print_with_rank(f"对应标签 (前{sample_count}个): {sample_labels}")
        else:
            debug_print("padded_batch没有payload属性")
            
    except Exception as e:
        debug_print(f"获取样本批次失败: {e}")
    debug_print(f"=== 数据加载调试信息结束 ===\n")

def print_with_rank(*args, **kwargs):
    """带进程信息的打印函数，根据进程信息输出到不同的日志文件"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    pid = os.getpid()
    
    # 格式化消息，添加进程信息前缀
    message = ' '.join(str(arg) for arg in args)
    formatted_message = f"[Rank {rank}/{world_size}, Local Rank {local_rank}, PID {pid}] {message}"
    
    # 根据进程信息创建日志文件名
    log_filename = f"process_rank_{rank}_local_{local_rank}_pid_{pid}.log"
    log_path = os.path.join("./logs_xqy", log_filename)
    
    # 确保日志目录存在
    os.makedirs("./logs_xqy", exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # 写入日志文件
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {formatted_message}\n")
            f.flush()  # 确保立即写入
    except Exception as e:
        # 如果写入文件失败，回退到终端输出
        print(f"日志写入失败: {e}")
        print(formatted_message, **kwargs)

@rank_zero_only
def print_main_only(*args, **kwargs):
    """只在主进程输出的打印函数"""
    print(*args, **kwargs)

def safe_main_print(*args, **kwargs):
    """主函数中的安全打印函数，只在主进程中执行"""
    print(*args, **kwargs)

if __name__ == '__main__':
    try:
        # 记录主程序开始时间
        main_start_time = print_time_point("=== 主程序开始执行 ===")
        
        # 记录配置加载开始时间
        config_start_time = print_time_point("开始加载配置")
        config = init_config()
        # debug_print(config)
        print_time_point("配置加载完成", config_start_time)
        
        # 训练模式配置
        # 设置为 True 启用增量训练模式（逐文件训练）
        # 设置为 False 使用普通训练模式（一次性加载所有数据）
        INCREMENTAL_MODE = True  
        
        # 训练和验证数据路径
        train_path = 'test_directory/train'  # 训练数据目录
        val_path = 'test_directory/val'      # 验证数据目录
        
        if INCREMENTAL_MODE:
            debug_print("\n=== 启用增量训练模式 ===")
            debug_print("将逐个文件进行训练，每个文件训练完成后保存检查点")
            
            # 记录增量训练模式开始时间
            incremental_mode_start_time = print_time_point("增量训练模式开始")
            
            # 增量训练参数
            checkpoint_dir = './checkpoints'  # 检查点保存目录
            
            model, metrics_tracker = train_incremental_coles_universal(
                train_dir=train_path,
                val_dir=val_path,
                config=config,
                checkpoint_dir=checkpoint_dir
            )
            
            # 记录增量训练模式完成时间
            print_time_point("增量训练模式完成", incremental_mode_start_time)
            
            debug_print(f"\n=== 增量训练总结 ===")
            debug_print(f"训练完成时间: {datetime.now()}")
            debug_print(f"总训练epoch数: {config['train_config']['epochs']}")
            debug_print(f"检查点保存目录: {checkpoint_dir}")
            if metrics_tracker.val_metrics:
                best_val_metric = max(metrics_tracker.val_metrics)
                debug_print(f"最佳验证指标: {best_val_metric:.4f}")
            debug_print(f"所有输出文件保存在: ./output/ 目录")
            
        else:
            debug_print("\n=== 启用普通训练模式 ===")
            debug_print("将一次性加载所有数据进行训练")
            
            # 记录普通训练模式开始时间
            normal_mode_start_time = print_time_point("普通训练模式开始")
            
            # 可以传入文件路径或目录路径
            # 示例：
            # - 单文件模式: train_and_eval_coles_universal('train.jsonl', 'val.jsonl', config)
            # - 多文件模式: train_and_eval_coles_universal('train/', 'val/', config)
            # model, metrics_tracker = train_and_eval_coles_universal(train_path, val_path, config)
            
            # 记录普通训练模式完成时间
            print_time_point("普通训练模式完成", normal_mode_start_time)
            
            # 训练完成后的额外信息
            debug_print(f"\n=== 普通训练总结 ===")
            debug_print(f"训练完成时间: {datetime.now()}")
            if metrics_tracker.val_metrics:
                best_epoch = metrics_tracker.epochs[metrics_tracker.val_metrics.index(max(metrics_tracker.val_metrics))]
                debug_print(f"最佳验证指标出现在第 {best_epoch} 个epoch")
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