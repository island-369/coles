import torch
import torch.nn as nn
from ptls.data_load.padded_batch import PaddedBatch
from encode_3 import UniversalFeatureEncoder

class UniversalTrxEncoder(nn.Module):
    """包装UniversalFeatureEncoder以兼容ptls的TrxEncoder接口
    
    1. 将ptls框架的PaddedBatch数据格式转换为UniversalFeatureEncoder期望的字典格式
    2. 处理特征编码，包括类别特征、时间特征和数值特征
    3. 添加可选的噪声和线性投影层
    4. 将编码结果重新包装为PaddedBatch格式，以便后续的序列编码器使用
    """
    
    def __init__(self, feature_config, emb_dim_cfg=None, num_fbr_cfg=None, 
                 feature_fusion='concat', field_transformer_args=None, 
                 embeddings_noise=0.0, linear_projection_size=None, debug_print_func=None):
        super().__init__()
        
        # 保存特征配置，用于数据验证和处理
        self.feature_config = feature_config
        self.embeddings_noise = embeddings_noise
        self.debug_print = debug_print_func if debug_print_func else lambda *args, **kwargs: None
        
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
        self.debug_print(f"\n=== UniversalTrxEncoder.forward ===")
        self.debug_print(f"Input x type: {type(x)}")
        self.debug_print(f"x attributes: {dir(x)}")
        if hasattr(x, 'payload'):
            self.debug_print(f"x.payload type: {type(x.payload)}")
            self.debug_print(f"x.payload keys: {list(x.payload.keys())}")
        if hasattr(x, 'seq_lens'):
            self.debug_print(f"x.seq_lens: {x.seq_lens}")
        
        # 1: 从PaddedBatch中提取实际数据
        # x.payload是一个字典，包含所有特征的张量数据
        # 每个特征的形状为 [batch_size, seq_len, feature_dim]
        payload = x.payload
        batch_size, seq_len = payload['event_time'].shape
        self.debug_print(f"Batch size: {batch_size}, Seq len: {seq_len}")
        
        # 2: 构建特征字典，进行数据验证和预处理
        feature_dict = {}
        
        for feature_name, feature_info in self.feature_config.items():
            if feature_name in payload:
                feature_values = payload[feature_name]
                
                # 调试信息：打印特征的统计信息
                self.debug_print(f"Feature {feature_name}: min={feature_values.min()}, max={feature_values.max()}, shape={feature_values.shape}")
                
                # 类别特征的范围检查和截断
                # 防止embedding层索引越界导致CUDA错误
                if feature_info['type'] == 'categorical':
                    n_choices = len(feature_info['choices'])  # 类别数量
                    if feature_values.max() >= n_choices:
                        self.debug_print(f"WARNING: Feature {feature_name} has max value {feature_values.max()} >= n_choices ({n_choices}), max value: {feature_values.max()}")
                        # 使用torch.clamp将超出范围的值截断到有效范围[0, n_choices-1]
                        feature_values = torch.clamp(feature_values, 0, n_choices - 1)
                        
                # 时间特征的范围检查和截断
                elif feature_info['type'] == 'time':
                    time_range = feature_info['range']  # 时间特征的范围
                    if feature_values.max() >= time_range:
                        self.debug_print(f"WARNING: Feature {feature_name} has max value {feature_values.max()}  > range ({time_range}), max value: {feature_values.max()}")
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
            # 确保输入数据类型与线性层权重一致，避免混合精度训练时的类型不匹配错误
            encoded = encoded.to(self.projection.weight.dtype)
            encoded = self.projection(encoded)
        
        # 5: 重新包装为PaddedBatch格式
        # 保持原有的序列长度信息，确保与后续的序列编码器兼容
        new_x = PaddedBatch(encoded, x.seq_lens)
        
        self.debug_print(f"Encoded tensor shape: {encoded.shape}")
        self.debug_print(f"Output PaddedBatch type: {type(new_x)}")
        self.debug_print(f"Output seq_lens: {new_x.seq_lens}")
        if hasattr(new_x, 'payload'):
            self.debug_print(f"Output payload shape: {new_x.payload.shape}")
        self.debug_print("=== End UniversalTrxEncoder.forward ===\n")
        
        return new_x