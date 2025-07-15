import torch
import torch.nn as nn
import math

# 数值型特征的频率编码表示
# 基于傅里叶变换的思想，将连续数值转换为周期性特征
class FrequencyBasedRepresentation(nn.Module):
    """数值型特征的频率编码器
    
    将连续的数值特征转换为频率域的表示，类似于位置编码的思想。
    通过不同频率的正弦和余弦函数来捕获数值的周期性模式。
    
    Args:
        L (int): 频率的数量，默认8。决定了编码的精度和复杂度
        out_dim (int): 输出维度，默认16。如果与输入维度(2*L)不同，会添加线性层
    """
    def __init__(self, L=8, out_dim=16):
        super().__init__()
        self.L = L  # 频率数量
        self.out_dim = out_dim  # 输出维度
        self.in_dim = 2 * L  # 输入维度 = 2*L (sin + cos)
        # 如果输出维度与输入维度相同，使用恒等映射；否则使用线性层
        self.lin = nn.Identity() if out_dim == self.in_dim else nn.Linear(self.in_dim, out_dim)

    def forward(self, x):
        """前向传播：将数值转换为频率编码
        
        处理流程：
        1. 扩展维度准备计算
        2. 生成不同频率的正弦和余弦值
        3. 拼接sin和cos特征
        4. 可选的线性变换到目标维度
        """
        x = x.unsqueeze(-1)  # 增加一个维度用于频率计算
        pi_x = x * math.pi  # 乘以π准备三角函数计算
        # 生成不同频率：2^0, 2^1, ..., 2^(L-1)
        freq = 2 ** torch.arange(self.L, device=x.device).float() * pi_x
        sin = torch.sin(freq)  # 正弦分量
        cos = torch.cos(freq)  # 余弦分量
        feat = torch.cat([sin, cos], dim=-1)  # 拼接sin和cos，形状: (..., 2L)
        # 确保输入数据类型与线性层权重一致，避免混合精度训练时的类型不匹配错误
        if hasattr(self.lin, 'weight'):
            feat = feat.to(self.lin.weight.dtype)
        return self.lin(feat)  # 线性变换到目标维度: (..., out_dim)

# 字段级别的Transformer编码器
# 用于学习不同特征字段之间的交互关系
class FieldTransformerBlock(nn.Module):
    """字段级Transformer编码器
    
    在特征字段维度上应用自注意力机制，学习不同特征之间的交互关系。
    这种方法可以捕获特征间的复杂依赖关系，提升模型的表达能力。
    
    Args:
        field_num (int): 特征字段的数量
        emb_dim (int): 每个特征的嵌入维度
        nhead (int): 多头注意力的头数，默认2
        ff_dim (int): 前馈网络的隐藏层维度，默认128
        dropout (float): Dropout概率，默认0.1
    """
    def __init__(self, field_num, emb_dim, nhead=2, ff_dim=128, dropout=0.1):
        super().__init__()
        # 创建Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,  # 模型维度
            nhead=nhead,      # 注意力头数
            dim_feedforward=ff_dim,  # 前馈网络维度
            dropout=dropout,  # Dropout概率
            batch_first=True  # 批次维度在第一位
        )
        # 使用单层Transformer编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.field_num = field_num  # 字段数量
        self.emb_dim = emb_dim      # 嵌入维度

    def forward(self, feats):  
        """前向传播：在字段维度上应用注意力机制
        
        Args:
            feats: 输入特征张量，形状为 (B, T, field, d)
                  B=批次大小, T=序列长度, field=字段数, d=嵌入维度
                  
        Returns:
            处理后的特征张量，形状保持 (B, T, field, d)
            
        处理流程：
        1. 重塑张量以适应Transformer输入格式
        2. 在字段维度上应用自注意力
        3. 恢复原始形状
        """
        B, T, F, D = feats.shape
        # 将(B,T,F,D)重塑为(B*T,F,D)，以便对每个时间步的字段应用注意力
        x = feats.view(B * T, F, D)
        # 在字段维度(F)上应用Transformer，学习字段间的交互
        out = self.encoder(x)
        # 恢复原始形状(B,T,F,D)
        out = out.view(B, T, F, D)
        return out

# 通用特征编码器 - 核心组件
# 支持多种类型特征的统一编码和融合
class UniversalFeatureEncoder(nn.Module):
    """通用特征编码器
    
    这是一个通用的特征编码器，能够处理多种类型的特征：
    - 数值型特征：使用频率编码(FrequencyBasedRepresentation)
    - 类别型特征：使用嵌入层(Embedding)
    - 时间型特征：使用嵌入层处理时间索引
    - 文本型特征：使用嵌入层+平均池化+MLP
    
    支持两种特征融合方式：
    - concat: 简单拼接所有特征
    - field_transformer: 使用Transformer学习特征间交互
    
    Args:
        feature_config (dict): 特征配置字典，定义每个特征的类型和参数
        emb_dim_cfg (dict): 嵌入维度配置，为每个特征指定嵌入维度
        num_fbr_cfg (dict): 数值特征的频率编码配置
        feature_fusion (str): 特征融合方式，'concat'或'field_transformer'
        field_transformer_args (dict): FieldTransformer的参数配置
    """
    def __init__(self, feature_config,
                 emb_dim_cfg=None,
                 num_fbr_cfg=None,
                 feature_fusion='concat',
                 field_transformer_args=None):
        super().__init__()
        self.feature_config = feature_config  # 特征配置
        self.emb_layers = nn.ModuleDict()     # 存储各特征的编码层
        self.feature_names = list(feature_config.keys())  # 特征名称列表
        self.feature_fusion = feature_fusion  # 特征融合方式
        emb_dim_cfg = emb_dim_cfg or {}      # 嵌入维度配置
        num_fbr_cfg = num_fbr_cfg or {}      # 数值特征频率编码配置
        out_dims = []                        # 记录各特征的输出维度
        self.field2outdim = {}               # 特征名到输出维度的映射

        # 遍历所有特征，为每种类型创建相应的编码层
        for key, cfg in feature_config.items():
            # 1. 数值型特征处理
            if cfg['type'] == 'numerical':
                # 获取频率编码的参数配置
                L = num_fbr_cfg.get(key, {}).get('L', 8)      # 频率数量，默认8
                d = num_fbr_cfg.get(key, {}).get('d', 16)     # 输出维度，默认16
                # 创建频率编码层，将连续数值转换为频率域表示
                self.emb_layers[key] = FrequencyBasedRepresentation(L=L, out_dim=d)
                self.field2outdim[key] = d
                out_dims.append(d)
                
            # 2. 类别型特征处理
            elif cfg['type'] == 'categorical':
                emb_dim = emb_dim_cfg.get(key, 16)  # 嵌入维度，默认16
                n_cat = len(cfg['choices'])         # 类别数量
                # 创建嵌入层，将类别索引映射到稠密向量
                self.emb_layers[key] = nn.Embedding(n_cat, emb_dim)
                self.field2outdim[key] = emb_dim
                out_dims.append(emb_dim)
                
            # 3. 时间型特征处理
            elif cfg['type'] == 'time':
                emb_dim = emb_dim_cfg.get(key, 16)  # 嵌入维度，默认16
                # 创建嵌入层，处理时间索引（如年、月、日、小时等）
                self.emb_layers[key] = nn.Embedding(cfg['range'], emb_dim)
                self.field2outdim[key] = emb_dim
                out_dims.append(emb_dim)
                
            # 4. 文本型特征处理
            elif cfg['type'] == 'text':
                emb_dim = emb_dim_cfg.get(key, 16)  # 嵌入维度，默认16
                # 创建词嵌入层，支持padding
                self.emb_layers[key] = nn.Embedding(cfg['vocab_size'], emb_dim, padding_idx=cfg['pad_idx'])
                # 创建MLP层，用于文本特征的进一步处理
                self.emb_layers[f"{key}_mlp"] = nn.Linear(emb_dim, emb_dim)
                self.field2outdim[key] = emb_dim
                out_dims.append(emb_dim)
            else:
                raise NotImplementedError(f"不支持的特征类型: {cfg['type']}")

        self.field_dims = out_dims.copy()  # 保存各特征的维度信息
        self.n_fields = len(self.feature_names)  # 特征字段数量

        # 配置特征融合方式
        if feature_fusion == 'field_transformer':
            # 使用Transformer进行特征融合
            emb_dim = out_dims[0]
            # 检查：field_transformer要求所有特征的嵌入维度必须相同
            assert all(x == emb_dim for x in out_dims), \
                f'使用field_transformer时，各特征Embedding维度需一致，当前维度: {out_dims}'
            self.emb_dim = emb_dim
            args = field_transformer_args or {}  # 获取Transformer参数
            # 创建字段级Transformer，学习特征间的交互关系
            self.field_transform = FieldTransformerBlock(self.n_fields, self.emb_dim, **args)
            # 输出维度 = 字段数 × 嵌入维度
            self.out_dim = self.n_fields * self.emb_dim
            
        elif feature_fusion == 'concat':
            # 使用简单拼接进行特征融合
            # 输出维度 = 所有特征维度之和
            self.out_dim = sum(out_dims)
        else:
            raise ValueError(f"未知的feature_fusion选项: {feature_fusion}，支持的选项: ['concat', 'field_transformer']")

    def forward(self, xdict):
        """前向传播：编码所有特征并进行融合
        
        Args:
            xdict (dict): 特征字典，键为特征名，值为对应的张量数据
                         每个特征张量的形状通常为 (batch_size, seq_len, ...)
                         
        Returns:
            torch.Tensor: 融合后的特征表示
                         形状取决于融合方式：
                         - concat: (B, T, sum_of_all_dims)
                         - field_transformer: (B, T, n_fields * emb_dim)
        """
        feats = []  # 存储各特征的编码结果
        
        # 遍历所有特征，按类型进行编码
        for key in self.feature_names:
            cfg = self.feature_config[key]
            
            # 文本特征的特殊处理
            if cfg['type'] == 'text':
                txt = xdict[key]  # 文本序列，形状: (B, T, text_len)
                # 1. 词嵌入
                emb = self.emb_layers[key](txt)  # (B, T, text_len, emb_dim)
                # 2. 创建mask，忽略padding token
                mask = (txt != cfg['pad_idx']).float().unsqueeze(-1)  # (B, T, text_len, 1)
                # 3. 加权求和（忽略padding）
                emb_sum = (emb * mask).sum(2)  # (B, T, emb_dim)
                denom = mask.sum(2).clamp(min=1)  # (B, T, 1)，防止除零
                # 4. 平均池化
                emb_mean = emb_sum / denom  # (B, T, emb_dim)
                # 5. MLP进一步处理
                mlp_layer = self.emb_layers[f"{key}_mlp"]
                # 确保输入数据类型与线性层权重一致，避免混合精度训练时的类型不匹配错误
                # emb_mean = emb_mean.to(mlp_layer.weight.dtype)
                # emb = mlp_layer(emb_mean)
                feats.append(emb)
                
            # 数值特征处理
            elif cfg['type'] == 'numerical':
                # 使用频率编码处理数值特征
                feats.append(self.emb_layers[key](xdict[key].float()))
                
            # 类别特征和时间特征处理
            else:
                # 直接使用嵌入层编码
                feats.append(self.emb_layers[key](xdict[key]))
        
        # 特征融合
        if self.feature_fusion == 'concat':
            # 方式1: 简单拼接所有特征
            outfeats = torch.cat(feats, dim=-1)  # (B, T, sum_dims)
            return outfeats
            
        elif self.feature_fusion == 'field_transformer':
            # 方式2: 使用Transformer学习特征间交互
            # 将特征列表堆叠为4D张量: (B, T, field, d)
            field_feats = torch.stack(feats, dim=2)
            # 应用字段级Transformer
            field_feats = self.field_transform(field_feats)
            # 展平为3D张量: (B, T, F*d)
            out = field_feats.view(field_feats.shape[0], field_feats.shape[1], -1)
            return out
        else:
            raise ValueError(f"未知的特征融合方式: {self.feature_fusion}，支持的方式: ['concat', 'field_transformer']")