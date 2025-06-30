# -*- coding: utf-8 -*-
"""
下游微调模型实现

基于预训练的CoLES模型进行下游任务的微调，支持二分类和多分类任务。
核心思路：
1. 主干部分（Backbone）：使用预训练的PTLS encoder（如TransformerSeqEncoder）
2. 任务头部（Head）：MLP分类器
3. LightningModule：封装训练/验证逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
from typing import Optional, Dict, Any


class MLPHead(nn.Module):
    """MLP分类头
    
    Args:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表，如[512, 256]
        n_classes: 分类类别数
        dropout: Dropout概率
        activation: 激活函数类型
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = None,
                 n_classes: int = 2,
                 dropout: float = 0.2,
                 activation: str = 'relu'):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
            
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.activation = activation_map[activation]
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)


class CoLESFinetuneModule(pl.LightningModule):
    """CoLES微调模块
    
    Args:
        seq_encoder: 预训练的序列编码器
        n_classes: 分类类别数
        head_hidden_dims: MLP头隐藏层维度列表
        dropout: Dropout概率
        lr: 学习率
        weight_decay: 权重衰减
        freeze_encoder: 是否冻结编码器参数
        freeze_epochs: 冻结编码器的epoch数（0表示不冻结）
        optimizer_type: 优化器类型 ('adam', 'adamw')
        scheduler_type: 学习率调度器类型 ('cosine', 'plateau', None)
        class_weights: 类别权重，用于处理类别不平衡：
        label_smoothing: 标签平滑参数
    """
    
    def __init__(self,
                 seq_encoder,
                 n_classes: int = 2,
                 head_hidden_dims: list = None,
                 dropout: float = 0.2,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 freeze_encoder: bool = False,
                 freeze_epochs: int = 0,
                 optimizer_type: str = 'adam',
                 scheduler_type: Optional[str] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 debug_print_func=None):
        super().__init__()
        
        # 保存超参数（排除seq_encoder避免序列化问题）
        self.save_hyperparameters(ignore=["seq_encoder", "class_weights", "debug_print_func"])
        
        self.seq_encoder = seq_encoder
        self.n_classes = n_classes
        self.freeze_epochs = freeze_epochs
        self.current_epoch_count = 0
        self.debug_print = debug_print_func if debug_print_func else print
        
        # 获取编码器输出维度
        if hasattr(seq_encoder, 'embedding_size'):
            encoder_dim = seq_encoder.embedding_size
        elif hasattr(seq_encoder, 'output_size'):
            encoder_dim = seq_encoder.output_size
        elif hasattr(seq_encoder, 'seq_encoder') and hasattr(seq_encoder.seq_encoder, 'embedding_size'):
            encoder_dim = seq_encoder.seq_encoder.embedding_size
        else:
            raise ValueError("Cannot determine encoder output dimension")
            
        # 创建分类头
        if head_hidden_dims is None:
            head_hidden_dims = [encoder_dim // 2]
            
        self.classifier = MLPHead(
            input_dim=encoder_dim,
            hidden_dims=head_hidden_dims,
            n_classes=n_classes,
            dropout=dropout
        )
        
        # 损失函数
        
        #如果存在class_weights，就把它注册到当前模型中
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
            
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )
        
        # 初始冻结编码器
        if freeze_encoder or freeze_epochs > 0:
            self._freeze_encoder()
            
        # 用于存储验证结果
        self.validation_step_outputs = []
        
    def _freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.seq_encoder.parameters():
            param.requires_grad = False
            
    def _unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.seq_encoder.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """前向传播
        
        Args:
            x: PaddedBatch格式的输入数据
            
        Returns:
            logits: 分类logits
        """
        # 通过编码器获取特征
        features = self.seq_encoder(x)
        
        # 处理PaddedBatch输出
        if hasattr(features, 'payload'):
            features = features.payload
            
        # 如果是序列输出，取CLS token或平均池化
        if features.dim() == 3:  # [B, T, D]
            features = features[:, 0, :]  # 取第一个token（CLS）
            # features = features.mean(dim=1)  # 平均池化
        # 分类
        logits = self.classifier(features)
        return logits
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算预测和概率
        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        
        # 存储结果用于epoch结束时计算指标
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': y,
            'probs': probs
        })
        
        return loss
        
    def on_validation_epoch_end(self):
        """验证epoch结束时计算指标"""
        if not self.validation_step_outputs:
            return
            
        # 获取当前步数
        current_step = self.global_step
        
        # 收集所有预测结果
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        
        # 计算平均损失
        avg_loss = all_losses.mean()
        
        # 计算准确率
        acc = (all_preds == all_targets).float().mean()
        
        # 计算F1分数
        f1 = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        
        # 计算AUC（仅对二分类）
        auc_value = 0.5  # 默认值
        if self.n_classes == 2:
            try:
                # 检查是否有多个类别
                unique_targets = torch.unique(all_targets)
                if len(unique_targets) > 1:
                    auc_value = roc_auc_score(all_targets.cpu().numpy(), all_probs[:, 1].cpu().numpy())
                    self.log('val_auc', auc_value, prog_bar=True, sync_dist=True)
                else:
                    self.debug_print(f"Warning: Only one class present in validation set: {unique_targets.tolist()}")
                    self.log('val_auc', auc_value, prog_bar=True, sync_dist=True)  # 默认值
            except Exception as e:
                self.debug_print(f"Error calculating AUC: {e}")
                self.log('val_auc', auc_value, prog_bar=True, sync_dist=True)  # 默认值
            
        # 记录指标
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, sync_dist=True)
        
        # 输出验证指标到日志
        self.debug_print(f"\n=== 验证指标 (Step {current_step}) ===")
        self.debug_print(f"验证损失 (val_loss): {avg_loss:.6f}")
        self.debug_print(f"验证准确率 (val_acc): {acc:.6f}")
        self.debug_print(f"验证F1分数 (val_f1): {f1:.6f}")
        if self.n_classes == 2:
            self.debug_print(f"验证AUC (val_auc): {auc_value:.6f}")
        self.debug_print(f"验证样本数量: {len(all_targets)}")
        self.debug_print(f"=== 验证完成 ===\n")
        
        # 清空输出
        self.validation_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)
        
    def on_test_epoch_end(self):
        """测试epoch结束"""
        self.on_validation_epoch_end()
        
    def on_train_epoch_end(self):
        """训练epoch结束时检查是否需要解冻编码器"""
        self.current_epoch_count += 1
        
        if (self.freeze_epochs > 0 and 
            self.current_epoch_count == self.freeze_epochs):
            print(f"\nUnfreezing encoder at epoch {self.current_epoch_count}")
            self._unfreeze_encoder()
            
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 选择优化器
        if self.hparams.optimizer_type == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
            
        # 选择学习率调度器
        if self.hparams.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.hparams.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
            
    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """加载预训练编码器权重
        
        Args:
            checkpoint_path: 预训练模型检查点路径
            strict: 是否严格匹配参数名称
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 提取编码器相关的状态字典
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 只取出主干seq_encoder部分，去掉"_seq_encoder."前缀
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_seq_encoder.'):
                new_k = k[len('_seq_encoder.'):]
                encoder_dict[new_k] = v
            # 如果有历史其它prefix，也可加
            elif k.startswith('seq_encoder.'):
                new_k = k[len('seq_encoder.'):]
                encoder_dict[new_k] = v
                
        # 加载权重
        missing_keys, unexpected_keys = self.seq_encoder.load_state_dict(
            encoder_dict, strict=strict
        )
        
        print(f"[INFO] missing keys: {missing_keys}")
        print(f"[INFO] unexpected keys: {unexpected_keys}")
        print(f"Successfully loaded pretrained encoder from {checkpoint_path}")
        
    def get_embeddings(self, dataloader):
        """获取数据的嵌入表示
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            embeddings: 嵌入向量
            labels: 对应的标签
        """
        self.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                
                # 获取编码器输出
                features = self.seq_encoder(x)
                if hasattr(features, 'payload'):
                    features = features.payload
                if features.dim() == 3:
                    features = features[:, 0, :]
                    
                embeddings.append(features.cpu())
                labels.append(y.cpu())
                
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def create_finetune_model(seq_encoder, 
                         n_classes: int = 2,
                         **kwargs) -> CoLESFinetuneModule:
    """创建微调模型的便捷函数
    
    Args:
        seq_encoder: 预训练的序列编码器
        n_classes: 分类类别数
        **kwargs: 其他参数
        
    Returns:
        微调模型实例
    """
    return CoLESFinetuneModule(
        seq_encoder=seq_encoder,
        n_classes=n_classes,
        **kwargs
    )