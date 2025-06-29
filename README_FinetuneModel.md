# CoLES下游微调模型使用指南

本文档介绍如何使用预训练的CoLES模型进行下游任务的微调训练。

## 📁 文件结构

```
├── downstream_finetune_model.py    # 微调模型核心实现
├── finetune_example.py            # 完整的微调训练示例
├── simple_finetune_test.py        # 简化的功能测试脚本
└── README_FinetuneModel.md        # 本文档
```

## 🚀 快速开始

### 1. 基本使用

```python
from downstream_finetune_model import CoLESFinetuneModule, create_finetune_model
from ptls.nn import TransformerSeqEncoder

# 创建或加载预训练编码器
seq_encoder = load_your_pretrained_encoder()

# 创建微调模型
model = create_finetune_model(
    seq_encoder=seq_encoder,
    n_classes=2,                    # 二分类
    head_hidden_dims=[256, 128],    # MLP头隐藏层维度
    dropout=0.3,
    lr=1e-4,
    freeze_epochs=5                 # 前5个epoch冻结编码器
)

# 使用PyTorch Lightning进行训练
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_dataloader, val_dataloader)
```

### 2. 运行测试

```bash
# 运行功能测试
python simple_finetune_test.py

# 运行完整训练示例
python finetune_example.py
```

## 🏗️ 模型架构

### 整体结构

```
输入数据 (PaddedBatch)
    ↓
预训练编码器 (Backbone)
    ↓
特征向量 [B, D]
    ↓
MLP分类头 (Head)
    ↓
分类logits [B, n_classes]
```

### 核心组件

1. **主干网络 (Backbone)**
   - 使用预训练的PTLS编码器（如TransformerSeqEncoder）
   - 支持冻结/解冻参数
   - 输出固定维度的特征向量

2. **分类头 (Head)**
   - 多层MLP结构
   - 支持自定义隐藏层维度
   - 包含Dropout和激活函数

3. **训练策略**
   - 支持渐进式解冻（先冻结backbone几个epoch）
   - 多种优化器和学习率调度器
   - 类别权重和标签平滑

## ⚙️ 配置参数

### CoLESFinetuneModule 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seq_encoder` | nn.Module | - | 预训练的序列编码器 |
| `n_classes` | int | 2 | 分类类别数 |
| `head_hidden_dims` | list | None | MLP头隐藏层维度 |
| `dropout` | float | 0.2 | Dropout概率 |
| `lr` | float | 1e-3 | 学习率 |
| `weight_decay` | float | 1e-5 | 权重衰减 |
| `freeze_encoder` | bool | False | 是否初始冻结编码器 |
| `freeze_epochs` | int | 0 | 冻结编码器的epoch数 |
| `optimizer_type` | str | 'adam' | 优化器类型 ('adam', 'adamw') |
| `scheduler_type` | str | None | 调度器类型 ('cosine', 'plateau', None) |
| `class_weights` | Tensor | None | 类别权重 |
| `label_smoothing` | float | 0.0 | 标签平滑参数 |

### MLPHead 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征维度 |
| `hidden_dims` | list | None | 隐藏层维度列表 |
| `n_classes` | int | 2 | 输出类别数 |
| `dropout` | float | 0.2 | Dropout概率 |
| `activation` | str | 'relu' | 激活函数类型 |

## 🔧 高级功能

### 1. 渐进式解冻

```python
model = CoLESFinetuneModule(
    seq_encoder=encoder,
    freeze_epochs=5  # 前5个epoch冻结编码器
)
```

### 2. 类别权重处理

```python
# 处理类别不平衡
class_weights = torch.tensor([0.3, 0.7])  # 类别0权重小，类别1权重大
model = CoLESFinetuneModule(
    seq_encoder=encoder,
    class_weights=class_weights
)
```

### 3. 学习率调度

```python
model = CoLESFinetuneModule(
    seq_encoder=encoder,
    optimizer_type='adamw',
    scheduler_type='cosine',  # 余弦退火
    lr=1e-4
)
```

### 4. 加载预训练权重

```python
model = CoLESFinetuneModule(seq_encoder=encoder)
model.load_pretrained_encoder(
    checkpoint_path='./pretrained_model.ckpt',
    strict=False  # 允许部分匹配
)
```

### 5. 获取特征嵌入

```python
# 获取数据的嵌入表示
embeddings, labels = model.get_embeddings(dataloader)
print(f"嵌入维度: {embeddings.shape}")
```

## 📊 训练监控

### 自动记录的指标

- **训练指标**: `train_loss`, `train_acc`
- **验证指标**: `val_loss`, `val_acc`, `val_f1`
- **二分类额外指标**: `val_auc`

### 使用TensorBoard

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir='./logs', name='finetune')
trainer = pl.Trainer(logger=logger)
```

### 模型检查点

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=3,
    filename='best_model_{epoch:02d}_{val_acc:.3f}'
)
```

## 🎯 最佳实践

### 1. 数据准备
- 确保输入数据格式为PaddedBatch
- 标签应为LongTensor类型
- 合理设置batch_size（推荐16-64）

### 2. 训练策略
- 使用渐进式解冻：先冻结backbone 3-10个epoch
- 设置较小的学习率（1e-4到1e-5）
- 使用权重衰减防止过拟合
- 启用梯度裁剪（gradient_clip_val=1.0）

### 3. 超参数调优
```python
# 推荐的超参数组合
config = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'dropout': 0.3,
    'freeze_epochs': 5,
    'head_hidden_dims': [256, 128],
    'optimizer_type': 'adamw',
    'scheduler_type': 'cosine'
}
```

### 4. 性能优化
- 使用混合精度训练：`precision='16-mixed'`
- 启用pin_memory：`pin_memory=True`
- 合理设置num_workers

## 🐛 常见问题

### Q1: 如何处理维度不匹配？
```python
# 检查编码器输出维度
print(f"编码器输出维度: {seq_encoder.embedding_size}")

# 确保MLP头输入维度正确
model = CoLESFinetuneModule(
    seq_encoder=seq_encoder,
    head_hidden_dims=[seq_encoder.embedding_size // 2, 64]
)
```

### Q2: 如何处理类别不平衡？
```python
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(labels), 
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
```

### Q3: 训练过程中loss不下降？
- 检查学习率是否过大或过小
- 确认数据预处理是否正确
- 尝试调整freeze_epochs
- 检查梯度是否正常传播

### Q4: 如何保存和加载模型？
```python
# 保存
trainer.save_checkpoint('model.ckpt')

# 加载
model = CoLESFinetuneModule.load_from_checkpoint(
    'model.ckpt',
    seq_encoder=seq_encoder  # 需要重新提供编码器
)
```

## 📈 性能基准

### 典型性能指标
- **收敛速度**: 通常在10-30个epoch内收敛
- **内存使用**: 取决于batch_size和序列长度
- **训练时间**: 与数据集大小和模型复杂度相关

### 优化建议
1. 使用预训练模型可显著提升性能
2. 渐进式解冻比直接微调效果更好
3. 适当的正则化可防止过拟合

## 🔗 相关文档

- [PTLS官方文档](https://github.com/dllllb/pytorch-lifestream)
- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)
- [下游数据加载器文档](./README_DownstreamDataLoader.md)

## 📝 更新日志

- **v1.0.0**: 初始版本，支持基本微调功能
- 支持多种优化器和调度器
- 支持渐进式解冻和类别权重
- 完整的训练监控和日志记录