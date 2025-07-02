# UniversalFeatureEncoder 集成到 CoLES 训练

本项目将 `UniversalFeatureEncoder` 成功集成到了 CoLES (Contrastive Learning for Event Sequences) 训练框架中，保持了原有的数据处理逻辑和子序列处理功能。

## 文件说明

### 核心文件
- `encode_3.py`: 包含 `UniversalFeatureEncoder` 的实现
- `train_coles_universal.py`: 集成了 UniversalFeatureEncoder 的 CoLES 训练脚本
- `test.py`: 主要的测试和训练入口文件
- `config_universal_example.json`: UniversalFeatureEncoder 的配置示例

### 关键组件

#### 1. UniversalTrxEncoder
这是一个包装类，将 `UniversalFeatureEncoder` 适配到 ptls 框架的 `TrxEncoder` 接口：

```python
class UniversalTrxEncoder(nn.Module):
    def __init__(self, feature_config, emb_dim_cfg=None, num_fbr_cfg=None, 
                 feature_fusion='concat', field_transformer_args=None, 
                 embeddings_noise=0.0, linear_projection_size=None):
```

**主要功能：**
- 兼容 ptls 的数据格式
- 支持 embedding 噪声注入
- 可选的线性投影层
- 保持与原 CoLES 框架的接口一致性

#### 2. 特征类型支持

`UniversalFeatureEncoder` 支持以下特征类型：

- **数值型特征 (numerical)**: 使用 frequency-based representation
- **类别型特征 (categorical)**: 使用 embedding 层
- **时间特征 (time)**: 使用专门的时间 embedding
- **文本特征 (text)**: 支持 BERT tokenizer 和平均池化

#### 3. 特征融合方式

支持两种特征融合方式：
- `concat`: 简单拼接所有特征
- `field_transformer`: 使用 Field Transformer 进行特征交互

## 使用方法

### 1. 基本使用

直接运行测试文件：
```bash
python test.py
```

这将使用 `UniversalFeatureEncoder` 进行 CoLES 训练。

### 2. 自定义配置

可以修改 `config.json` 或创建新的配置文件：

```json
{
    "model_config": {
        "feature_fusion": "concat",  // 或 "field_transformer"
        "model_dim": 512,
        "n_heads": 8,
        "n_layers": 6
    },
    "universal_encoder_config": {
        "emb_dim_cfg": {
            "交易渠道": 16,
            "cups_交易金额": 32
        },
        "num_fbr_cfg": {
            "cups_交易金额": {"L": 8, "d": 32}
        }
    }
}
```

### 3. 直接调用训练函数

```python
from train_coles_universal import train_and_eval_coles_universal
from config import init_config

config = init_config()
model = train_and_eval_coles_universal('train.jsonl', 'val.jsonl', config)
```

## 配置参数说明

### emb_dim_cfg
为每个特征指定 embedding 维度：
```python
emb_dim_cfg = {
    '交易渠道': 16,      # 类别特征的embedding维度
    '卡等级': 8,         # 类别特征的embedding维度
    'merchant_desc': 64  # 文本特征的embedding维度
}
```

### num_fbr_cfg
为数值特征配置 frequency-based representation：
```python
num_fbr_cfg = {
    'cups_交易金额': {
        'L': 8,    # 频率基数的数量
        'd': 32    # 输出维度
    }
}
```

### field_transformer_args
配置 Field Transformer 参数（当 feature_fusion='field_transformer' 时）：
```python
field_transformer_args = {
    'nhead': 4,        # 注意力头数
    'ff_dim': 256,     # 前馈网络维度
    'dropout': 0.1     # dropout率
}
```

## 与原始 CoLES 的区别

### 保持不变的部分
- 数据预处理逻辑 (`PandasDataPreprocessor`)
- 子序列切分策略 (`SampleSlices`)
- 训练循环和优化器配置
- 模型保存和加载

### 改进的部分
- **更灵活的特征编码**: 支持多种特征类型和编码方式
- **更强的特征表示**: frequency-based representation 用于数值特征
- **特征交互**: 可选的 Field Transformer 进行特征间交互
- **更好的可配置性**: 通过配置文件灵活调整各种参数

## 性能优势

1. **更好的数值特征处理**: frequency-based representation 比简单的 embedding 更适合连续数值
2. **特征交互**: Field Transformer 可以学习特征间的复杂交互
3. **灵活的架构**: 支持不同的特征融合策略
4. **保持原有优势**: 继承了 CoLES 的对比学习和子序列处理能力

## 注意事项

1. **特征配置**: 确保 `feature_config` 中的特征名称与数据中的列名一致
2. **维度匹配**: 使用 `field_transformer` 时，所有特征的 embedding 维度必须相同
3. **内存使用**: Field Transformer 会增加内存使用，根据硬件情况调整 batch_size
4. **数据格式**: 保持与原始 CoLES 相同的数据格式（jsonl 文件）

## 示例输出

训练完成后，模型会保存为 `final_model_universal.pth`，可以用于后续的推理和评估。

```
save_model
模型已保存到: ./output/final_model_universal.pth
```

这样就成功将 `UniversalFeatureEncoder` 集成到了 CoLES 训练框架中，既保持了原有的强大功能，又增加了更灵活和强大的特征编码能力。