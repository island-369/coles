# 下游二分类任务数据加载器

本文档介绍如何使用 `DownstreamBinaryClassificationDataset` 和 `DownstreamTestDataset` 进行下游二分类任务的数据加载。

## 概述

这些数据加载器专门为使用预训练的CoLES模型进行下游二分类任务而设计，支持：

- **文件夹标签结构**：每个文件夹名代表一个类别标签
- **流式处理**：内存占用低，适合大规模数据
- **分布式训练**：支持多GPU和多进程数据加载
- **兼容性**：与现有的预处理器和数据集构建器完全兼容

## 数据格式要求

### 目录结构

```
downstream_data/
├── train/
│   ├── positive/          # 正样本文件夹
│   │   ├── pos_file1.jsonl
│   │   ├── pos_file2.jsonl
│   │   └── ...
│   └── negative/          # 负样本文件夹
│       ├── neg_file1.jsonl
│       ├── neg_file2.jsonl
│       └── ...
└── test/
    ├── positive/
    │   ├── test_pos1.jsonl
    │   └── ...
    └── negative/
        ├── test_neg1.jsonl
        └── ...
```

### 数据文件格式

JSONL文件内的数据格式与训练时完全相同：

```json
{"trans": [["发卡机构地址值", "发卡机构银行值", "卡等级值", "2024", "12", "25", "14", "30", "45", "1735107045", "收单机构地址值", "收单机构银行值", "交易代码值", "交易渠道值", "输入方式值", "应答码值", "商户类型值", "连接方式值", "受卡方名称地址值", "100.50"], [...]], "user_id": "optional_user_id"}
```

## 使用方法

### 1. 基本使用

```python
from downstream_data_loader import DownstreamBinaryClassificationDataset, DownstreamTestDataset
from torch.utils.data import DataLoader

# 创建训练数据集
train_dataset = DownstreamBinaryClassificationDataset(
    data_root="/path/to/train_data",
    preprocessor=your_preprocessor,
    dataset_builder=your_dataset_builder,
    label_mapping={'positive': 1, 'negative': 0},  # 可选
    shuffle_files=True
)

# 创建测试数据集
test_dataset = DownstreamTestDataset(
    data_root="/path/to/test_data",
    preprocessor=your_preprocessor,
    dataset_builder=your_dataset_builder,
    label_mapping=train_dataset.get_label_mapping(),
    shuffle_files=False
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    collate_fn=DownstreamBinaryClassificationDataset.collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=4,
    collate_fn=DownstreamTestDataset.collate_fn
)
```

### 2. 预处理器配置

```python
from ptls.preprocessing import PandasDataPreprocessor

preprocessor = PandasDataPreprocessor(
    col_id='client_id',
    col_event_time='unix_timestamp',
    event_time_transformation='none',
    cols_category=[
        '发卡机构地址', '发卡机构银行', '卡等级', '收单机构地址', '收单机构银行',
        'cups_交易代码', '交易渠道', 'cups_服务点输入方式', 'cups_应答码',
        'cups_商户类型', 'cups_连接方式', 'cups_受卡方名称地址'
    ],
    cols_numerical=['交易金额'],
    cols_datetime=['year', 'month', 'day', 'hour', 'minutes', 'seconds'],
    return_records=True
)
```

### 3. 数据集构建器配置

```python
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices

def create_dataset_builder():
    def dataset_builder(processed_data):
        return ColesDataset(
            data=processed_data,
            splitter=SampleSlices(
                split_count=5,
                cnt_min=10,
                cnt_max=200
            )
        )
    return dataset_builder
```

## 类说明

### DownstreamBinaryClassificationDataset

用于训练阶段的数据集类，特点：

- **无限循环**：使用 `itertools.cycle` 无限循环文件，避免分布式训练死锁
- **标签自动识别**：根据文件夹名自动生成标签映射
- **分布式支持**：支持多GPU训练和多进程数据加载

**主要参数：**
- `data_root`: 数据根目录路径
- `preprocessor`: 数据预处理器
- `dataset_builder`: 数据集构建函数
- `label_mapping`: 标签映射字典（可选）
- `shuffle_files`: 是否随机打乱文件顺序

### DownstreamTestDataset

用于测试/验证阶段的数据集类，特点：

- **单次遍历**：不使用无限循环，每个文件只处理一次
- **确定性**：适合测试和验证阶段的确定性评估
- **继承兼容**：继承自训练数据集，保持接口一致性

## 重要方法

### 获取数据集信息

```python
# 获取标签映射
label_mapping = dataset.get_label_mapping()
print(f"标签映射: {label_mapping}")  # {'positive': 1, 'negative': 0}

# 获取类别数量
num_classes = dataset.get_num_classes()
print(f"类别数量: {num_classes}")  # 2

# 获取文件分布
file_counts = dataset.get_file_count_by_label()
print(f"文件分布: {file_counts}")  # {'positive': 10, 'negative': 8}
```

### 批处理函数

两个数据集都提供了专门的 `collate_fn` 静态方法：

```python
# 返回格式：(features, labels)
features, labels = batch
# features: PaddedBatch 对象，包含特征数据
# labels: torch.LongTensor，包含对应的标签
```

## 注意事项

### 1. 标签映射一致性

确保训练集和测试集使用相同的标签映射：

```python
# 正确做法
train_dataset = DownstreamBinaryClassificationDataset(...)
test_dataset = DownstreamTestDataset(
    ...,
    label_mapping=train_dataset.get_label_mapping()  # 使用训练集的标签映射
)
```

### 2. 预处理器兼容性

使用与预训练模型相同的预处理器配置，确保特征处理的一致性。

### 3. 分布式训练

在分布式环境中，数据集会自动处理文件分片，确保每个进程处理不同的数据子集。

### 4. 内存管理

数据集采用流式处理，逐用户加载数据，有效控制内存使用。

## 完整示例

参考 `downstream_example.py` 文件获取完整的使用示例，包括：

- 数据集创建
- 数据加载器配置
- 训练循环示例
- 评估循环示例

## 故障排除

### 常见问题

1. **文件路径错误**
   - 检查 `data_root` 路径是否正确
   - 确保文件夹结构符合要求

2. **标签映射问题**
   - 确保文件夹名与标签映射一致
   - 检查是否有未知的文件夹名

3. **数据格式错误**
   - 确保JSONL文件格式正确
   - 检查交易字段数量是否匹配

4. **分布式训练问题**
   - 确保所有进程都能访问数据文件
   - 检查文件分布是否均匀

### 调试建议

1. 使用 `debug_print_func` 参数启用详细日志
2. 先在单进程环境下测试
3. 检查数据加载器的第一个批次
4. 验证标签分布是否合理

## 性能优化

1. **调整批次大小**：根据GPU内存调整 `batch_size`
2. **多进程加载**：适当设置 `num_workers`
3. **内存固定**：使用 `pin_memory=True`
4. **文件分布**：确保各类别文件数量相对均衡

通过以上配置和使用方法，你可以高效地进行下游二分类任务的数据加载和模型训练。