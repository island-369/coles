# 优化的调试输出系统

## 概述

这个优化的调试系统提供了更好的控制和格式化的调试输出，解决了之前调试信息混乱和难以控制的问题。

## 主要特性

### 1. 分级调试控制
- **DEBUG**: 显示所有调试信息，包括详细的数据内容
- **INFO**: 显示重要的处理步骤和批次信息
- **WARNING**: 只显示警告和错误
- **ERROR**: 只显示错误信息
- **NONE**: 关闭所有调试输出

### 2. 组件级别控制
可以为不同的组件设置不同的调试级别：
- `ColesDataset`: 控制ColesDataset相关的调试输出
- `MemoryMapDataset`: 控制MemoryMapDataset相关的调试输出
- `collate_fn`: 控制数据整理函数的调试输出

### 3. 智能格式化
- 自动识别PyTorch张量、NumPy数组、字典、列表等数据类型
- 根据数据维度智能选择显示方式
- 可配置的显示元素数量限制
- 支持只显示张量形状而不显示具体数值

### 4. 性能监控
- 可选的性能监控功能
- 记录操作耗时
- 存储性能数据用于分析

## 文件结构

```
├── debug_utils.py          # 核心调试工具类
├── debug_config.py         # 调试配置文件
├── debug_usage_example.py  # 使用示例
└── DEBUG_README.md         # 本文档
```

## 配置说明

### debug_config.py 主要配置项

```python
# 全局调试级别
DEBUG_LEVEL = 'INFO'

# 性能监控
ENABLE_PERFORMANCE_MONITORING = True

# 显示限制
MAX_TENSOR_ELEMENTS = 3  # 张量显示的最大元素数
MAX_DICT_FIELDS = 6      # 字典显示的最大字段数
MAX_LIST_ELEMENTS = 5    # 列表显示的最大元素数

# 组件特定调试级别
COMPONENT_DEBUG_LEVELS = {
    'ColesDataset': 'INFO',
    'MemoryMapDataset': 'DEBUG',
    'collate_fn': 'INFO',
}

# 输出控制
SUPPRESS_EMPTY_DATA_WARNINGS = True  # 抑制空数据警告
SHOW_TENSOR_SHAPES_ONLY = False      # 只显示张量形状

# 日志文件（可选）
LOG_TO_FILE = False
LOG_FILE_PATH = 'debug_output.log'
```

## 使用方法

### 1. 基本使用

```python
from debug_utils import get_debugger

# 获取调试器实例
debugger = get_debugger()

# 记录数据样本
debugger.log_data_sample(
    data_dict,
    "数据样本标题",
    level='DEBUG',
    component='MyComponent'
)

# 记录批次信息
debugger.log_batch_info(
    batch_size=32,
    title="批次处理",
    additional_info={"key": "value"},
    level='INFO'
)

# 记录处理步骤
debugger.log_processing_step(
    "步骤名称",
    input_data,
    "处理结果描述",
    level='INFO'
)

# 记录性能
debugger.log_performance(
    "操作名称",
    duration_seconds,
    level='INFO'
)

# 记录错误
debugger.log_error(
    "错误描述",
    exception=e,
    component='MyComponent'
)
```

### 2. 在现有代码中的集成

调试系统已经集成到以下组件中：

- **ColesDataset.__getitem__()**: 记录数据访问和分割操作
- **ColesDataset.collate_fn()**: 记录批次整理过程
- **MemoryMapDataset.__getitem__()**: 记录内存数据访问

### 3. 调试级别控制示例

```python
# 修改 debug_config.py
DEBUG_LEVEL = 'DEBUG'  # 显示所有调试信息

# 或者为特定组件设置调试级别
COMPONENT_DEBUG_LEVELS = {
    'ColesDataset': 'DEBUG',      # 显示ColesDataset的详细信息
    'MemoryMapDataset': 'INFO',   # 只显示MemoryMapDataset的重要信息
    'collate_fn': 'WARNING',      # 只显示collate_fn的警告和错误
}
```

## 输出格式示例

### 数据样本输出
```
[14:30:25] ColesDataset[0] - Raw Features
  event_time: shape=(5,), data=[1.0, 2.0, 3.0]
  amount: shape=(5, 2), data=[[100.0, 200.0], [150.0, 250.0], [120.0, 180.0]]
  mcc_code: shape=(5,), data=[5411, 5812, 5999]
```

### 批次信息输出
```
[14:30:25] [BATCH] Collate Function - Input Batch
  Batch Size: 32
  event_time: shape=(32, 10)
  amount: shape=(32, 10, 2)
  mcc_code: shape=(32, 10)
```

### 处理步骤输出
```
[14:30:25] [STEP] Split Operation (idx=0)
  Output: 3 splits generated
```

### 性能监控输出
```
[14:30:25] [PERF] Data Loading: 0.1234s
```

## 优势对比

### 优化前的问题
- 调试输出混乱，难以阅读
- 无法控制输出级别
- 格式不统一
- 性能影响大
- 无法针对特定组件控制

### 优化后的改进
- ✅ 清晰的分级输出控制
- ✅ 组件级别的精细控制
- ✅ 统一的格式化输出
- ✅ 可配置的显示限制
- ✅ 可选的性能监控
- ✅ 支持文件日志输出
- ✅ 智能的数据类型识别
- ✅ 时间戳标记

## 最佳实践

1. **开发阶段**: 设置 `DEBUG_LEVEL = 'DEBUG'` 查看详细信息
2. **测试阶段**: 设置 `DEBUG_LEVEL = 'INFO'` 关注重要步骤
3. **生产环境**: 设置 `DEBUG_LEVEL = 'WARNING'` 或 `'ERROR'`
4. **性能调优**: 启用 `ENABLE_PERFORMANCE_MONITORING = True`
5. **问题排查**: 针对特定组件设置 `DEBUG` 级别

## 运行示例

```bash
# 运行使用示例
python debug_usage_example.py

# 修改配置后重新运行训练脚本
python train_coles_universal.py
```

这个优化的调试系统提供了更好的控制性和可读性，帮助您更有效地监控和调试数据加载过程。