# Debug Configuration
# 调试配置文件 - 控制调试输出的级别和行为

# 调试级别设置
# 'DEBUG': 显示所有调试信息，包括详细的数据内容
# 'INFO': 显示重要的处理步骤和批次信息
# 'WARNING': 只显示警告和错误
# 'ERROR': 只显示错误信息
# 'NONE': 关闭所有调试输出
DEBUG_LEVEL = 'DEBUG'  # 设置为DEBUG级别以查看分割子序列详情

# 性能监控设置
ENABLE_PERFORMANCE_MONITORING = True

# 输出格式设置
MAX_TENSOR_ELEMENTS = 3  # 张量显示的最大元素数
MAX_DICT_FIELDS =22      # 字典显示的最大字段数
MAX_LIST_ELEMENTS = 5    # 列表显示的最大元素数

# 特定组件的调试控制
COMPONENT_DEBUG_LEVELS = {
    'ColesDataset': 'INFO',        # ColesDataset的调试级别
    'MemoryMapDataset': 'DEBUG',   # MemoryMapDataset的调试级别
    'collate_fn': 'INFO',          # collate_fn的调试级别
    'data_loading': 'INFO',        # 数据加载的调试级别
    'split_operation': 'DEBUG',    # 分割操作的调试级别
}

# 输出控制
SUPPRESS_EMPTY_DATA_WARNINGS = True  # 是否抑制空数据警告
SHOW_TENSOR_SHAPES_ONLY = False      # 是否只显示张量形状而不显示内容

# 日志文件设置（可选）
LOG_TO_FILE = True
LOG_FILE_PATH = 'logs_xqy/debug_utils_output.log'