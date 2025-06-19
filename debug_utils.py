import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union




try:
    from debug_config import (
        DEBUG_LEVEL, ENABLE_PERFORMANCE_MONITORING, 
        MAX_TENSOR_ELEMENTS, MAX_DICT_FIELDS, MAX_LIST_ELEMENTS,
        COMPONENT_DEBUG_LEVELS, SUPPRESS_EMPTY_DATA_WARNINGS,
        SHOW_TENSOR_SHAPES_ONLY, LOG_TO_FILE, LOG_FILE_PATH
    )
except ImportError:
    # 默认配置
    DEBUG_LEVEL = 'INFO'
    ENABLE_PERFORMANCE_MONITORING = True
    MAX_TENSOR_ELEMENTS = 3
    MAX_DICT_FIELDS = 6
    MAX_LIST_ELEMENTS = 5
    COMPONENT_DEBUG_LEVELS = {}
    SUPPRESS_EMPTY_DATA_WARNINGS = True
    SHOW_TENSOR_SHAPES_ONLY = False
    LOG_TO_FILE = False
    LOG_FILE_PATH = 'debug_output.log'

class DebugLogger:
    """优化的调试输出工具类"""
    
    def __init__(self, level=None):
        """
        初始化调试器
        
        Args:
            level: 调试级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'NONE')
        """
        self.level = level or DEBUG_LEVEL
        self.levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'NONE': 4}
        self.current_level = self.levels.get(self.level, 1)
        self.performance_data = {} if ENABLE_PERFORMANCE_MONITORING else None
        self.log_file = None
        
        if LOG_TO_FILE:
            try:
                self.log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
            except Exception:
                pass
        
        self.logger = None
    
    def _should_log(self, level, component=None):
        """检查是否应该记录此级别的日志"""
        # 检查组件特定的调试级别
        if component and component in COMPONENT_DEBUG_LEVELS:
            component_level = self.levels.get(COMPONENT_DEBUG_LEVELS[component], 1)
            return self.levels.get(level, 1) >= component_level
        
        return self.levels.get(level, 1) >= self.current_level
    
    def _format_tensor_data(self, value: Any, max_elements: int = 5) -> str:
        """格式化张量数据"""
        try:
            if hasattr(value, 'numel') and value.numel() > 0:
                # PyTorch tensor
                if len(value.shape) == 0:  # scalar
                    return f"scalar({value.item()})"
                elif len(value.shape) == 1:
                    if value.numel() <= max_elements:
                        return f"[{value.tolist()}]"
                    else:
                        return f"[{value[:max_elements].tolist()}...] (len={value.numel()})"
                elif len(value.shape) == 2:
                    rows_to_show = min(2, value.shape[0])
                    return f"[{value[:rows_to_show].tolist()}...] (shape={list(value.shape)})"
                else:
                    return f"tensor(shape={list(value.shape)}, first_10={value.flatten()[:10].tolist()})"
            
            elif hasattr(value, 'size') and value.size > 0:
                # numpy array
                if len(value.shape) == 0:  # scalar
                    return f"scalar({value.item()})"
                elif len(value.shape) == 1:
                    if value.size <= max_elements:
                        return f"[{value.tolist()}]"
                    else:
                        return f"[{value[:max_elements].tolist()}...] (len={value.size})"
                elif len(value.shape) == 2:
                    rows_to_show = min(2, value.shape[0])
                    return f"[{value[:rows_to_show].tolist()}...] (shape={list(value.shape)})"
                else:
                    return f"array(shape={list(value.shape)}, first_10={value.flatten()[:10].tolist()})"
            
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                # 其他可迭代对象
                try:
                    sample_data = list(value)[:max_elements] if hasattr(value, '__len__') else []
                    if len(sample_data) < len(value):
                        return f"{sample_data}... (len={len(value)})"
                    else:
                        return str(sample_data)
                except:
                    return str(type(value))
            
            else:
                return str(value)
        
        except Exception as e:
            return f"<error formatting: {e}>"
    
    def log_data_sample(self, data, title="Data Sample", level='DEBUG', max_fields=None, max_elements=None, component=None):
        """记录数据样本"""
        if not self._should_log(level, component):
            return
        
        if max_fields is None:
            max_fields = MAX_DICT_FIELDS
        if max_elements is None:
            max_elements = MAX_TENSOR_ELEMENTS
            
        timestamp = time.strftime("%H:%M:%S")
        message = f"\n[{timestamp}] {title}"
        
        if isinstance(data, dict):
            if not data:
                if not SUPPRESS_EMPTY_DATA_WARNINGS:
                    message += "\n  (empty dict)"
            else:
                items = list(data.items())[:max_fields]
                for key, value in items:
                    formatted_value = self._format_tensor_data(value, max_elements)
                    message += f"\n  {key}: {formatted_value}"
                
                if len(data) > max_fields:
                    message += f"\n  ... ({len(data) - max_fields} more fields)"
        else:
            formatted_data = self._format_tensor_data(data, max_elements)
            message += f"\n  {formatted_data}"
        
        message += "\n"
        self._output_message(message)
    
    def _output_message(self, message):
        """输出消息到日志文件，不输出到终端"""
        # 只输出到日志文件，避免终端重复输出
        if self.log_file:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass
    
    def log_batch_info(self, batch_size, title, additional_info=None, level='INFO', component=None):
        """记录批次信息"""
        if not self._should_log(level, component):
            return
        
        timestamp = time.strftime("%H:%M:%S")
        message = f"\n[{timestamp}] [BATCH] {title}\n"
        message += f"  Batch Size: {batch_size}\n"
        
        if additional_info and isinstance(additional_info, dict):
            for key, value in additional_info.items():
                message += f"  {key}: {value}\n"
        
        self._output_message(message)
    
    def log_processing_step(self, step_name, input_data, output_info, level='INFO', component=None):
        """记录处理步骤"""
        if not self._should_log(level, component):
            return
        
        timestamp = time.strftime("%H:%M:%S")
        message = f"\n[{timestamp}] [STEP] {step_name}\n"
        
        # 特殊处理Split Operation的输出
        if step_name == 'Split Operation' and isinstance(output_info, list):
            message += f"  Generated {len(output_info)} subsequences\n"
            message += "  Subsequence Details:\n"
            for i, sub_seq in enumerate(output_info):
                message += f"    Subsequence {i+1}:\n"
                if isinstance(sub_seq, dict):
                    for key, value in list(sub_seq.items())[:MAX_DICT_FIELDS]:
                        formatted_value = self._format_tensor_data(value, MAX_TENSOR_ELEMENTS)
                        message += f"      {key}: {formatted_value}\n"
                    if len(sub_seq) > MAX_DICT_FIELDS:
                        message += f"      ... ({len(sub_seq) - MAX_DICT_FIELDS} more fields)\n"
                else:
                    formatted_value = self._format_tensor_data(sub_seq, MAX_TENSOR_ELEMENTS)
                    message += f"      {formatted_value}\n"
        else:
            message += f"  Output: {output_info}\n"
        
        # 可选：显示输入数据的简要信息
        if level == 'DEBUG' and input_data:
            if isinstance(input_data, dict) and len(input_data) <= 3:
                for key, value in input_data.items():
                    if hasattr(value, 'shape'):
                        message += f"  Input {key}: {value.shape}\n"
                    else:
                        message += f"  Input {key}: {type(value).__name__}\n"
        
        self._output_message(message)
    
    def log_performance(self, operation, duration, level='INFO', component=None):
        """记录性能信息"""
        if not self._should_log(level, component) or not ENABLE_PERFORMANCE_MONITORING:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        message = f"\n[{timestamp}] [PERF] {operation}: {duration:.4f}s\n"
        
        # 存储性能数据用于分析
        if self.performance_data is not None:
            if operation not in self.performance_data:
                self.performance_data[operation] = []
            self.performance_data[operation].append(duration)
        
        self._output_message(message)
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None, component=None):
        """记录错误信息"""
        if not self._should_log('ERROR', component):
            return
        
        timestamp = time.strftime("%H:%M:%S")
        message = f"\n[{timestamp}] [ERROR] {error_msg}\n"
        if exception:
            message += f"  Exception: {str(exception)}\n"
        
        self._output_message(message)


# 全局调试器实例
_global_debugger = None

def get_debugger(level='INFO') -> DebugLogger:
    """获取全局调试器实例"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = DebugLogger(level)
    return _global_debugger

def set_debug_level(level: str):
    """设置调试级别"""
    debugger = get_debugger()
    debugger.level = level
    debugger.current_level = debugger.levels.get(level, 1)