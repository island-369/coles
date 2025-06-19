#!/usr/bin/env python3

from debug_utils import get_debugger

print("Testing debug_utils...")
debugger = get_debugger()
print(f"Debugger created, log_file: {debugger.log_file}")

# 测试数据样本记录
debugger.log_data_sample({'test': 'data', 'number': 123}, 'Test Message')
print("log_data_sample called")

# 测试批次信息记录
debugger.log_batch_info(32, 'Test Batch')
print("log_batch_info called")

# 测试处理步骤记录
debugger.log_processing_step('Test Step', {'input': 'data'}, 'output info')
print("log_processing_step called")

print("Debug test completed")