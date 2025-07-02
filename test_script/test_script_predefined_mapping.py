#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：使用预定义映射进行类别编码

这个示例展示了如何使用feature_config中的idx_map来对类别字段进行编码，
而不是根据输入数据动态生成映射。
"""

import pandas as pd
from ptls.preprocessing import (
    PandasDataPreprocessor, 
    extract_predefined_mappings_from_feature_config,
    get_categorical_columns_from_feature_config
)

# 示例feature_config（类似于debug_output_20250612_163425.log中的内容）
feature_config = {
    '交易渠道': {
        'type': 'categorical',
        'idx_map': {'网银': 1, 'ATM': 2, '柜台': 3, '手机银行': 4}
    },
    'cups_商户类型': {
        'type': 'categorical', 
        'idx_map': {'超市': 1, '餐饮': 2, '加油站': 3, '医院': 4}
    },
    '交易金额': {
        'type': 'numerical'
    },
    'unix_timestamp': {
        'type': 'time'
    }
}

# 示例数据
data = pd.DataFrame({
    'client_id': [1, 1, 2, 2, 3],
    'unix_timestamp': [1640995200, 1641081600, 1641168000, 1641254400, 1641340800],
    '交易渠道': ['网银', 'ATM', '手机银行', '网银', '未知渠道'],  # 包含未知值
    'cups_商户类型': ['超市', '餐饮', '加油站', '超市', '其他'],  # 包含未知值
    '交易金额': [100.0, 50.0, 200.0, 75.0, 300.0]
})

print("原始数据:")
print(data)
print()

# 提取预定义映射
predefined_mappings = extract_predefined_mappings_from_feature_config(feature_config)
categorical_columns = get_categorical_columns_from_feature_config(feature_config)

print("提取的预定义映射:")
for col, mapping in predefined_mappings.items():
    print(f"{col}: {mapping}")
print()

print("类别列:")
print(categorical_columns)
print()

# 创建预处理器
preprocessor = PandasDataPreprocessor(
    col_id='client_id',
    col_event_time='unix_timestamp',
    event_time_transformation='none',
    cols_category=categorical_columns,
    cols_category_with_mapping=predefined_mappings,
    cols_numerical=['交易金额'],
    return_records=True
)

# 进行预处理
processed_data = preprocessor.fit_transform(data)

print("预处理后的数据:")
for i, record in enumerate(processed_data):
    print(f"Record {i+1}:")
    for key, value in record.items():
        print(f"  {key}: {value}")
    print()

# 检查字典大小
print("类别字典大小:")
for ct in preprocessor.cts_category:
    print(f"{ct.col_name_original}: {ct.dictionary_size}")