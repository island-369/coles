import json
import os
import glob
import pandas as pd
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import ptls


@rank_zero_only
def print_ptls_info():
    print(f"\n=== PTLS库信息 ===")
    print(f"PTLS路径: {ptls.__file__}")
    print(f"=== PTLS库信息结束 ===\n")


def load_jsonl_as_dataframe_new_format(jsonl_path):
    """从单个jsonl文件加载新格式数据为DataFrame
    新格式：每行是一个字典，字典中的'trans'键存储用户交易序列
    每个交易序列中的元素是列表，按固定顺序存储字段值
    """
    # 定义字段名称和顺序
    field_names = [
        '发卡机构地址', '发卡机构银行', '卡等级', 'year', 'month', 'day', 
        'hour', 'minutes', 'seconds', 'unix_timestamp', '收单机构地址', 
        '收单机构银行', 'cups_交易代码', '交易渠道', 'cups_服务点输入方式', 
        'cups_应答码', 'cups_商户类型', 'cups_连接方式', 'cups_受卡方名称地址', 
        '交易金额', 
    ]
    
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            user_data = json.loads(line)  # 每行是一个字典
            # 从字典中提取'trans'键对应的交易序列
            if 'trans' in user_data:
                user_trx = user_data['trans']
                # 将列表格式转换为字典格式
                converted_trx_list = []
                for trx_array in user_trx:
                    if isinstance(trx_array, list) and len(trx_array) >= len(field_names):
                        # 将列表转换为字典
                        trx_dict = {}
                        for i, field_name in enumerate(field_names):
                            trx_dict[field_name] = trx_array[i]
                        converted_trx_list.append(trx_dict)
                    else:
                        print(f"Warning: Invalid transaction format in line: {line.strip()}")
                        continue
                data.append(converted_trx_list)
            else:
                # 如果没有'trans'键，跳过这行或者抛出警告
                print(f"Warning: 'trans' key not found in line: {line.strip()}")
                continue

    # 展平成 DataFrame，同时添加 client_id
    all_records = []
    for client_id, trx_list in enumerate(data):       
        for trx in trx_list:
            trx['client_id'] = client_id
            all_records.append(trx)
    df = pd.DataFrame(all_records)

    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_jsonl_as_dataframe(jsonl_path):
    """从单个jsonl文件加载数据为DataFrame"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            user_trx = json.loads(line)  # 每行是一个用户的交易序列
            data.append(user_trx)

    # 展平成 DataFrame，同时添加 client_id
    all_records = []
    for client_id, trx_list in enumerate(data):       
        for trx in trx_list:
            trx['client_id'] = client_id
            all_records.append(trx)
    df = pd.DataFrame(all_records)

    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 构造标准化时间字段 'trans_date' , df[trans_date][0]:  2025-04-01 19:27:40
    # df['trans_date'] = pd.to_datetime(
    #     df[['year', 'month', 'day', 'hour', 'minutes', 'seconds']],
    #     errors='coerce'  # 避免不合法时间导致 crash
    # )
    return df


def load_jsonl_from_directory(directory_path):
    """从目录下的多个jsonl文件加载数据为DataFrame
    
    Args:
        directory_path (str): 包含jsonl文件的目录路径
        
    Returns:
        pd.DataFrame: 合并后的DataFrame，包含所有文件的数据
    """
    # 获取目录下所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"在目录 {directory_path} 中没有找到任何jsonl文件")
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件: {[os.path.basename(f) for f in jsonl_files]}")
    
    all_data = []
    global_client_id = 0  # 全局客户ID计数器，确保不同文件间的client_id不重复
    
    for file_path in sorted(jsonl_files):  # 排序确保处理顺序一致
        print(f"正在处理文件: {os.path.basename(file_path)}")
        
        # 读取单个文件的数据
        file_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                user_trx = json.loads(line)  # 每行是一个用户的交易序列
                file_data.append(user_trx)
        
        # 展平成记录，同时分配全局唯一的client_id
        for trx_list in file_data:
            for trx in trx_list:
                trx['client_id'] = global_client_id
                all_data.append(trx)
            global_client_id += 1  # 每个用户序列分配一个唯一ID
        
        print(f"文件 {os.path.basename(file_path)} 处理完成，包含 {len(file_data)} 个用户序列")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    print(f"总共加载了 {len(df)} 条交易记录，来自 {global_client_id} 个用户")
    
    # 将字符串格式的时间字段转换为数值类型
    time_cols = ['year', 'month', 'day', 'hour', 'minutes', 'seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df