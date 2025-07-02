import os
import json

# 配置区
DATA_DIR = './downstream_dataset/train'   # jsonl文件目录
RESULT_FILE = 'valid_users_stat.txt'     # 统计结果文件
TRX_LEN_THRESH = 2                      # 统计"交易数大于等于这个值"的用户

def count_valid_users_in_file(filepath, min_trx_len):
    """统计单个文件中交易数大于等于阈值的用户数"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                data = json.loads(line)
                trans = data.get('trans', [])
                if isinstance(trans, list) and len(trans) >= min_trx_len:
                    count += 1
            except Exception as e:
                continue  # skip malformed lines
    return count

def stat_all_files(data_dir, min_trx_len, result_file):
    """统计所有jsonl文件的有效用户数"""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
    total = 0
    
    with open(result_file, 'w', encoding='utf-8') as fout:
        fout.write(f"交易数大于等于{min_trx_len}的用户统计（每文件/合计）：\n\n")
        
        for fname in files:
            fpath = os.path.join(data_dir, fname)
            n = count_valid_users_in_file(fpath, min_trx_len)
            fout.write(f"{fname}\t{n}\n")
            print(f"{fname}\t{n}")
            total += n
            
        fout.write(f"\n所有文件总用户数（交易数≥{min_trx_len}）：{total}\n")
        print(f"\n所有文件总用户数（交易数≥{min_trx_len}）：{total}")
        
    print(f"\n统计结果已保存到: {result_file}")

if __name__ == '__main__':
    # 检查目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误：目录 {DATA_DIR} 不存在")
        exit(1)
        
    # 检查目录中是否有jsonl文件
    jsonl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
    if not jsonl_files:
        print(f"错误：目录 {DATA_DIR} 中没有找到jsonl文件")
        exit(1)
        
    print(f"开始统计目录 {DATA_DIR} 中的有效用户数...")
    print(f"交易数阈值: {TRX_LEN_THRESH}")
    print(f"找到 {len(jsonl_files)} 个jsonl文件")
    print("-" * 50)
    
    stat_all_files(DATA_DIR, TRX_LEN_THRESH, RESULT_FILE)