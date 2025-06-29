import os
import sys
import time
from datetime import datetime
import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tqdm import tqdm

# 1. 各类配置和特征工程
from config import init_config
from encode_3 import UniversalFeatureEncoder
from ptls.preprocessing import (
    extract_predefined_mappings_from_feature_config,
    get_categorical_columns_from_feature_config,
    PandasDataPreprocessor
)
from ptls.nn import TransformerSeqEncoder
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import NoSplit

# 2. 下游相关
from downstream_finetune_model import create_finetune_model
from downstream_data_loader import DownstreamTestDataset

# 3. 日志
def setup_logging():
    log_dir = 'logs_finetune'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/test_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    return log_file, log_filename

log_file, log_filename = setup_logging()
@rank_zero_only
def debug_print(*args, **kwargs):
    if log_file is not None:
        print(*args, file=log_file, **kwargs)
        log_file.flush()

def main():
    config = init_config()
    model_config = config['model_config']
    feature_config = config['feature_config']
    universal_encoder_config = config.get('universal_encoder_config', {})
    train_config = config['train_config']
    finetune_config = config.get('finetune_config', {})

    # ---- 路径参数 ----
    test_path = 'downstream_dataset/test'
    batch_size = train_config['batch_size']
    model_dim = model_config['model_dim']
    n_heads = model_config['n_heads']
    n_layers = model_config['n_layers']
    emb_dim_cfg = universal_encoder_config.get('emb_dim_cfg', {})
    num_fbr_cfg = universal_encoder_config.get('num_fbr_cfg', {})
    feature_fusion = model_config.get('feature_fusion', 'concat')

    # ---- 特征与预处理 ----
    predefined_mappings = extract_predefined_mappings_from_feature_config(feature_config)
    categorical_columns = get_categorical_columns_from_feature_config(feature_config)
    preprocessor = PandasDataPreprocessor(
        col_id='client_id',
        col_event_time='unix_timestamp',
        event_time_transformation='none',
        cols_category=categorical_columns,
        cols_category_with_mapping=predefined_mappings,
        cols_numerical=['交易金额'],
        return_records=True,
    )
    def build_dataset_from_df(df):
        return ColesDataset(
            MemoryMapDataset(data=df, i_filters=[SeqLenFilter(min_seq_len=1)]),
            splitter=NoSplit(),
        )

    # ---- 数据集/Loader ----
    test_ds = DownstreamTestDataset(
        data_root=test_path,
        preprocessor=preprocessor,
        dataset_builder=build_dataset_from_df,
        debug_print_func=debug_print,
        shuffle_files=False,
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=DownstreamTestDataset.collate_fn
    )

    # ---- 构建模型（结构参数全部与训练一致） ----
    from train_coles_universal import UniversalTrxEncoder
    trx_encoder = UniversalTrxEncoder(
        feature_config=feature_config,
        emb_dim_cfg=emb_dim_cfg,
        num_fbr_cfg=num_fbr_cfg,
        feature_fusion=feature_fusion,
        linear_projection_size=model_dim,
        embeddings_noise=0.0
    )
    seq_encoder = TransformerSeqEncoder(
        trx_encoder=trx_encoder,
        input_size=None,
        is_reduce_sequence=True,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1,
        dim_hidden=model_dim*4
    )
    n_classes = test_ds.get_num_classes()
    downstream_model = create_finetune_model(
        seq_encoder=seq_encoder,
        n_classes=n_classes,
        lr=finetune_config.get('learning_rate', 1e-4),
        weight_decay=finetune_config.get('weight_decay', 1e-5),
        head_hidden_dims=finetune_config.get('mlp_hidden_dims', [model_dim // 2]),
        dropout=finetune_config.get('mlp_dropout', 0.2),
        freeze_encoder=finetune_config.get('freeze_encoder', False),
        freeze_epochs=finetune_config.get('freeze_epochs', 0),
        optimizer_type=finetune_config.get('optimizer_type', 'adam'),
        scheduler_type=finetune_config.get('scheduler_type', None),
        label_smoothing=finetune_config.get('label_smoothing', 0.0),
    )
    # ---- 加载finetune权重 ----
    finetune_ckpt = './output_finetune/finetuned_downstream.ckpt'
    state = torch.load(finetune_ckpt, map_location='cpu', weights_only=False)
    downstream_model.load_state_dict(state['state_dict'] if 'state_dict' in state else state, strict=False)
    downstream_model.eval()
    downstream_model = downstream_model.cuda() if torch.cuda.is_available() else downstream_model

    debug_print(f"[{datetime.now()}] 开始测试")
    # ---- 正式推理 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds, all_labels, all_probs = [], [], []
    
    debug_print("开始处理测试数据...")
    print("开始处理测试数据...")
    
    with torch.no_grad():
        batch_iter = tqdm(test_loader, desc="Testing", unit="batch")
        for i_batch, batch in enumerate(batch_iter, 1):
            x, y = batch
            x = x.to(device)
            logits = downstream_model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            
            # 实时刷新指标
            if i_batch % 1 == 0:
                y_true_tmp = torch.cat(all_labels).numpy()
                y_pred_tmp = torch.cat(all_preds).numpy()
                acc = accuracy_score(y_true_tmp, y_pred_tmp)
                f1 = f1_score(y_true_tmp, y_pred_tmp, average='weighted')
                postfix = {
                    "batch": i_batch,
                    "acc": f"{acc:.4f}",
                    "f1": f"{f1:.4f}"
                }
                if n_classes == 2 and len(y_true_tmp) > 20:
                    try:
                        y_prob_tmp = torch.cat(all_probs).numpy()
                        auc = roc_auc_score(y_true_tmp, y_prob_tmp[:, 1])
                        postfix["auc"] = f"{auc:.4f}"
                    except Exception:
                        postfix["auc"] = "nan"
                batch_iter.set_postfix(postfix)
                
                # 同时记录到日志
                log_msg = f"Batch {i_batch}: Acc={acc:.4f}, F1={f1:.4f}"
                if "auc" in postfix:
                    log_msg += f", AUC={postfix['auc']}"
                debug_print(log_msg)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # ---- 输出指标 ----
    output_dir = './output_finetune_test'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_pred.npy'), y_pred)
    np.save(os.path.join(output_dir, 'test_true.npy'), y_true)
    np.save(os.path.join(output_dir, 'test_prob.npy'), y_prob)

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # 汇总指标到字典
    result_summary = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),  # 转list便于json化
        'model_weight': os.path.abspath(finetune_ckpt),  # 权重的绝对路径
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_classes': int(n_classes),
        'num_samples': len(y_true)
    }
    
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:,1])
        result_summary['auc'] = float(auc)
    
    # 保存整体测试结果到JSON
    result_path = os.path.join(output_dir, 'test_evaluation_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=4, ensure_ascii=False)
    
    debug_print("Test Accuracy:", accuracy)
    debug_print("Test F1-score:", f1)
    if n_classes == 2:
        debug_print("Test AUC:", result_summary['auc'])
    debug_print("Confusion matrix:\n" + str(cm))
    debug_print("Saved overall test metrics to", result_path)
    
    print('Test Accuracy:', accuracy)
    print('Test F1:', f1)
    if n_classes == 2:
        print('Test AUC:', result_summary['auc'])
    print('Confusion matrix:\n', cm)
    print('Saved overall test metrics to', result_path)

    if log_file is not None:
        log_file.close()

if __name__ == "__main__":
    main()