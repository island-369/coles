import os
import sys
import time
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.profilers import SimpleProfiler


# 1. 配置模块和特征工程导入
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

# 2. 下游模型和数据集导入
from downstream_finetune_model import create_finetune_model
from downstream_data_loader import DownstreamBinaryClassificationDataset, DistributedValTestDataset

# 3. 日志设置
def setup_logging():
    log_dir = 'logs_finetune'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/debug_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    return log_file, log_filename

# 初始化日志
log_file, log_filename = setup_logging()

@rank_zero_only
def debug_print(*args, **kwargs):
    if log_file is not None:
        print(*args, file=log_file, **kwargs); log_file.flush()

# ---- 定义从 DF 到 ColesDataset 的转换 ----
def build_dataset_from_df(df):
    return ColesDataset(
        MemoryMapDataset(data=df, i_filters=[SeqLenFilter(min_seq_len=1,max_seq_len=5000)]),
        
        splitter=NoSplit(),   # 下游任务使用NoSplit，保持完整序列
    )



# 4. 下游训练流程
def main():
    config = init_config()
    model_config = config['model_config']
    feature_config = config['feature_config']
    universal_encoder_config = config.get('universal_encoder_config', {})
    train_config = config['train_config']
    finetune_config = config.get('finetune_config', {})
    data_config = config['data_config']

    # 路径配置
    train_path = 'downstream_dataset/train'
    val_path = 'downstream_dataset/val'

    debug_print(f"[{datetime.now()}] 下游微调开始")
    debug_print(f"加载特征配置&模型参数...")
    emb_dim_cfg = universal_encoder_config.get('emb_dim_cfg', {})
    num_fbr_cfg = universal_encoder_config.get('num_fbr_cfg', {})
    feature_fusion = model_config.get('feature_fusion', 'concat')
    batch_size = train_config['batch_size']
    model_dim = model_config['model_dim']
    n_heads = model_config['n_heads']
    n_layers = model_config['n_layers']

    # ---- 数据预处理器 ----
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



    # ---- 数据集/数据加载器（与预训练同风格，使用你自己的下游数据集类） ----
    # 训练集使用无限循环的DownstreamBinaryClassificationDataset
    train_ds = DownstreamBinaryClassificationDataset(
        data_root=train_path,
        preprocessor=preprocessor,
        dataset_builder=build_dataset_from_df,
        debug_print_func=debug_print,
        shuffle_files=True,
        num_classes=2,  # 二分类任务
    )
    # 验证集使用全局分片的DistributedValTestDataset，避免多进程数据不均衡问题
    val_ds = DistributedValTestDataset(
        data_root=val_path,
        preprocessor=preprocessor,
        dataset_builder=build_dataset_from_df,
        debug_print_func=debug_print,
        num_classes=2,  # 二分类任务
    )
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=DownstreamBinaryClassificationDataset.collate_fn,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=DistributedValTestDataset.collate_fn,
        persistent_workers=True,
    )

    # ---- 构建和加载模型（与预训练代码中保持一致） ----
    from universal_trx_encoder import UniversalTrxEncoder  # 使用独立模块避免日志冲突
    
    trx_encoder = UniversalTrxEncoder(
        feature_config=feature_config,
        emb_dim_cfg=emb_dim_cfg,
        num_fbr_cfg=num_fbr_cfg,
        feature_fusion=feature_fusion,
        linear_projection_size=model_dim,
        embeddings_noise=0.0,  # 下游一般不要噪声
        debug_print_func=debug_print
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

    # ---- 下游分类头和LightningModule ----
    n_classes = train_ds.get_num_classes()  # 2 for 二分类

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
        debug_print_func=debug_print,
    )

    # ---- 加载预训练pt文件 ----
    pretrained_ckpt = finetune_config.get('pretrained_ckpt', './output_pretrain/continuous_coles_model.ckpt')
    debug_print(f"加载预训练权重: {pretrained_ckpt}")
    downstream_model.load_pretrained_encoder(
        checkpoint_path=pretrained_ckpt,
        strict=False   # 大多数情况下非strict模式
    )
    debug_print("预训练权重加载完毕")

    # ---- Lightning Trainer，基于max_steps训练 ----
    pl.seed_everything(42)
    
    # 基于步数的训练配置
    max_steps = finetune_config.get('max_steps', 10000)  # 默认10000步
    val_check_interval = finetune_config.get('val_check_interval', 500)  # 每500步验证一次
    
    debug_print(f"使用基于步数的训练: max_steps={max_steps}, val_check_interval={val_check_interval}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./output_finetune/checkpoints",
        filename="finetune-step{step:06d}-recall{val_recall:.4f}",  # 文件名可含step
        monitor="val_loss",                            # 按验证集指标保留
        mode="min",
        save_top_k=3,                                  # 最多保留3个最佳
        save_last=True                                 # 总保留最后一次
    )
    
    
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=2,
        strategy='ddp'
        # strategy='deepspeed',
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=True,
        val_check_interval=val_check_interval,
        enable_model_summary=True,
        accumulate_grad_batches=4,           # <- 比如累积4个batch
    )

    debug_print("开始下游Finetune训练...")
    
    # ckpt_last = './output_finetune/finetuned_downstream.ckpt' 
    
    
    trainer.validate(downstream_model,val_loader)
    
    trainer.fit(downstream_model, train_loader, val_loader)
    
    trainer.validate(downstream_model,val_loader)

    # ---- 保存下游模型 ----
    output_dir = './output_finetune'
    os.makedirs(output_dir, exist_ok=True)
    downstream_model_path = os.path.join(output_dir, 'finetuned_downstream.ckpt')
    trainer.save_checkpoint(downstream_model_path)
    debug_print(f"[{datetime.now()}] 下游finetune模型保存于: {downstream_model_path}")

    # 恢复stdout并关闭日志
    if log_file is not None:
        log_file.close()

if __name__ == "__main__":
    main()