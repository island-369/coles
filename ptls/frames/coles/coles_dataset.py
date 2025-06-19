from functools import reduce
from operator import iadd

import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit


class ColesDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.CoLESModule

    Parameters
    ----------
    data:
        source data with feature dicts
    splitter:
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time='event_time',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from debug_utils import get_debugger
        debugger = get_debugger()
        
        feature_arrays = self.data[idx]
        
        # 记录原始特征数组（仅在DEBUG级别）
        debugger.log_data_sample(
            feature_arrays, 
            f"ColesDataset[{idx}] - Raw Features", 
            level='DEBUG',
            max_fields=20,
            max_elements=3,
            component='ColesDataset'
        )
        
        splits = self.get_splits(feature_arrays)
        
        # 记录分割结果（INFO级别，但限制输出）
        if len(splits) > 0:
            debugger.log_processing_step(
                f"Split Operation (idx={idx})",
                feature_arrays,
                f"{len(splits)} splits generated",
                level='INFO',
                component='ColesDataset'
            )
            
            # 只在DEBUG级别显示第一个分割的详细内容
            if isinstance(splits[0], dict):
                debugger.log_data_sample(
                    splits[0],
                    f"ColesDataset[{idx}] - Split[0] Sample",
                    level='DEBUG',
                    max_fields=20,
                    max_elements=3,
                    component='ColesDataset'
                )
        
        return splits

    def __iter__(self):
        for feature_arrays in self.data:
            # print(f"__iter__: {feature_arrays}")
            yield self.get_splits(feature_arrays)

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        
        # 创建分割后的子序列
        subsequences = [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]
        
        # 添加调试输出显示分割后的子序列
        from debug_utils import get_debugger
        debugger = get_debugger()
        
        # 记录分割操作的详细信息
        debugger.log_processing_step(
            "Split Operation",
            {"original_length": len(local_date), "split_count": len(indexes)},
            subsequences,  # 直接传递子序列列表
            level='DEBUG',
            component='split_operation'
        )
        
        return subsequences

    @staticmethod
    def collate_fn(batch):
        from debug_utils import get_debugger
        debugger = get_debugger()
        
        # 记录批次信息（INFO级别）
        debugger.log_batch_info(
            len(batch),
            f"Collate Function - Input Batch",
            level='INFO',
            component='collate_fn'
        )
        
        # 记录第一个样本（仅DEBUG级别）
        if len(batch) > 0 and len(batch[0]) > 0:
            debugger.log_data_sample(
                batch[0][0] if isinstance(batch[0][0], dict) else batch[0],
                "Collate - First Sample",
                level='DEBUG',
                max_fields=20,
                max_elements=3,
                component='collate_fn'
            )
        
        class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples]
        
        # 记录类标签信息
        debugger.log_processing_step(
            "Class Labels Generation",
            {"total_labels": len(class_labels), "unique_classes": len(set(class_labels))},
            f"Generated {len(class_labels)} class labels",
            level='INFO',
            component='collate_fn'
        )
        
        batch = reduce(iadd, batch)
        
        # 记录reduce后的批次
        if len(batch) > 0:
            debugger.log_processing_step(
                "Batch Reduction",
                batch[0] if isinstance(batch[0], dict) else {"batch_size": len(batch)},
                f"Reduced to {len(batch)} samples",
                level='INFO',
                component='collate_fn'
            )
        
        padded_batch = collate_feature_dict(batch)
        
        # 记录填充后的批次信息（INFO级别）
        if hasattr(padded_batch, 'payload') and isinstance(padded_batch.payload, dict):
            batch_info = {}
            for key, value in padded_batch.payload.items():
                if hasattr(value, 'shape'):
                    batch_info[key] = f"shape={value.shape}"
            
            debugger.log_batch_info(
                len(batch),
                "Collate - Padded Batch",
                additional_info=batch_info,
                level='INFO',
                component='collate_fn'
            )
            
            # 详细的张量内容（仅DEBUG级别）
            debugger.log_data_sample(
                padded_batch.payload,
                "Collate - Padded Batch Content",
                level='DEBUG',
                max_fields=20,
                max_elements=2,
                component='collate_fn'
            )
        
        return padded_batch, torch.LongTensor(class_labels)


class ColesIterableDataset(ColesDataset, torch.utils.data.IterableDataset):
    # __len__ = None
    pass
