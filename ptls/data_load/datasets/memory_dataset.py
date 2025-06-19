import logging
from typing import Iterable, List

import torch

from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

logger = logging.getLogger(__name__)


class MemoryMapDataset(torch.utils.data.Dataset):
    def __init__(self, data, i_filters: List[Iterable] = None):
        if i_filters is None:
            self.processed_data = [rec for rec in data]
        else:
            post_processor_filter = IterableChain(*i_filters)
            self.processed_data = [rec for rec in post_processor_filter(data)]
        logger.info(f'Loaded {len(self.processed_data)} records')

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        from debug_utils import get_debugger
        debugger = get_debugger()
        
        processed_data = self.processed_data[item]
        
        # 记录数据访问（仅在DEBUG级别显示详细内容）
        debugger.log_data_sample(
            processed_data,
            f"MemoryMapDataset[{item}] - Data Access",
            level='DEBUG',
            max_fields=20,
            max_elements=5,
            component='MemoryMapDataset'
        )
        
        return processed_data



class MemoryIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, i_filters=None):
        """Memory-based iterable dataset with optional filters.
        
        Parameters
        ----------
        data : iterable
            Source data to iterate over
        i_filters : List[Iterable], optional
            List of filters to apply to the data pipeline
        """
        if i_filters is None:
            i_filters = []
        self.data = data
        self.post_processor_filter = IterableChain(ToTorch(), *i_filters)
        

    def __iter__(self):
        """Iterate over the data with applied filters."""
        from debug_utils import get_debugger
        debugger = get_debugger()
        
        count = 0
        for rec in self.post_processor_filter(self.data):
            # 记录数据访问（仅在DEBUG级别显示详细内容）
            # print(f"MemoryIterableDataset[{count}] - Data Access")
            debugger.log_data_sample(
                rec,
                f"MemoryIterableDataset[{count}] - Data Access",
                level='DEBUG',
                max_fields=20,
                max_elements=5,
                component='MemoryMapDataset'
            )
            count += 1
            yield rec

