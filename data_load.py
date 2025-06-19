
import json
import random
import torch
from torch.utils.data import Dataset

from config import get_processor


class SeqTransJsonMultiWindowDataset(Dataset):
    def __init__(self, json_path, feature_config, min_hist=2, max_hist=4, min_pred=1, max_pred=3, base_year=2014):
        with open(json_path, 'r', encoding='utf8') as f:
            self.all_users = json.load(f)
        self.feature_config = feature_config
        self.base_year = base_year
        self.min_hist = min_hist
        self.max_hist = max_hist
        self.min_pred = min_pred
        self.max_pred = max_pred

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        user_seq = self.all_users[idx]
        T = len(user_seq)
        

        hist_len = random.randint(self.min_hist, min(self.max_hist, T-2))
        pred_len = random.randint(self.min_pred, min(self.max_pred, T-hist_len))
        max_start = T - hist_len - pred_len
        if max_start < 0:
            start = 0
            hist_len = max(1, T - pred_len)
            pred_len = T - hist_len
        else:
            start = random.randint(0, max_start)
        his_idxs = [start + i for i in range(hist_len)]
        pred_idxs = [start + hist_len + i for i in range(pred_len)]
        x_input = [self.encode_item(user_seq[i]) for i in his_idxs]
        x_target = [self.encode_item(user_seq[i]) for i in pred_idxs]
        return x_input, x_target

    def encode_item(self, item):
        out = {}
        for key, cfg in self.feature_config.items():
            processor = get_processor(cfg)
            out[key] = processor.encode_item(item[key])
        return out

#将不同长度的序列样本整理成 batch，并进行 padding    
def prepare_collate_batch(feature_config):

    def collate_batch(batch):
        nonlocal feature_config
        batch_size = len(batch)
        hist_lens = [len(item[0]) for item in batch]
        pred_lens = [len(item[1]) for item in batch]
        max_hist = max(hist_lens)
        max_pred = max(pred_lens)
        batch_inputs = {}
        batch_targets = {}
        for key, cfg in feature_config.items():
            typ = cfg['type']
            if typ == 'text':
                L = feature_config[key]['max_length']
                batch_inputs[key] = torch.full((batch_size, max_hist, L), feature_config[key]['pad_idx'], dtype=torch.long)
                batch_targets[key] = torch.full((batch_size, max_pred, L), feature_config[key]['pad_idx'], dtype=torch.long)
            else:
                batch_inputs[key] = torch.zeros(batch_size, max_hist).long()
                batch_targets[key] = torch.zeros(batch_size, max_pred).long()
        for b, (x_in, x_tar) in enumerate(batch):
            for t, dic in enumerate(x_in):
                for k in dic:
                    if feature_config[k]['type'] == 'text':
                        batch_inputs[k][b, t, :] = dic[k]
                    else:
                        batch_inputs[k][b, t] = dic[k]
            for h, dic in enumerate(x_tar):
                for k in dic:
                    if feature_config[k]['type'] == 'text':
                        batch_targets[k][b, h, :] = dic[k]
                    else:
                        batch_targets[k][b, h] = dic[k]
        return batch_inputs, batch_targets, torch.tensor(hist_lens), torch.tensor(pred_lens)
    return collate_batch