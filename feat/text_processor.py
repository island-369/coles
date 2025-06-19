import torch
from feat.feat_processor import FeatureProcessor
import torch.nn.functional as F

class TextlProcessor(FeatureProcessor):
    def __init__(self, cfg):
        super().__init__(key='text', cfg=cfg)
        self.func = cfg['func']

    def feat_loss(self, pre, tar):
        pad_idx = self.cfg['pad_idx']
        pre = pre[:, :, :, :]
        tar = tar[:, :, 1:]  # (B,P,L-1)
        pre = pre.reshape(-1, pre.size(-1))
        tar = tar.reshape(-1)
        mask = (tar != pad_idx)
        if mask.sum() == 0:
            return .0, .0
        filtered_pre = pre[mask]
        filtered_tar = tar[mask]
        loss = F.cross_entropy(filtered_pre, filtered_tar)
        pred_cls = filtered_pre.argmax(-1)
        matches = (pred_cls == filtered_tar).float().mean()
        return loss, matches

    def encode(self, data):
        tokens = self.func(data, add_special_tokens=False, max_length=self.cfg['max_length']-2,
                                        padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze(0)
        tokens = torch.cat([
            torch.tensor([self.cfg['bos_token_id']]),
            tokens,
            torch.tensor([self.cfg['eos_token_id']])
        ])
        tokens = tokens[:self.cfg['max_length']]
        pad_len = self.cfg['max_length'] - tokens.size(0)
        if pad_len > 0:
            tokens = torch.cat([tokens, torch.full((pad_len,), self.cfg['pad_idx'])])
        return tokens
                