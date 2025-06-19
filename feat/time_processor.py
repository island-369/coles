import torch
from feat.feat_processor import FeatureProcessor
import torch.nn.functional as F

class TimelProcessor(FeatureProcessor):
    def __init__(self, cfg):
        super().__init__(key='time', cfg=cfg)
        self.base_year = cfg['base_year'] if 'base_year' in cfg else 2014

    def feat_loss(self, predict, target):
        pre = predict.reshape(-1, predict.size(-1))
        tar = target.reshape(-1)
        loss = F.cross_entropy(pre, tar)
        pred_cls = pre.argmax(-1)
        matches = (pred_cls == tar).float().mean()
        return loss, matches

    #统一编码为整数索引
    def encode(self, data):
        data = int(data)
        sub_type = self.cfg['sub_type']
        if sub_type == 'year':
            data = int(data) - self.base_year
        elif sub_type  == 'month' or sub_type == 'day':
            data -= 1
        return torch.tensor(data).long()
        