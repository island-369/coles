import torch
from feat.feat_processor import FeatureProcessor
import torch.nn.functional as F

class CategoricalProcessor(FeatureProcessor):
    def __init__(self, cfg):
        super().__init__(key='categorical', cfg=cfg)

    def feat_loss(self, predict, target):
        pre = predict.reshape(-1, predict.size(-1))
        tar = target.reshape(-1)
        loss = F.cross_entropy(pre, tar)
        pred_cls = pre.argmax(-1)
        matches = (pred_cls == tar).float().mean()
        return loss, matches

    def encode(self, data):
        # 类名称到索引的映射
        return torch.tensor(self.cfg['idx_map'][data] if not isinstance(data, int) else data).long()