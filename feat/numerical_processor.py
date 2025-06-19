import torch
from feat.feat_processor import FeatureProcessor
import torch.nn.functional as F

class NumericalProcessor(FeatureProcessor):
    def __init__(self, cfg):
        super().__init__(key='numerical', cfg=cfg)

    def feat_loss(self, predict, target):
        pre = predict.reshape(-1, predict.size(-1))
        tar = target.reshape(-1)
        loss = F.cross_entropy(pre, tar)
        pred_cls = pre.argmax(-1)
        matches = (pred_cls == tar).float().mean()
        return loss, matches

    # 数值型数据转换成分桶索引
    def encode(self, data):
        data = torch.tensor(data).long()
        bucket = torch.bucketize(data.float(), torch.tensor(self.cfg['bins'], device=data.device))-1
        bucket = torch.clamp(bucket, min=0, max=len(self.cfg['bins'])-2)
        return bucket