class FeatureProcessor:
    def __init__(self, key, cfg):
        self.key = key
        self.cfg = cfg

    def preprocess(self, data):
        return data

    def encode(self, data):
        pass

    def encode_item(self, data):
        data = self.preprocess(data)
        return self.encode(data)
    
    def feat_loss(self, predict, target):
        pass
    
    def compute_loss(self, predict, target):
        return self.feat_loss(predict, target)
    
