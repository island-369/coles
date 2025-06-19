import torch.nn as nn


#输入来自UniversalFeatureEncoder的向量，它将多类型交易特征融合并编码成统一向量
class TransactionTransformer(nn.Module):
    def __init__(self, feature_encoder_dim, model_dim=2048, n_heads=32, n_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(feature_encoder_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(model_dim, n_heads, model_dim*2, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, feats, attn_mask=None):
        x = self.input_proj(feats)
        if attn_mask is not None:
            return self.trans(x, src_key_padding_mask=attn_mask)
        else:
            return self.trans(x)