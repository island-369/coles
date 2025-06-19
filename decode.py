import torch.nn as nn
import torch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # logits: [batch, vocab]
    batch, vocab = logits.size()
    top_k = min(top_k, vocab)
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        thresholds = values[:, -1].unsqueeze(-1).expand_as(logits)
        logits = torch.where(logits < thresholds, torch.full_like(logits, filter_value), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 0] = 0  # 保证最少有一个保留
        sorted_logits[sorted_mask] = filter_value
        logits.scatter_(1, sorted_indices, sorted_logits)
    return logits

class ConditionedGPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, pad_idx):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, input_tokens, cond_emb):
        # input_tokens: [B, L], cond_emb: [B, D]
        B, L = input_tokens.shape
        tok = self.token_emb(input_tokens)              # (B,L,D)
        pos = self.pos_emb(torch.arange(L, device=input_tokens.device).unsqueeze(0))     # (1,L,D)
        pos = pos.expand(B, -1, -1)                    # (B,L,D)
        cond = cond_emb.unsqueeze(1).expand(B,L,-1)    # (B,L,D)
        h = tok + pos + cond
        # causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=h.device), 1).bool()
        for lyr in self.layers:
            h = lyr(h, src_mask=causal_mask)
        h = self.norm(h)
        return self.lm_head(h)   # (B,L,vocab)

    def generate(
        self, cond_emb, bos_token_id, eos_token_id, pad_token_id, device,
        max_gen_len=None, top_k=0, top_p=0.0
    ):
        B, D = cond_emb.shape
        max_len = self.max_len if max_gen_len is None else max_gen_len
        tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_len - 1):
            logits = self.forward(tokens, cond_emb)  # (B,seq,V)
            next_logits = logits[:, -1, :]           # (B,V)
            if top_k > 0 or top_p > 0.0:  # 采样
                filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:                           # 贪婪
                next_token = next_logits.argmax(-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)  #对已生成 <EOS> 的样本，不再生成有效 token，而是一直填充 <PAD>
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break
        if tokens.shape[1] < max_len:
            pad_length = max_len - tokens.shape[1]
            tokens = torch.cat([tokens, torch.full((B, pad_length), pad_token_id, device=device, dtype=torch.long)], dim=1)
        return tokens

class MultiStepHeadDecoder(nn.Module):
    def __init__(self, input_dim, feature_config, gpt_config=None, device='cpu'):
        super().__init__()
        self.feature_config = feature_config
        self.device = device

        self.feat_decodes = nn.ModuleDict()
        for key, cfg in feature_config.items():
            if cfg['type'] == 'numerical':
                self.feat_decodes[key] = nn.Linear(input_dim, len(cfg['bins'])-1)
            elif cfg['type'] == 'categorical':
                self.feat_decodes[key] = nn.Linear(input_dim, len(cfg['choices']))
            elif cfg['type'] == 'time':
                self.feat_decodes[key] = nn.Linear(input_dim, cfg['range'])
            elif cfg['type'] == 'text':
                self.feat_decodes[key] = ConditionedGPTDecoder(
                vocab_size=cfg['vocab_size'],
                d_model=input_dim,
                n_heads=32,
                n_layers=16,
                max_len=cfg['max_length'],
                pad_idx=cfg['pad_idx']
            )

    def forward(self, feats, target=None, teacher_forcing=True, gen_args=None):
        outs = {}
        B, P, D = feats.shape
        # 结构化特征头
        for k, head in self.feat_decodes.items():
            cfg = self.feature_config[k]
            if cfg['type'] != 'text':
                outs[k] = head(feats)
            else:
                cond_emb = feats.reshape(B*P, D)
                if teacher_forcing and target is not None:
                    tgt_tokens = target[k].reshape(B*P, -1)     # [B*P, L]
                    # 输入： [bos, token1.... tokenL-1] 预测token1~tokenL
                    input_tokens = tgt_tokens[:, :-1]    # [B*P,L-1]
                    targets = tgt_tokens[:, 1:]          # [B*P,L-1]
                    logits = head(input_tokens, cond_emb) # [B*P,L-1, V]
                    # reshape回多步
                    logits = logits.reshape(B, P, -1, logits.size(-1))
                    outs[k] = logits
                else:
                    # 推理自回归generate
                    gen_kwargs = gen_args or {}
                    gen_tokens = head.generate(
                        cond_emb, bos_token_id=cfg['bos_token_id'], eos_token_id=cfg['eos_token_id'],
                        pad_token_id=cfg['pad_idx'], device=feats.device, **gen_kwargs
                    )   # [B*P, L]
                    outs[k] = gen_tokens.reshape(B, P, -1)
        return outs