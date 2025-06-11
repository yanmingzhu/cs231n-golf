import torch
import torch.nn as nn
import math

def build_rope_cache(seq_len, dim, device):
    half_dim = dim // 2
    theta = 10000 ** (-torch.arange(0, half_dim, device=device).float() / half_dim)
    pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
    freqs = pos * theta
    return torch.sin(freqs), torch.cos(freqs)

def apply_rope(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    sin, cos = sin.unsqueeze(0), cos.unsqueeze(0)  # [1, seq_len, dim//2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class RoPESelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.mha, attention = nn.MultiheadAttention(embed_dim, num_heads, need_weights=True, batch_first=True)
        print(f"attention: {attention}")
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [batch, seq_len, embed_dim]
        B, L, D = x.shape
        sin, cos = build_rope_cache(L, D, x.device)

        q = k = apply_rope(x, sin, cos)  # RoPE only on Q and K
        v = x

        out, _ = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.self_attn = RoPESelfAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src: [batch, seq_len, embed_dim]
        src2 = self.self_attn(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
