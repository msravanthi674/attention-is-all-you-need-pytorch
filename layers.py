import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # type: ignore

# -------------------------------
# Scaled Dot-Product Attention
# -------------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: [batch, heads, seq_len, head_dim]
    mask:    [batch, 1, seq_len, seq_len]
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


# -------------------------------
# Multi-Head Attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        concat = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.out_proj(concat)
        return output, attn_weights


# -------------------------------
# Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# -------------------------------
# Feed-Forward Network
# -------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Encoder Layer
# -------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x

# -------------------------------
# Decoder Layer
# -------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 2. Encoder-Decoder Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 3. Feed Forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Norm + Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention (decoder attends to itself with causal mask)
        attn_out, _ = self.self_attn(x, mask=tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # 2. Cross-Attention (decoder attends to encoder outputs)
        cross_out, _ = self.cross_attn(x, mask=src_mask)
        x = x + self.dropout(cross_out)
        x = self.norm2(x)

        # 3. Feed-Forward
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)

        return x
