import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 1),
        )
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class RotaryMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, base: int = 10000):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.scale = self.head_dim ** -0.5
        self.base = base
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _rope_cache(self, seq_len: int, device, dtype):
        half_dim = self.head_dim // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = self.base ** (-freq_seq / half_dim)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_cos, q_sin = self._rope_cache(q_len, q.device, q.dtype)
        k_cos, k_sin = self._rope_cache(k_len, k.device, k.dtype)
        q = self._apply_rope(q, q_cos, q_sin)
        k = self._apply_rope(k, k_cos, k_sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        return self.out_proj(attn_out), attn_weights


class TinyNetworkBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, att="self_att"):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.att = att
        if att == "self_att":
            self.attn = RotaryMultiheadAttention(embed_dim, num_heads)
        elif att == "mlp":
            self.attn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
        else:
            raise ValueError(f"arg parse att should be att or mlp , not {att}")
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * 4
        self.ffn_value = nn.Linear(embed_dim, hidden_dim)
        self.ffn_gate = nn.Linear(embed_dim, hidden_dim)
        self.ffn_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, query):
        if self.att == "self_att":
            attn_output, _ = self.attn(query, query, query)
        else:
            mixed = torch.cat([query, query], dim=-1)
            attn_output = self.attn(mixed)
        query = self.norm1(query + attn_output)
        ff_output = self.ffn_out(self.ffn_value(query) * F.silu(self.ffn_gate(query)))
        query = self.norm2(query + ff_output)
        return query

