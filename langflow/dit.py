"""DiT backbone shared by LangFlow and all baselines (AR, MDLM, SEDD, UDLM, Duo).

Pure PyTorch (no flash-attn). Parameter names and registration order match the
original training code, so existing `.ckpt` files (including their optimizer and
EMA states) and the released HuggingFace weights load unchanged.
"""

import math
import typing

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


class Rotary(nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum('i,j->ij', t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      # This makes the transformation on v an identity.
      self.cos_cached[:, :, 2, :, :].fill_(1.)
      self.sin_cached[:, :, 2, :, :].fill_(0.)
    return self.cos_cached, self.sin_cached


def _apply_rotary_emb(x, cos, sin):
  """Non-interleaved rotary embedding (same as flash-attn's torch reference).

  x: [batch, seq_len, n_heads, head_dim]; cos, sin: [seq_len, rotary_dim // 2].
  """
  ro_dim = cos.shape[-1] * 2
  cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
  sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
  x_rot = x[..., :ro_dim]
  x1, x2 = x_rot.chunk(2, dim=-1)
  x_rotated = torch.cat([-x2, x1], dim=-1)
  return torch.cat([x_rot * cos + x_rotated * sin, x[..., ro_dim:]], dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.autocast(device_type='cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    q, k, v = qkv.chunk(3, dim=2)
    q = _apply_rotary_emb(q.squeeze(dim=2), cos, sin)
    k = _apply_rotary_emb(k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def attention(q, k, v, causal):
  # q, k, v: [batch, seq_len, n_heads, head_dim]
  out = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=causal)
  return einops.rearrange(out.transpose(1, 2), 'b s h d -> b s (h d)')


class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim

  def forward(self, x):
    with torch.autocast(device_type='cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
  """Embeds scalar timesteps (or log-SNRs) into vector representations."""

  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DDiTBlock(nn.Module):
  """Transformer block; adaLN-conditioned (diffusion) or causal (AR)."""

  def __init__(self, dim, n_heads, cond_dim=None, adaLN=True, causal=False,
               mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.causal = causal

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)
    if self.adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
    x = attention(q, k, v, causal=self.causal)

    if self.adaLN:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


def _normalize_embedding_layernorm(weight: torch.Tensor) -> torch.Tensor:
  """Normalize each embedding to unit norm, then scale by sqrt(dim)."""
  normalized = F.normalize(weight.float(), dim=-1)
  return (normalized * math.sqrt(weight.shape[-1])).to(weight.dtype)


class EmbeddingLayer(nn.Module):
  """Token embedding; accepts token ids [B, L] or probabilities [B, L, V]."""

  def __init__(self, dim, vocab_dim, use_normalized_embedding=False):
    super().__init__()
    self.dim = dim
    self.vocab_dim = vocab_dim
    self.use_normalized_embedding = use_normalized_embedding
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def _get_embedding(self):
    if self.use_normalized_embedding:
      return _normalize_embedding_layernorm(self.embedding)
    return self.embedding

  def forward(self, x):
    embedding = self._get_embedding()
    if x.ndim == 2:
      return embedding[x]
    assert x.ndim == 3  # probabilities
    return torch.einsum(
      'blv,ve->ble', x.float(), embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c):
    x = self.norm_final(x)
    if self.adaLN:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
    return self.linear(x)


class DIT(nn.Module):
  """Diffusion Transformer.

  Args:
    vocab_size: Number of tokens (rows of the embedding / output logits).
    causal: Causal attention without time conditioning (AR baseline).
    self_conditioning: Adds `x = x + proj([x, x_self_cond])` (LangFlow).
    normalized_embedding: Unit-norm embeddings scaled by sqrt(dim) (LangFlow).
  """

  def __init__(self, vocab_size, hidden_size=768, cond_dim=128, n_blocks=12,
               n_heads=12, dropout=0.1, causal=False, self_conditioning=False,
               normalized_embedding=False):
    super().__init__()
    self.vocab_size = vocab_size
    self.causal = causal
    self.adaLN = not causal
    self.vocab_embed = EmbeddingLayer(
      hidden_size, vocab_size,
      use_normalized_embedding=normalized_embedding)
    if not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(hidden_size // n_heads)
    self.blocks = nn.ModuleList([
      DDiTBlock(dim=hidden_size, n_heads=n_heads, cond_dim=cond_dim,
                adaLN=self.adaLN, causal=causal, dropout=dropout)
      for _ in range(n_blocks)])
    self.output_layer = DDiTFinalLayer(
      hidden_size=hidden_size, out_channels=vocab_size, cond_dim=cond_dim,
      adaLN=self.adaLN)
    self.self_conditioning = self_conditioning
    if self.self_conditioning:
      # Zero-init: the model starts as if self-conditioning were disabled.
      self.self_cond_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
      nn.init.zeros_(self.self_cond_proj.weight)

  @classmethod
  def from_config(cls, config, vocab_size):
    """Builds the backbone from the Hydra training config."""
    return cls(
      vocab_size=vocab_size,
      hidden_size=config.model.hidden_size,
      cond_dim=config.model.cond_dim,
      n_blocks=config.model.n_blocks,
      n_heads=config.model.n_heads,
      dropout=config.model.dropout,
      causal=config.algo.causal_attention,
      self_conditioning=config.algo.get('self_conditioning', False),
      normalized_embedding=config.algo.get('normalized_embedding', False))

  def forward(self, x, sigma):
    """x: token ids [B, L] or token probabilities [B, L, V]; sigma: [B]."""
    return self.forward_from_embed(self.vocab_embed(x), sigma)

  def forward_from_embed(self, x, sigma, x_self_cond=None,
                         output_hidden_states=False):
    """x: (noisy) embeddings [B, L, D]; sigma: [B] noise level (log-SNR for LangFlow)."""
    hidden_states = [x]
    if self.self_conditioning:
      if x_self_cond is None:
        x_self_cond = torch.zeros_like(x)
      x = x + self.self_cond_proj(torch.cat([x, x_self_cond], dim=-1))

    t_cond = None if self.causal else F.silu(self.sigma_map(sigma))
    rotary_cos_sin = self.rotary_emb(x)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, c=t_cond)
        hidden_states.append(x)
      x = self.output_layer(x, c=t_cond)

    if output_hidden_states:
      return x, hidden_states
    return x
