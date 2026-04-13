"""HuggingFace model implementation for LangFlow.

LangFlow is a continuous diffusion language model that operates in embedding space.
"""

import math
import typing

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .config import LangFlowConfig


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
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def _apply_rotary_emb(x, cos, sin):
    # x: [batch, seqlen, nheads, headdim]
    # cos, sin: [seqlen, headdim//2]
    ro_dim = cos.shape[-1] * 2
    # Expand to [1, seqlen, 1, ro_dim] for broadcasting
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
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2]
        q, k, v = qkv.chunk(3, dim=2)
        q = _apply_rotary_emb(q.squeeze(dim=2), cos, sin)
        k = _apply_rotary_emb(k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def regular_attention_multi_headed(q, k, v):
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False)
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


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
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(
            self.attn_qkv(x),
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
        x = regular_attention_multi_headed(q, k, v)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x


def _normalize_embedding_layernorm(weight: torch.Tensor) -> torch.Tensor:
    """Normalize embedding weights to unit norm per row, then scale by sqrt(dim)."""
    normalized = F.normalize(weight.float(), dim=-1)
    return (normalized * math.sqrt(weight.shape[-1])).to(weight.dtype)


class EmbeddingLayer(nn.Module):
    """Embedding layer with optional layernorm normalization."""
    
    def __init__(self, dim, vocab_dim, use_normalized_embedding=True):
        super().__init__()
        self.dim = dim
        self.vocab_dim = vocab_dim
        self.use_normalized_embedding = use_normalized_embedding
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def _get_embedding(self):
        if self.use_normalized_embedding:
            return _normalize_embedding_layernorm(self.embedding)
        return self.embedding

    def forward(self, x):
        embedding = self._get_embedding()
        if x.ndim == 2:
            return embedding[x]
        assert x.ndim == 3  # probabilities
        return torch.einsum("blv,ve->ble", x.float(), embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class GumbelProposal(nn.Module):
    """Learnable Gumbel distribution proposal for sampling gamma (log-SNR)."""
    
    def __init__(self, loc: float = 4.723, scale: float = 0.852, 
                 cutoff: float = 1e-5, entropy: float = 7.02):
        super().__init__()
        self.loc = nn.Parameter(torch.tensor(loc))
        self.scale = nn.Parameter(torch.tensor(scale))
        self.cutoff = cutoff
        self.entropy = nn.Parameter(torch.tensor(entropy))

    def _get_distribution(self) -> torch.distributions.Gumbel:
        return torch.distributions.Gumbel(self.loc, self.scale)

    @property
    def gamma_min(self) -> float:
        return float(self.loc - math.log(-math.log(self.cutoff)) * self.scale)

    @property
    def gamma_max(self) -> float:
        return float(self.loc - math.log(self.cutoff) * self.scale)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Convert uniform samples to gamma values via inverse CDF."""
        gamma = self._get_distribution().icdf(q)
        return gamma.clamp(min=self.gamma_min, max=self.gamma_max)

    def log_pdf(self, gamma: torch.Tensor) -> torch.Tensor:
        """Compute log probability density at gamma."""
        return self._get_distribution().log_prob(gamma)


class LangFlowBackbone(nn.Module):
    """DiT backbone for LangFlow."""
    
    def __init__(self, config: LangFlowConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        cond_dim = config.cond_dim

        self.vocab_embed = EmbeddingLayer(
            dim, config.vocab_size,
            use_normalized_embedding=config.use_normalized_embedding)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(dim=dim, n_heads=config.n_heads, cond_dim=cond_dim, dropout=config.dropout)
            for _ in range(config.n_blocks)
        ])

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim, out_channels=config.vocab_size, cond_dim=cond_dim)

        # Self-conditioning projection
        if config.self_conditioning:
            self.self_cond_proj = nn.Linear(dim * 2, dim, bias=False)
            nn.init.zeros_(self.self_cond_proj.weight)

    def forward(self, x_embed, sigma, x_self_cond=None, output_hidden_states=False):
        """Forward pass from embeddings.
        
        Args:
            x_embed: [B, L, D] - Input embeddings (possibly noisy)
            sigma: [B] - Gamma values (log-SNR)
            x_self_cond: [B, L, D] - Self-conditioning embeddings (optional)
            output_hidden_states: Whether to return all hidden states
        
        Returns:
            logits: [B, L, vocab_size]
            hidden_states: List of hidden states if output_hidden_states=True
        """
        all_hidden_states = []
        x = x_embed
        
        if output_hidden_states:
            all_hidden_states.append(x)

        # Self-conditioning
        if self.config.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = x + self.self_cond_proj(torch.cat([x, x_self_cond], dim=-1))

        t_cond = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c=t_cond)
                if output_hidden_states:
                    all_hidden_states.append(x)
            x = self.output_layer(x, c=t_cond)

        return x, all_hidden_states


class LangFlow(transformers.PreTrainedModel):
    """HuggingFace-compatible LangFlow model.
    
    LangFlow is a continuous diffusion language model that operates in embedding space.
    It uses a DiT (Diffusion Transformer) backbone with:
    - Self-conditioning: uses previous predictions as additional input
    - Bias (preconditioning): skip connection for improved generation
    - Normalized embeddings: layernorm on embedding vectors
    - Learnable Gumbel proposal for gamma (log-SNR) sampling
    """
    config_class = LangFlowConfig
    base_model_prefix = "langflow"

    def __init__(self, config: LangFlowConfig):
        super().__init__(config)
        self.config = config
        self.backbone = LangFlowBackbone(config)
        self.proposal = GumbelProposal(
            loc=config.gumbel_loc,
            scale=config.gumbel_scale,
            cutoff=config.gumbel_cutoff,
            entropy=config.gumbel_entropy)

    def _get_embedding_matrix(self) -> torch.Tensor:
        """Get the embedding matrix for bias skip connection."""
        return self.backbone.vocab_embed._get_embedding()

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens or probabilities to continuous embeddings."""
        return self.backbone.vocab_embed(x)

    def _forward_diffusion(self, x_embed: torch.Tensor, 
                           gamma: torch.Tensor) -> torch.Tensor:
        """Add noise to embeddings (forward diffusion process)."""
        gamma = gamma.float()
        alpha = torch.sigmoid(-gamma).sqrt()[:, None, None]
        sigma = torch.sigmoid(gamma).sqrt()[:, None, None]
        noise = torch.randn_like(x_embed)
        return (x_embed * alpha + noise * sigma).to(x_embed.dtype)

    def _euler_edm_step(self, z: torch.Tensor, x_pred: torch.Tensor,
                        t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Single Euler step for EDM sampling."""
        t_ = t.double()
        s_ = s.double()
        cur = z.double() * ((F.softplus(t_) - F.softplus(s_)) / 2).exp()
        end = torch.sigmoid(-s_).sqrt() * x_pred.double()
        z = end.lerp(cur, ((s_ - t_) / 2).exp()).to(z.dtype)
        return z

    def forward(
        self,
        input_ids: typing.Optional[torch.LongTensor] = None,
        noisy_embeds: typing.Optional[torch.FloatTensor] = None,
        timesteps: typing.Optional[torch.FloatTensor] = None,
        x_self_cond: typing.Optional[torch.FloatTensor] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
    ) -> typing.Union[torch.Tensor, typing.Tuple, transformers.modeling_outputs.MaskedLMOutput]:
        """Forward pass for LangFlow.
        
        Args:
            input_ids: [B, L] - Token IDs (will be embedded and noised if timesteps provided)
            noisy_embeds: [B, L, D] - Pre-noised embeddings (alternative to input_ids)
            timesteps: [B] - Gamma values (log-SNR) for conditioning
            x_self_cond: [B, L, D] - Self-conditioning embeddings
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return MaskedLMOutput
        
        Returns:
            logits or MaskedLMOutput
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get embeddings
        if noisy_embeds is not None:
            z = noisy_embeds
        elif input_ids is not None:
            x_embed = self._embed_tokens(input_ids)
            if timesteps is not None:
                z = self._forward_diffusion(x_embed, timesteps)
            else:
                z = x_embed
        else:
            raise ValueError("Either input_ids or noisy_embeds must be provided")

        if timesteps is None:
            # Use minimum gamma for clean input
            timesteps = torch.full((z.shape[0],), self.proposal.gamma_min, device=z.device)

        # Process sigma
        sigma = timesteps
        if sigma.ndim == 2:
            sigma = sigma.mean(-1)

        # Get model output
        logits, all_hidden_states = self.backbone(
            z, sigma, x_self_cond=x_self_cond, output_hidden_states=output_hidden_states)

        # Add bias (preconditioning) skip connection
        if self.config.use_bias:
            c_skip = ((F.softplus(-sigma) - sigma) / 2).exp()
            embedding = self._get_embedding_matrix()
            skip_logits = torch.matmul(z.float(), embedding.t().float())
            logits = logits + c_skip[:, None, None] * skip_logits.to(logits.dtype)

        if return_dict:
            return transformers.modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                loss=None)
        elif output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 1,
        seq_length: typing.Optional[int] = None,
        num_steps: int = 128,
        device: typing.Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """Generate samples using Euler-EDM solver.
        
        Args:
            num_samples: Number of samples to generate
            seq_length: Sequence length (defaults to config.model_length)
            num_steps: Number of denoising steps
            device: Device to generate on
        
        Returns:
            samples: [num_samples, seq_length] - Generated token IDs
        """
        if seq_length is None:
            seq_length = self.config.model_length
        if device is None:
            device = next(self.parameters()).device

        embed_dim = self.config.hidden_size
        eps = 1e-5

        # Initialize with Gaussian noise
        z = torch.randn(num_samples, seq_length, embed_dim, device=device)

        # Create gamma schedule from t=1-eps to t=eps
        t = torch.linspace(1.0 - eps, eps, num_steps, device=device)
        gamma = self.proposal(t)

        # Self-conditioning state
        x_self_cond = None

        # Euler-EDM sampling loop
        for i in range(len(gamma) - 1):
            gamma_t = gamma[i]
            gamma_s = gamma[i + 1]

            # Get model prediction
            gamma_expanded = gamma_t.unsqueeze(0).expand(num_samples)
            logits = self.forward(
                noisy_embeds=z,
                timesteps=gamma_expanded,
                x_self_cond=x_self_cond,
                return_dict=False)

            # Convert logits to embedding prediction
            probs = F.softmax(logits.float(), dim=-1)
            x_pred = self._embed_tokens(probs)

            # Update self-conditioning
            if self.config.self_conditioning:
                x_self_cond = x_pred

            # Euler step
            z = self._euler_edm_step(z, x_pred, gamma_t, gamma_s)

        # Final step: get logits and take argmax
        gamma_final = gamma[-1]
        gamma_expanded = gamma_final.unsqueeze(0).expand(num_samples)
        logits = self.forward(
            noisy_embeds=z,
            timesteps=gamma_expanded,
            x_self_cond=x_self_cond,
            return_dict=False)
        samples = logits.argmax(dim=-1)

        return samples
