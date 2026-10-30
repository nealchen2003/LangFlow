"""LangFlow: continuous flow-matching language model in embedding space.

`LangFlowMixin` holds everything LangFlow-specific (denoiser with the skip
connection, ODE / SDE samplers, ODE likelihood). It is shared by the
HuggingFace model below and by the Lightning trainer in `algo.py`.
"""

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .config import LangFlowConfig
from .dit import DIT


class GumbelProposal(nn.Module):
  """Gumbel distribution over gamma (log-NSR), used as noise schedule.

  With `learnable=True` (the LangFlow default), `loc`, `scale` and `entropy`
  are parameters fitted by `fit_loss` so that the CDF tracks the per-gamma
  cross-entropy. With `learnable=False` the distribution is fixed.
  """

  def __init__(self, loc: float, scale: float, cutoff: float = 1e-5,
               entropy: float = 7.02, learnable: bool = True):
    super().__init__()
    self.cutoff = cutoff
    if learnable:
      self.loc = nn.Parameter(torch.tensor(loc))
      self.scale = nn.Parameter(torch.tensor(scale))
      self.entropy = nn.Parameter(torch.tensor(entropy))
    else:
      self.loc, self.scale, self.entropy = loc, scale, None

  def _dist(self) -> torch.distributions.Gumbel:
    return torch.distributions.Gumbel(self.loc, self.scale)

  def _quantile(self, coef):
    value = self.loc - coef * self.scale
    return float(value.detach() if torch.is_tensor(value) else value)

  @property
  def gamma_min(self) -> float:
    return self._quantile(math.log(-math.log(self.cutoff)))

  @property
  def gamma_max(self) -> float:
    return self._quantile(math.log(self.cutoff))

  def forward(self, q: torch.Tensor) -> torch.Tensor:
    """Maps uniform samples / quantiles in (0, 1) to gamma (inverse CDF)."""
    return self._dist().icdf(q).clamp(min=self.gamma_min, max=self.gamma_max)

  def log_pdf(self, gamma: torch.Tensor) -> torch.Tensor:
    return self._dist().log_prob(gamma)

  def fit_loss(self, gamma: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    """(entropy * CDF(gamma) - loss)^2; `loss` must be detached."""
    if self.entropy is None:
      return torch.zeros_like(gamma)
    return (self.entropy * self._dist().cdf(gamma) - loss) ** 2


class LangFlowMixin:
  """LangFlow computations shared by the HF model and the Lightning trainer.

  The host class provides `backbone` (a `DIT` with self-conditioning),
  `proposal` (a `GumbelProposal`), `self_conditioning` and `_bias_scale()`.
  Noise levels are log-NSRs `gamma`: z = sqrt(sigmoid(-gamma)) x
  + sqrt(sigmoid(gamma)) eps.
  """

  def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
    """Token ids [B, L] or probabilities [B, L, V] -> embeddings [B, L, D]."""
    return self.backbone.vocab_embed(x)

  def _forward_diffusion(self, x_embed, gamma):
    gamma = gamma.float()
    alpha = torch.sigmoid(-gamma).sqrt()[:, None, None]
    sigma = torch.sigmoid(gamma).sqrt()[:, None, None]
    noise = torch.randn_like(x_embed)
    return (x_embed * alpha + noise * sigma).to(x_embed.dtype)

  def denoise(self, z, gamma, x_self_cond=None, output_hidden_states=False):
    """Logits [B, L, V] of the clean tokens given noisy embeddings z [B, L, D].

    gamma: [B] (a [B, L] tensor is averaged over L). Adds the preconditioning
    skip connection c_skip(gamma) * z @ E^T.
    """
    if gamma.ndim == 2:
      gamma = gamma.mean(-1)
    if gamma.ndim == 0:
      gamma = gamma.unsqueeze(0)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
      logits, hidden_states = self.backbone.forward_from_embed(
        z, gamma, x_self_cond=x_self_cond, output_hidden_states=True)
      bias_scale = self._bias_scale()
      if bias_scale > 0:
        c_skip = ((F.softplus(-gamma) - gamma) / 2).exp() * bias_scale
        embedding = self.backbone.vocab_embed._get_embedding()
        skip_logits = torch.matmul(z.float(), embedding.t().float())
        logits = logits + c_skip[:, None, None] * skip_logits.to(logits.dtype)
    if output_hidden_states:
      return logits, hidden_states
    return logits

  def _x0_pred(self, z, gamma, x_self_cond, temperature=1.0):
    """Denoised embedding E^T softmax(logits / temperature)."""
    logits = self.denoise(z, gamma, x_self_cond).float()
    if temperature != 1.0:
      logits = logits / temperature
    return self._embed_tokens(F.softmax(logits, dim=-1))

  @staticmethod
  def _euler_step(z, x_pred, t, s):
    """Euler step of the probability-flow ODE from log-NSR t to s."""
    t_, s_ = t.double(), s.double()
    cur = z.double() * ((F.softplus(t_) - F.softplus(s_)) / 2).exp()
    end = torch.sigmoid(-s_).sqrt() * x_pred.double()
    return end.lerp(cur, ((s_ - t_) / 2).exp()).to(z.dtype)

  @torch.no_grad()
  def sample(self, num_samples, seq_length, num_steps, sampler='ode',
             temperature=1.0, sde_offset=0.1, sde_mode='elf', eps=1e-5,
             device=None):
    """Generates token ids [num_samples, seq_length].

    sampler='ode': deterministic Euler solver on `num_steps` Gumbel quantiles
      (num_steps - 1 Euler steps + a final denoising pass = num_steps NFEs).
    sampler='sde': each of the `num_steps` steps first re-noises z to
      gamma + sde_offset, denoises there and takes an Euler step; 'elf' starts
      the Euler step from the original (gamma, z), 'edm' from the re-noised
      state (num_steps + 1 NFEs). sde_offset=0 recovers the ODE.
    temperature: scales the logits of every x0 prediction (softmax(logits / T)).
    """
    if sampler not in ('ode', 'sde'):
      raise ValueError(f"sampler must be 'ode' or 'sde', got {sampler!r}")
    if sampler == 'sde' and sde_mode not in ('elf', 'edm'):
      raise ValueError(f"sde_mode must be 'elf' or 'edm', got {sde_mode!r}")
    assert temperature > 0, temperature
    device = device or next(self.parameters()).device
    z = torch.randn(num_samples, seq_length, self.backbone.vocab_embed.dim,
                    device=device)
    num_points = num_steps if sampler == 'ode' else num_steps + 1
    gamma = self.proposal(
      torch.linspace(1.0 - eps, eps, num_points, device=device))

    x_self_cond = None
    for i in range(num_points - 1):
      gamma_t, gamma_s = gamma[i], gamma[i + 1]
      if sampler == 'ode':
        x_pred = self._x0_pred(z, gamma_t.expand(num_samples), x_self_cond,
                               temperature)
        z = self._euler_step(z, x_pred, gamma_t, gamma_s)
      else:
        # Step back to a noisier level, denoise there, then step forward.
        gamma_back = gamma_t + sde_offset
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        alpha_back = torch.sigmoid(-gamma_back).sqrt()
        sigma_back = torch.sigmoid(gamma_back).sqrt()
        noise = torch.randn_like(z)
        offset = torch.as_tensor(sde_offset, device=z.device, dtype=z.dtype)
        z_back = (alpha_back / alpha_t * z
                  + sigma_back * torch.sqrt(-torch.expm1(-offset)) * noise)
        x_pred = self._x0_pred(z_back, gamma_back.expand(num_samples),
                               x_self_cond, temperature)
        if sde_mode == 'edm':
          z = self._euler_step(z_back, x_pred, gamma_back, gamma_s)
        else:
          z = self._euler_step(z, x_pred, gamma_t, gamma_s)
      if self.self_conditioning:
        x_self_cond = x_pred

    logits = self.denoise(z, gamma[-1].expand(num_samples), x_self_cond)
    return logits.argmax(dim=-1)

  def _x0_and_div(self, z_t, gamma, self_cond, attention_mask=None):
    """x0 prediction and its Hutchinson divergence estimate w.r.t. z_t."""
    B = z_t.shape[0]
    u = torch.randn_like(z_t)
    if attention_mask is not None:
      # Padding positions contribute neither to the divergence nor to the ODE.
      token_mask = attention_mask[:, :, None].to(u.dtype)
      u = u * token_mask
    z = z_t.clone().detach().requires_grad_(True)
    gamma = gamma.expand(B)
    x_self_cond = None
    if self_cond != 'off' and self.self_conditioning:
      x_self_cond = self._embed_tokens(
        F.softmax(self.denoise(z, gamma), dim=-1))
      if self_cond == 'detached':
        x_self_cond = x_self_cond.detach()
    probs = F.softmax(self.denoise(z, gamma, x_self_cond), dim=-1)
    x_reconst = self._embed_tokens(probs)
    if attention_mask is not None:
      x_reconst = x_reconst * token_mask
    grad = torch.autograd.grad((u * x_reconst).sum(), z)[0]
    return x_reconst, (u * grad).sum(dim=[1, 2])

  def flow_nll(self, x0, num_steps=128, ode_method='heun2',
               self_cond='attached', attention_mask=None):
    """Upper bound on the NLL via the probability-flow ODE (per sequence, nats).

    NLL = reconstruction CE at gamma_min + divergence integral from gamma_min
    to gamma_max + Gaussian prior NLL at gamma_max. `self_cond` sets how the
    self-conditioning input enters the divergence: 'attached' (gradients flow
    through it, i.e. full chain rule; default), 'detached' (treated as a
    constant) or 'off' (self-conditioning disabled).
    """
    import torchdiffeq  # Only needed for likelihood evaluation.
    assert self_cond in ('attached', 'detached', 'off'), self_cond
    B, L = x0.shape
    dtype = torch.float32
    with torch.enable_grad():
      x_embed = self._embed_tokens(x0).to(dtype)
      D = x_embed.shape[2]
      gamma_0 = torch.tensor(self.proposal.gamma_min, device=x0.device,
                             dtype=dtype)
      sigma_0 = torch.sigmoid(gamma_0).sqrt()
      z_0 = (torch.sigmoid(-gamma_0).sqrt() * x_embed
             + sigma_0 * torch.randn_like(x_embed)).detach()

      # 1. Reconstruction loss at gamma_min.
      x_self_cond = None
      if self_cond != 'off' and self.self_conditioning:
        with torch.no_grad():
          x_self_cond = self._embed_tokens(
            F.softmax(self.denoise(z_0, gamma_0.expand(B)), dim=-1))
      logits = self.denoise(z_0, gamma_0.expand(B), x_self_cond)
      reconst = F.cross_entropy(
        logits.flatten(0, 1), x0.flatten(), reduction='none'
      ).reshape(B, L).detach()
      if attention_mask is not None:
        reconst = reconst * attention_mask.to(dtype)
      reconst_nll = reconst.sum(dim=1)

      # 2. Divergence integral, solved in (z / sigma, sqrt(SNR)) coordinates.
      def ode_func(sqrt_snr, y):
        gamma = -2.0 * torch.log(sqrt_snr)
        sigma = torch.sigmoid(gamma).sqrt()
        z_t = y[:, 1:].reshape(B, L, D) * sigma
        x_reconst, div = self._x0_and_div(z_t, gamma, self_cond, attention_mask)
        return torch.cat([(sigma * div).view(B, 1), x_reconst.reshape(B, -1)],
                         dim=1).detach()

      y0 = torch.cat([torch.zeros(B, 1, device=x0.device, dtype=dtype),
                      (z_0 / sigma_0).reshape(B, -1)], dim=1)
      t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=x0.device)
      gamma_grid = self.proposal(t_grid.clamp(min=1e-5, max=1.0 - 1e-5)).to(dtype)
      y1 = torchdiffeq.odeint(ode_func, y0, torch.exp(-0.5 * gamma_grid),
                              method=ode_method)[-1]
      flow_nll = -y1[:, 0].detach()

      # 3. Prior NLL at gamma_max (up to the constant shared by all models).
      z_1 = y1[:, 1:].reshape(B, L, D)
      if attention_mask is not None:
        mask = attention_mask.to(dtype)
        prior_nll = ((0.5 * z_1 ** 2 * mask[:, :, None]).sum(dim=[1, 2])
                     - mask.sum(dim=1) * D / 2)
        num_tokens = attention_mask.sum(dim=1).long()
      else:
        prior_nll = 0.5 * (z_1 ** 2).sum(dim=[1, 2]) - L * D / 2
        num_tokens = torch.full((B,), L, dtype=torch.long, device=x0.device)
    return {'nll': reconst_nll + flow_nll + prior_nll,
            'reconst_nll': reconst_nll,
            'flow_nll': flow_nll,
            'prior_nll': prior_nll,
            'num_tokens': num_tokens}


class LangFlow(LangFlowMixin, transformers.PreTrainedModel):
  """HuggingFace LangFlow model (DiT backbone + learnable Gumbel schedule)."""
  config_class = LangFlowConfig
  base_model_prefix = 'langflow'

  def __init__(self, config: LangFlowConfig):
    super().__init__(config)
    self.config = config
    self.backbone = DIT(
      vocab_size=config.vocab_size,
      hidden_size=config.hidden_size,
      cond_dim=config.cond_dim,
      n_blocks=config.n_blocks,
      n_heads=config.n_heads,
      dropout=config.dropout,
      self_conditioning=config.self_conditioning,
      normalized_embedding=config.use_normalized_embedding)
    self.proposal = GumbelProposal(
      loc=config.gumbel_loc,
      scale=config.gumbel_scale,
      cutoff=config.gumbel_cutoff,
      entropy=config.gumbel_entropy)

  @property
  def self_conditioning(self):
    return self.config.self_conditioning

  def _bias_scale(self):
    return 1.0 if self.config.use_bias else 0.0

  def forward(
      self,
      input_ids: typing.Optional[torch.LongTensor] = None,
      noisy_embeds: typing.Optional[torch.FloatTensor] = None,
      timesteps: typing.Optional[torch.FloatTensor] = None,
      x_self_cond: typing.Optional[torch.FloatTensor] = None,
      output_hidden_states: typing.Optional[bool] = None,
      return_dict: typing.Optional[bool] = None,
  ) -> typing.Union[torch.Tensor, typing.Tuple,
                    transformers.modeling_outputs.MaskedLMOutput]:
    """Clean-token logits.

    Args:
      input_ids: [B, L] token ids (noised to `timesteps` if given).
      noisy_embeds: [B, L, D] pre-noised embeddings (alternative to input_ids).
      timesteps: [B] log-NSR gamma (defaults to gamma_min, i.e. clean input).
      x_self_cond: [B, L, D] self-conditioning embeddings.
    """
    output_hidden_states = bool(output_hidden_states)
    return_dict = (return_dict if return_dict is not None
                   else self.config.use_return_dict)
    if noisy_embeds is not None:
      z = noisy_embeds
    elif input_ids is not None:
      z = self._embed_tokens(input_ids)
      if timesteps is not None:
        z = self._forward_diffusion(z, timesteps)
    else:
      raise ValueError('Either input_ids or noisy_embeds must be provided')
    if timesteps is None:
      timesteps = torch.full((z.shape[0],), self.proposal.gamma_min,
                             device=z.device)

    logits, hidden_states = self.denoise(
      z, timesteps, x_self_cond, output_hidden_states=True)
    if return_dict:
      return transformers.modeling_outputs.MaskedLMOutput(
        logits=logits,
        hidden_states=hidden_states if output_hidden_states else None,
        loss=None)
    if output_hidden_states:
      return logits, hidden_states
    return logits

  @torch.no_grad()
  def generate_samples(self, num_samples=1, seq_length=None, num_steps=128,
                       device=None, sampler='ode', temperature=1.0,
                       sde_offset=0.1, sde_mode='elf'):
    """Generates token ids [num_samples, seq_length]; see `LangFlowMixin.sample`."""
    return self.sample(
      num_samples=num_samples,
      seq_length=seq_length or self.config.model_length,
      num_steps=num_steps,
      sampler=sampler,
      temperature=temperature,
      sde_offset=sde_offset,
      sde_mode=sde_mode,
      device=device)
