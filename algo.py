"""Lightning modules: LangFlow and the AR / MDLM / SEDD / UDLM / Duo baselines.

The trainer and the baselines are adapted from Duo
(https://github.com/s-sahoo/duo, Apache-2.0).
"""

import collections
import dataclasses
import itertools
import os
import pickle

import fsspec
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from duo import dataloader
from duo import utils
from langflow.dit import DIT
from langflow.model import GumbelProposal
from langflow.model import LangFlowMixin


@dataclasses.dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  prior_loss: torch.FloatTensor
  num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
  """alpha_t = 1 - t for the discrete diffusion baselines."""

  def __init__(self):
    super().__init__()
    self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t


def sample_categorical(categorical_probs):
  gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
  """Training / validation loop, EMA, optimizer and fault-tolerant resuming."""

  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer,
               vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.ignore_bos = config.algo.get('ignore_bos', False)
    self.loss_type = config.algo.get('loss_type', None)
    self.tokenizer = tokenizer
    self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
    self.sampler = config.sampling.predictor
    self.antithetic_sampling = config.training.antithetic_sampling
    self.parameterization = config.algo.parameterization
    if config.algo.backbone == 'dit':
      self.backbone = DIT.from_config(config, vocab_size=self.vocab_size)
    elif config.algo.backbone == 'hf_dit':
      # Public HuggingFace checkpoints of the baselines (e.g. s-sahoo/duo).
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)

    self.T = config.algo.T
    self.num_tokens = config.model.length
    self.p_nucleus = config.sampling.p_nucleus
    self.noise = LogLinear()
    self.metrics = utils.Metrics(
      gen_ppl_eval_model_name_or_path=config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=config.eval.perplexity_batch_size)
    self._init_ema()
    self.lr = config.optim.lr
    self.sampling_eps = config.training.sampling_eps
    self.time_conditioning = config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _init_ema(self):
    self.ema = None
    if self.config.training.ema > 0:
      self.ema = utils.ExponentialMovingAverage(
        self._get_parameters(), decay=self.config.training.ema)

  def _validate_configuration(self):
    assert self.config.algo.backbone in {'dit', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.metrics.to(*args, **kwargs)
    return self

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(), self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress']['current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    fit_loop = checkpoint['loops']['fit_loop']
    optim_step = fit_loop['epoch_loop.automatic_optimization.optim_progress'][
      'optimizer']['step']
    fit_loop['epoch_loop.batch_progress']['total']['completed'] = (
      optim_step['total']['completed'] * self.trainer.accumulate_grad_batches)
    fit_loop['epoch_loop.batch_progress']['current']['completed'] = (
      optim_step['current']['completed'] * self.trainer.accumulate_grad_batches)
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with accumulate_grad_batches here.
    fit_loop['epoch_loop.state_dict']['_batches_that_stepped'] = (
      optim_step['total']['completed'])
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
      sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
      checkpoint['sampler']['random_state'] = sampler_state_dict.get(
        'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def forward(self, xt, sigma):
    sigma = self._process_sigma(sigma)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
      model_output = self.backbone(xt, sigma)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def on_train_epoch_start(self):
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    losses = self._loss(batch['input_ids'], batch['attention_mask'],
                        current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
    self.log(name='trainer/loss', value=losses.loss.item(), on_step=True,
             on_epoch=False, sync_dist=True)
    return losses.loss

  def on_train_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False, on_epoch=True,
               sync_dist=True)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self._eval_mode()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
    return losses.loss

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False, on_epoch=True,
               sync_dist=True)
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      samples, text_samples = None, None
      for _ in range(self.config.sampling.num_sample_batches):
        samples = self.generate_samples(
          num_samples=self.config.loader.eval_batch_size)
        self.metrics.record_entropy(samples)
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.metrics.record_generative_perplexity(
            text_samples, self.num_tokens, self.device)
      if text_samples is not None:
        if (self.trainer.global_rank == 0
            and hasattr(self.trainer.logger, 'log_table')):
          # Log the last generated samples
          text_samples = text_samples[: self.config.sampling.num_sample_log]
          self.trainer.logger.log_table(
            key=f'samples@global_step{self.global_step}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
        if self.config.eval.compute_generative_perplexity:
          self.log('val/gen_ppl', self.metrics.gen_ppl.compute(),
                   on_epoch=True, on_step=False, sync_dist=True)
          self.log('val/sample_entropy', self.metrics.sample_entropy.compute(),
                   on_epoch=True, on_step=False, sync_dist=True)
    self._train_mode()

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1, self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)
    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
    raise NotImplementedError

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generates samples with the EMA weights."""
    # Lightning auto-casting is not working in this method for some reason
    self._eval_mode()
    samples = self.generate_samples(
      num_samples=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self._train_mode()
    return samples

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False):
    raise NotImplementedError

  def _loss(self, x0, valid_tokens, current_accumulation_step=None,
            train_mode=False):
    input_tokens, output_tokens, valid_tokens = self._process_model_input(
      x0, valid_tokens)
    loss = self.nll(input_tokens, output_tokens, current_accumulation_step,
                    train_mode)
    assert loss.ndim == 2
    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    return Loss(loss=nlls / num_tokens,
                nlls=nlls,
                prior_loss=0.0,
                num_tokens=num_tokens)


##############################################################################
#                            Autoregressive model                            #
##############################################################################


class AR(TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning

  def _process_model_input(self, x0, valid_tokens):
    return x0[:, :-1], x0[:, 1:], valid_tokens[:, 1:]

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del current_accumulation_step, train_mode
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(-1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros((num_samples, num_pred_tokens + 1), dtype=torch.long,
                    device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      x[:, i + 1] = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


##############################################################################
#                         Discrete diffusion models                          #
##############################################################################


class Diffusion(TrainerBase):
  """Continuous-time discrete diffusion with the log-linear schedule."""

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.noise_removal in {
      'none', 'ancestral', 'greedy'}
    assert self.loss_type in {'elbo', 'low_var'}
    if self.config.sampling.noise_removal == 'greedy':
      assert self.sampler != 'analytic'
      assert self.parameterization in {'mean', 'subs'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var):
    raise NotImplementedError

  def nll(self, x0, output_tokens, current_accumulation_step=None,
          train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      # t \in {1/T, 2/T, ..., 1}
      t = (t * self.T).to(torch.int) / self.T + 1 / self.T
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)
    xt = self.q_xt(x0, alpha_t)
    log_x_theta = self.forward(xt, sigma=sigma)
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

  def _get_score(self, **kwargs):
    raise NotImplementedError

  def _denoiser_update(self, x, t):
    raise NotImplementedError

  def _analytic_update(self, x, t, dt):
    raise NotImplementedError

  def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
    raise NotImplementedError

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
      if self.sampler == 'ancestral':
        _, x = self._ancestral_update(x=x, t=t, dt=dt, p_x0=None)
      elif self.sampler == 'ancestral_cache':
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache)
        if not torch.allclose(x_next, x) or self.time_conditioning:
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x=x, t=t, dt=dt)

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      if self.sampler == 'analytic':
        x = self._denoiser_update(x=x, t=t0)
      else:
        _, x = self._ancestral_update(x=x, t=t0, dt=None, p_x0=p_x0_cache,
                                      noise_removal_step=True)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
    return x


class AbsorbingState(Diffusion):
  """Masked (absorbing-state) diffusion; base of MDLM and SEDD."""

  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do vocab_size = len(tokenizer), so that we
    # account for the special tokens added in dataloader.py. But we use
    # tokenizer.vocab_size so as to to be consistent with the prior checkpoints.
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    self.subs_masking = config.algo.subs_masking
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not (self.parameterization == 'mean' and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'mean'

  def q_xt(self, x, alpha_t):
    """Masks each token of x with probability 1 - alpha_t [B, 1]."""
    move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
    xt = torch.where(move_indices, self.mask_index, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return self.mask_index * torch.ones(
      *batch_dims, dtype=torch.int64, device=self.device)

  def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      p_x0 = self.forward(x, self._sigma_from_alphat(alpha_t)).exp()
    q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    _x = sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_alphat(self.noise(t)[1])
    sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self._get_score(x, sigma_t)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    score = self._get_score(x, sigma)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    return sample_categorical(probs)

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge


class MDLM(AbsorbingState):
  def _validate_configuration(self):
    # ancestral sampling isn't desirable because it's slow
    assert self.sampler == 'ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    # Normalize the model_output such that x.exp() is a probability
    # distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix. For the logits of the
    # unmasked tokens, set all values to -infinity except for the indices
    # corresponding to the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta, dim=-1, index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0
    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1

    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(model_output)
    unmasked_score = torch.scatter(
      unmasked_score, -1, x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))

    masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
    model_output = (masked_score * masked_indices
                    + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class SEDD(AbsorbingState):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
    ).log().to(model_output.dtype)
    # logits shape (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score for the input word to 0.
    return torch.scatter(model_output, -1, xt[..., None],
                         torch.zeros_like(model_output[..., :1]))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False):
    """Score entropy loss [B, L]; log_x_theta is the log score."""
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]
    words_that_were_masked = x0[masked_indices]
    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices], -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = (score[:, : self.mask_index].sum(dim=-1)
                  + score[:, self.mask_index + 1:].sum(dim=-1))
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(*xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class UDLM(Diffusion):
  """Uniform-state diffusion (the base of Duo)."""

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.time_conditioning
    assert self.parameterization == 'mean'
    assert self.T == 0

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def q_xt(self, x, alpha_t):
    """Replaces each token of x by a uniform one w.p. 1 - alpha_t [B, 1]."""
    move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
    uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
    xt = torch.where(move_indices, uniform_tensor, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return torch.randint(0, self.vocab_size, batch_dims, dtype=torch.int64,
                         device=self.device)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot), or model's predicted `x_theta` of
        shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
      :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)

    const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1) - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    q_xs = self._compute_posterior(
      x=self.forward(x, sigma_t).exp(), xt=x, alpha_s=alpha_s, alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(q_xs.log(), top_p=self.p_nucleus)
    return None, sample_categorical(q_xs)


class Integral(torch.autograd.Function):
  """Looks up Duo's cached diffusion-transformation integral p_t(gamma)."""

  @staticmethod
  def forward(ctx, gamma_t, data):
    gamma_max = data['gamma_max']
    gamma_min = data['gamma_min']
    if (gamma_t.max() > gamma_max) or (gamma_t.min() < gamma_min):
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
      gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
    indices = torch.round(
      (data['num_points'] - 1) * (gamma_t - gamma_min) / (
        gamma_max - gamma_min)).long()
    ctx.grad_pt = data['grad_pt'][indices]
    pt = data['pt'][indices]
    assert pt.shape == gamma_t.shape
    return pt

  @staticmethod
  def backward(ctx, grad_output):
    return ctx.grad_pt * grad_output, None


class DUO(UDLM):
  """Duo: UDLM trained with the Gaussian-relaxation curriculum."""

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    with fsspec.open(self.config.algo.integral_cache_path, 'rb') as f:
      self.integral_cache = pickle.load(f)
    self.integral_cache['pt'] = torch.from_numpy(self.integral_cache['pt'])
    self.integral_cache['grad_pt'] = torch.from_numpy(
      self.integral_cache['grad_pt'])
    self.gamma_min = self.config.algo.gamma_min
    self.gamma_max = self.config.algo.gamma_max
    self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
    self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
    self.curriculum_start = self.config.algo.curriculum_start
    self.curriculum_end = self.config.algo.curriculum_end
    self.loss_type = self.config.algo.loss_type
    self._validate_configuration()

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache['pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache['grad_pt'].to(
      *args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
    start = self.gumbel_tau_log10_start
    end = self.gumbel_tau_log10_end
    delta = end - start
    if self.global_step < self.curriculum_start:
      tau = start
    elif self.global_step < self.curriculum_end:
      frac = (self.global_step - self.curriculum_start) / (
        self.curriculum_end - self.curriculum_start)
      tau = start + frac * delta
    else:
      tau = -10
    return 10 ** (-tau)

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
             value=1 / self._compute_gumbel_tau_inverse(),
             on_step=True, on_epoch=False, sync_dist=True)
    return super().training_step(batch, batch_idx)

  def _gamma_to_alphat(self, gamma_t):
    integral = Integral.apply(gamma_t, self.integral_cache)
    return (self.vocab_size * integral - 1) / (self.vocab_size - 1)

  def _q_xt_gaussian(self, x, gamma_t):
    """Gaussian-diffused one-hot x at log-NSR gamma_t [B]."""
    assert gamma_t.ndim == 1
    assert x.ndim == 3
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    epsilon = torch.randn(x.shape, dtype=torch.float32, device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, output_tokens, current_accumulation_step=None,
          train_mode=False):
    use_true_nll = (self.global_step > self.curriculum_end or not train_mode)
    if use_true_nll:
      return super().nll(x0, output_tokens, current_accumulation_step)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)

    x0_one_hot = F.one_hot(x0, self.vocab_size)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = xt * self._compute_gumbel_tau_inverse()
    xt_usdm = xt.argmax(-1)
    log_x_theta = self.forward(xt, sigma=sigma)
    return self.nll_per_token(log_x_theta=log_x_theta,
                              xt=xt_usdm,
                              x0=x0,
                              alpha_t=usdm_alpha_t,
                              dalpha_t=usdm_dalpha_t,
                              low_var=False)


##############################################################################
#                                  LangFlow                                  #
##############################################################################


class LangFlow(LangFlowMixin, TrainerBase):
  """LangFlow: continuous flow matching in (normalized) embedding space.

  Trained with token cross-entropy at log-NSRs gamma drawn from a learnable
  Gumbel schedule (fitted to the per-gamma loss), self-conditioning and a
  preconditioning skip connection. Validation reports the variational (SDE)
  NLL bound; `mode=ppl_eval eval.ppl_bound=ode` reports the ODE bound instead.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer, vocab_size=tokenizer.vocab_size)
    self.proposal = GumbelProposal(
      loc=config.algo.gumbel_loc,
      scale=config.algo.gumbel_scale,
      cutoff=config.algo.gumbel_cutoff,
      entropy=config.algo.gumbel_entropy,
      learnable=config.algo.learnable_schedule)
    self._init_ema()
    self.self_conditioning = config.algo.self_conditioning
    self.self_cond_prob = config.algo.self_cond_prob
    # Linear warmup (in steps) of the skip connection; null disables it.
    self.bias_warmup = config.algo.bias_warmup
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    assert self.config.algo.backbone == 'dit'
    assert self.config.algo.time_conditioning
    assert self.config.eval.ppl_bound in ('ode', 'sde')

  def _init_ema(self):
    # TrainerBase calls this before the schedule exists; the EMA tracks the
    # backbone and the schedule parameters, so build it once both exist.
    self.ema = None
    if hasattr(self, 'proposal'):
      super()._init_ema()

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(),
                           self.proposal.parameters())

  def _bias_scale(self) -> float:
    """0 without skip connection, 1 at inference, linear warmup in training."""
    if self.bias_warmup is None:
      return 0.0
    if not self.training:
      return 1.0
    return min((self.global_step + 1e-6) / (self.bias_warmup + 1e-6), 1.0)

  def forward(self, xt, sigma, x_self_cond=None):
    """Logits from noisy embeddings xt [B, L, D] (or token ids) at gamma sigma."""
    if xt.ndim == 2:
      xt = self._embed_tokens(xt)
    return self.denoise(xt, sigma, x_self_cond)

  def _sample_gamma(self, batch_size):
    """Stratified gamma ~ Gumbel schedule."""
    eps = 1e-5  # Matches the schedule cutoff.
    t = (torch.arange(batch_size, device=self.device)
         + torch.rand(batch_size, device=self.device)) / batch_size
    gamma = self.proposal(t.clamp(min=eps, max=1.0 - eps))
    assert torch.isfinite(gamma).all(), f'Non-finite gamma: {gamma}'
    return gamma

  def _weighted_mse(self, logits, x_clean, gamma):
    """Per-token diffusion term of the variational bound (no gradient)."""
    x_recon = self._embed_tokens(F.softmax(logits.float(), dim=-1))
    mse = ((x_clean.float() - x_recon.float()) ** 2).sum(dim=-1)
    # Importance weight 0.5 * exp(-gamma) / p(gamma).
    weight = 0.5 * torch.exp(-gamma - self.proposal.log_pdf(gamma))
    return mse * weight[:, None]

  def _loss(self, x0, valid_tokens, current_accumulation_step=None,
            train_mode=False):
    """Cross-entropy (+ schedule fit) loss; `nlls` is the variational bound."""
    del current_accumulation_step
    gamma = self._sample_gamma(x0.shape[0]).detach()
    x_clean = self._embed_tokens(x0)
    z = self._forward_diffusion(x_clean, gamma)

    # Self-conditioning: always in evaluation, with prob. self_cond_prob in
    # training. Zeros (instead of None) keep the projection in the graph.
    x_self_cond = torch.zeros_like(z)
    use_self_cond = self.self_conditioning and (
      not train_mode
      or float(torch.rand(1, device=self.device).item()) < self.self_cond_prob)
    if use_self_cond:
      # cache_enabled=False avoids autocast cache issues with no_grad + DDP.
      with torch.no_grad(), torch.autocast(device_type='cuda',
                                           cache_enabled=False):
        logits_sc = self.forward(z, sigma=gamma)
        x_self_cond = self._embed_tokens(F.softmax(logits_sc.float(), dim=-1))

    logits = self.forward(z, sigma=gamma, x_self_cond=x_self_cond)
    ce_loss = F.cross_entropy(
      logits.flatten(0, 1), x0.flatten(), reduction='none').reshape(*x0.shape)
    mse_nll = self._weighted_mse(logits.detach(), x_clean, gamma)
    if self.ignore_bos:
      ce_loss = ce_loss[:, 1:]
      mse_nll = mse_nll[:, 1:]
      valid_tokens = valid_tokens[:, 1:]

    num_tokens = valid_tokens.sum()
    # Schedule fit loss on the per-sequence CE (detached, so it only trains
    # the schedule; zero for a fixed schedule).
    ce_per_seq = ((ce_loss * valid_tokens).sum(dim=1)
                  / valid_tokens.sum(dim=1).clamp(min=1))
    fit_loss = self.proposal.fit_loss(gamma, ce_per_seq.detach()).mean()
    total_loss = (ce_loss * valid_tokens).sum() / num_tokens + fit_loss

    # Variational NLL: weighted MSE + reconstruction CE, the latter estimated
    # from the sequence with the smallest gamma and rescaled to the batch.
    idx = torch.argmin(gamma)
    ce_value = ((ce_loss[idx] * valid_tokens[idx]).sum()
                * (float(num_tokens) / float(valid_tokens[idx].sum())))
    nll = (mse_nll * valid_tokens).sum() + ce_value
    return Loss(loss=total_loss,
                nlls=nll.detach(),
                prior_loss=fit_loss.detach(),
                num_tokens=num_tokens)

  def training_step(self, batch, batch_idx):
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    losses = self._loss(batch['input_ids'], batch['attention_mask'],
                        current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
    log = dict(on_step=True, on_epoch=False, sync_dist=True)
    self.log(name='trainer/loss', value=losses.loss.item(), **log)
    self.log(name='trainer/nll',
             value=(losses.nlls / losses.num_tokens).item(), **log)
    if self.proposal.entropy is not None:
      for name in ('loc', 'scale', 'entropy'):
        self.log(name=f'proposal/{name}',
                 value=getattr(self.proposal, name).item(), **log)
    return losses.loss

  def validation_step(self, batch, batch_idx):
    del batch_idx
    if self.config.mode == 'ppl_eval' and self.config.eval.ppl_bound == 'ode':
      out = self.flow_nll(
        batch['input_ids'],
        num_steps=self.config.eval.ode_steps,
        ode_method=self.config.eval.ode_method,
        self_cond=self.config.eval.self_cond,
        attention_mask=batch['attention_mask'])
      num_tokens = out['num_tokens'].sum()
      self.metrics.update_valid(out['nll'].sum(), 0.0, num_tokens)
      for k in ('reconst_nll', 'flow_nll', 'prior_nll'):
        self.log(f'val/{k}', out[k].sum() / num_tokens, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=int(num_tokens))
      return out['nll'].sum() / num_tokens
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
    self.log(name='val/loss', value=losses.loss.item(), on_step=False,
             on_epoch=True, sync_dist=True)
    return losses.loss

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
    sampling = self.config.sampling
    return self.sample(
      num_samples=num_samples,
      seq_length=self.num_tokens,
      num_steps=num_steps or sampling.steps,
      sampler=sampling.predictor,
      temperature=sampling.temperature,
      sde_offset=sampling.sde_offset,
      sde_mode=sampling.sde_mode,
      eps=eps)

  def load_hf_weights(self, path):
    """Loads released HuggingFace weights (file, directory or Hub repo id)."""
    import huggingface_hub
    import safetensors.torch
    if os.path.isdir(path):
      path = os.path.join(path, 'model.safetensors')
    elif not os.path.isfile(path):
      path = huggingface_hub.hf_hub_download(path, 'model.safetensors')
    self.load_state_dict(safetensors.torch.load_file(path), strict=True)
    self.ema = None  # The released weights are already the EMA weights.
    return self


ALGOS = {'ar': AR, 'mdlm': MDLM, 'sedd': SEDD, 'udlm': UDLM, 'duo': DUO,
         'langflow': LangFlow}
