"""Utilities: logging, LR schedulers, EMA, metrics (NLL / Gen. PPL / entropy).

Adapted from Duo (https://github.com/s-sahoo/duo, Apache-2.0).

Run `python -m duo.utils --vocab_size=N` to cache the integral used by Duo for
a new tokenizer (caches for gpt2 and bert-base-uncased ship in `integral/`).
"""

import argparse
import logging
import math
import os
import pickle
import time
import typing

import fsspec
import lightning
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from timm.scheduler import CosineLRScheduler

LOG2 = math.log(2)


def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""
  logger = logging.getLogger(name)
  logger.setLevel(level)
  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger, level,
            lightning.pytorch.utilities.rank_zero_only(getattr(logger, level)))
  return logger


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on whether we're using the
    # scheduler every epoch or every step. Otherwise, lightning will always
    # call step (i.e., meant for each epoch), and if we set scheduler interval
    # to "step", then the learning rate update will be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0,
                          filter_value=-float('Inf'), dim=-1):
  """Filter a distribution of logits using top-k/top-p (nucleus) filtering."""
  if dim != -1:
    logits = torch.transpose(logits, dim, -1)

  assert top_k < logits.size(dim)
  if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    to_remove_mask = logits < torch.min(values, dim=-1, keepdim=True)[0]
    logits[to_remove_mask] = filter_value

  if top_p > 0.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    # Ensures at least one token is kept
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask_to_remove = torch.empty_like(sorted_indices_to_remove)
    mask_to_remove.scatter_(dim=-1, index=sorted_indices,
                            src=sorted_indices_to_remove)
    logits[mask_to_remove] = filter_value

  if dim != -1:
    logits = torch.transpose(logits, dim, -1)
  return logits


class ExponentialMovingAverage:
  """Maintains (exponential) moving average of a set of parameters."""

  def __init__(self, parameters, decay, use_num_updates=True):
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def move_shadow_params_to_device(self, device):
    self.shadow_params = [i.to(device) for i in self.shadow_params]

  def update(self, parameters):
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """Copy the moving averages into `parameters`."""
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      param.data.copy_(s_param.data)

  def store(self, parameters):
    """Save the current parameters for restoring later."""
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """Restore the parameters stored with the `store` method."""
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']


class NLL(torchmetrics.aggregation.MeanMetric):
  def update(self, value: typing.Union[float, torch.Tensor],
             weight: typing.Union[float, torch.Tensor] = 1.0) -> None:
    """Accumulates sum(value) and sum(weight) (weight broadcast to value)."""
    if not isinstance(value, torch.Tensor):
      value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
    if weight is not None and not isinstance(weight, torch.Tensor):
      weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
    weight = torch.broadcast_to(weight, value.shape)
    value, weight = self._cast_and_nan_check_input(value, weight)
    if value.numel() == 0:
      return
    self.mean_value += value.sum()
    self.weight += weight.sum()


class BPD(NLL):
  def compute(self) -> torch.Tensor:
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> torch.Tensor:
    return torch.exp(self.mean_value / self.weight)


class Metrics:
  """NLL / BPD / PPL during training and Gen. PPL / entropy of samples."""

  def __init__(self, gen_ppl_eval_model_name_or_path=None,
               eval_ppl_batch_size=None) -> None:
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
    metrics.set_dtype(torch.float64)
    self.train_nlls = metrics.clone(prefix='train/')
    self.train_aux = BPD()
    self.valid_nlls = metrics.clone(prefix='val/')
    self.valid_aux = BPD()
    self.gen_ppl = Perplexity()
    self.sample_entropy = torchmetrics.aggregation.MeanMetric()
    self.eval_ppl_batch_size = eval_ppl_batch_size
    self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
      gen_ppl_eval_model_name_or_path)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    self._eval_model = None

  def to(self, *args, **kwargs):
    self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
    self.sample_entropy = self.sample_entropy.to(*args, **kwargs)
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.train_aux = self.train_aux.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.valid_aux = self.valid_aux.to(*args, **kwargs)

  def reset(self):
    self.gen_ppl.reset()
    self.sample_entropy.reset()
    self.train_nlls.reset()
    self.train_aux.reset()
    self.valid_nlls.reset()
    self.valid_aux.reset()

  def update_train(self, nll, aux_loss, num_tokens):
    self.train_nlls.update(nll, num_tokens)
    self.train_aux.update(aux_loss, num_tokens)

  def update_valid(self, nll, aux_loss, num_tokens):
    self.valid_nlls.update(nll, num_tokens)
    self.valid_aux.update(aux_loss, num_tokens)

  @torch.no_grad()
  def record_entropy(self, tokens):
    """Unigram entropy of each sample's token ids."""
    for sample in tokens:
      _, counts = torch.unique(sample, return_counts=True, sorted=False)
      entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
      self.sample_entropy.update(entropy)

  @torch.no_grad()
  def record_generative_perplexity(self, text_samples: typing.List[str],
                                   max_length: int, device='cuda') -> None:
    """Gen. PPL of decoded samples under the eval LM (GPT-2 Large).

    Samples are re-tokenized with the eval tokenizer, split into chunks of the
    eval context size, and scored up to (and including) the first EOS.
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if self._eval_model is None:
      self._eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        self.gen_ppl_eval_model_name_or_path).eval()
    eval_model = self._eval_model.to(device)
    samples = self.tokenizer(
      text_samples, return_tensors='pt', return_token_type_ids=False,
      return_attention_mask=True, truncation=True, padding=True,
      max_length=max_length)
    attn_mask = samples['attention_mask'].to(device)
    samples = samples['input_ids'].to(device)
    eval_context_size = 1024
    batch_size = min(self.eval_ppl_batch_size, samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size], eval_context_size, dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size], eval_context_size, dim=-1)
      for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
        logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
        nlls = F.cross_entropy(logits.transpose(-1, -2)[..., :-1],
                               sample_chunk[..., 1:], reduction='none')
        first_eos = (sample_chunk == self.tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = sample_chunk != self.tokenizer.eos_token_id
        valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
        self.gen_ppl.update(nlls * valid_tokens, valid_tokens)


def _discrete_prob_map(gamma_t, N=10):
  from scipy.stats import norm
  snr_sqrt = np.exp(-gamma_t / 2)
  def value(x):
    return norm.pdf(x, loc=snr_sqrt, scale=1) * norm.cdf(x, scale=1) ** (N - 1)
  return value


def _discrete_prob_grad(gamma_t, N=10):
  from scipy.stats import norm
  snr_sqrt = np.exp(-gamma_t / 2)
  def value(x):
    coef = -0.5 * snr_sqrt * (x - snr_sqrt)
    return coef * norm.pdf(x, loc=snr_sqrt, scale=1) * norm.cdf(x, scale=1) ** (N - 1)
  return value


def cache_duo_integral(vocab_size=30522, partition_index=0, num_partitions=1,
                       log10_num_points=5, path='integral'):
  """Caches Duo's Diffusion Transformation operator integral on a gamma grid."""
  from scipy.integrate import quad
  print(f'Caching partition:{partition_index} / {num_partitions}')
  gamma_min, gamma_max = -5, -1
  num_points = 10 ** log10_num_points
  p_cache, grad_p_cache = [], []
  start_time = time.time()
  gammas = np.linspace(gamma_min, gamma_max, num_points)
  n = num_points // num_partitions
  for gamma in gammas[partition_index * n: (partition_index + 1) * n]:
    p_cache.append(quad(_discrete_prob_map(gamma, vocab_size), -np.inf, np.inf)[0])
    grad_p_cache.append(
      quad(_discrete_prob_grad(gamma, vocab_size), -np.inf, np.inf)[0])
    if len(p_cache) % 100 == 0:
      print('{}% completed. Time elapsed:{:.2f} mins'.format(
        int(100 * len(p_cache) / num_points), (time.time() - start_time) / 60))
  filename = os.path.join(path, '{}_{}_{}-{}.pkl'.format(
    vocab_size, log10_num_points, partition_index, num_partitions))
  with open(filename, 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'gamma_min': gamma_min,
                 'gamma_max': gamma_max, 'num_points': num_points,
                 'pt': np.asarray(p_cache),
                 'grad_pt': np.asarray(grad_p_cache)}, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Caches the integral appearing in the Diffusion '
                'Transformation operator of Duo.')
  parser.add_argument('--vocab_size', type=int, default=50257,
                      help='Vocabulary size (default: 50257, gpt2)')
  parser.add_argument('--partition_index', type=int, default=0)
  parser.add_argument('--num_partitions', type=int, default=1)
  parser.add_argument('--log10_num_points', type=int, default=5)
  args = parser.parse_args()
  cache_duo_integral(vocab_size=args.vocab_size,
                     partition_index=args.partition_index,
                     num_partitions=args.num_partitions,
                     log10_num_points=args.log10_num_points)
