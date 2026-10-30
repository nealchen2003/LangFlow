"""Train and evaluate LangFlow and the baselines (AR, MDLM, SEDD, UDLM, Duo).

  python main.py -cn langflow-owt                          # train
  python main.py -cn langflow-owt mode=sample_eval ...     # samples + Gen. PPL
  python main.py -cn langflow-owt mode=ppl_eval ...        # perplexity

See README.md for all options.
"""

import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import tqdm

import algo
from duo import dataloader
from duo import utils

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(model_cls, config, tokenizer):
  path = config.eval.checkpoint_path
  if 'hf' in config.algo.backbone:
    return model_cls(config, tokenizer=tokenizer).to('cuda')
  if path.endswith('.ckpt'):
    return model_cls.load_from_checkpoint(
      path, tokenizer=tokenizer, config=config, weights_only=False)
  # Released HuggingFace weights: .safetensors file, directory or Hub repo id.
  if model_cls is not algo.LangFlow:
    raise ValueError(f'Expected a .ckpt checkpoint, got {path}')
  return model_cls(config, tokenizer=tokenizer).load_hf_weights(path).to('cuda')


@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True,
                  save_cfg: bool = True) -> None:
  """Prints the config as a Rich tree and saves it next to the checkpoints."""
  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
  for field in config.keys():
    branch = tree.add(field, style=style, guide_style=style)
    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)
    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open('{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _wandb_logger(config):
  if config.get('wandb', None) is None:
    return None
  return L.pytorch.loggers.WandbLogger(
    config=omegaconf.OmegaConf.to_object(config), **config.wandb)


def _trainer(config, logger, **kwargs):
  callbacks = [hydra.utils.instantiate(callback)
               for callback in config.get('callbacks', {}).values()]
  return hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=logger,
    **kwargs)


def _generate_samples(model_cls, config, logger, tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(model_cls, config, tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  all_samples = []
  pbar = tqdm.tqdm(range(config.sampling.num_sample_batches),
                   desc='Sampling', dynamic_ncols=True)
  for _ in pbar:
    samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
    model.metrics.record_entropy(samples)
    text_samples = model.tokenizer.batch_decode(samples)
    model.metrics.record_generative_perplexity(
      text_samples, config.model.length, model.device)
    all_samples.extend(list(text_samples))
    pbar.set_postfix(
      entropy=f'{model.metrics.sample_entropy.compute().item():.4f}')

  generative_ppl = model.metrics.gen_ppl.compute().item()
  entropy = model.metrics.sample_entropy.compute().item()
  print('Generative perplexity:', generative_ppl)
  print('Sample entropy:', entropy)
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)


def _eval_ppl(model_cls, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')
  with omegaconf.open_dict(config):
    config.eval.generate_samples = False
  model = _load_from_checkpoint(model_cls, config, tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  if config.algo.name == 'langflow' and config.eval.ppl_bound == 'ode':
    # TF32 matmuls, as used for the ODE-bound numbers in the paper.
    torch.set_float32_matmul_precision('high')
  # The ODE bound of LangFlow differentiates through the model.
  trainer = _trainer(config, _wandb_logger(config), inference_mode=False)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(model_cls, config, logger, tokenizer):
  logger.info('Starting Training.')
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = model_cls.load_from_checkpoint(
      config.training.finetune_path, tokenizer=tokenizer, config=config,
      weights_only=False)
  else:
    model = model_cls(config, tokenizer=valid_ds.tokenizer)

  trainer = _trainer(config, _wandb_logger(config))
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
  """Main entry point for training and evaluation."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name not in algo.ALGOS:
    raise ValueError(f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'model_cls': algo.ALGOS[config.algo.name],
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
