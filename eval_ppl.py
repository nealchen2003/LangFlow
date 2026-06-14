"""LangFlow Perplexity Evaluation Script — Multi-GPU edition.

Computes NLL via the probability flow ODE / SDE:
  - Reconstruction loss (CE at gamma_min)
  - Flow integral (ODE integration of divergence) / SDE Monte Carlo
  - Prior loss (Gaussian log-prob at gamma_max)

Multi-GPU usage:
    torchrun --nproc_per_node=<NUM_GPUS> eval_ppl.py ...

Single-GPU usage:
    python eval_ppl.py -cn langflow-owt-small eval.checkpoint_path=...

Features:
  - Results saved to JSON file in checkpoint directory
"""

import json
import os
from datetime import datetime
from datetime import timedelta

import hydra
import lightning as L
import omegaconf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchdiffeq
from tqdm import tqdm

from duo import dataloader
from duo import utils
from langflow.model import LangFlow

torch.set_float32_matmul_precision('high')

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _is_main_process() -> bool:
    return _rank() == 0


def _setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=10)
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if _rank() == 0:
            print("NCCL initialized successfully")


def _teardown_distributed():
    if _is_distributed():
        dist.destroy_process_group()


def _gather_tensor(local_tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a 1-D tensor from every rank and concatenate on rank-0.

    Args:
        local_tensor: 1-D float tensor of arbitrary length on the current rank.

    Returns:
        Concatenated tensor on rank-0; local_tensor unchanged on other ranks.
    """
    if not _is_distributed():
        return local_tensor

    torch.cuda.synchronize()

    world = _world_size()

    # Share lengths so we can handle uneven splits
    local_len = torch.tensor([local_tensor.numel()], device=local_tensor.device)
    all_lens = [torch.zeros_like(local_len) for _ in range(world)]
    dist.all_gather(all_lens, local_len)

    max_len = max(t.item() for t in all_lens)

    # Pad to common length then gather
    padded = torch.zeros(max_len, dtype=local_tensor.dtype, device=local_tensor.device)
    padded[: local_tensor.numel()] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)

    if _is_main_process():
        return torch.cat([gathered[r][: all_lens[r].item()] for r in range(world)])

    return local_tensor  # non-main ranks don't use this


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LangFlowNLL(LangFlow):
    """LangFlow with NLL computation."""

    def _pred_x_and_div(self, z_t, gamma, use_self_cond=False, attention_mask=None,
                        sc_chain_rule=False):
        B, L, D = z_t.shape
        u = torch.randn_like(z_t)
        if attention_mask is not None:
            # Zero out padding positions so they don't contribute to the
            # Hutchinson divergence estimate or the ODE z-trajectory.
            token_mask = attention_mask[:, :, None].to(u.dtype)
            u = u * token_mask

        z = z_t.clone().detach().requires_grad_(True)
        gamma_batch = gamma.expand(B)

        x_self_cond = None
        if use_self_cond and self.config.self_conditioning:
            logits_sc = self.forward(noisy_embeds=z, timesteps=gamma_batch, x_self_cond=None, return_dict=False)
            probs_sc = F.softmax(logits_sc, dim=-1)
            x_self_cond = self._embed_tokens(probs_sc)
            if not sc_chain_rule:
                x_self_cond = x_self_cond.detach()

        logits = self.forward(noisy_embeds=z, timesteps=gamma_batch, x_self_cond=x_self_cond, return_dict=False)
        probs = F.softmax(logits, dim=-1)
        x_reconst = self._embed_tokens(probs)
        if attention_mask is not None:
            x_reconst = x_reconst * token_mask

        grad = torch.autograd.grad(
            (u * x_reconst).sum(), z, create_graph=False, retain_graph=False
        )[0]

        div = (u * grad).sum(dim=[1, 2])
        return x_reconst, div

    def compute_flow_nll(self, x0, n_steps=128, ode_method="euler", use_self_cond=True,
                         attention_mask=None, sc_chain_rule=False):
        B, L = x0.shape
        device = x0.device
        dtype = torch.float32

        x_embed = self._embed_tokens(x0).to(dtype)
        D = x_embed.shape[2]
        N = L * D

        gamma_0 = torch.tensor(self.proposal.gamma_min, device=device, dtype=dtype)
        alpha_0 = torch.sigmoid(-gamma_0).sqrt()
        sigma_0 = torch.sigmoid(gamma_0).sqrt()

        eps = torch.randn_like(x_embed)
        z_0 = alpha_0 * x_embed + sigma_0 * eps
        z_0 = z_0.detach()

        # === 1. Reconstruction loss at gamma_0 ===
        gamma_batch = gamma_0.expand(B)

        x_self_cond = None
        if use_self_cond and self.config.self_conditioning:
            with torch.no_grad():
                logits_sc = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=None, return_dict=False)
                probs_sc = F.softmax(logits_sc, dim=-1)
                x_self_cond = self._embed_tokens(probs_sc).detach()

        logits = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=x_self_cond, return_dict=False)
        reconst_nll_per_token = F.cross_entropy(
            logits.flatten(0, 1), x0.flatten(), reduction="none"
        ).reshape(B, L).detach()
        if attention_mask is not None:
            reconst_nll = (reconst_nll_per_token * attention_mask.to(dtype)).sum(dim=1)
        else:
            reconst_nll = reconst_nll_per_token.sum(dim=1)

        # === 2. Flow integral via ODE ===
        def ode_func(sqrt_snr, y):
            gamma = -2.0 * torch.log(sqrt_snr)
            sigma = torch.sigmoid(gamma).sqrt()
            z_scaled = y[:, 1:].reshape(B, L, D)
            z_t = z_scaled * sigma
            x_reconst, x_div = self._pred_x_and_div(
                z_t, gamma, use_self_cond=use_self_cond, attention_mask=attention_mask,
                sc_chain_rule=sc_chain_rule)
            div_scaled = sigma * x_div
            v = x_reconst.reshape(B, -1)
            return torch.cat([div_scaled.view(B, 1), v], dim=1).detach()

        y0 = torch.cat([
            torch.zeros(B, 1, device=device, dtype=dtype),
            (z_0 / sigma_0).reshape(B, -1)
        ], dim=1)

        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        t_grid = t_grid.clamp(min=1e-5, max=1.0 - 1e-5)
        gamma_grid = self.proposal(t_grid).to(dtype)
        sqrt_snr_grid = torch.exp(-0.5 * gamma_grid)

        y1 = torchdiffeq.odeint(ode_func, y0, sqrt_snr_grid, method=ode_method)[-1]

        flow_nll = -y1[:, 0].detach()

        # === 3. Prior loss at gamma_1 ===
        z_scaled_final = y1[:, 1:].reshape(B, L, D)
        if attention_mask is not None:
            token_mask = attention_mask[:, :, None].to(dtype)
            num_valid_per_seq = attention_mask.sum(dim=1).to(dtype)   # (B,)
            prior_nll = (0.5 * (z_scaled_final ** 2) * token_mask).sum(dim=[1, 2]) \
                        - num_valid_per_seq * D / 2
        else:
            prior_nll = 0.5 * (z_scaled_final ** 2).sum(dim=[1, 2]) - N / 2

        num_valid_tokens = (
            attention_mask.sum(dim=1).long()
            if attention_mask is not None
            else torch.full((B,), L, dtype=torch.long, device=device)
        )

        return {
            "nll": reconst_nll + flow_nll + prior_nll,
            "reconst_nll": reconst_nll,
            "int_nll": flow_nll,
            "prior_nll": prior_nll,
            "num_valid_tokens": num_valid_tokens,
        }

    @torch.no_grad()
    def compute_sde_nll(self, x0, n_monte_carlo=128, use_self_cond=True,
                         attention_mask=None):
        B, L = x0.shape
        device = x0.device
        dtype = torch.float32

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=dtype)
        else:
            attention_mask = attention_mask.to(device=device, dtype=dtype)

        x_embed = self._embed_tokens(x0).to(dtype)
        
        def _add_noise(gamma):
            gamma = gamma.view(*gamma.shape, *[1] * (x_embed.ndim - gamma.ndim))
            alpha = torch.sigmoid(-gamma).sqrt()
            sigma = torch.sigmoid(gamma).sqrt()
            return alpha * x_embed + sigma * torch.randn_like(x_embed)

        gamma_0 = torch.tensor(self.proposal.gamma_min, device=device, dtype=dtype)
        z_0 = _add_noise(gamma_0)

        # === 1. Reconstruction loss at gamma_0 ===
        gamma_batch = gamma_0.expand(B)

        x_self_cond = None
        if use_self_cond and self.config.self_conditioning:
            logits_sc = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=None, return_dict=False)
            probs_sc = F.softmax(logits_sc, dim=-1)
            x_self_cond = self._embed_tokens(probs_sc)

        logits = self.forward(noisy_embeds=z_0, timesteps=gamma_batch, x_self_cond=x_self_cond, return_dict=False)
        reconst_nll_per_token = F.cross_entropy(
            logits.flatten(0, 1), x0.flatten(), reduction="none"
        ).reshape(B, L)
        reconst_nll = (reconst_nll_per_token * attention_mask).sum(dim=1)

        # === 2. SDE Monte Carlo ===
        int_nll = 0.0
        for _ in range(n_monte_carlo):
            gamma = self.proposal(torch.rand(B, device=device))
            log_pdf = self.proposal.log_pdf(gamma)
            z_t = _add_noise(gamma)
            x_self_cond = None
            if use_self_cond and self.config.self_conditioning:
                logits_sc = self.forward(noisy_embeds=z_t, timesteps=gamma, x_self_cond=None, return_dict=False)
                probs_sc = F.softmax(logits_sc, dim=-1)
                x_self_cond = self._embed_tokens(probs_sc)

            logits = self.forward(noisy_embeds=z_t, timesteps=gamma, x_self_cond=x_self_cond, return_dict=False)
            z_hat = self._embed_tokens(F.softmax(logits, dim=-1))
            int_nll_per_token = 0.5 * torch.exp(-gamma - log_pdf)[:, None] * (z_hat - x_embed).square().sum(dim=2)
            int_nll += (int_nll_per_token * attention_mask).sum(dim=1)

        int_nll /= n_monte_carlo

        # === 3. Prior loss at gamma_1 ===
        import numpy as np
        prior_nll_per_token = 0.5 * np.exp(-self.proposal.gamma_max) * x_embed.square().sum(2)
        prior_nll = (prior_nll_per_token * attention_mask).sum(dim=1)

        num_valid_tokens = attention_mask.sum(dim=1).long()

        return {
            "nll": reconst_nll + int_nll + prior_nll,
            "reconst_nll": reconst_nll,
            "int_nll": int_nll,
            "prior_nll": prior_nll,
            "num_valid_tokens": num_valid_tokens,
        }


# ---------------------------------------------------------------------------
# Checkpoint / config helpers
# ---------------------------------------------------------------------------

def _load_from_checkpoint(config, tokenizer):
    local_device = f"cuda:{_rank()}"
    model = LangFlowNLL.from_pretrained(
        config.eval.checkpoint_path,
        torch_dtype=torch.float32,
    )
    model.to(local_device)
    return model


# Build eval dataloader directly, bypassing get_dataloaders() which asserts
# training batch-size constraints that don't apply to evaluation.
# Does not use DistributedSampler — interleaved indices give each rank a
# non-overlapping shard without padding/duplication.
def _build_distributed_loader(config, tokenizer, rank, world_size):
    if config.data.valid in ['text8', 'lm1b', 'ag_news']:
        validation_split = 'test'
    else:
        validation_split = 'validation'

    dataset_kwargs = dict(
        dataset_name=config.data.valid,
        tokenizer=tokenizer,
        wrap=config.data.wrap,
        mode=validation_split,
        cache_dir=config.data.cache_dir,
        insert_eos=config.data.insert_valid_eos,
        block_size=config.model.length,
        streaming=config.data.streaming,
        num_proc=config.loader.num_workers,
        revision=config.data.get("valid_revision", None),
    )

    if world_size > 1:
        # Rank 0 prepares/caches the dataset first; others wait, then load.
        if rank == 0:
            dataloader.get_dataset(**dataset_kwargs)
        dist.barrier()

    valid_set = dataloader.get_dataset(**dataset_kwargs)

    eval_batch_size = config.loader.eval_batch_size

    if (first_n := config.eval.get("first_n", None)) is not None:
        valid_set = torch.utils.data.Subset(valid_set, range(min(len(valid_set), first_n)))

    # Multi-GPU: interleaved, non-overlapping indices — no padding needed.
    # Uneven tails are handled naturally: some ranks get one fewer sample,
    # and _gather_tensor handles variable-length tensors across ranks.
    total = len(valid_set)
    indices = list(range(rank, total, world_size))
    subset = torch.utils.data.Subset(valid_set, indices)

    return torch.utils.data.DataLoader(
        subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def _eval_langflow_ppl(config, logger, tokenizer):
    """Evaluate LangFlow perplexity using Flow/SDE NLL — multi-GPU aware.

    Each GPU processes its own shard of the validation set in parallel.
    Per-device average NLL values are token-weighted when aggregating across
    ranks, so unequal last batches and uneven shards are handled correctly.
    """
    mode = config.eval.get("mode", "flow")
    nfe = getattr(config.eval, 'n_steps', 128)
    ode_method = getattr(config.eval, 'ode_method', 'euler')
    n_monte_carlo = getattr(config.eval, 'n_monte_carlo', 128)

    rank = _rank()
    world_size = _world_size()
    local_device = torch.device(f"cuda:{rank}")

    if _is_main_process():
        logger.info("Starting LangFlow PPL Evaluation.")
        logger.info(f"World size (GPUs): {world_size}")
        logger.info(f"Mode: {mode}")
        if mode == "sde":
            logger.info(f"Monte Carlo samples: {n_monte_carlo}")
        else:
            logger.info(f"NFE: {nfe}, ODE method: {ode_method}")

    # --- Load model (every rank gets its own copy, already on local_device) ---
    model = _load_from_checkpoint(config, tokenizer)
    model.eval()

    if _is_main_process():
        logger.info(f"Self-conditioning: {'ENABLED' if model.config.self_conditioning else 'DISABLED'}")

    disable_ema = getattr(config.eval, 'disable_ema', False)
    if disable_ema and hasattr(model, 'ema'):
        model.ema = None

    if getattr(model, 'ema', None) is not None:
        if _is_main_process():
            logger.info("Using EMA weights for evaluation.")
        model.ema.store(model.parameters())
        model.ema.copy_to(model.parameters())

    use_self_cond = config.eval.get("use_self_cond", model.config.self_conditioning)
    sc_chain_rule = mode == "sc_chain_rule"

    loader = _build_distributed_loader(config, tokenizer, rank, world_size)

    # Accumulate per-sequence NLL sums and total token counts.
    # Summing per-sequence NLL (not averaging) ensures unequal last-batch
    # sizes are handled correctly when computing the token-weighted average.
    local_nll_sum     = 0.0
    local_reconst_sum = 0.0
    local_int_sum     = 0.0
    local_prior_sum   = 0.0
    local_tokens      = 0
    local_num_seqs    = 0
    seq_len           = None

    pbar_desc = (
        f"[rank {rank}] NLL (NMC={n_monte_carlo})"
        if mode == "sde"
        else f"[rank {rank}] NLL (NFE={nfe})"
    )
    with tqdm(loader, desc=pbar_desc, disable=not _is_main_process()) as pbar:
        for batch in pbar:
            x0 = batch['input_ids'].to(local_device)   # (B, L)
            B, seq_len = x0.shape
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(local_device)

            if mode == "sde":
                losses = model.compute_sde_nll(
                    x0, n_monte_carlo=n_monte_carlo, use_self_cond=use_self_cond,
                    attention_mask=attention_mask,
                )
            else:
                losses = model.compute_flow_nll(
                    x0, n_steps=nfe, ode_method=ode_method, use_self_cond=use_self_cond,
                    attention_mask=attention_mask, sc_chain_rule=sc_chain_rule,
                )

            # losses['nll'] etc. are per-sequence sums (shape: [B])
            local_nll_sum     += losses['nll'].detach().sum().item()
            local_reconst_sum += losses['reconst_nll'].detach().sum().item()
            local_int_sum += losses['int_nll'].detach().sum().item()
            local_prior_sum   += losses['prior_nll'].detach().sum().item()
            local_tokens      += losses['num_valid_tokens'].sum().item()
            local_num_seqs    += B

            if _is_main_process():
                pbar.set_postfix(
                    {'nll/tok': local_nll_sum / local_tokens}
                )

    torch.cuda.synchronize()
    if _is_distributed():
        dist.barrier()

    # ------------------------------------------------------------------
    # Gather per-rank (nll_sum, token_count) to rank-0, then compute the
    # global token-weighted average:
    #
    #   global_avg_per_token = sum_r(nll_sum_r) / sum_r(tokens_r)
    #
    # Summing NLL sums (not per-rank averages) and dividing by total tokens
    # is equivalent to a single global mean over every token, regardless of
    # how samples are split across ranks or how large each last batch is.
    # ------------------------------------------------------------------
    def _scalar_to_cuda(v):
        return torch.tensor([v], dtype=torch.float32, device=local_device)

    gathered_nll_sum     = _gather_tensor(_scalar_to_cuda(local_nll_sum))
    gathered_reconst_sum = _gather_tensor(_scalar_to_cuda(local_reconst_sum))
    gathered_int_sum     = _gather_tensor(_scalar_to_cuda(local_int_sum))
    gathered_prior_sum   = _gather_tensor(_scalar_to_cuda(local_prior_sum))
    gathered_tokens      = _gather_tensor(_scalar_to_cuda(local_tokens))
    gathered_num_seqs    = _gather_tensor(_scalar_to_cuda(local_num_seqs))

    if not _is_main_process():
        return {}

    # --- Compute token-weighted global averages on rank-0 ---
    total_tokens = gathered_tokens.sum().item()

    global_avg_nll_per_token     = gathered_nll_sum.sum().item()     / total_tokens
    global_avg_reconst_per_token = gathered_reconst_sum.sum().item() / total_tokens
    global_avg_int_per_token     = gathered_int_sum.sum().item()     / total_tokens
    global_avg_prior_per_token   = gathered_prior_sum.sum().item()    / total_tokens

    num_samples = int(gathered_num_seqs.sum().item())
    avg_valid_tokens_per_seq = total_tokens / num_samples

    # Convert per-token NLL back to per-sequence for reporting
    global_avg_nll_per_seq     = global_avg_nll_per_token     * avg_valid_tokens_per_seq
    global_avg_reconst_per_seq = global_avg_reconst_per_token * avg_valid_tokens_per_seq
    global_avg_int_per_seq     = global_avg_int_per_token     * avg_valid_tokens_per_seq
    global_avg_prior_per_seq   = global_avg_prior_per_token   * avg_valid_tokens_per_seq
    ppl = torch.exp(torch.tensor(global_avg_nll_per_token)).item()

    loc = model.proposal.loc
    scale = model.proposal.scale
    gumbel_loc = loc.item() if hasattr(loc, 'item') else float(loc)
    gumbel_scale = scale.item() if hasattr(scale, 'item') else float(scale)

    print("=" * 60)
    print("LangFlow PPL Evaluation Results (Multi-GPU)")
    print("=" * 60)
    print(f"Checkpoint: {config.eval.checkpoint_path}")
    print(f"GPUs used: {world_size}")
    print(f"Mode: {mode}")
    if mode == "sde":
        print(f"Monte Carlo samples: {n_monte_carlo}")
    else:
        print(f"NFE: {nfe}, ODE method: {ode_method}")
    print(f"Self-conditioning: {'ENABLED' if model.config.self_conditioning else 'DISABLED'}")
    print(f"Gumbel loc: {gumbel_loc:.4f}, scale: {gumbel_scale:.4f}")
    print(f"Samples: {num_samples}, Seq len: {seq_len}")
    print("-" * 60)
    print(f"Reconstruction NLL (per seq): {global_avg_reconst_per_seq:.4f}")
    print(f"Integral NLL (per seq):       {global_avg_int_per_seq:.4f}")
    print(f"Prior NLL (per seq):          {global_avg_prior_per_seq:.4f}")
    print(f"Total NLL (per seq):          {global_avg_nll_per_seq:.4f}")
    print(f"NLL (per token):              {global_avg_nll_per_token:.4f}")
    print(f"Perplexity:                   {ppl:.4f}")
    print("=" * 60)

    results = {
        'nll_per_seq': global_avg_nll_per_seq,
        'nll_per_token': global_avg_nll_per_token,
        'ppl': ppl,
        'reconst_nll': global_avg_reconst_per_seq,
        'int_nll': global_avg_int_per_seq,
        'prior_nll': global_avg_prior_per_seq,
        'mode': mode,
        'num_samples': num_samples,
        'world_size': world_size,
        'seq_len': seq_len,
        'gumbel_loc': gumbel_loc,
        'gumbel_scale': gumbel_scale,
    }
    if mode == "sde":
        results['n_monte_carlo'] = n_monte_carlo
    else:
        results['nfe'] = nfe
        results['ode_method'] = ode_method
    return results


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _save_results_json(results, config, logger):
    """Save evaluation results to a JSON file in the checkpoint directory."""
    checkpoint_path = config.eval.checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = results.get('mode', config.eval.get("mode", "flow"))
    if mode == "sde":
        n_monte_carlo = results.get('n_monte_carlo', 128)
        output_filename = f"{mode}_ppl_{checkpoint_name}_nmc{n_monte_carlo}_{timestamp}.json"
    else:
        nfe = results.get('nfe', 128)
        ode_method = results.get('ode_method', 'euler')
        output_filename = f"{mode}_ppl_{checkpoint_name}_nfe{nfe}_{ode_method}_{timestamp}.json"
    output_path = os.path.join(checkpoint_dir, output_filename)
    
    results_with_config = {
        'results': results,
        'config': {
            'checkpoint_path': checkpoint_path,
            'seed': int(config.seed),
            'data': config.data.name if hasattr(config.data, 'name') else str(config.data),
            'model_length': int(config.model.length),
            'batch_size': int(config.loader.eval_batch_size),
        },
        'timestamp': timestamp,
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_with_config, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    print(f"Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    _setup_distributed()
    try:
        L.seed_everything(config.seed + _rank())
        logger = utils.get_logger(__name__)
        tokenizer = dataloader.get_tokenizer(config)
        results = _eval_langflow_ppl(config, logger, tokenizer)
        
        # Only rank-0 saves results
        if _is_main_process() and results:
            _save_results_json(results, config, logger)
    finally:
        _teardown_distributed()
    return results


if __name__ == "__main__":
    main()