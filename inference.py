"""Standalone inference with the released LangFlow HuggingFace checkpoints.

Usage:
    python inference.py --checkpoint /path/to/model.safetensors

The model code lives in this repo. You only need to download the
safetensors checkpoint from HuggingFace — no need to clone the HF repo.
`--checkpoint` may also be a local directory or a Hub repo id
(e.g. Continuous-Rivals-Discrete/langflow-lm1b).
"""

import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from langflow import LangFlow, LangFlowConfig

# Tokenizer of each released checkpoint, keyed by vocabulary size.
TOKENIZERS = {50257: 'gpt2', 30522: 'bert-base-uncased'}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with LangFlow")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Safetensors checkpoint (e.g. model.safetensors), directory or Hub repo id")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Total number of samples to generate (default: 5)")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of samples to generate per forward pass (default: 1)")
    parser.add_argument(
        "--num_steps", type=int, default=128,
        help="Number of denoising steps (default: 128)")
    parser.add_argument(
        "--seq_length", type=int, default=1024,
        help="Sequence length in tokens (default: 1024; 128 for LM1B)")
    parser.add_argument(
        "--sampler", type=str, default="ode", choices=["ode", "sde"],
        help="Deterministic ODE (Euler) or stochastic SDE sampler (default: ode)")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Logit temperature of the x0 prediction softmax(logits / T) (default: 1.0)")
    parser.add_argument(
        "--sde_offset", type=float, default=0.1,
        help="SDE sampler: re-noising offset in log-NSR; 0 recovers the ODE (default: 0.1)")
    parser.add_argument(
        "--sde_mode", type=str, default="elf", choices=["elf", "edm"],
        help="SDE sampler: Euler step from the original (elf) or re-noised (edm) state")
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer for decoding (default: inferred from the vocabulary size)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save generated texts as a .txt file (optional)")
    return parser.parse_args()


def load_state_dict(checkpoint, device):
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "model.safetensors")
    elif not os.path.isfile(checkpoint):
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(checkpoint, "model.safetensors")
    return load_file(checkpoint, device=str(device))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Architecture from the bundled config.json; the vocabulary size follows
    # the checkpoint (50257 for OWT, 30522 for LM1B).
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_state_dict(args.checkpoint, device)
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langflow")
    config = LangFlowConfig.from_pretrained(config_dir)
    config.vocab_size = state_dict["backbone.vocab_embed.embedding"].shape[0]
    config.model_length = args.seq_length
    model = LangFlow(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or TOKENIZERS[config.vocab_size])

    print(f"Generating {args.num_samples} sample(s) "
          f"(batch_size={args.batch_size}, steps={args.num_steps}, "
          f"sampler={args.sampler}, temperature={args.temperature})...")

    all_samples = []
    remaining = args.num_samples
    with torch.no_grad():
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            samples = model.generate_samples(
                num_samples=batch,
                seq_length=args.seq_length,
                num_steps=args.num_steps,
                device=device,
                sampler=args.sampler,
                temperature=args.temperature,
                sde_offset=args.sde_offset,
                sde_mode=args.sde_mode,
            )
            all_samples.append(samples)
            remaining -= batch

    all_samples = torch.cat(all_samples, dim=0)
    texts = tokenizer.batch_decode(all_samples, skip_special_tokens=True)
    for i, text in enumerate(texts):
        print(f"\n--- Sample {i + 1} ---")
        print(text[:500] + ("..." if len(text) > 500 else ""))

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(f"--- Sample {i + 1} ---\n{text}\n\n")
        print(f"\nSaved {len(texts)} sample(s) to {args.output}")


if __name__ == "__main__":
    main()
