"""Standalone inference script for LangFlow.

Usage:
    python inference.py --checkpoint /path/to/model.safetensors

The model code lives in this repo. You only need to download the
safetensors checkpoint from HuggingFace — no need to clone the HF repo.
"""

import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from langflow import LangFlow, LangFlowConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with LangFlow")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the safetensors checkpoint file (e.g. model.safetensors)")
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
        help="Sequence length in tokens (default: 1024)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save generated texts as a .txt file (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture config from the bundled config.json
    config_dir = os.path.join(os.path.dirname(__file__), "langflow")
    config = LangFlowConfig.from_pretrained(config_dir)

    # Build model and load weights from the downloaded safetensors file
    print(f"Loading checkpoint: {args.checkpoint}")
    model = LangFlow(config)
    state_dict = load_file(args.checkpoint, device=str(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Generating {args.num_samples} sample(s) "
          f"(batch_size={args.batch_size}, steps={args.num_steps})...")

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
