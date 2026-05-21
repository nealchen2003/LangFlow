"""Generate LangFlow samples and report generated-sample perplexity.

LM1B Evaluation:
    python gen_ppl.py \
        --model Continuous-Rivals-Discrete/langflow-lm1b \
        --tokenizer bert-base-uncased \
        --num-steps 128 \
        --seq-length 128 \
        --output gen_ppl_lm1b.json

OWT Evaluation:
    python gen_ppl.py \
        --model Continuous-Rivals-Discrete/langflow-owt \
        --tokenizer gpt2-large \
        --num-steps 1024 \
        --seq-length 1024 \
        --output gen_ppl_owt.json
"""

import argparse
import json
import math
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from langflow import LangFlow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated-sample perplexity for a LangFlow HF model.")
    parser.add_argument(
        "--model",
        help="Hugging Face model id or local directory for the LangFlow model.")
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer used to decode generated LangFlow token ids.")
    parser.add_argument(
        "--ppl-model",
        default="gpt2-large",
        help="Hugging Face causal LM used to score generated text.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Total number of samples to generate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size.")
    parser.add_argument(
        "--ppl-batch-size",
        type=int,
        default=32,
        help="Batch size for the perplexity scorer.")
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of LangFlow denoising steps.")
    parser.add_argument(
        "--seq-length",
        type=int,
        help="Generated sample length in LangFlow token ids.")
    parser.add_argument(
        "--max-ppl-length",
        type=int,
        default=1024,
        help="Maximum tokenized length for the causal LM scorer.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for metrics and generated samples.")
    parser.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="Number of generated samples to print.")
    return parser.parse_args()


def load_generator(model_name_or_path, device):
    print(f"Loading LangFlow generator: {model_name_or_path}")
    model = LangFlow.from_pretrained(model_name_or_path, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model


def generate_texts(model, tokenizer, args, device):
    samples = []
    remaining = args.num_samples

    print(
        f"Generating {args.num_samples} sample(s) "
        f"(batch_size={args.batch_size}, steps={args.num_steps}, length={args.seq_length})")
    with torch.inference_mode():
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            token_ids = model.generate_samples(
                num_samples=batch_size,
                seq_length=args.seq_length,
                num_steps=args.num_steps,
                device=device,
            )
            samples.append(token_ids.cpu())
            remaining -= batch_size

    sample_ids = torch.cat(samples, dim=0)
    # Remark: Following the baselines' evaluation protocol, we don't skip special tokens like [CLS].
    # When evaluating Gen. PPL for LM1B, [CLS] will be verbatim re-tokenized as [ CL S ].
    # While this seems weird, we preserve the original behavior for consistency.
    texts = tokenizer.batch_decode(sample_ids, skip_special_tokens=False)
    return sample_ids, texts


def sample_entropy(sample_ids):
    entropies = []
    for ids in sample_ids:
        _, counts = torch.unique(ids, return_counts=True)
        probs = counts.float() / counts.sum()
        entropies.append(-(probs * torch.log(probs)).sum())
    return torch.stack(entropies).mean().item()


def compute_generated_ppl(texts, scorer_name_or_path, batch_size, max_length, device):
    print(f"Loading PPL scorer: {scorer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(scorer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    scorer = AutoModelForCausalLM.from_pretrained(
        scorer_name_or_path,
        dtype=dtype,
    )
    scorer.to(device)
    scorer.eval()

    total_nll = 0.0
    total_tokens = 0
    per_sample_nll = []
    per_sample_tokens = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            outputs = scorer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_tokens = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            batch_tokens = int(valid_tokens.sum().item())
            if batch_tokens == 0:
                continue

            batch_nll = outputs.loss.item() * batch_tokens
            total_nll += batch_nll
            total_tokens += batch_tokens

            # Re-score each text to keep per-sample values exact despite padding.
            for text in batch_texts:
                single = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                single_input_ids = single["input_ids"].to(device)
                single_attention_mask = single["attention_mask"].to(device)
                single_tokens = int((single_attention_mask.sum() - 1).clamp(min=0).item())
                if single_tokens == 0:
                    per_sample_nll.append(float("nan"))
                    per_sample_tokens.append(0)
                    continue
                single_out = scorer(
                    input_ids=single_input_ids,
                    attention_mask=single_attention_mask,
                    labels=single_input_ids,
                )
                per_sample_nll.append(single_out.loss.item())
                per_sample_tokens.append(single_tokens)

    if total_tokens == 0:
        raise ValueError("No scoreable tokens were generated.")

    nll_per_token = total_nll / total_tokens
    return {
        "gen_ppl": math.exp(nll_per_token),
        "nll_per_token": nll_per_token,
        "num_scored_tokens": total_tokens,
        "per_sample_nll": per_sample_nll,
        "per_sample_tokens": per_sample_tokens,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    generator = load_generator(args.model, device)
    generator_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sample_ids, texts = generate_texts(generator, generator_tokenizer, args, device)

    metrics = compute_generated_ppl(
        texts=texts,
        scorer_name_or_path=args.ppl_model,
        batch_size=args.ppl_batch_size,
        max_length=args.max_ppl_length,
        device=device,
    )
    metrics["sample_entropy"] = sample_entropy(sample_ids)

    print("=" * 60)
    print("Generated PPL Results")
    print("=" * 60)
    print(f"Generator:       {args.model}")
    print(f"PPL scorer:      {args.ppl_model}")
    print(f"Samples:         {args.num_samples}")
    print(f"Scored tokens:   {metrics['num_scored_tokens']}")
    print(f"NLL/token:       {metrics['nll_per_token']:.4f}")
    print(f"Gen. PPL:        {metrics['gen_ppl']:.4f}")
    print(f"Sample entropy:  {metrics['sample_entropy']:.4f}")

    for idx, text in enumerate(texts[:args.print_samples]):
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(f"\n--- Sample {idx + 1} ---")
        print(preview)

    if args.output is not None:
        payload = {
            "metrics": metrics,
            "args": vars(args),
            "generated_texts": texts,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()