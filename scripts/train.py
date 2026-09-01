"""Train a MicroServe transformer on TinyShakespeare.

Two models are needed for speculative decoding: a ``target`` (the model whose
distribution we must reproduce exactly) and a much smaller ``draft``. Both
share one tokenizer, which is written next to the checkpoints.

Usage::

    python scripts/train.py --preset target
    python scripts/train.py --preset draft --steps 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microserve.config import ModelConfig, TrainConfig
from microserve.data import get_batch, load_corpus
from microserve.model import Transformer
from microserve.tokenizer import CharTokenizer

PRESETS: dict[str, dict[str, object]] = {
    "target": {
        "model": {"n_layers": 4, "d_model": 128, "n_heads": 4, "dropout": 0.1},
        "train": {"steps": 4000, "batch_size": 16, "seq_len": 128, "lr": 3e-3},
    },
    "draft": {
        "model": {"n_layers": 2, "d_model": 64, "n_heads": 2, "dropout": 0.1},
        "train": {"steps": 4000, "batch_size": 16, "seq_len": 128, "lr": 4e-3},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup into cosine decay down to ``lr * min_lr_ratio``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate(
    model: Transformer,
    data: torch.Tensor,
    cfg: TrainConfig,
    generator: torch.Generator,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.seq_len, generator)
        logits, _ = model(x)
        total += nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        ).item()
    model.train()
    return total / cfg.eval_batches


def build_optimizer(model: Transformer, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Weight-decay the matrices, leave norms and embeddings undecayed."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 and "embed" not in name else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target")
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="time a few steps, print the projected wall clock, and exit",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    preset = PRESETS[args.preset]
    train_cfg = TrainConfig(**preset["train"], seed=args.seed)  # type: ignore[arg-type]
    for field_name in ("steps", "batch_size", "seq_len", "lr"):
        override = getattr(args, field_name)
        if override is not None:
            setattr(train_cfg, field_name, override)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One tokenizer for both models; build it once and reuse it thereafter.
    tok_path = out_dir / "tokenizer.json"
    tokenizer = CharTokenizer.from_file(tok_path) if tok_path.exists() else None
    corpus = load_corpus(args.data, train_cfg.val_fraction, tokenizer)
    if tokenizer is None:
        corpus.tokenizer.save(tok_path)

    model_cfg = ModelConfig(
        vocab_size=corpus.vocab_size,
        max_seq_len=max(1024, train_cfg.seq_len),
        **preset["model"],  # type: ignore[arg-type]
    )
    model = Transformer(model_cfg)
    model.train()
    print(
        f"preset={args.preset} params={model.num_parameters():,} "
        f"vocab={corpus.vocab_size} train_tokens={len(corpus.train):,}"
    )

    optimizer = build_optimizer(model, train_cfg)
    batch_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)

    if args.profile_only:
        for _ in range(3):  # warm up allocator and thread pool
            x, y = get_batch(corpus.train, train_cfg.batch_size, train_cfg.seq_len, batch_gen)
            logits, _ = model(x)
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).backward()
            optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        trials = 5
        for _ in range(trials):
            x, y = get_batch(corpus.train, train_cfg.batch_size, train_cfg.seq_len, batch_gen)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        per_step = (time.perf_counter() - start) / trials
        print(
            f"{per_step * 1000:.1f} ms/step -> "
            f"{per_step * train_cfg.steps / 60:.1f} min for {train_cfg.steps} steps"
        )
        return

    history: list[dict[str, float]] = []
    best_val = float("inf")
    start = time.perf_counter()

    for step in range(train_cfg.steps):
        lr = lr_at(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(corpus.train, train_cfg.batch_size, train_cfg.seq_len, batch_gen)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            val_loss = evaluate(model, corpus.val, train_cfg, eval_gen)
            elapsed = time.perf_counter() - start
            history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "elapsed_s": elapsed,
                }
            )
            print(
                f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} "
                f"| lr {lr:.2e} | {elapsed:6.1f}s"
            )
            if val_loss < best_val:
                best_val = val_loss
                model.save(
                    out_dir / f"{args.preset}.pt",
                    extra={"step": step, "val_loss": val_loss, "preset": args.preset},
                )

        if train_cfg.ckpt_every and step and step % train_cfg.ckpt_every == 0:
            model.save(out_dir / f"{args.preset}_step{step}.pt", extra={"step": step})

    (out_dir / f"{args.preset}_losses.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "params": model.num_parameters(),
                "best_val_loss": best_val,
                "wall_clock_s": time.perf_counter() - start,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done: best val {best_val:.4f} -> {out_dir / f'{args.preset}.pt'}")


if __name__ == "__main__":
    main()
