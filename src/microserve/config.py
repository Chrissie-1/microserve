"""Configuration dataclasses for MicroServe.

Every tunable in the engine lives here so that experiments are reproducible
from a single serialisable object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _round_to_multiple(value: float, multiple: int) -> int:
    return int(multiple * round(value / multiple))


@dataclass
class ModelConfig:
    """Shape of a decoder-only transformer.

    Note: the tokenizer is character level, so ``vocab_size`` is small (~65).
    A GPT-2 sized vocabulary would put 6.4M parameters in the embedding alone
    and dominate a model whose whole point is to be tiny.
    """

    vocab_size: int = 65
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 1024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_weights: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        if self.d_ff is None:
            # SwiGLU uses three matrices instead of two, so the usual 4x
            # expansion is scaled by 2/3 to keep the parameter count equal.
            self.d_ff = _round_to_multiple(8 * self.d_model / 3, 32)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        assert self.d_ff is not None
        return self.d_ff

    def n_params(self) -> int:
        """Analytic parameter count (excludes buffers)."""
        emb = self.vocab_size * self.d_model
        attn = self.n_layers * 4 * self.d_model * self.d_model
        ffn = self.n_layers * 3 * self.d_model * self.ffn_hidden
        norms = self.n_layers * 2 * self.d_model + self.d_model
        head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return emb + attn + ffn + norms + head


@dataclass
class CacheConfig:
    """Paged KV cache geometry.

    Total cache bytes = 2 * n_layers * num_blocks * block_size * n_heads *
    d_head * dtype_size.
    """

    block_size: int = 16
    num_blocks: int = 512
    dtype: str = "float32"

    def bytes_for(self, model: ModelConfig) -> int:
        itemsize = {"float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]
        per_slot = model.n_heads * model.d_head
        return 2 * model.n_layers * self.num_blocks * self.block_size * per_slot * itemsize

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size


@dataclass
class SchedulerConfig:
    """Admission control and batching policy."""

    policy: str = "fcfs"  # fcfs | sjf | lifo
    batching: str = "continuous"  # continuous | static
    max_batch_size: int = 8
    max_batched_tokens: int = 2048
    # Fraction of the cache that must stay free before we admit a new request.
    watermark: float = 0.01
    preemption_mode: str = "recompute"  # recompute | swap (swap unimplemented)

    def __post_init__(self) -> None:
        if self.policy not in {"fcfs", "sjf", "lifo"}:
            raise ValueError(f"unknown scheduling policy {self.policy!r}")
        if self.batching not in {"continuous", "static"}:
            raise ValueError(f"unknown batching mode {self.batching!r}")
        if self.preemption_mode != "recompute":
            raise ValueError("only 'recompute' preemption is implemented")


@dataclass
class SamplingConfig:
    """Per-request sampling parameters."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    max_new_tokens: int = 32
    seed: int | None = None
    stop_token: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class SpeculativeConfig:
    """Draft-then-verify settings. ``gamma`` is the proposal length."""

    enabled: bool = False
    gamma: int = 4
    draft_checkpoint: str | None = None


@dataclass
class TrainConfig:
    """Hyper-parameters for scripts/train.py."""

    steps: int = 4000
    batch_size: int = 16
    seq_len: int = 128
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    ckpt_every: int = 1000
    seed: int = 1337
    val_fraction: float = 0.1
    num_threads: int = 1


@dataclass
class EngineConfig:
    """Top-level configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    seed: int = 0
    device: str = "cpu"

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EngineConfig:
        sub = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        table = {
            "model": ModelConfig,
            "cache": CacheConfig,
            "scheduler": SchedulerConfig,
            "speculative": SpeculativeConfig,
        }
        for key, value in raw.items():
            if key not in sub:
                raise ValueError(f"unknown config key {key!r}")
            kwargs[key] = table[key](**value) if key in table else value
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> EngineConfig:
    """Load an :class:`EngineConfig` from a ``.json`` or ``.yaml`` file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        import yaml  # imported lazily so JSON-only users need no dependency

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return EngineConfig.from_dict(raw or {})
