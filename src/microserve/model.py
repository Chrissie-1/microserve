"""Decoder-only transformer with two interchangeable attention paths.

``forward`` is the dense reference implementation: ordinary padded batches, an
optional list-of-tensors KV cache. It is what training uses and what the paged
path is checked against.

``forward_paged`` reads and writes a :class:`~microserve.cache.PagedKVCache`.
It consumes a *flattened* batch, so prefill and decode sequences of different
lengths can share one forward pass -- the property continuous batching needs.

Architecture: pre-norm RMSNorm, rotary position embeddings, SwiGLU MLP, tied
input/output embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cache import PagedKVCache
from .config import ModelConfig

KVCache = list[tuple[Tensor, Tensor]]


@dataclass
class PagedBatch:
    """Flattened description of one paged forward pass.

    ``total_tokens`` query tokens belong to ``B`` sequences. Sequence ``b``
    contributes ``q_lens[b]`` consecutive tokens to the flat arrays.
    """

    input_ids: Tensor  # (total_tokens,) int64
    positions: Tensor  # (total_tokens,) absolute position of each query token
    slot_mapping: Tensor  # (total_tokens,) physical cache slot for each token
    block_tables: Tensor  # (B, max_blocks) int64
    context_lens: Tensor  # (B,) context length including this pass
    q_lens: Tensor  # (B,) query tokens contributed by each sequence

    @property
    def batch_size(self) -> int:
        return int(self.q_lens.numel())

    @property
    def max_q(self) -> int:
        return int(self.q_lens.max().item())

    def scatter_index(self) -> Tensor:
        """Map flat token index -> row-major index into a ``(B, max_q)`` grid."""
        max_q = self.max_q
        rows = torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.q_lens.device), self.q_lens
        )
        starts = torch.cumsum(self.q_lens, 0) - self.q_lens
        cols = (
            torch.arange(self.input_ids.numel(), device=self.q_lens.device)
            - torch.repeat_interleave(starts, self.q_lens)
        )
        return rows * max_q + cols


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary tables, indexed by absolute position."""

    cos: Tensor
    sin: Tensor

    def __init__(self, d_head: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("rotary embeddings need an even head dimension")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(pos, inv_freq)  # (max_seq_len, d_head / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos[positions], self.sin[positions]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``(..., n_heads, d_head)``.

    ``cos``/``sin`` have shape ``(..., d_head // 2)`` and are broadcast over the
    head axis, which is inserted here.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class Attention(nn.Module):
    """Multi-head causal attention shared by the dense and paged paths."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.scale = 1.0 / math.sqrt(cfg.d_head)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shape = (*x.shape[:-1], self.n_heads, self.d_head)
        return (
            self.q_proj(x).view(shape),
            self.k_proj(x).view(shape),
            self.v_proj(x).view(shape),
        )

    # -- dense reference path --------------------------------------------
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        bsz, seq_len, _ = x.shape
        q, k, v = self._project(x)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=1)
            v = torch.cat([past[1], v], dim=1)
        present = (k, v)

        q = q.transpose(1, 2)  # (B, H, T, D)
        keys = k.transpose(1, 2)
        values = v.transpose(1, 2)
        total = keys.size(2)
        offset = total - seq_len
        scores = torch.matmul(q, keys.transpose(-1, -2)) * self.scale
        causal = torch.ones(seq_len, total, dtype=torch.bool, device=x.device).tril(
            diagonal=offset
        )
        scores = scores.masked_fill(~causal, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), values)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out), present

    # -- paged path -------------------------------------------------------
    def forward_paged(
        self,
        x: Tensor,
        batch: PagedBatch,
        cache: PagedKVCache,
        scatter_idx: Tensor,
        q_positions: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        q, k, v = self._project(x)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        cache.write(self.layer_idx, batch.slot_mapping, k, v)

        bsz, max_q = batch.batch_size, batch.max_q
        max_context = int(batch.context_lens.max().item())
        keys, values = cache.gather(self.layer_idx, batch.block_tables, max_context)

        # Scatter the flat queries into a (B, max_q, H, D) grid so that every
        # sequence can be attended in one batched matmul.
        q_pad = q.new_zeros((bsz * max_q, self.n_heads, self.d_head))
        q_pad[scatter_idx] = q
        q_pad = q_pad.view(bsz, max_q, self.n_heads, self.d_head).transpose(1, 2)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(q_pad, keys.transpose(-1, -2)) * self.scale

        # Key index j in the gathered tensor *is* logical position j, so the
        # causal mask is simply j <= (absolute position of the query token).
        key_pos = torch.arange(scores.size(-1), device=x.device)
        allowed = key_pos[None, None, None, :] <= q_positions[:, None, :, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), values)
        out = out.transpose(1, 2).reshape(bsz * max_q, self.n_heads * self.d_head)
        projected: Tensor = self.o_proj(out[scatter_idx])
        return projected


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn_out, present = self.attn(self.attn_norm(x), cos, sin, past)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, present

    def forward_paged(
        self,
        x: Tensor,
        batch: PagedBatch,
        cache: PagedKVCache,
        scatter_idx: Tensor,
        q_positions: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        x = x + self.attn.forward_paged(
            self.attn_norm(x), batch, cache, scatter_idx, q_positions, cos, sin
        )
        out: Tensor = x + self.ffn(self.ffn_norm(x))
        return out


class Transformer(nn.Module):
    """The model. See module docstring for the two forward paths."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # Scale residual-output projections so activation variance stays flat
        # with depth (GPT-2 initialisation).
        residual_scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.o_proj.weight, std=residual_scale)
            nn.init.normal_(block.ffn.down_proj.weight, std=residual_scale)

    @property
    def layers(self) -> list[Block]:
        """``nn.ModuleList`` iteration is typed ``Tensor | Module``; narrow it."""
        return cast("list[Block]", list(self.blocks))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        seen: set[int] = set()
        total = 0
        for param in self.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                total += param.numel()
        return total

    # -- dense reference path --------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        past_kv: KVCache | None = None,
        return_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Dense forward over ``(B, T)`` ids.

        With ``past_kv`` supplied, ``input_ids`` holds only the new tokens and
        positions continue from the cached length.
        """
        _, seq_len = input_ids.shape
        past_len = 0 if past_kv is None else past_kv[0][0].size(1)
        if past_len + seq_len > self.cfg.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        cos, sin = self.rope(positions)

        x = self.dropout(self.embed(input_ids))
        present: KVCache = []
        for i, block in enumerate(self.layers):
            x, kv = block(x, cos, sin, None if past_kv is None else past_kv[i])
            if return_cache:
                present.append(kv)
        logits = self.lm_head(self.norm(x))
        return logits, (present if return_cache else None)

    # -- paged path -------------------------------------------------------
    def forward_paged(self, batch: PagedBatch, cache: PagedKVCache) -> Tensor:
        """Paged forward. Returns ``(total_tokens, vocab)`` logits."""
        cos, sin = self.rope(batch.positions)
        scatter_idx = batch.scatter_index()
        # Pad rows attend to position 0 only; their outputs are discarded.
        q_positions = torch.zeros(
            batch.batch_size * batch.max_q, dtype=torch.long, device=batch.positions.device
        )
        q_positions[scatter_idx] = batch.positions
        q_positions = q_positions.view(batch.batch_size, batch.max_q)

        x = self.embed(batch.input_ids)
        for block in self.layers:
            x = block.forward_paged(x, batch, cache, scatter_idx, q_positions, cos, sin)
        logits: Tensor = self.lm_head(self.norm(x))
        return logits

    # -- checkpointing ----------------------------------------------------
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "config": self.cfg.__dict__,
            "state_dict": self.state_dict(),
            **(extra or {}),
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> Transformer:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig(**payload["config"])
        model = cls(cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
