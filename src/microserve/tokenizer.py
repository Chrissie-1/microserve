"""Character-level tokenizer.

Deliberately trivial: the point of MicroServe is the serving stack, not the
vocabulary. A char tokenizer keeps the embedding table small enough that a
1M-parameter model is actually mostly transformer.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Bijective mapping between characters and integer ids."""

    def __init__(self, chars: list[str]) -> None:
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        self.itos: list[str] = list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    # -- construction -----------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @classmethod
    def from_file(cls, path: str | Path) -> CharTokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["itos"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    # -- core API ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Encode ``text``; unknown characters raise rather than pass silently."""
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"character {exc.args[0]!r} not in vocabulary") from exc

    def encode_lossy(self, text: str, fallback: str = " ") -> list[int]:
        """Encode, substituting ``fallback`` for out-of-vocabulary characters."""
        default = self.stoi[fallback]
        return [self.stoi.get(c, default) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def __len__(self) -> int:
        return len(self.itos)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CharTokenizer(vocab_size={self.vocab_size})"
