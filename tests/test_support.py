"""Tokenizer, config, data, and statistics helpers."""

from __future__ import annotations

import json

import pytest
import torch

from microserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingConfig,
    load_config,
)
from microserve.data import Corpus, get_batch, load_corpus, sample_prompts
from microserve.stats import chi2_sf, chi_square_two_sample, percentile
from microserve.tokenizer import CharTokenizer


class TestTokenizer:
    def test_round_trip(self) -> None:
        text = "To be, or not to be."
        tok = CharTokenizer.from_text(text)
        assert tok.decode(tok.encode(text)) == text

    def test_vocabulary_is_sorted_and_unique(self) -> None:
        tok = CharTokenizer.from_text("banana")
        assert tok.itos == ["a", "b", "n"]
        assert tok.vocab_size == 3 == len(tok)

    def test_duplicate_characters_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            CharTokenizer(["a", "a"])

    def test_unknown_character_raises(self) -> None:
        tok = CharTokenizer.from_text("abc")
        with pytest.raises(ValueError, match="not in vocabulary"):
            tok.encode("abz")

    def test_lossy_encoding_substitutes(self) -> None:
        tok = CharTokenizer.from_text("ab ")
        assert tok.decode(tok.encode_lossy("azb")) == "a b"

    def test_save_and_load(self, tmp_path) -> None:
        tok = CharTokenizer.from_text("hello world")
        path = tmp_path / "tok.json"
        tok.save(path)
        assert CharTokenizer.from_file(path).itos == tok.itos

    def test_repr_mentions_size(self) -> None:
        assert "5" in repr(CharTokenizer.from_text("abcde"))


class TestConfig:
    def test_head_dimension_must_divide(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            ModelConfig(d_model=10, n_heads=4)

    def test_ffn_hidden_defaults_to_two_thirds_of_4x(self) -> None:
        cfg = ModelConfig(d_model=128)
        assert cfg.ffn_hidden == 352  # 8/3 * 128 = 341.3, rounded to a multiple of 32
        assert ModelConfig(d_model=128, d_ff=200).ffn_hidden == 200

    def test_derived_head_dim(self) -> None:
        assert ModelConfig(d_model=128, n_heads=4).d_head == 32

    def test_cache_size_arithmetic(self) -> None:
        model = ModelConfig(n_layers=2, d_model=64, n_heads=4)
        cache = CacheConfig(block_size=16, num_blocks=10, dtype="float32")
        assert cache.num_slots == 160
        # 2 (K and V) * 2 layers * 160 slots * 64 features * 4 bytes
        assert cache.bytes_for(model) == 2 * 2 * 160 * 64 * 4

    def test_greedy_flag(self) -> None:
        assert SamplingConfig(temperature=0.0).greedy
        assert not SamplingConfig(temperature=0.1).greedy

    def test_round_trip_through_json(self, tmp_path) -> None:
        cfg = EngineConfig(
            model=ModelConfig(n_layers=3), cache=CacheConfig(block_size=32), seed=7
        )
        path = tmp_path / "cfg.json"
        cfg.save(path)
        restored = load_config(path)
        assert restored.model.n_layers == 3
        assert restored.cache.block_size == 32
        assert restored.seed == 7

    def test_yaml_is_supported(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump({"seed": 11, "cache": {"block_size": 8}}))
        cfg = load_config(path)
        assert cfg.seed == 11 and cfg.cache.block_size == 8

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown config key"):
            EngineConfig.from_dict({"nonsense": 1})

    def test_parameter_count_formula(self) -> None:
        cfg = ModelConfig(vocab_size=100, n_layers=2, d_model=64, n_heads=4)
        assert cfg.n_params() > 0
        untied = ModelConfig(**{**cfg.__dict__, "tie_weights": False})
        assert untied.n_params() == cfg.n_params() + 100 * 64


class TestData:
    @pytest.fixture
    def corpus_file(self, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("abcdefghij" * 200, encoding="utf-8")
        return path

    def test_split_proportions(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file, val_fraction=0.2)
        total = len(corpus.train) + len(corpus.val)
        assert total == 2000
        assert len(corpus.val) == pytest.approx(400, abs=1)

    def test_reuses_a_supplied_tokenizer(self, corpus_file) -> None:
        tok = CharTokenizer.from_text("abcdefghijZ")
        corpus = load_corpus(corpus_file, tokenizer=tok)
        assert corpus.vocab_size == 11
        assert corpus.tokenizer is tok

    def test_split_lookup(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file)
        assert corpus.split("train") is corpus.train
        with pytest.raises(ValueError, match="unknown split"):
            corpus.split("test")

    def test_batch_targets_are_inputs_shifted_by_one(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file)
        gen = torch.Generator().manual_seed(0)
        x, y = get_batch(corpus.train, 4, 16, gen)
        assert x.shape == y.shape == (4, 16)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])

    def test_batches_are_reproducible(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file)
        a = get_batch(corpus.train, 2, 8, torch.Generator().manual_seed(3))
        b = get_batch(corpus.train, 2, 8, torch.Generator().manual_seed(3))
        torch.testing.assert_close(a[0], b[0])

    def test_window_longer_than_split_raises(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file, val_fraction=0.01)
        with pytest.raises(ValueError, match="shorter than one training window"):
            get_batch(corpus.val, 1, 10_000, torch.Generator())

    def test_sample_prompts_honours_lengths(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file)
        lengths = [3, 7, 11]
        prompts = sample_prompts(corpus, 3, lengths, torch.Generator().manual_seed(0))
        assert [len(p) for p in prompts] == lengths
        assert all(0 <= t < corpus.vocab_size for p in prompts for t in p)

    def test_corpus_is_a_dataclass_with_vocab_size(self, corpus_file) -> None:
        corpus = load_corpus(corpus_file)
        assert isinstance(corpus, Corpus)
        assert corpus.vocab_size == corpus.tokenizer.vocab_size


class TestStats:
    @pytest.mark.parametrize(
        "statistic,dof",
        [(3.841, 1), (5.991, 2), (7.815, 3), (11.070, 5), (18.307, 10), (31.410, 20)],
    )
    def test_matches_published_critical_values(self, statistic: float, dof: int) -> None:
        assert chi2_sf(statistic, dof) == pytest.approx(0.05, abs=5e-4)

    def test_tail_is_monotone(self) -> None:
        values = [chi2_sf(x, 4) for x in (0.5, 1.0, 4.0, 10.0, 30.0)]
        assert values == sorted(values, reverse=True)

    def test_edges(self) -> None:
        assert chi2_sf(0.0, 3) == 1.0
        assert chi2_sf(-1.0, 3) == 1.0
        assert chi2_sf(1e4, 1) == pytest.approx(0.0, abs=1e-12)
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)

    def test_identical_samples_are_indistinguishable(self) -> None:
        sample = [i % 5 for i in range(1000)]
        assert chi_square_two_sample(sample, sample).p_value > 0.99

    def test_shifted_samples_are_distinguishable(self) -> None:
        a = [0] * 500 + [1] * 500
        b = [0] * 900 + [1] * 100
        assert chi_square_two_sample(a, b).p_value < 1e-6

    def test_rare_categories_are_merged(self) -> None:
        a = [0] * 400 + [1] * 400 + list(range(2, 20))
        b = [0] * 400 + [1] * 400 + list(range(2, 20))
        result = chi_square_two_sample(a, b)
        assert result.pooled_low_count == 18
        assert result.num_categories == 3  # two kept plus the merged bucket

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            chi_square_two_sample([], [1, 2])

    def test_result_is_printable(self) -> None:
        assert "chi2=" in str(chi_square_two_sample([0, 1] * 50, [0, 1] * 50))

    def test_percentiles(self) -> None:
        values = [float(i) for i in range(101)]
        assert percentile(values, 0) == 0.0
        assert percentile(values, 50) == 50.0
        assert percentile(values, 100) == 100.0
        assert percentile([5.0], 42) == 5.0
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_percentile_interpolates(self) -> None:
        assert percentile([0.0, 10.0], 25) == pytest.approx(2.5)


def test_package_json_config_files_are_valid(tmp_path) -> None:
    """The shipped configs must load without hand-editing."""
    cfg = EngineConfig()
    path = tmp_path / "default.json"
    cfg.save(path)
    assert json.loads(path.read_text())["scheduler"]["policy"] == "fcfs"
    assert load_config(path).scheduler.policy == "fcfs"
