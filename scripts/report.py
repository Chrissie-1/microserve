"""Turn the benchmark JSON into plots and a RESULTS.md summary.

Reads whatever exists under ``artifacts/`` -- training curves, benchmark
sweeps, speculative statistics -- and writes ``artifacts/plots/*.png`` plus a
``RESULTS.md`` at the repository root. Missing inputs are skipped, so this can
be run after a partial benchmark.

Usage::

    python scripts/report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (see the data-viz palette reference).
# Slots are assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQUENTIAL = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["params"][key], []).append(row)
    return out


def line(ax: Any, xs: list[float], ys: list[float], color: str, label: str) -> None:
    ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=5, label=label)


# -- individual figures ---------------------------------------------------
def plot_training(art: Path, out: Path) -> str | None:
    curves = {
        name: load(art / f"{name}_losses.json") for name in ("target", "draft")
    }
    if not any(curves.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (name, data) in zip(axes, curves.items(), strict=True):
        if data is None:
            ax.set_visible(False)
            continue
        steps = [h["step"] for h in data["history"]]
        line(ax, steps, [h["train_loss"] for h in data["history"]], SERIES[0], "train")
        line(ax, steps, [h["val_loss"] for h in data["history"]], SERIES[1], "validation")
        style(
            ax,
            f"{name} ({data['params']:,} params)",
            "step",
            "cross-entropy (nats/char)",
        )
        ax.legend(loc="upper right")
    fig.tight_layout()
    path = out / "training.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_batching(data: dict[str, Any], out: Path) -> str:
    groups = group_by(data["results"], "batching")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for slot, (mode, rows) in enumerate(sorted(groups.items())):
        rows.sort(key=lambda r: r["params"]["arrival_rate"])
        rates = [r["params"]["arrival_rate"] for r in rows]
        line(axes[0], rates, [r["throughput_tok_s"] for r in rows], SERIES[slot], mode)
        line(axes[1], rates, [r["ttft_p95"] for r in rows], SERIES[slot], mode)
    style(axes[0], "Throughput", "arrival rate (req/s)", "output tokens/s")
    style(axes[1], "Time to first token (p95)", "arrival rate (req/s)", "seconds")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend()
    # TTFT spans two orders of magnitude; a linear axis would hide the
    # low-load region, which is exactly where continuous batching wins.
    axes[1].set_yscale("log")
    fig.tight_layout()
    path = out / "batching.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_policy(data: dict[str, Any], out: Path) -> str:
    groups = group_by(data["results"], "policy")
    order = [p for p in ("fcfs", "sjf", "lifo") if p in groups]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for slot, policy in enumerate(order):
        rows = sorted(groups[policy], key=lambda r: r["params"]["arrival_rate"])
        rates = [r["params"]["arrival_rate"] for r in rows]
        line(axes[0], rates, [r["ttft_p95"] for r in rows], SERIES[slot], policy.upper())
        line(
            axes[1], rates, [r["latency_p95"] for r in rows], SERIES[slot], policy.upper()
        )
    style(axes[0], "Time to first token (p95)", "arrival rate (req/s)", "seconds")
    style(axes[1], "End-to-end latency (p95)", "arrival rate (req/s)", "seconds")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    path = out / "policy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_blocksize(data: dict[str, Any], out: Path) -> str:
    rows = sorted(data["results"], key=lambda r: r["params"]["block_size"])
    sizes = [str(r["params"]["block_size"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    frag = [100 * float(r["params"].get("fragmentation", 0.0)) for r in rows]
    axes[0].bar(sizes, frag, color=SEQUENTIAL, width=0.68)
    for x, value in zip(sizes, frag, strict=True):
        axes[0].text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    style(axes[0], "Internal fragmentation", "block size (tokens)", "% of slots wasted")

    tput = [r["throughput_tok_s"] for r in rows]
    axes[1].bar(sizes, tput, color=SERIES[1], width=0.68)
    style(axes[1], "Throughput", "block size (tokens)", "output tokens/s")
    fig.tight_layout()
    path = out / "blocksize.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_batchsize(data: dict[str, Any], out: Path) -> str:
    rows = sorted(data["results"], key=lambda r: r["params"]["max_batch_size"])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xs = [r["throughput_tok_s"] for r in rows]
    ys = [r["latency_p95"] for r in rows]
    ax.plot(xs, ys, color=SEQUENTIAL, linewidth=1.8, marker="o", markersize=6)
    for row, x, y in zip(rows, xs, ys, strict=True):
        ax.annotate(
            f"batch {row['params']['max_batch_size']}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=INK_MUTED,
        )
    style(ax, "Latency / throughput frontier", "output tokens/s", "p95 latency (s)")
    fig.tight_layout()
    path = out / "batchsize.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_capacity(data: dict[str, Any], out: Path) -> str:
    rows = sorted(data["results"], key=lambda r: r["params"]["num_blocks"])
    labels = [str(r["params"]["num_blocks"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(labels, [r["preemptions"] for r in rows], color=SERIES[1], width=0.68)
    style(axes[0], "Preemptions", "cache blocks", "recompute events")
    line(
        axes[1],
        list(range(len(rows))),
        [r["throughput_tok_s"] for r in rows],
        SERIES[0],
        "",
    )
    axes[1].set_xticks(range(len(rows)), labels)
    style(axes[1], "Throughput", "cache blocks", "output tokens/s")
    fig.tight_layout()
    path = out / "capacity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_speculative(spec: dict[str, Any], out: Path) -> str:
    rows = [r for r in spec["speedup"] if r["gamma"] > 0]
    gammas = [r["gamma"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    # One series: the title names it, so no legend box.
    line(axes[0], gammas, [r["acceptance_rate"] for r in rows], SERIES[0], "")
    style(axes[0], "Draft acceptance rate", "gamma (proposals per round)", "fraction accepted")
    axes[0].set_ylim(0, 1)

    # The dashed break-even line plus the direct labels carry the meaning, so
    # the bars stay one colour rather than borrowing a status palette.
    speedups = [r["speedup"] for r in rows]
    axes[1].bar([str(g) for g in gammas], speedups, color=SEQUENTIAL, width=0.68)
    axes[1].axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    for x, value in zip([str(g) for g in gammas], speedups, strict=True):
        axes[1].text(x, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
    style(axes[1], "Wall-clock speedup vs baseline", "gamma", "speedup (higher is better)")
    fig.tight_layout()
    path = out / "speculative.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# -- markdown tables ------------------------------------------------------
def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def sweep_table(data: dict[str, Any], key: str, label: str) -> str:
    rows = []
    for r in data["results"]:
        rows.append(
            [
                str(r["params"].get(key, r["label"])),
                f"{r['params'].get('arrival_rate', '-')}",
                f"{r['throughput_tok_s']:.1f}",
                f"{r['ttft_p50'] * 1000:.0f}",
                f"{r['ttft_p95'] * 1000:.0f}",
                f"{r['latency_p95']:.3f}",
                f"{r['mean_batch_size']:.2f}",
                f"{r['mean_utilization']:.2f}",
                str(r["preemptions"]),
            ]
        )
    return table(
        [label, "req/s", "tok/s", "TTFT p50 (ms)", "TTFT p95 (ms)",
         "lat p95 (s)", "mean batch", "mean util", "preempt"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="RESULTS.md")
    args = parser.parse_args()

    art = Path(args.artifacts)
    bench = art / "benchmark"
    plots = art / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# Results",
        "",
        "Generated by `python scripts/report.py`. Every number here comes from the "
        "JSON under `artifacts/`, produced on a single CPU thread "
        "(`--threads 1`); absolute rates are therefore small, and the shape of "
        "each curve is the point rather than its height.",
        "",
    ]

    training = plot_training(art, plots)
    if training:
        rows = []
        for name in ("target", "draft"):
            data = load(art / f"{name}_losses.json")
            if data:
                rows.append(
                    [
                        name,
                        f"{data['params']:,}",
                        str(data["model"]["n_layers"]),
                        str(data["model"]["d_model"]),
                        str(data["model"]["n_heads"]),
                        f"{data['best_val_loss']:.4f}",
                        f"{data['wall_clock_s'] / 60:.1f}",
                    ]
                )
        sections += [
            "## Models",
            "",
            table(
                ["model", "params", "layers", "d_model", "heads",
                 "best val loss", "train time (min)"],
                rows,
            ),
            "",
            f"![training curves](artifacts/plots/{training})",
            "",
        ]

    plan: list[tuple[str, str, str, str, Any]] = [
        ("batching", "Continuous vs static batching", "batching", "mode", plot_batching),
        ("policy", "Scheduling policy", "policy", "policy", plot_policy),
        ("blocksize", "KV block size", "block_size", "block size", plot_blocksize),
        ("batchsize", "Batch size frontier", "max_batch_size", "max batch", plot_batchsize),
        ("capacity", "Cache capacity and preemption", "num_blocks", "blocks", plot_capacity),
    ]
    for name, title, key, label, plotter in plan:
        data = load(bench / f"{name}.json")
        if not data:
            continue
        image = plotter(data, plots)
        sections += [
            f"## {title}",
            "",
            sweep_table(data, key, label),
            "",
            f"![{name}](artifacts/plots/{image})",
            "",
        ]

    spec = load(art / "spec_stats.json")
    if spec:
        image = plot_speculative(spec, plots)
        greedy = table(
            ["gamma", "prompts", "exact matches", "acceptance rate"],
            [
                [str(r["gamma"]), str(r["prompts"]), f"{r['exact_matches']}/{r['prompts']}",
                 f"{r['acceptance_rate']:.3f}"]
                for r in spec["greedy_equivalence"]
            ],
        )
        chi = table(
            ["gamma", "trials", "chi2", "dof", "p-value", "verdict"],
            [
                [str(r["gamma"]), str(r["trials"]), f"{r['chi2']:.2f}", str(r["dof"]),
                 f"{r['p_value']:.4f}",
                 "indistinguishable" if r["passes"] else "DIFFERENT"]
                for r in spec["distribution_test"]
            ],
        )
        speed = table(
            ["gamma", "tok/s", "speedup", "acceptance rate"],
            [
                [str(r["gamma"]) if r["gamma"] else "baseline",
                 f"{r['tokens_per_s']:.1f}", f"{r['speedup']:.2f}x",
                 f"{r['acceptance_rate']:.3f}" if r["acceptance_rate"] else "-"]
                for r in spec["speedup"]
            ],
        )
        sections += [
            "## Speculative decoding",
            "",
            "**Correctness.** At temperature 0 the speculative output is identical, "
            "token for token, to the dense reference:",
            "",
            greedy,
            "",
            "**Distributional equivalence.** At temperature 1, the first emitted token "
            "is compared against exact draws from the target's own next-token "
            "distribution (chi-square test of homogeneity; p > 0.05 means the two "
            "samples are indistinguishable):",
            "",
            chi,
            "",
            "**Speed.**",
            "",
            speed,
            "",
            f"![speculative](artifacts/plots/{image})",
            "",
        ]

    Path(args.out).write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {len(list(plots.glob('*.png')))} plots in {plots}")


if __name__ == "__main__":
    main()
