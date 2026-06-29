"""Cross-model per-typology heatmap — where suppression lands.

Under the peer-benchmark incentive, plots under-escalation per typology across all
models in the ladder run. The point: bright-line *overt* structuring stays at 0%
in every model, while the judgment-call typologies bend — the failure is
typology-shaped, and that shape generalizes across providers. Reads an existing
ladder run (default results/runs/ladder_5model/multimodel.json); no model calls.

    uv run python -m eval.typology_multimodel
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from common.config import load_config, resolve

# Typology display order: bright-line first, then the judgment calls.
TYP = [
    ("structuring_overt", "Structuring\novert\n(bright-line)"),
    ("structuring_subtle", "Structuring\nsubtle"),
    ("rapid_passthrough", "Rapid\npass-through"),
    ("layering_gather", "Layering /\ngather-scatter"),
    ("fan_out_dispersion", "Fan-out\ndispersion"),
]
ROW_ORDER = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
             "gpt-4o", "replicate/meta/meta-llama-3-70b-instruct"]
# White -> project red (#E45756): 0% reads clean, high under-escalation reads red.
CMAP = LinearSegmentedColormap.from_list("wr", ["#FFFFFF", "#E45756"])


def plot(results: dict, path: str, cond: str = "quota") -> str:
    models = [m for m in ROW_ORDER if m in results] + [m for m in results if m not in ROW_ORDER]
    typs = [(k, lbl) for k, lbl in TYP
            if any(k in c.get(cond, {}).get("per_typology", {}) for c in results.values())]
    M = np.array([[results[m].get(cond, {}).get("per_typology", {}).get(k, np.nan)
                   for k, _ in typs] for m in models], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(typs))); ax.set_xticklabels([lbl for _, lbl in typs], fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=9.5)
    for i in range(len(models)):
        for j in range(len(typs)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 else "#222222",
                        fontweight="bold" if v >= 0.5 else "normal")
    fig.suptitle("Where suppression lands — per-typology under-escalation under the peer benchmark",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.925,
             "Bright-line overt structuring stays at 0% in every model; the judgment-call typologies bend.\n"
             "(GPT-4o and Llama also miss subtle structuring at neutral baseline — partly capability, not incentive.)",
             ha="center", va="top", fontsize=8.5, color="#444444")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("under-escalation of reportable alerts", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:.0%}"))
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ladder_5model")
    args = ap.parse_args()
    cfg = load_config()
    data = json.loads(resolve("results", "runs", args.run, "multimodel.json").read_text())
    out = plot(data["results"], str(resolve(cfg["reporting"]["plots_dir"], "typology_multimodel.png")))
    print(f"[typology_multimodel] wrote {out}")


if __name__ == "__main__":
    main()
