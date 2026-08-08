"""Figure builders for the localization demo. Kept separate from app.py so the
plotting logic can be unit-tested / reused without a running Streamlit session.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _normalise(a: np.ndarray) -> np.ndarray:
    a_min, a_max = float(a.min()), float(a.max())
    return (a - a_min) / (a_max - a_min + 1e-8)


def sample_panel(
    image: np.ndarray,
    recon: np.ndarray,
    ssim_map: np.ndarray,
    attn_map: np.ndarray | None,
    cond: str,
    label: str,
    score: float,
    disc_score: float | None,
) -> plt.Figure:
    """One row: original | reconstruction | SSIM error heatmap | attention map
    (if the condition has one). Used per-condition in the localization demo."""
    n_panels = 4 if attn_map is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title(f"Original ({label})")

    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstruction")

    im = axes[2].imshow(_normalise(ssim_map), cmap="inferno")
    title = f"{cond} SSIM error (localization)"
    if score is not None:
        title += f"\nscore={score:.3f}"
    axes[2].set_title(title)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if attn_map is not None:
        im2 = axes[3].imshow(_normalise(attn_map), cmap="viridis")
        title2 = f"{cond} attention mask"
        if disc_score is not None:
            title2 += f"\ndisc={disc_score:.3f}"
        axes[3].set_title(title2)
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def metrics_bar_chart(df, metric: str, palette: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [palette.get(c, "#888888") for c in df.index]
    ax.bar(df.index, df[metric], color=colors)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.upper()} by condition")
    fig.tight_layout()
    return fig


def loss_curve_chart(loss_history: dict, palette: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, losses in loss_history.items():
        if cond not in palette:
            continue
        ax.plot(losses, label=cond, color=palette.get(cond))
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Loss curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save fig into out_dir with a clear, collision-safe filename. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
