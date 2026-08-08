"""Loads the exported asset bundle (config/metrics/scores/demo arrays/weights)
produced by Cell 19 ("Save streamlit assets") in re_attn_aae_kaggle-RSNA-ResNet.py.

Directory contract this expects, under DATA_DIR:
    config.json           hyperparams, palette, per-condition SSIM/disc thresholds
    metrics.json           per-condition auc_roc / auc_pr / f1 / bootstrap CI
    loss_history.json      per-condition per-epoch training loss
    scores/                scores_c{1..7}.npy, disc_c{3..7}.npy, binary_test.npy
    demo/                  images.npy, labels.npy, recon_c{4,6}.npy,
                            ssim_c{4,6}.npy, attn_c{4,6}.npy, disc_c{4,6}.npy
    weights/                *.pth model weights (optional; only needed for live inference)
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def load_config() -> dict:
    p = DATA_DIR / "config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    p = DATA_DIR / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_loss_history() -> dict:
    p = DATA_DIR / "loss_history.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_scores(cond: str) -> np.ndarray | None:
    p = DATA_DIR / "scores" / f"scores_{cond.lower()}.npy"
    return np.load(p) if p.exists() else None


@st.cache_data
def load_disc_scores(cond: str) -> np.ndarray | None:
    p = DATA_DIR / "scores" / f"disc_{cond.lower()}.npy"
    return np.load(p) if p.exists() else None


@st.cache_data
def load_binary_test() -> np.ndarray | None:
    p = DATA_DIR / "scores" / "binary_test.npy"
    return np.load(p) if p.exists() else None


@st.cache_data
def load_demo_bundle() -> dict | None:
    """Returns the precomputed per-sample demo arrays: images, labels, and
    per-condition reconstruction / SSIM-error-map / attention-map / disc-score,
    for whichever conditions have files present. Returns None if the demo
    bundle hasn't been exported yet (e.g. only scalar-score conditions exist
    so far) rather than crashing — see README "progressive" data contract."""
    demo_dir = DATA_DIR / "demo"
    if not (demo_dir / "images.npy").exists():
        return None
    bundle = {
        "images": np.load(demo_dir / "images.npy"),
        "labels": np.load(demo_dir / "labels.npy"),
        "conditions": {},
    }
    for p in sorted(demo_dir.glob("recon_c*.npy")):
        cond = p.stem.replace("recon_", "").upper()
        bundle["conditions"][cond] = {
            "recon": np.load(demo_dir / f"recon_{cond.lower()}.npy"),
            "ssim":  np.load(demo_dir / f"ssim_{cond.lower()}.npy"),
            "attn":  _load_optional(demo_dir / f"attn_{cond.lower()}.npy"),
            "disc":  _load_optional(demo_dir / f"disc_{cond.lower()}.npy"),
        }
    return bundle


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def available_conditions() -> list[str]:
    """Primary condition keys (C1..C7) present in metrics.json, in order."""
    metrics = load_metrics()
    return [k for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"] if k in metrics]
