"""SSIM WINDOW SWEEP — the mechanism test for Limitation 1.

The central result is that rescoring a frozen autoencoder with SSIM helps on RSNA and
hurts on LAG. The sharpest alternative reading is that this is an artefact of 64x64
resolution: SSIM's default 11x11 window spans about a sixth of the image, so it may be
measuring something closer to a global statistic than a local structural one.

A 128x128 replication changes resolution AND the window-to-image ratio at once. This
script changes ONLY the ratio, by recomputing the SSIM score at window sizes 11/7/5/3 on
the SAME frozen weights, the SAME reconstructions and the SAME images. If shrinking the
window attenuates the reversal, the ratio is implicated. If it does not, the reversal is
not about the window and the 128px run should confirm that.

No training. No GPU required. Runs from the downloaded result bundles.
"""
import os, sys, json, glob, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

SRC = "../../DL/dl_project.py"          # relative to this file's directory
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "../../../.."))

# ---- pull the network + SSIM definitions out of the notebook source, without running it
def load_defs(path):
    src = open(path).read().split("\n")
    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "warnings": warnings,
          "math": __import__("math"), "Image": Image}
    # cell 1.3b holds the network classes; cell 3.0 holds SSIM. Take both verbatim.
    def grab(start_marker, end_marker):
        i = next(k for k, l in enumerate(src) if l.startswith(start_marker))
        j = next(k for k, l in enumerate(src) if k > i and l.startswith(end_marker))
        return "\n".join(src[i:j])
    exec(grab("# %% [CELL 1.3b]", "# %% [CELL 1.4]"), ns)
    exec(grab("# %% [CELL 3.0]", "# %% [CELL 3.0b]"), ns)
    return ns

NS = load_defs(os.path.join(ROOT, SRC))
AE, ssim = NS["AE"], NS["ssim"]

BUNDLES = {
    "RSNA":   ("dl_results_RSNA_dl-v1_20260818-1013",        "ckpt_dl-v1",         "",             "RSNA"),
    "VinCXR": ("dl_results_VinCXR_dl-v1-vincxr_20260818-1047","ckpt_dl-v1-vincxr", "_ds-VinCXR",   "VinCXR"),
    "LAG":    ("dl_results_LAG_dl-v1-lag_20260818-1111",      "ckpt_dl-v1-lag",    "_ds-LAG",      "LAG"),
}
DATA = os.path.join(REPO, "MedIAnomaly-Data-64")
WINDOWS = [11, 7, 5, 3]
SEEDS = [42, 43, 44]


def load_test(name, size=64):
    d = json.load(open(os.path.join(DATA, name, "data.json")))
    te0, te1 = d["test"]["0"], d["test"]["1"]
    def rd(names):
        out = []
        for n in names:
            p = os.path.join(DATA, name, "images", n)
            im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
            out.append(np.asarray(im, dtype=np.float32) / 255.0)
        return np.stack(out)
    x = np.concatenate([rd(te0), rd(te1)])[:, None]          # (N,1,H,W) in [0,1]
    y = np.concatenate([np.zeros(len(te0), int), np.ones(len(te1), int)])
    return torch.from_numpy(x * 2 - 1), y                     # to [-1,1], the model's range


def recon(net, x, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(net(x[i:i + bs])["x_hat"])
    return torch.cat(outs)


def ssim_score(x, xh, win):
    """Anomaly score = 1 - SSIM, per image. Matches SSIMLoss(anomaly_score=True)."""
    a, b = (x + 1) / 2, (xh + 1) / 2                          # SSIM wants [0,1], data_range=1
    out = []
    for i in range(0, len(a), 256):
        s = ssim(a[i:i+256], b[i:i+256], data_range=1.0, size_average=True, win_size=win)
        out.append((1.0 - s))
    return torch.cat(out).numpy()


def l2_score(x, xh):
    return ((x - xh) ** 2).mean(dim=[1, 2, 3]).numpy()


if __name__ == "__main__":
    print("=" * 78)
    print("SSIM WINDOW SWEEP at fixed 64x64 — does the reversal depend on window/image ratio?")
    print("Same frozen weights, same reconstructions, same images. Only win_size changes.")
    print("=" * 78)
    rows = {}
    for ds, (bdir, ck, tag, dname) in BUNDLES.items():
        x, y = load_test(dname)
        print(f"\n{ds}  (n={len(y)}, {int(y.sum())} abnormal)")
        per_win = {w: [] for w in WINDOWS}; base = []
        for sd in SEEDS:
            wp = os.path.join(ROOT, bdir, "results_dl", ck, f"ae{tag}_s{sd}", "net.pth")
            if not os.path.exists(wp):
                print(f"  seed {sd}: weights missing"); continue
            # arguments mirror build_net() exactly, with the config fingerprint's values
            net = AE(input_size=64, in_planes=1, base_width=16, expansion=1,
                     mid_num=1024, latent_size=16, en_num_layers=1, de_num_layers=1)
            net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
            xh = recon(net, x)
            base.append(roc_auc_score(y, l2_score(x, xh)) * 100)
            for w in WINDOWS:
                per_win[w].append(roc_auc_score(y, ssim_score(x, xh, w)) * 100)
            del net
        b = float(np.mean(base))
        print(f"  {'L2 score (baseline)':<26}{b:7.2f}")
        rows[ds] = {"base": b}
        for w in WINDOWS:
            m = float(np.mean(per_win[w])); rows[ds][w] = m
            frac = w / 64.0
            print(f"  SSIM win={w:<2} ({frac*100:4.1f}% of image){m:9.2f}   "
                  f"delta vs L2 {m-b:+7.2f}")
    print("\n" + "=" * 78)
    print("THE QUESTION: does shrinking the window attenuate the reversal?")
    print("=" * 78)
    print(f"  {'window':<10}" + "".join(f"{d:>12}" for d in BUNDLES))
    for w in WINDOWS:
        print(f"  {'win='+str(w):<10}" + "".join(f"{rows[d][w]-rows[d]['base']:>+12.2f}" for d in BUNDLES))
    print("\n  (values are AUROC delta of the SSIM rescore against the L2 baseline)")
    json.dump(rows, open(os.path.join(ROOT, "ssim_window_sweep.json"), "w"), indent=2)
    print(f"\n  written -> ssim_window_sweep.json")
