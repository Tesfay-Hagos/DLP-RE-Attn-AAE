"""SSIM BANDWIDTH AND COMPONENT DECOMPOSITION.

Two checks the earlier window sweep did not perform.

(1) BANDWIDTH. `_fspecial_gauss_1d(size, sigma)` builds a Gaussian of width `sigma` and
    truncates it to `size`. Our first sweep varied `size` with `win_sigma` fixed at 1.5, so
    it truncated the kernel without changing its scale -- 11 -> 7 discards a few percent of
    the mass and the analysis scale barely moves. Here we vary sigma itself, with the
    window sized to the kernel (ws = 2*ceil(3*sigma)+1), which is the real scale parameter.

(2) COMPONENTS. SSIM factorises as l * cs, where
        l  = (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1)     luminance
        cs = (2*sig_xy   + C2) / (sig_x^2 + sig_y^2 + C2)    contrast/structure
    `cs` is a normalised local correlation: scale-free, and blind to residual magnitude.
    Scoring with each factor alone shows which one the substitution is actually using.

Identical frozen weights, identical reconstructions, only the scoring function changes.
Larger windows crop more under valid convolution, so every comparison is against an
l2 baseline computed on the SAME crop.
"""
import os, json, warnings, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "../../DL/dl_project.py")).read().split("\n")
ns = {"torch": torch, "nn": nn, "F": F, "np": np, "warnings": warnings, "math": math}
def grab(a, b):
    i = next(k for k, l in enumerate(src) if l.startswith(a))
    j = next(k for k, l in enumerate(src) if k > i and l.startswith(b))
    return "\n".join(src[i:j])
exec(grab("# %% [CELL 1.3b]", "# %% [CELL 1.4]"), ns)
AE = ns["AE"]

B = {"RSNA":   ("dl_results_RSNA_dl-v1_20260818-1013/results_dl/ckpt_dl-v1", "", "RSNA"),
     "VinCXR": ("dl_results_VinCXR_dl-v1-vincxr_20260818-1047/results_dl/ckpt_dl-v1-vincxr", "_ds-VinCXR", "VinCXR"),
     "LAG":    ("dl_results_LAG_dl-v1-lag_20260818-1111/results_dl/ckpt_dl-v1-lag", "_ds-LAG", "LAG")}
SIGMAS = [0.5, 1.0, 1.5, 3.0, 6.0]
C1, C2 = 0.01 ** 2, 0.03 ** 2          # data_range = 1


def gauss1d(sigma, size):
    c = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return g.view(1, 1, 1, -1)


def blur(x, w):
    x = F.conv2d(x, w, padding=0)
    return F.conv2d(x, w.transpose(2, 3), padding=0)


def ssim_parts(x, y, sigma):
    """Returns (l, cs) maps under a Gaussian of width `sigma`, valid convolution."""
    ws = 2 * int(math.ceil(3 * sigma)) + 1
    w = gauss1d(sigma, ws)
    mx, my = blur(x, w), blur(y, w)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx = blur(x * x, w) - mx2
    sy = blur(y * y, w) - my2
    sxy = blur(x * y, w) - mxy
    l  = (2 * mxy + C1) / (mx2 + my2 + C1)
    cs = (2 * sxy + C2) / (sx + sy + C2)
    return l, cs, ws


def load_test(name, size=64):
    D = os.path.abspath(os.path.join(ROOT, f"../../../../MedIAnomaly-Data-64/{name}"))
    d = json.load(open(f"{D}/data.json")); te0, te1 = d["test"]["0"], d["test"]["1"]
    rd = lambda ns_: np.stack([np.asarray(Image.open(f"{D}/images/{n}").convert("L")
                                          .resize((size, size)), dtype=np.float32) / 255. for n in ns_])
    x = np.concatenate([rd(te0), rd(te1)])[:, None]
    y = np.concatenate([np.zeros(len(te0), int), np.ones(len(te1), int)])
    return torch.from_numpy(x * 2 - 1), y


def recon(net, x, bs=256):
    with torch.no_grad():
        return torch.cat([net(x[i:i+bs])["x_hat"] for i in range(0, len(x), bs)])


if __name__ == "__main__":
    print("=" * 86)
    print("SSIM BANDWIDTH SWEEP — varying the Gaussian width, not just the truncation")
    print("All deltas are against an l2 score computed on the SAME valid-convolution crop.")
    print("=" * 86)
    out = {}
    for ds, (ck, tag, name) in B.items():
        x, y = load_test(name); a01 = (x + 1) / 2
        print(f"\n{ds}")
        print(f"  {'sigma':>6}{'ws':>5}{'% width':>9}{'l2 (crop)':>11}{'SSIM':>9}{'delta':>9}"
              f"{'1-l only':>10}{'1-cs only':>11}")
        out[ds] = {}
        for sg in SIGMAS:
            aucs = {"l2c": [], "ssim": [], "l": [], "cs": []}
            for sd in (42, 43, 44):
                wp = os.path.join(ROOT, ck, f"ae{tag}_s{sd}", "net.pth")
                if not os.path.exists(wp): continue
                net = AE(input_size=64, in_planes=1, base_width=16, expansion=1,
                         mid_num=1024, latent_size=16, en_num_layers=1, de_num_layers=1)
                net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
                b01 = (recon(net, x) + 1) / 2
                l, cs, ws = ssim_parts(a01, b01, sg)
                # l2 on the SAME crop, so window size cannot flatter the comparison
                k = (64 - ws + 1)
                off = (64 - k) // 2
                r = ((a01 - b01) ** 2)[:, :, off:off+k, off:off+k]
                aucs["l2c"].append(roc_auc_score(y, r.mean(dim=[1,2,3]).numpy()) * 100)
                aucs["ssim"].append(roc_auc_score(y, (1 - l*cs).mean(dim=[1,2,3]).numpy()) * 100)
                aucs["l"].append(roc_auc_score(y, (1 - l).mean(dim=[1,2,3]).numpy()) * 100)
                aucs["cs"].append(roc_auc_score(y, (1 - cs).mean(dim=[1,2,3]).numpy()) * 100)
                del net
            m = {k_: float(np.mean(v)) for k_, v in aucs.items()}
            out[ds][sg] = m
            print(f"  {sg:>6.1f}{ws:>5}{ws/64*100:>8.1f}%{m['l2c']:>11.2f}{m['ssim']:>9.2f}"
                  f"{m['ssim']-m['l2c']:>+9.2f}{m['l']-m['l2c']:>+10.2f}{m['cs']-m['l2c']:>+11.2f}")
    json.dump(out, open(os.path.join(ROOT, "ssim_bandwidth_sweep.json"), "w"), indent=2)
    print("\n" + "=" * 86)
    print("DOES THE SIGN DISAGREEMENT SURVIVE THE BANDWIDTH SWEEP?")
    print("=" * 86)
    print(f"  {'sigma':>6}" + "".join(f"{d:>12}" for d in B) + "   signs")
    for sg in SIGMAS:
        ds_ = [out[d][sg]["ssim"] - out[d][sg]["l2c"] for d in B]
        agree = len({np.sign(v) for v in ds_}) == 1
        print(f"  {sg:>6.1f}" + "".join(f"{v:>+12.2f}" for v in ds_)
              + f"   {'all same' if agree else 'DISAGREE'}")
