r"""Rebuild the distance/support decomposition from the two measured sweeps.

The decomposition was originally produced by a one-off pass whose script was never
committed -- the artifact shipped support_decomposition.json without the code behind
it. This recovers it from series that ARE committed, and checks the recovery.

The split is an identity. With A(s) the AUROC of score s,

    A(s_val) - A(s_l2)  =  [A(s_pad) - A(s_l2)]  +  [A(s_val) - A(s_pad)]
                            \_____ distance _____/   \_____ support _____/

s_pad cancels, so the equality holds for ANY intermediate score and carries no
empirical content by itself. The content is the choice of pivot: s_pad is SSIM at the
same bandwidth, replicate-padded to full support, so it differs from s_l2 only in the
distance and from s_val only in the region scored.

Provenance of each term:
    l2_full     ssim_window_sweep.json  ["base"]        full-image pixel score
    ssim_pad    sigma_crossing.json     l2 + ["1.5"]    replicate-padded, sigma=1.5
    ssim_valid  ssim_window_sweep.json  ["11"]          valid convolution, 11x11

Both sweeps clamp to [0,1] and use sigma=1.5, so the three are commensurable.
NOTE ssim_bandwidth_sweep.json is a DIFFERENT series -- valid convolution against a
CROPPED l2 baseline, unclamped -- and must not be mixed in. The two disagree by 2.7
AUROC at their one shared bandwidth.

    python3 support_decomposition.py          # rebuild and check against the stored file
    python3 support_decomposition.py --write   # rebuild and overwrite it
"""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-3   # the two sweeps accumulate in different orders; display precision is 1e-2


def load(name):
    return json.load(open(os.path.join(ROOT, name), encoding="utf-8"))


def rebuild():
    sc, ws = load("sigma_crossing.json"), load("ssim_window_sweep.json")
    out = {}
    for ds in ("RSNA", "VinCXR", "LAG"):
        l2_full    = ws[ds]["base"]
        ssim_pad   = sc[ds]["l2"] + sc[ds]["1.5"]
        ssim_valid = ws[ds]["11"]
        out[ds] = {"replicate": {
            "l2_full": l2_full, "ssim_pad": ssim_pad, "ssim_valid": ssim_valid,
            "distance": ssim_pad - l2_full,
            "support":  ssim_valid - ssim_pad,
        }}
    return out


def main():
    new = rebuild()
    print("=" * 74)
    print("DISTANCE / SUPPORT DECOMPOSITION  (replicate padding, sigma=1.5)")
    print("=" * 74)
    print(f"  {'dataset':<9}{'l2_full':>10}{'ssim_pad':>10}{'ssim_val':>10}"
          f"{'distance':>11}{'support':>10}{'sum':>10}")
    for ds, v in new.items():
        r = v["replicate"]
        print(f"  {ds:<9}{r['l2_full']:>10.4f}{r['ssim_pad']:>10.4f}"
              f"{r['ssim_valid']:>10.4f}{r['distance']:>+11.4f}"
              f"{r['support']:>+10.4f}{r['distance']+r['support']:>+10.4f}")

    # The identity must hold to machine precision, by construction.
    worst = max(abs((r["replicate"]["distance"] + r["replicate"]["support"])
                    - (r["replicate"]["ssim_valid"] - r["replicate"]["l2_full"]))
                for r in new.values())
    print(f"\n  identity residual (must be ~0): {worst:.2e}")
    assert worst < 1e-9, "the identity failed -- arithmetic is wrong"

    if "--write" in sys.argv:
        json.dump(new, open(os.path.join(ROOT, "support_decomposition.json"), "w"), indent=1)
        print("  wrote support_decomposition.json")
        return 0

    stored = load("support_decomposition.json")
    bad = 0
    for ds, v in new.items():
        for k, val in v["replicate"].items():
            got = stored[ds]["replicate"][k]
            if abs(val - got) > TOL:
                print(f"  MISMATCH {ds}.{k}: rebuilt {val:.6f} vs stored {got:.6f}")
                bad += 1
    print(f"  vs stored file: {'all terms agree within %.0e' % TOL if not bad else '%d MISMATCHES' % bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
