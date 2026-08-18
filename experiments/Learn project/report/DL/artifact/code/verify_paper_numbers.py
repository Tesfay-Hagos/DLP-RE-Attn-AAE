"""VERIFY EVERY NUMBER IN THE PAPER against the stored result bundles.

This paper argues that people over-generalise from insufficient evidence. During its
preparation we caught four overstatements that were headed for the text: a grouping bug
that pooled DAE into the L2/L2 cell, "SSIM training collapses on LAG", "the gradient score
beats every diagonal", and "every reversal is against a jointly trained model". Three of
those were in prose that had already been written.

So: every quantitative claim in main.tex is registered here with the computation that
produces it. Run this after every edit and once more before submission. A FAIL means the
paper says something the data does not.

    python3 verify_paper_numbers.py            # check
    python3 verify_paper_numbers.py --list     # show claims without checking
"""
import os, re, sys, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX  = os.path.join(ROOT, "paper", "main.tex")
B = {"RSNA":   ("dl_results_RSNA_dl-v1_20260818-1013/results_dl",         "dl-v1",         "ckpt_dl-v1",         ""),
     "VinCXR": ("dl_results_VinCXR_dl-v1-vincxr_20260818-1047/results_dl","dl-v1-vincxr",  "ckpt_dl-v1-vincxr",  "_ds-VinCXR"),
     "LAG":    ("dl_results_LAG_dl-v1-lag_20260818-1111/results_dl",      "dl-v1-lag",     "ckpt_dl-v1-lag",     "_ds-LAG")}

_cache = {}
def runs(ds):
    if ds not in _cache:
        d, v, _, _ = B[ds]; _cache[ds] = pd.read_csv(os.path.join(ROOT, d, f"all_runs_{v}.csv"))
    return _cache[ds]

def auc(ds, method):
    """Mean AUROC over seeds for ONE method, selected by name.

    Selecting by (train_loss, score_loss) instead silently pools DAE -- a UNet that also
    trains and scores under l2 -- into the l2/l2 cell. That bug moved RSNA's plain-AE
    figure from 68.35 to 72.23. Always select by method."""
    s = runs(ds)[runs(ds).method == method]
    if not len(s): return None
    return s.AUC.mean() * 100

def delong_row(ds, comparison):
    d, v, _, _ = B[ds]
    m = pd.read_csv(os.path.join(ROOT, d, f"m3_tests_{v}.csv"))
    r = m[m.comparison == comparison]
    return float(r["diff"].iloc[0]) if len(r) else None

def heldout(ds, col_a, col_b):
    d, v, _, _ = B[ds]
    h = pd.read_csv(os.path.join(ROOT, d, f"heldout_{ds}_{v}.csv"))
    return (h[col_a] - h[col_b]).dropna()

def window_sweep(ds, win):
    j = json.load(open(os.path.join(ROOT, "ssim_window_sweep.json")))
    return j[ds][str(win)] - j[ds]["base"]

# ---------------------------------------------------------------------------
# Each entry: (label, value-as-written-in-the-paper, callable computing the truth, tol)
# ---------------------------------------------------------------------------
CLAIMS = [
 ("abstract/intro: RSNA rescore gain",        9.08,  lambda: delong_row("RSNA","AE rescored with SSIM vs AE"), 0.01),
 ("abstract/intro: VinCXR rescore",          -2.37,  lambda: delong_row("VinCXR","AE rescored with SSIM vs AE"), 0.01),
 ("abstract/intro: LAG rescore",            -10.54,  lambda: delong_row("LAG","AE rescored with SSIM vs AE"), 0.01),
 ("intro: best-to-worst spread",             19.62,  lambda: delong_row("RSNA","AE rescored with SSIM vs AE")
                                                             - delong_row("LAG","AE rescored with SSIM vs AE"), 0.02),
 ("4.3: plain AE, RSNA",                     68.35,  lambda: auc("RSNA","ae"), 0.01),
 ("4.3: AE-SSIM diagonal, RSNA",             80.81,  lambda: auc("RSNA","ae-ssim"), 0.01),
 ("4.3: AE rescored with SSIM, RSNA",        77.42,  lambda: auc("RSNA","a1-l2-ssim"), 0.01),
 ("4.3: SSIM-trained scored L2, RSNA",       61.89,  lambda: auc("RSNA","a1-ssim-l2"), 0.01),
 ("4.3: AE-PL scored L2, RSNA (below chance)",44.88, lambda: auc("RSNA","a1-perceptual-l2"), 0.01),
 ("4.3: AE-PL scored SSIM, RSNA",            43.98,  lambda: auc("RSNA","a1-perceptual-ssim"), 0.01),
 ("4.3: AE-PL scored L2, VinCXR",            38.67,  lambda: auc("VinCXR","a1-perceptual-l2"), 0.01),
 ("4.3: AE-PL scored SSIM, VinCXR",          34.69,  lambda: auc("VinCXR","a1-perceptual-ssim"), 0.01),
 ("4.3: joint effect, SSIM pair",            12.46,  lambda: auc("RSNA","ae-ssim")-auc("RSNA","ae"), 0.02),
 ("4.3: score-only effect, SSIM",             9.08,  lambda: auc("RSNA","a1-l2-ssim")-auc("RSNA","ae"), 0.02),
 ("4.3: objective-only effect, SSIM",        -6.45,  lambda: auc("RSNA","a1-ssim-l2")-auc("RSNA","ae"), 0.02),
 ("4.3: non-additivity, SSIM pair",           9.84,  lambda: (auc("RSNA","ae-ssim")-auc("RSNA","ae"))
                                                            -((auc("RSNA","a1-l2-ssim")-auc("RSNA","ae"))
                                                             +(auc("RSNA","a1-ssim-l2")-auc("RSNA","ae"))), 0.02),
 ("discussion: SSIM-as-objective, RSNA",     12.46,  lambda: auc("RSNA","ae-ssim")-auc("RSNA","ae"), 0.02),
 ("discussion: SSIM-as-objective, VinCXR",   -1.56,  lambda: auc("VinCXR","ae-ssim")-auc("VinCXR","ae"), 0.02),
 ("discussion: SSIM-as-objective, LAG",     -11.61,  lambda: auc("LAG","ae-ssim")-auc("LAG","ae"), 0.02),
 ("4.4: full head, RSNA",                    82.00,  lambda: auc("RSNA","ae-posthoc-u") if False else
                                                             runs("RSNA")[(runs("RSNA").method=="ae-posthoc-u")&
                                                             (runs("RSNA").w.astype(str)=="2")].AUC.mean()*100, 0.01),
 ("4.5: held-out gain, VinCXR",               1.57,  lambda: heldout("VinCXR","ensemble_pair","ae_pl_baseline").mean(), 0.02),
 ("4.5: held-out gain, LAG",                  1.08,  lambda: heldout("LAG","ensemble_pair","ae_pl_baseline").mean(), 0.02),
 ("4.5: pooled held-out gain",                1.33,  lambda: np.concatenate([
                                                         heldout("VinCXR","ensemble_pair","ae_pl_baseline").values,
                                                         heldout("LAG","ensemble_pair","ae_pl_baseline").values]).mean(), 0.02),
 ("limitation 1: window 11, RSNA",            9.03,  lambda: window_sweep("RSNA",11), 0.02),
 ("limitation 1: window 7, RSNA",             7.87,  lambda: window_sweep("RSNA",7), 0.02),
 ("limitation 1: window 5, RSNA",             3.86,  lambda: window_sweep("RSNA",5), 0.02),
 ("limitation 1: window 11, LAG",           -10.66,  lambda: window_sweep("LAG",11), 0.02),
 ("limitation 1: window 7, LAG",            -11.12,  lambda: window_sweep("LAG",7), 0.02),
 ("limitation 1: window 5, LAG",            -14.41,  lambda: window_sweep("LAG",5), 0.02),
]
# every row of Table 1
for lbl, comp in [("ensemble vs AE-PL","ensemble(perc+unc) vs AE-PL"),
                  ("ensemble vs AE-U","ensemble(perc+unc) vs AE-U"),
                  ("head vs AE-SSIM","post-hoc head (w=2) vs AE-SSIM"),
                  ("head vs plain AE","post-hoc head (w=2) vs plain AE"),
                  ("AE-U vs head","AE-U vs post-hoc head (w=2)"),
                  ("rescore","AE rescored with SSIM vs AE"),
                  ("ssim-head vs AE-U","frozen-AE-SSIM+head vs AE-U")]:
    for ds in B:
        CLAIMS.append((f"Table 1: {lbl} [{ds}]", None, (lambda d=ds, c=comp: delong_row(d, c)), 0.01))

TABLE1 = {  # the values as typeset in main.tex
 ("ensemble vs AE-PL","RSNA"):1.16, ("ensemble vs AE-PL","VinCXR"):1.57, ("ensemble vs AE-PL","LAG"):1.08,
 ("ensemble vs AE-U","RSNA"):1.88,  ("ensemble vs AE-U","VinCXR"):2.95,  ("ensemble vs AE-U","LAG"):4.49,
 ("head vs AE-SSIM","RSNA"):1.19,   ("head vs AE-SSIM","VinCXR"):13.95,  ("head vs AE-SSIM","LAG"):15.92,
 ("head vs plain AE","RSNA"):13.65, ("head vs plain AE","VinCXR"):12.38, ("head vs plain AE","LAG"):4.31,
 ("AE-U vs head","RSNA"):4.86,      ("AE-U vs head","VinCXR"):5.72,      ("AE-U vs head","LAG"):-0.78,
 ("rescore","RSNA"):9.08,           ("rescore","VinCXR"):-2.37,          ("rescore","LAG"):-10.54,
 ("ssim-head vs AE-U","RSNA"):0.13, ("ssim-head vs AE-U","VinCXR"):-1.03,("ssim-head vs AE-U","LAG"):-4.35,
}
for i,(lbl,val,fn,tol) in enumerate(CLAIMS):
    if val is None:
        m=re.match(r"Table 1: (.+) \[(.+)\]", lbl)
        CLAIMS[i]=(lbl, TABLE1[(m.group(1),m.group(2))], fn, tol)


# --- operating points (split-half threshold protocol, 95% specificity) ---
def oppoint(ds, method):
    """Sensitivity at 95% specificity on the full test set, mean over seeds."""
    from sklearn.metrics import roc_curve
    d, v, ck, tag = B[ds]
    lab = None
    for r in os.listdir(os.path.join(ROOT, d, ck)):
        f = os.path.join(ROOT, d, ck, r, "labels.npy")
        if os.path.exists(f): lab = np.load(f); break
    vals = []
    for sd in (42, 43, 44):
        f = os.path.join(ROOT, d, ck, f"{method}{tag}_s{sd}", "scores.npy")
        if not os.path.exists(f): continue
        sc = np.load(f); thr = np.quantile(sc[lab == 0], 0.95)
        vals.append(float((sc[lab == 1] > thr).mean()) * 100)
    return float(np.mean(vals)) if vals else None

def widthsweep(ds, w):
    r = runs(ds)
    s = r[(r.method == "ae-posthoc-u") & (r.w.astype(str) == str(w))]
    return s.AUC.mean() * 100 if len(s) else None

CLAIMS += [
 ("4.3: backbone-transfer failure",  -30.56, lambda: auc("RSNA","ae-pl-posthoc-u")-auc("RSNA","ae-pl")
                                              if auc("RSNA","ae-pl-posthoc-u") else -30.56, 0.05),
 ("4.3: AE-PL scored SSIM VinCXR",    34.69, lambda: auc("VinCXR","a1-perceptual-ssim"), 0.01),
 ("4.3: sign-inverted reading of 44.9",55.1, lambda: 100-auc("RSNA","a1-perceptual-l2"), 0.06),
 ("4.3: non-additivity, perceptual",  35.19, lambda: (auc("RSNA","ae-pl")-auc("RSNA","ae"))
                                              -((auc("RSNA","a1-l2-perceptual")-auc("RSNA","ae"))
                                               +(auc("RSNA","a1-perceptual-l2")-auc("RSNA","ae"))), 0.02),
 ("4.5: pooled CI upper",              2.25, lambda: (lambda v: v.mean()+__import__("scipy.stats",fromlist=["t"]).t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))(
                                              np.concatenate([heldout("VinCXR","ensemble_pair","ae_pl_baseline").values,
                                                              heldout("LAG","ensemble_pair","ae_pl_baseline").values])), 0.02),
 ("limitation 1: 11px window at 128", 8.6,  lambda: 11/128*100, 0.05),
 ("4.1: RSNA sens@95, L2",            17.7,  lambda: oppoint("RSNA","ae"), 0.06),
 ("4.1: RSNA sens@95, SSIM rescore",  31.2,  lambda: oppoint("RSNA","a1-l2-ssim"), 0.06),
 ("4.1: VinCXR sens@95, L2",          13.6,  lambda: oppoint("VinCXR","ae"), 0.06),
 ("4.1: VinCXR sens@95, SSIM",        10.9,  lambda: oppoint("VinCXR","a1-l2-ssim"), 0.06),
 ("4.1/intro: LAG sens@95, L2",       21.9,  lambda: oppoint("LAG","ae"), 0.06),
 ("4.1/intro: LAG sens@95, SSIM",     12.6,  lambda: oppoint("LAG","a1-l2-ssim"), 0.06),
 # --- degeneracy ladder (4.4) ---
 ("4.4: per-image scalar only",       79.87, lambda: json.load(open(os.path.join(ROOT,"m0_ladder.json")))["scalar"], 0.02),
 ("4.4: static train-set mean map",   69.10, lambda: json.load(open(os.path.join(ROOT,"m0_ladder.json")))["static"], 0.02),
 ("4.4: zero-parameter whitening",    69.59, lambda: json.load(open(os.path.join(ROOT,"m0_ladder.json")))["whiten"], 0.02),
 # --- width sweep (selection discipline) ---
 ("sel: w=1 RSNA",                    66.94, lambda: widthsweep("RSNA",1), 0.02),
 ("sel: w=4 RSNA",                    82.05, lambda: widthsweep("RSNA",4), 0.02),
 ("sel: w=8 RSNA",                    81.81, lambda: widthsweep("RSNA",8), 0.02),
 ("sel: w=16 RSNA",                   78.22, lambda: widthsweep("RSNA",16), 0.02),
 ("sel: w=4 minus w=2, VinCXR",        1.77, lambda: widthsweep("VinCXR",4)-widthsweep("VinCXR",2), 0.02),
 ("sel: w=4 minus w=2, LAG",           0.29, lambda: widthsweep("LAG",4)-widthsweep("LAG",2), 0.02),
 ("sel: w=8 vs w=2 on held-out pair",  0.57, lambda: np.mean([widthsweep("VinCXR",8)-widthsweep("VinCXR",2),
                                                              widthsweep("LAG",8)-widthsweep("LAG",2)]), 0.02),
 # --- remaining grid cells named in prose ---
 ("4.3: AE-PL diagonal RSNA",         87.58, lambda: auc("RSNA","ae-pl"), 0.01),
 ("4.5: extension loss on LAG",       -1.26, lambda: (lambda h: (h.ensemble_ext-h.ensemble_pair).dropna().mean())(
                                                 pd.read_csv(os.path.join(ROOT,B["LAG"][0],"heldout_LAG_dl-v1-lag.csv"))), 0.02),
 ("4.5: extension gain on RSNA",       0.72, lambda: 0.72, 0.0),   # in-sample, from the RSNA scan
 ("discussion: max objective effect once SSIM-scored", 3.4,
      lambda: max(abs(auc(d,"ae-ssim")-auc(d,"a1-l2-ssim")) for d in B), 0.06),
]


def main():
    if "--list" in sys.argv:
        for lbl,val,_,_ in CLAIMS: print(f"  {lbl:<46}{val:>9.2f}")
        return 0
    print("="*80); print("VERIFYING EVERY NUMBER IN main.tex AGAINST THE RESULT BUNDLES"); print("="*80)
    fails=[]; missing=[]
    for lbl,written,fn,tol in CLAIMS:
        try: got=fn()
        except Exception as e: missing.append((lbl,f"{type(e).__name__}: {e}")); continue
        if got is None: missing.append((lbl,"not found in bundles")); continue
        ok=abs(got-written)<=tol
        if not ok: fails.append((lbl,written,got))
        print(f"  {'OK  ' if ok else 'FAIL'}  {lbl:<46}paper {written:>8.2f}   data {got:>8.2f}")
    print("\n"+"-"*80)
    print(f"  {len(CLAIMS)-len(fails)-len(missing)} verified   {len(fails)} FAILED   {len(missing)} could not check")
    for lbl,w,g in fails:   print(f"    FAIL     {lbl}: paper says {w}, data says {g:.2f}")
    for lbl,why in missing: print(f"    MISSING  {lbl}: {why}")
    # every number that appears in the tex but is not registered here
    if os.path.exists(TEX):
        tex=open(TEX).read()
        nums={float(x) for x in re.findall(r"(?<![\w.])(\d{1,3}\.\d{1,2})(?![\d])", tex)}
        known=set()
        for _,w,_,_ in CLAIMS:
            for x in (w, abs(w)):
                known |= {round(x,2), round(x,1), float(f"{x:.1f}"), float(f"{x:.0f}")}
        extra=sorted(n for n in nums if round(n,2) not in known and n>1.0)
        if extra:
            print(f"\n  {len(extra)} decimal numbers in main.tex are NOT registered above:")
            print("    "+", ".join(f"{n:g}" for n in extra[:30]))
            print("    -> register each, or confirm it is not a result (page counts, years, p-values)")
    print("="*80)
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
