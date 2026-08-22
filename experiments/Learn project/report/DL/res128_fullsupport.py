"""Full-support (replicate-padded) SSIM at 128px, on the frozen 128px weights.

The appendix bandwidth rows -- and therefore the sigma=0.75 prediction -- are
replicate-padded at FULL support. The 128px runs scored with the library default,
i.e. valid convolution, which also drops the border. Comparing the two directly
mixes conventions. This measures the missing arm so the comparison is like for like.
No training: weights are loaded frozen, only the scoring functional changes.
"""
import os, json, math, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = "/home/tesfayh/Artificial_inteligence/DL/project/Applications-of-AI-for-Anomaly-Detection/Anomaly-detection-project"
SRC  = open(os.path.join(ROOT, "experiments/Learn project/DL/dl_project.py")).read().split("\n")
ns = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math}
def grab(a, b):
    i = next(k for k, l in enumerate(SRC) if l.startswith(a))
    j = next(k for k, l in enumerate(SRC) if k > i and l.startswith(b))
    return "\n".join(SRC[i:j])
exec(grab("# %% [CELL 1.3b]", "# %% [CELL 1.4]"), ns)
AE = ns["AE"]

S = "/tmp/claude-1000/-home-tesfayh-Artificial-inteligence-DL-project-Applications-of-AI-for-Anomaly-Detection-Anomaly-detection-project/3de5e6cd-f0b7-41b6-af41-48266be9df74/scratchpad/v128"
BUN = {'RSNA':  ('dl_results_RSNA_res128-v1-rsna_20260820-2151','res128-v1-rsna','',''),
       'VinCXR':('dl_results_VinCXR_res128-v1-vincxr_20260820-2213','res128-v1-vincxr','_ds-VinCXR','VinCXR'),
       'LAG':   ('dl_results_LAG_res128-v1-lag_20260820-2303','res128-v1-lag','_ds-LAG','LAG')}
DATA = os.path.join(ROOT, "MedIAnomaly-Data-128")
C1, C2 = 0.01**2, 0.03**2

def g1(sg, ws):
    c = torch.arange(ws, dtype=torch.float) - ws // 2
    g = torch.exp(-(c**2) / (2 * sg**2)); g /= g.sum()
    return g.view(1, 1, 1, -1)

def blur(x, w):                      # replicate padding on BOTH axes -> full support
    p = w.shape[-1] // 2
    x = F.pad(x, (p, p, 0, 0), mode='replicate'); x = F.conv2d(x, w, padding=0)
    x = F.pad(x, (0, 0, p, p), mode='replicate'); return F.conv2d(x, w.transpose(2, 3), padding=0)

def ssim_full(x, y, sg=1.5):
    ws = 2 * int(math.ceil(3 * sg)) + 1; w = g1(sg, ws)
    mx, my = blur(x, w), blur(y, w); mx2, my2, mxy = mx*mx, my*my, mx*my
    sx = blur(x*x, w) - mx2; sy = blur(y*y, w) - my2; sxy = blur(x*y, w) - mxy
    m = ((2*mxy + C1) / (mx2 + my2 + C1)) * ((2*sxy + C2) / (sx + sy + C2))
    return (1 - m).mean(dim=[1,2,3]).numpy()

def load(name):
    d = os.path.join(DATA, name)
    dd = json.load(open(f"{d}/data.json")); t0, t1 = dd["test"]["0"], dd["test"]["1"]
    rd = lambda ns_: np.stack([np.asarray(Image.open(f"{d}/images/{n}").convert("L")
                                          .resize((128,128), Image.BILINEAR),
                                          dtype=np.float32)/255. for n in ns_])
    x = np.concatenate([rd(t0), rd(t1)])[:, None]
    return torch.from_numpy(x*2-1), np.concatenate([np.zeros(len(t0),int), np.ones(len(t1),int)])

out = {}
for ds, (bun, ver, tag, dname) in BUN.items():
    x, y = load(dname or 'RSNA'); a01 = ((x+1)/2).clamp(0,1)
    l2s, sfs = [], []
    for sd in (42, 43, 44):
        ck = os.path.join(S, bun, 'results_dl', 'ckpt_'+ver, f'ae{tag}_s{sd}', 'net.pth')
        net = AE(input_size=128, in_planes=1, base_width=16, expansion=1, mid_num=1024,
                 latent_size=16, en_num_layers=1, de_num_layers=1)
        net.load_state_dict(torch.load(ck, map_location='cpu')); net.eval()
        with torch.no_grad():
            xh = torch.cat([net(x[i:i+128])["x_hat"] for i in range(0, len(x), 128)])
        del net
        b01 = ((xh+1)/2).clamp(0,1)
        l2s.append(roc_auc_score(y, ((x-xh)**2).mean(dim=[1,2,3]).numpy())*100)
        sfs.append(roc_auc_score(y, ssim_full(a01, b01))*100)
    l2s, sfs = np.array(l2s), np.array(sfs)
    out[ds] = {'l2_full': l2s.tolist(), 'ssim_pad': sfs.tolist(),
               'l2_mean': float(l2s.mean()), 'ssim_pad_mean': float(sfs.mean()),
               'distance_delta': float((sfs-l2s).mean()),
               'distance_sd': float((sfs-l2s).std()),
               'per_seed': (sfs-l2s).tolist()}
    print(f"{ds} done", flush=True)
json.dump(out, open(os.path.join(ROOT, "experiments/Learn project/report/DL/res128_fullsupport.json"), "w"), indent=1)
PRED = {'RSNA':-3.31,'VinCXR':-8.77,'LAG':-18.39}
print("\n" + "="*78)
print("FULL-SUPPORT (replicate-padded) SSIM at 128px -- like for like with sigma=0.75")
print("="*78)
print(f"  {'ds':8}{'l2':>8}{'ssim_pad':>10}{'distance':>10}{'sd':>7}{'sigma=0.75':>12}{'err':>7}")
for ds, v in out.items():
    print(f"  {ds:8}{v['l2_mean']:8.2f}{v['ssim_pad_mean']:10.2f}{v['distance_delta']:+10.2f}"
          f"{v['distance_sd']:7.2f}{PRED[ds]:+12.2f}{abs(v['distance_delta']-PRED[ds]):7.2f}")
