"""Close two open audit items.

(1) CLAMP MISMATCH. ssim_window_sweep.py omitted the .clamp(0,1) that SSIMLoss applies, so
    the appendix window table reads +9.03 for the same quantity Table 1 calls +9.08. One
    quantity, two released numbers. Regenerate with the clamp.

(2) SEED BASIS. dark_region_check.json was computed on seed 42 only, while the paper says
    "on all three datasets" without qualifying the seed. Run all three seeds.
"""
import os, json, warnings, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
ROOT=os.path.dirname(os.path.abspath(__file__))
src=open(os.path.join(ROOT,"../../DL/dl_project.py")).read().split("\n")
ns={"torch":torch,"nn":nn,"F":F,"np":np,"warnings":warnings,"math":math}
def grab(a,b):
    i=next(k for k,l in enumerate(src) if l.startswith(a)); j=next(k for k,l in enumerate(src) if k>i and l.startswith(b))
    return "\n".join(src[i:j])
exec(grab("# %% [CELL 1.3b]","# %% [CELL 1.4]"),ns)
exec(grab("# %% [CELL 3.0]","# %% [CELL 3.0b]"),ns)
AE, ssim = ns["AE"], ns["ssim"]
B={"RSNA":("dl_results_RSNA_dl-v1_20260818-1013/results_dl/ckpt_dl-v1","","RSNA"),
   "VinCXR":("dl_results_VinCXR_dl-v1-vincxr_20260818-1047/results_dl/ckpt_dl-v1-vincxr","_ds-VinCXR","VinCXR"),
   "LAG":("dl_results_LAG_dl-v1-lag_20260818-1111/results_dl/ckpt_dl-v1-lag","_ds-LAG","LAG")}
def mk(): return AE(input_size=64,in_planes=1,base_width=16,expansion=1,mid_num=1024,
                    latent_size=16,en_num_layers=1,de_num_layers=1)
def load(name,split="test"):
    D=os.path.abspath(os.path.join(ROOT,f"../../../../MedIAnomaly-Data-64/{name}"))
    d=json.load(open(f"{D}/data.json"))
    rd=lambda n_: np.stack([np.asarray(Image.open(f"{D}/images/{n}").convert("L").resize((64,64)),dtype=np.float32)/255. for n in n_])
    if split=="train":
        x=rd(d["train"]["0"][:256]); return torch.from_numpy(x*2-1)[:,None], None
    t0,t1=d["test"]["0"],d["test"]["1"]
    x=np.concatenate([rd(t0),rd(t1)])[:,None]
    return torch.from_numpy(x*2-1), np.concatenate([np.zeros(len(t0),int),np.ones(len(t1),int)])
def recon(net,x):
    with torch.no_grad(): return torch.cat([net(x[i:i+256])["x_hat"] for i in range(0,len(x),256)])

print("(1) WINDOW SWEEP, WITH THE CLAMP  (win-11 must now match Table 1's +9.08)\n")
win_out={}
for ds,(ck,tag,name) in B.items():
    x,y=load(name); win_out[ds]={}
    per={w:[] for w in (11,7,5,3)}; base=[]
    for sd in (42,43,44):
        net=mk(); net.load_state_dict(torch.load(os.path.join(ROOT,ck,f"ae{tag}_s{sd}","net.pth"),map_location="cpu")); net.eval()
        xh=recon(net,x); del net
        base.append(roc_auc_score(y,((x-xh)**2).mean(dim=[1,2,3]).numpy())*100)
        a=((x+1)/2).clamp(0,1); b=((xh+1)/2).clamp(0,1)      # THE CLAMP, as SSIMLoss does
        for w in (11,7,5,3):
            sc=torch.cat([1-ssim(a[i:i+256],b[i:i+256],data_range=1.0,size_average=True,win_size=w)
                          for i in range(0,len(a),256)]).numpy()
            per[w].append(roc_auc_score(y,sc)*100)
    bm=float(np.mean(base)); win_out[ds]={"base":bm, **{str(w):float(np.mean(v)) for w,v in per.items()}}
    print(f"  {ds:<9}base {bm:6.2f}   " + "  ".join(f"w{w}: {np.mean(per[w])-bm:+6.2f}" for w in (11,7,5,3)))
json.dump(win_out,open(os.path.join(ROOT,"ssim_window_sweep.json"),"w"),indent=2)

print("\n(2) DARK-REGION CHECK, ALL THREE SEEDS\n")
dark={}
for ds,(ck,tag,name) in B.items():
    x,_=load(name,"train"); sh=[]
    for sd in (42,43,44):
        net=mk(); net.load_state_dict(torch.load(os.path.join(ROOT,ck,f"ae-ssim{tag}_s{sd}","net.pth"),map_location="cpu")); net.eval()
        xh=recon(net,x); del net
        mu=F.avg_pool2d(F.pad((x+1)/2,(5,5,5,5),mode='replicate'),11,stride=1).numpy()
        oor=(xh.abs()>1).numpy()
        q=np.quantile(mu,[0.2,0.4,0.6,0.8]); bk=np.digitize(mu,q)
        sh.append(float(oor[bk==0].sum())/max(float(oor.sum()),1)*100)
    dark[ds]={"darkest_quintile_share_pct":[round(v,1) for v in sh],
              "mean":round(float(np.mean(sh)),1),"min":round(float(np.min(sh)),1)}
    print(f"  {ds:<9}share of out-of-range pixels in the darkest quintile, per seed: "
          f"{', '.join(f'{v:.1f}%' for v in sh)}")
json.dump(dark,open(os.path.join(ROOT,"dark_region_check.json"),"w"),indent=2)
lo=min(d["min"] for d in dark.values())
print(f"\n  minimum across all 9 (dataset, seed) pairs: {lo:.1f}%")
print(f"  -> the paper may say '{int(lo)}% to 99.5%' across all three datasets AND all three seeds")
