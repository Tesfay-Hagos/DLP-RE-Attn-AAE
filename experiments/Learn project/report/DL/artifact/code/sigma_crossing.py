"""Where does the SSIM substitution cross zero, as a function of Gaussian bandwidth?

Limitation 1 predicts the RSNA gain vanishes at 128x128. That prediction rests on
sigma=1.5 at 128px being scale-equivalent to sigma~0.75 at 64px, and on 0.75 lying in
the all-negative region. Our earlier grid sampled {0.5, 1.0, 1.5, 3.0, 6.0}: RSNA is
negative at 0.5 and ALREADY POSITIVE at 1.0, so the crossing is somewhere in between
and 0.75 was never measured. This measures it.

Full-support (replicate-padded) SSIM against the full-image l2 score, identical frozen
weights, no retraining.
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
exec(grab("# %% [CELL 1.3b]","# %% [CELL 1.4]"),ns); AE=ns["AE"]
B={"RSNA":("dl_results_RSNA_dl-v1_20260818-1013/results_dl/ckpt_dl-v1","","RSNA"),
   "VinCXR":("dl_results_VinCXR_dl-v1-vincxr_20260818-1047/results_dl/ckpt_dl-v1-vincxr","_ds-VinCXR","VinCXR"),
   "LAG":("dl_results_LAG_dl-v1-lag_20260818-1111/results_dl/ckpt_dl-v1-lag","_ds-LAG","LAG")}
C1,C2=0.01**2,0.03**2
SIG=[0.5,0.65,0.75,0.85,1.0,1.5,2.0,3.0,6.0]
def g1(sg,ws):
    c=torch.arange(ws,dtype=torch.float)-ws//2; g=torch.exp(-(c**2)/(2*sg**2)); g/=g.sum(); return g.view(1,1,1,-1)
def blur(x,w):
    p=w.shape[-1]//2
    x=F.pad(x,(p,p,0,0),mode='replicate'); x=F.conv2d(x,w,padding=0)
    x=F.pad(x,(0,0,p,p),mode='replicate'); return F.conv2d(x,w.transpose(2,3),padding=0)
def ssim_sc(x,y,sg):
    ws=2*int(math.ceil(3*sg))+1; w=g1(sg,ws)
    mx,my=blur(x,w),blur(y,w); mx2,my2,mxy=mx*mx,my*my,mx*my
    sx=blur(x*x,w)-mx2; sy=blur(y*y,w)-my2; sxy=blur(x*y,w)-mxy
    m=((2*mxy+C1)/(mx2+my2+C1))*((2*sxy+C2)/(sx+sy+C2))
    return (1-m).mean(dim=[1,2,3]).numpy()
def load(name):
    D=os.path.abspath(os.path.join(ROOT,f"../../../../MedIAnomaly-Data-64/{name}"))
    d=json.load(open(f"{D}/data.json")); t0,t1=d["test"]["0"],d["test"]["1"]
    rd=lambda n_: np.stack([np.asarray(Image.open(f"{D}/images/{n}").convert("L").resize((64,64)),dtype=np.float32)/255. for n in n_])
    x=np.concatenate([rd(t0),rd(t1)])[:,None]
    return torch.from_numpy(x*2-1), np.concatenate([np.zeros(len(t0),int),np.ones(len(t1),int)])
if __name__=="__main__":
    out={}
    for ds,(ck,tag,name) in B.items():
        x,y=load(name); a01=((x+1)/2).clamp(0,1)
        acc={s:[] for s in SIG}; l2=[]
        for sd in (42,43,44):
            net=AE(input_size=64,in_planes=1,base_width=16,expansion=1,mid_num=1024,
                   latent_size=16,en_num_layers=1,de_num_layers=1)
            net.load_state_dict(torch.load(os.path.join(ROOT,ck,f"ae{tag}_s{sd}","net.pth"),map_location="cpu")); net.eval()
            with torch.no_grad(): xh=torch.cat([net(x[i:i+256])["x_hat"] for i in range(0,len(x),256)])
            del net
            b01=((xh+1)/2).clamp(0,1)
            l2.append(roc_auc_score(y,((x-xh)**2).mean(dim=[1,2,3]).numpy())*100)
            for s_ in SIG: acc[s_].append(roc_auc_score(y,ssim_sc(a01,b01,s_))*100)
        base=float(np.mean(l2)); out[ds]={"l2":base, **{str(s_):float(np.mean(v)-base) for s_,v in acc.items()}}
        print(f"{ds} done", flush=True)
    json.dump(out,open(os.path.join(ROOT,"sigma_crossing.json"),"w"),indent=2)
    print("\n"+"="*74)
    print("SSIM-minus-l2, full support, by Gaussian bandwidth")
    print("="*74)
    print(f"  {'dataset':<9}"+"".join(f"{'s='+str(s_):>10}" for s_ in SIG))
    for ds in B: print(f"  {ds:<9}"+"".join(f"{out[ds][str(s_)]:>+10.2f}" for s_ in SIG))
    r=out["RSNA"]["0.75"]
    print(f"\n  RSNA at sigma=0.75, the 128px scale-equivalent of the default: {r:+.2f}")
    print(f"  -> the 'RSNA gain vanishes at 128px' prediction is "
          f"{'SUPPORTED' if r<=0 else 'NOT SUPPORTED (still positive)'}")
    for ds in B:
        v=[out[ds][str(s_)] for s_ in SIG]
        cr=[f"{SIG[i]}-{SIG[i+1]}" for i in range(len(SIG)-1) if v[i]*v[i+1]<0]
        print(f"  {ds:<9} crossing in sigma = {', '.join(cr) if cr else 'none in [0.5,1.5]'}")
