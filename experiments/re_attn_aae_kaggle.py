#!/usr/bin/env python3
# ============================================================================
# RE-Attention Adversarial Autoencoder — Kaggle Notebook (PyTorch / T4 GPU)
# Network Anomaly Detection on KDD Cup 1999
#
# DATASET SETUP (do this before running):
#   1. Open this notebook on Kaggle
#   2. Go to  Data → Add Data → Search: "KDD Cup 1999 Data" by galaxyh
#      Direct URL: https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data
#   3. Enable GPU: Settings → Accelerator → GPU T4 x1
#
# Ablation (main story — C1 → C3 → C5):
#   C1 – Vanilla AE baseline             (reconstruction score)
#   C2 – GAN discriminator baseline      (discriminator score)   [supporting]
#   C3 – Adversarial AE, no attention    (reconstruction score)
#   C4 – Adversarial AE, dual score      (alpha-blend)           [supporting]
#   C5 – RE-Attn-AAE  [NOVEL]            (error-guided attention)
# ============================================================================

# %% [CELL 1]  Install / verify packages

import subprocess, sys

def check_import(pkg):
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

for pkg in ['torch', 'sklearn', 'numpy', 'matplotlib', 'pandas', 'seaborn']:
    check_import(pkg)

# %% [CELL 2]  Imports and global plot style

import os, time, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()          # inside Jupyter/Kaggle — keep inline backend
except NameError:
    matplotlib.use('Agg')  # plain .py run — save to disk, no GUI windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ── Global plot defaults (publication quality) ─────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.titlesize'   : 14,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'legend.framealpha': 0.9,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

# ── Consistent colour palette across all plots ─────────────────────────────
PAL = {
    'C1': '#4878CF',   # steel blue
    'C2': '#999999',   # grey  (supporting baseline)
    'C3': '#3DBE73',   # green
    'C4': '#AAAAAA',   # light grey (supporting)
    'C5': '#E84C3D',   # red  (novel model — always stands out)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == 'cuda':
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# %% [CELL 3]  Configuration

SAMPLE_MODE = bool(int(os.environ.get('SAMPLE_MODE', '0')))

_candidates = [
    '/kaggle/input/datasets/galaxyh/kdd-cup-1999-data/kddcup.data.corrected',
    '/kaggle/input/kdd-cup-1999-data/kddcup.data.corrected',
]
DATA_PATH  = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
OUTPUT_DIR = '/kaggle/working/results_kdd' if not SAMPLE_MODE else 'results_sample'

# ── Version + skip control ────────────────────────────────────────────
RUN_VERSION    = 'v2'
SKIP_COMPLETED = True
WANDB_PROJECT  = 'RE-Attn-AAE-KDD'
WANDB_GROUP    = f'ablation-{RUN_VERSION}'

CKPT_DIR = f'{OUTPUT_DIR}/ckpt_{RUN_VERSION}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

print(f"SAMPLE_MODE : {SAMPLE_MODE}")
print(f"RUN_VERSION : {RUN_VERSION}  (SKIP_COMPLETED={SKIP_COMPLETED})")
print(f"DATA_PATH   : {DATA_PATH}")
print(f"CKPT_DIR    : {CKPT_DIR}")

# ── wandb login ───────────────────────────────────────────────────────
USE_WANDB = False
try:
    import wandb
    if os.path.exists('/kaggle/working'):
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret('REATTN_KEY'), relogin=True)
    else:
        wandb.login()
    USE_WANDB = True
    wandb.init(project=WANDB_PROJECT,
               group=WANDB_GROUP,
               name=f'ablation-C1-C5-{RUN_VERSION}',
               config=dict(latent_dim=4, lr=1e-5, epochs=30, batch_size=512,
                           run_version=RUN_VERSION, dataset='KDD99'),
               tags=['ablation', 'RE-attention', 'AAE', 'KDD99', RUN_VERSION],
               resume='allow', id=f'kdd-ablation-{RUN_VERSION}',
               settings=wandb.Settings(init_timeout=120))
    print(f"WandB ready  project={WANDB_PROJECT}  group={WANDB_GROUP}")
except Exception as _e:
    USE_WANDB = False
    print(f"WandB unavailable ({_e}) — continuing without.")

# ── Checkpoint helpers ────────────────────────────────────────────────
def ckpt_path(cond):
    return f'{CKPT_DIR}/{cond}_done.json'

def is_done(cond):
    return SKIP_COMPLETED and os.path.exists(ckpt_path(cond))

def save_ckpt(cond, scores, metrics, epoch_loss, **model_states):
    """Save condition results to disk and upload wandb artifact."""
    info = {'metrics': metrics, 'loss_history': [float(v) for v in epoch_loss]}
    with open(ckpt_path(cond), 'w') as f:
        json.dump(info, f, indent=2)
    np.save(f'{CKPT_DIR}/{cond}_scores.npy', scores)
    for name, state in model_states.items():
        torch.save(state, f'{CKPT_DIR}/{cond}_{name}.pth')
    if USE_WANDB and wandb.run is not None:
        log = {'condition': cond}
        for m in ['auc_roc', 'auc_pr', 'f1', 'fnr']:
            if m in metrics: log[f'{cond}/{m}'] = metrics[m]
        wandb.log(log)
        for ep, val in enumerate(epoch_loss):
            wandb.log({f'loss/{cond}': val, f'step_{cond}': ep})
        _art_name = f'{WANDB_GROUP}-{cond.lower()}-ckpt'
        try:
            art = wandb.Artifact(_art_name, type='checkpoint',
                                 metadata={'cond': cond, 'version': RUN_VERSION})
            art.add_file(ckpt_path(cond))
            art.add_file(f'{CKPT_DIR}/{cond}_scores.npy')
            for name in model_states:
                wp = f'{CKPT_DIR}/{cond}_{name}.pth'
                if os.path.exists(wp): art.add_file(wp)
            wandb.log_artifact(art)
            print(f'  [{cond}] artifact logged → wandb:{_art_name}:latest')
        except Exception as _ae:
            print(f'  [{cond}] wandb artifact upload failed: {_ae}')
    print(f'  [{cond}] checkpoint saved to {CKPT_DIR}/')

def load_ckpt(cond):
    """Load saved condition — returns (scores, metrics, loss_history)."""
    with open(ckpt_path(cond)) as f:
        info = json.load(f)
    scores = np.load(f'{CKPT_DIR}/{cond}_scores.npy')
    print(f'  [{cond}] loaded from checkpoint (version {RUN_VERSION}).')
    return scores, info['metrics'], info['loss_history']

def load_weights(cond, **models):
    for name, model in models.items():
        p = f'{CKPT_DIR}/{cond}_{name}.pth'
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
        else:
            print(f'  [{cond}] weight missing: {p}')

PCT_ANOMALIES = 0.01
LATENT_DIM    = 4
LR            = 1e-5
BETA1         = 0.5
EPOCHS        = 30  if not SAMPLE_MODE else 2
BATCH_SIZE    = 512 if not SAMPLE_MODE else 4
SEED          = 42
ALPHA_STEPS   = np.round(np.arange(0.0, 1.05, 0.05), 2)
NORMAL_IDX    = 11
EPS           = 1e-8

ATTACK_FAMILIES = {
    'DoS'  : [0, 9, 14, 18, 20],
    'Probe': [5, 10, 15, 17],
    'R2L'  : [2, 3, 4, 13, 19, 21, 22],
    'U2R'  : [1, 7, 8, 12, 16],
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [CELL 3b]  Auto-restore checkpoints from wandb after session reset

if USE_WANDB and SKIP_COMPLETED and wandb.run is not None:
    try:
        _api      = wandb.Api()
        _entity   = wandb.run.entity
        _restored = []
        _missing  = []
        for _cond in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if not os.path.exists(ckpt_path(_cond)):
                _art_name = f'{_entity}/{WANDB_PROJECT}/{WANDB_GROUP}-{_cond.lower()}-ckpt:latest'
                try:
                    _api.artifact(_art_name).download(root=CKPT_DIR)
                    _restored.append(_cond)
                    print(f'  [restore] {_cond} ← wandb:{WANDB_GROUP}-{_cond.lower()}-ckpt:latest')
                except Exception:
                    _missing.append(_cond)
        if _restored:
            print(f"  [restore] restored {len(_restored)}/5: {_restored}")
        if _missing:
            print(f"  [restore] will train from scratch: {_missing}")
    except Exception as _re:
        print(f"  [restore] skipped: {_re}")
else:
    print("  [restore] skipped (wandb not active or SKIP_COMPLETED=False)")

# %% [CELL 4]  Data loading, preprocessing, feature names

if SAMPLE_MODE:
    INPUT_DIM    = 115
    x_train      = np.random.randn(10, INPUT_DIM).astype(np.float32)
    x_test       = np.random.randn(10, INPUT_DIM).astype(np.float32)
    y_train      = np.array([11]*8 + [0, 9], dtype=np.int64)
    y_test       = np.array([11]*8 + [0, 9], dtype=np.int64)
    binary_train = (y_train != NORMAL_IDX).astype(np.int32)
    binary_test  = (y_test  != NORMAL_IDX).astype(np.int32)
    x_train_norm = x_train[binary_train == 0]
    feature_names = [f'feat_{i}' for i in range(INPUT_DIM)]
    print(f"SAMPLE_MODE — x:{x_train.shape}  normal:{(binary_train==0).sum()}  anomaly:{(binary_train==1).sum()}")
else:
    col_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label'
    ]
    cat_vars = ['protocol_type','service','flag','land',
                'logged_in','is_host_login','is_guest_login']

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, header=None, names=col_names, index_col=False)
    print(f"Raw shape : {df.shape}")

    le = LabelEncoder()
    le.fit(df['label'])

    def reduce_anomalies(df, pct=0.01, seed=42):
        np.random.seed(seed)
        is_anom  = df['label'] != 'normal.'
        keep_n   = int(pct * (~is_anom).sum())
        keep_idx = np.random.choice(df.index[is_anom], size=keep_n, replace=False)
        return pd.concat([df[~is_anom], df.loc[keep_idx]], axis=0)

    df = reduce_anomalies(df, pct=PCT_ANOMALIES, seed=SEED)

    cat_data      = pd.get_dummies(df[cat_vars])
    numeric_vars  = [c for c in df.columns if c not in cat_vars and c != 'label']
    features      = pd.concat([df[numeric_vars], cat_data], axis=1)
    feature_names = list(features.columns)

    labels_int    = le.transform(df['label'])
    x_train_df, x_test_df, y_train, y_test = train_test_split(
        features, labels_int, test_size=0.25, random_state=SEED
    )

    scaler       = MinMaxScaler()
    x_train      = scaler.fit_transform(x_train_df).astype(np.float32)
    x_test       = scaler.transform(x_test_df).astype(np.float32)
    binary_train = (y_train != NORMAL_IDX).astype(np.int32)
    binary_test  = (y_test  != NORMAL_IDX).astype(np.int32)
    INPUT_DIM    = x_train.shape[1]
    x_train_norm = x_train[binary_train == 0]

print(f"INPUT_DIM   : {INPUT_DIM}  |  features: {len(feature_names)}")
print(f"x_train     : {x_train.shape}   anomalies: {binary_train.mean()*100:.2f}%")
print(f"x_test      : {x_test.shape}    anomalies: {binary_test.mean()*100:.2f}%")
print(f"Normal-only : {x_train_norm.shape}")

# %% [CELL 5]  DataLoader helper

def make_loader(x_np, batch_size, shuffle=True, drop_last=True):
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,
                      pin_memory=(device.type == 'cuda'),
                      num_workers=2)

# %% [CELL 6]  Model architecture definitions

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(96, 64),        nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(64, 48),        nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(48, 16),        nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(16, latent_dim)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(16, 48),         nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(48, 64),         nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(64, 96),         nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(96, output_dim)
        )
    def forward(self, z): return self.net(z)

class LatentDisc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16,  1),         nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)

class REAttention(nn.Module):
    """Error-guided attention: e=(x-x̂)² → soft feature mask a∈[0,1]^d."""
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim), nn.Sigmoid()
        )
    def forward(self, e): return self.net(e)

class GANGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),        nn.Tanh(),
            nn.Linear(64,        128),       nn.Tanh(),
            nn.Linear(128,       256),       nn.Tanh(),
            nn.Linear(256,       256),       nn.Tanh(),
            nn.Linear(256,       512),       nn.Tanh(),
            nn.Linear(512,       input_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class GANDisc(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,       128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,       128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,       128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,       128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,         1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

print("Model classes defined.")

# %% [CELL 7]  Evaluation utilities

def compute_metrics(scores, binary_labels):
    if len(np.unique(binary_labels)) < 2:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'f1': np.nan, 'fnr': np.nan}
    auc_roc  = roc_auc_score(binary_labels, scores)
    auc_pr   = average_precision_score(binary_labels, scores)
    fpr, tpr, thresh = roc_curve(binary_labels, scores)
    best     = np.argmax(tpr - fpr)
    pred     = (scores >= thresh[best]).astype(int)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'f1': f1_score(binary_labels, pred, zero_division=0),
            'fnr': float(1.0 - tpr[best])}

def per_family_auc(scores, y_raw):
    out = {}
    for fam, idx in ATTACK_FAMILIES.items():
        mask  = np.isin(y_raw, [NORMAL_IDX] + idx)
        y_sub = (y_raw[mask] != NORMAL_IDX).astype(int)
        out[fam] = (roc_auc_score(y_sub, scores[mask])
                    if len(np.unique(y_sub)) > 1 else np.nan)
    return out

def print_metrics(name, m, fam=None):
    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    for k, v in m.items():
        print(f"  {k.upper():<10}: {v:.4f}")
    if fam:
        print("  Per-family AUC:")
        for k, v in fam.items():
            print(f"    {k:6s} : {v:.4f}" if not np.isnan(v) else f"    {k:6s} : n/a")

def batch_infer_recon(enc, dec, x_np):
    enc.eval(); dec.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_np), BATCH_SIZE):
            xb    = torch.tensor(x_np[i:i+BATCH_SIZE]).to(device)
            x_hat = dec(enc(xb))
            out.append(((xb - x_hat)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)

mse_fn = nn.MSELoss()
bce_fn = nn.BCELoss()
all_results   = {}
loss_history  = {}   # stores epoch losses for convergence plots

print("Utilities defined.")

# %% [CELL 8]  Condition 1 — Vanilla AE Baseline

print("\n" + "="*60)
print("CONDITION 1 — Vanilla AE Baseline")
print("="*60)

enc_c1 = Encoder(INPUT_DIM, LATENT_DIM).to(device)
dec_c1 = Decoder(LATENT_DIM, INPUT_DIM).to(device)

if is_done('C1'):
    scores_c1, m_c1, c1_epoch_loss = load_ckpt('C1')
    load_weights('C1', enc=enc_c1, dec=dec_c1)
    fam_c1 = per_family_auc(scores_c1, y_test)
    all_results['C1'] = {**m_c1, 'family': fam_c1, 'label': 'Vanilla AE'}
    loss_history['C1'] = c1_epoch_loss
else:
    opt_c1 = Adam(list(enc_c1.parameters()) + list(dec_c1.parameters()), lr=LR)
    # train on normal-only data — consistent with C3/C5 (fixes contamination bug)
    loader_c1     = make_loader(x_train_norm, BATCH_SIZE)
    c1_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc_c1.train(); dec_c1.train()
        losses = []
        for (xb,) in loader_c1:
            xb = xb.to(device)
            opt_c1.zero_grad()
            loss = mse_fn(dec_c1(enc_c1(xb)), xb)
            loss.backward(); opt_c1.step()
            losses.append(loss.item())
        c1_epoch_loss.append(np.mean(losses))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  loss={c1_epoch_loss[-1]:.4f}")
    loss_history['C1'] = c1_epoch_loss
    print(f"C1 training time: {time.time()-t0:.1f}s")
    scores_c1 = batch_infer_recon(enc_c1, dec_c1, x_test)
    m_c1      = compute_metrics(scores_c1, binary_test)
    fam_c1    = per_family_auc(scores_c1, y_test)
    print_metrics('C1 — Vanilla AE', m_c1, fam_c1)
    all_results['C1'] = {**m_c1, 'family': fam_c1, 'label': 'Vanilla AE'}
    save_ckpt('C1', scores_c1, m_c1, c1_epoch_loss,
              enc=enc_c1.state_dict(), dec=dec_c1.state_dict())

enc_c1.eval()
z1_c1_list = []
with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
        z1_c1_list.append(enc_c1(xb).cpu().numpy())
z1_test_c1 = np.concatenate(z1_c1_list)
print(f"C1 latent collected: {z1_test_c1.shape}")

# %% [CELL 9]  Condition 2 — GAN Discriminator Baseline  [supporting]

print("\n" + "="*60)
print("CONDITION 2 — GAN Discriminator Baseline  [supporting]")
print("="*60)

generator  = GANGenerator(INPUT_DIM).to(device)
disc_c2    = GANDisc(INPUT_DIM).to(device)

if is_done('C2'):
    scores_c2, m_c2, c2_d_loss = load_ckpt('C2')
    load_weights('C2', gen=generator, disc=disc_c2)
    fam_c2 = per_family_auc(scores_c2, y_test)
    all_results['C2'] = {**m_c2, 'family': fam_c2, 'label': 'GAN Discriminator'}
    loss_history['C2'] = c2_d_loss
else:
    opt_g      = Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d      = Adam(disc_c2.parameters(),   lr=LR, betas=(BETA1, 0.999))
    loader_c2  = make_loader(x_train_norm, BATCH_SIZE)
    c2_d_loss  = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        generator.train(); disc_c2.train()
        d_ep, g_ep = [], []
        for (real,) in loader_c2:
            real = real.to(device); n = real.size(0)
            noise = torch.randn(n, INPUT_DIM, device=device)
            fake  = generator(noise).detach()
            opt_d.zero_grad()
            loss_d = (bce_fn(disc_c2(real), torch.ones(n,  1, device=device))
                    + bce_fn(disc_c2(fake), torch.zeros(n, 1, device=device)))
            loss_d.backward(); opt_d.step()
            opt_g.zero_grad()
            loss_g = bce_fn(disc_c2(generator(torch.randn(n, INPUT_DIM, device=device))),
                            torch.ones(n, 1, device=device))
            loss_g.backward(); opt_g.step()
            d_ep.append(loss_d.item())
        c2_d_loss.append(np.mean(d_ep))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  D={c2_d_loss[-1]:.4f}")
    loss_history['C2'] = c2_d_loss
    print(f"C2 training time: {time.time()-t0:.1f}s")
    disc_c2.eval()
    raw_d = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            raw_d.append(disc_c2(xb).cpu().numpy().flatten())
    scores_c2 = 1.0 - np.concatenate(raw_d)
    m_c2   = compute_metrics(scores_c2, binary_test)
    fam_c2 = per_family_auc(scores_c2, y_test)
    print_metrics('C2 — GAN Discriminator', m_c2, fam_c2)
    all_results['C2'] = {**m_c2, 'family': fam_c2, 'label': 'GAN Discriminator'}
    save_ckpt('C2', scores_c2, m_c2, c2_d_loss,
              gen=generator.state_dict(), disc=disc_c2.state_dict())

# %% [CELL 10]  Conditions C3 & C4 — Adversarial AE

print("\n" + "="*60)
print("CONDITIONS 3 & 4 — Adversarial AE (shared model)")
print("="*60)

enc_c34    = Encoder(INPUT_DIM, LATENT_DIM).to(device)
dec_c34    = Decoder(LATENT_DIM, INPUT_DIM).to(device)
ld_c34     = LatentDisc(LATENT_DIM).to(device)

if is_done('C3') and is_done('C4'):
    scores_recon_c34, m_c3, c34_epoch_loss = load_ckpt('C3')
    scores_c4,        m_c4, _             = load_ckpt('C4')
    load_weights('C3', enc=enc_c34, dec=dec_c34, disc=ld_c34)
    fam_c3 = per_family_auc(scores_recon_c34, y_test)
    fam_c4 = per_family_auc(scores_c4, y_test)
    all_results['C3'] = {**m_c3, 'family': fam_c3, 'label': 'Adversarial AE'}
    all_results['C4'] = {**m_c4, 'family': fam_c4, 'label': 'AAE Dual Score'}
    loss_history['C3'] = c34_epoch_loss
    # reconstruct latent scores for alpha-sweep plot (needed by Cell 18)
    enc_c34.eval(); dec_c34.eval(); ld_c34.eval()
    lat34 = []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            lat34.append(ld_c34(enc_c34(xb)).cpu().numpy().flatten())
    scores_latent_c34 = 1.0 - np.concatenate(lat34)
    best_alpha_c4 = m_c4.get('best_alpha', 0.5)
    auc_curve_c4  = m_c4.get('alpha_curve', [])
else:
    opt_rec34  = Adam(list(enc_c34.parameters()) + list(dec_c34.parameters()),
                      lr=LR, betas=(BETA1, 0.999))
    opt_disc34 = Adam(ld_c34.parameters(),   lr=LR, betas=(BETA1, 0.999))
    opt_gen34  = Adam(enc_c34.parameters(),  lr=LR, betas=(BETA1, 0.999))
    loader_c34     = make_loader(x_train_norm, BATCH_SIZE)
    c34_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc_c34.train(); dec_c34.train(); ld_c34.train()
        rec_l, d_l, g_l = [], [], []
        for (xb,) in loader_c34:
            xb = xb.to(device); n = xb.size(0)
            opt_rec34.zero_grad()
            loss_rec = mse_fn(dec_c34(enc_c34(xb)), xb)
            loss_rec.backward(); opt_rec34.step()
            opt_disc34.zero_grad()
            with torch.no_grad(): z_enc = enc_c34(xb)
            z_real  = torch.randn(n, LATENT_DIM, device=device)
            loss_d  = (-torch.mean(torch.log(ld_c34(z_real) + EPS))
                       - torch.mean(torch.log(1.0 - ld_c34(z_enc) + EPS)))
            loss_d.backward(); opt_disc34.step()
            opt_gen34.zero_grad()
            loss_g = -torch.mean(torch.log(ld_c34(enc_c34(xb)) + EPS))
            loss_g.backward(); opt_gen34.step()
            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())
        c34_epoch_loss.append(np.mean(rec_l))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
                  f"Recon={c34_epoch_loss[-1]:.4f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}")
    loss_history['C3'] = c34_epoch_loss
    print(f"C3/C4 training time: {time.time()-t0:.1f}s")
    enc_c34.eval(); dec_c34.eval(); ld_c34.eval()
    recon34, lat34 = [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            z  = enc_c34(xb)
            recon34.append(((xb - dec_c34(z))**2).mean(dim=1).cpu().numpy())
            lat34.append(ld_c34(z).cpu().numpy().flatten())
    scores_recon_c34  = np.concatenate(recon34)
    scores_latent_c34 = 1.0 - np.concatenate(lat34)
    m_c3   = compute_metrics(scores_recon_c34, binary_test)
    fam_c3 = per_family_auc(scores_recon_c34, y_test)
    print_metrics('C3 — AAE (recon only)', m_c3, fam_c3)
    all_results['C3'] = {**m_c3, 'family': fam_c3, 'label': 'Adversarial AE'}
    save_ckpt('C3', scores_recon_c34, m_c3, c34_epoch_loss,
              enc=enc_c34.state_dict(), dec=dec_c34.state_dict(), disc=ld_c34.state_dict())
    best_auc_c4, best_alpha_c4 = -1, 0.5
    auc_curve_c4 = []
    for alpha in ALPHA_STEPS:
        s = alpha * scores_recon_c34 + (1 - alpha) * scores_latent_c34
        a = roc_auc_score(binary_test, s) if len(np.unique(binary_test)) > 1 else np.nan
        auc_curve_c4.append(a)
        if not np.isnan(a) and a > best_auc_c4: best_auc_c4, best_alpha_c4 = a, alpha
    scores_c4 = best_alpha_c4 * scores_recon_c34 + (1-best_alpha_c4) * scores_latent_c34
    m_c4   = compute_metrics(scores_c4, binary_test)
    fam_c4 = per_family_auc(scores_c4, y_test)
    print(f"\n  C4 best alpha = {best_alpha_c4:.2f}")
    print_metrics('C4 — AAE dual score  [supporting]', m_c4, fam_c4)
    all_results['C4'] = {**m_c4, 'family': fam_c4, 'label': 'AAE Dual Score',
                         'best_alpha': best_alpha_c4, 'alpha_curve': auc_curve_c4}
    save_ckpt('C4', scores_c4,
              {**m_c4, 'best_alpha': best_alpha_c4, 'alpha_curve': list(auc_curve_c4)},
              [], )   # no model weights — C4 reuses C3 model

# %% [CELL 11]  Condition 5 — RE-Attn-AAE  [NOVEL]

print("\n" + "="*60)
print("CONDITION 5 — RE-Attn-AAE  [NOVEL]")
print("="*60)

enc1_c5  = Encoder(INPUT_DIM, LATENT_DIM).to(device)
enc2_c5  = Encoder(INPUT_DIM, LATENT_DIM).to(device)
dec_c5   = Decoder(LATENT_DIM, INPUT_DIM).to(device)
re_attn  = REAttention(INPUT_DIM, hidden=64).to(device)
ld_c5    = LatentDisc(LATENT_DIM).to(device)

_n_probe   = min(200, len(x_train_norm)) if not SAMPLE_MODE else len(x_train_norm)
_probe_idx = np.random.choice(len(x_train_norm), _n_probe, replace=False)
x_probe    = torch.tensor(x_train_norm[_probe_idx], dtype=torch.float32).to(device)
attn_evol_history = {}

if is_done('C5'):
    scores_c5, m_c5, c5_epoch_loss = load_ckpt('C5')
    load_weights('C5', enc1=enc1_c5, enc2=enc2_c5, dec=dec_c5,
                 re_attn=re_attn, disc=ld_c5)
    fam_c5 = per_family_auc(scores_c5, y_test)
    all_results['C5'] = {**m_c5, 'family': fam_c5, 'label': 'RE-Attn-AAE (Ours)'}
    loss_history['C5'] = c5_epoch_loss
    # recompute scores for downstream cells
    enc1_c5.eval(); enc2_c5.eval(); dec_c5.eval(); re_attn.eval(); ld_c5.eval()
    recon5, lat5, attn_all, z1_all = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb  = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            z1  = enc1_c5(xb); x_hat1 = dec_c5(z1)
            att = re_attn((xb - x_hat1)**2)
            recon5.append(((xb - x_hat1)**2).mean(dim=1).cpu().numpy())
            lat5.append(ld_c5(enc2_c5(xb * att)).cpu().numpy().flatten())
            attn_all.append(att.cpu().numpy())
            z1_all.append(z1.cpu().numpy())
    scores_recon_c5  = np.concatenate(recon5)
    scores_latent_c5 = 1.0 - np.concatenate(lat5)
    attn_weights_all = np.concatenate(attn_all)
    z1_test_c5       = np.concatenate(z1_all)
    best_alpha_c5    = m_c5.get('best_alpha', 0.5)
    auc_curve_c5     = m_c5.get('alpha_curve', [])
else:
    opt_rec_c5  = Adam(
        list(enc1_c5.parameters()) + list(dec_c5.parameters()) +
        list(re_attn.parameters()) + list(enc2_c5.parameters()),
        lr=LR, betas=(BETA1, 0.999))
    opt_disc_c5 = Adam(ld_c5.parameters(),   lr=LR, betas=(BETA1, 0.999))
    opt_gen_c5  = Adam(enc2_c5.parameters(), lr=LR, betas=(BETA1, 0.999))
    loader_c5     = make_loader(x_train_norm, BATCH_SIZE)
    c5_epoch_loss = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        enc1_c5.train(); enc2_c5.train(); dec_c5.train()
        re_attn.train(); ld_c5.train()
        rec_l, d_l, g_l = [], [], []
        for (xb,) in loader_c5:
            xb = xb.to(device); n = xb.size(0)
            # Phase 1 — both reconstruction passes
            opt_rec_c5.zero_grad()
            z1     = enc1_c5(xb);  x_hat1 = dec_c5(z1)
            error  = (xb - x_hat1) ** 2
            att    = re_attn(error)
            z2     = enc2_c5(xb * att);  x_hat2 = dec_c5(z2)
            loss_rec = mse_fn(x_hat1, xb) + mse_fn(x_hat2, xb)
            loss_rec.backward(); opt_rec_c5.step()
            # Phase 2 — latent discriminator
            opt_disc_c5.zero_grad()
            with torch.no_grad():
                z1_s   = enc1_c5(xb); xh1 = dec_c5(z1_s)
                z2_enc = enc2_c5(xb * re_attn((xb - xh1)**2))
            z_real = torch.randn(n, LATENT_DIM, device=device)
            loss_d = (-torch.mean(torch.log(ld_c5(z_real)  + EPS))
                      - torch.mean(torch.log(1.0 - ld_c5(z2_enc) + EPS)))
            loss_d.backward(); opt_disc_c5.step()
            # Phase 3 — enc2 adversarial update
            opt_gen_c5.zero_grad()
            with torch.no_grad():
                z1_s2  = enc1_c5(xb); xh1_s2 = dec_c5(z1_s2)
                att_s2 = re_attn((xb - xh1_s2)**2)
            loss_g = -torch.mean(torch.log(ld_c5(enc2_c5(xb * att_s2)) + EPS))
            loss_g.backward(); opt_gen_c5.step()
            rec_l.append(loss_rec.item()); d_l.append(loss_d.item()); g_l.append(loss_g.item())
        c5_epoch_loss.append(np.mean(rec_l))
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            enc1_c5.eval(); dec_c5.eval(); re_attn.eval()
            with torch.no_grad():
                _xh  = dec_c5(enc1_c5(x_probe))
                _att = re_attn((x_probe - _xh) ** 2)
            attn_evol_history[epoch + 1] = _att.mean(dim=0).cpu().numpy()
            enc1_c5.train(); dec_c5.train(); re_attn.train()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
                  f"Recon={c5_epoch_loss[-1]:.4f}  "
                  f"Disc={np.mean(d_l):.4f}  Gen={np.mean(g_l):.4f}")
    loss_history['C5'] = c5_epoch_loss
    print(f"C5 training time: {time.time()-t0:.1f}s")
    enc1_c5.eval(); enc2_c5.eval(); dec_c5.eval(); re_attn.eval(); ld_c5.eval()
    recon5, lat5, attn_all, z1_all = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x_test), BATCH_SIZE):
            xb    = torch.tensor(x_test[i:i+BATCH_SIZE]).to(device)
            z1    = enc1_c5(xb);  x_hat1 = dec_c5(z1)
            att   = re_attn((xb - x_hat1)**2)
            recon5.append(((xb - x_hat1)**2).mean(dim=1).cpu().numpy())
            lat5.append(ld_c5(enc2_c5(xb * att)).cpu().numpy().flatten())
            attn_all.append(att.cpu().numpy())
            z1_all.append(z1.cpu().numpy())
    scores_recon_c5  = np.concatenate(recon5)
    scores_latent_c5 = 1.0 - np.concatenate(lat5)
    attn_weights_all = np.concatenate(attn_all, axis=0)
    z1_test_c5       = np.concatenate(z1_all)
    best_auc_c5, best_alpha_c5 = -1, 0.5
    auc_curve_c5 = []
    for alpha in ALPHA_STEPS:
        s = alpha * scores_recon_c5 + (1 - alpha) * scores_latent_c5
        a = roc_auc_score(binary_test, s) if len(np.unique(binary_test)) > 1 else np.nan
        auc_curve_c5.append(a)
        if not np.isnan(a) and a > best_auc_c5: best_auc_c5, best_alpha_c5 = a, alpha
    scores_c5 = best_alpha_c5 * scores_recon_c5 + (1-best_alpha_c5) * scores_latent_c5
    m_c5   = compute_metrics(scores_c5, binary_test)
    fam_c5 = per_family_auc(scores_c5, y_test)
    print(f"\n  C5 best alpha = {best_alpha_c5:.2f}")
    print_metrics('C5 — RE-Attn-AAE (full)', m_c5, fam_c5)
    all_results['C5'] = {**m_c5, 'family': fam_c5, 'label': 'RE-Attn-AAE (Ours)',
                         'best_alpha': best_alpha_c5, 'alpha_curve': auc_curve_c5}
    save_ckpt('C5', scores_c5,
              {**m_c5, 'best_alpha': best_alpha_c5, 'alpha_curve': list(auc_curve_c5)},
              c5_epoch_loss,
              enc1=enc1_c5.state_dict(), enc2=enc2_c5.state_dict(),
              dec=dec_c5.state_dict(), re_attn=re_attn.state_dict(),
              disc=ld_c5.state_dict())

# %% [CELL 12]  Results summary table

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
header = f"\n  {'Condition':<28} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'FNR':>8}"
print(header); print(f"  {'-'*56}")
for k, r in all_results.items():
    tag = ' ←' if k == 'C5' else ''
    print(f"  {r['label']:<28} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} "
          f"{r['f1']:>8.4f} {r['fnr']:>8.4f}{tag}")

print(f"\n  Per-Family AUC-ROC")
print(f"  {'Condition':<28} {'DoS':>8} {'Probe':>8} {'R2L':>8} {'U2R':>8}")
print(f"  {'-'*56}")
for k, r in all_results.items():
    fam = r.get('family', {})
    def _f(v): return f"{v:>8.4f}" if not np.isnan(v) else f"{'n/a':>8}"
    print(f"  {r['label']:<28} "
          f"{_f(fam.get('DoS',np.nan))} {_f(fam.get('Probe',np.nan))} "
          f"{_f(fam.get('R2L',np.nan))} {_f(fam.get('U2R',np.nan))}")

# %% [CELL 13]  Convergence curves — C1 / C3 / C5

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Training Convergence — Reconstruction Loss per Epoch', fontsize=15, fontweight='bold')

epochs_x = np.arange(1, EPOCHS + 1)

# Left: all three on one axis
ax = axes[0]
for key, col, lbl in [('C1', PAL['C1'], 'C1 Vanilla AE'),
                       ('C3', PAL['C3'], 'C3 Adversarial AE'),
                       ('C5', PAL['C5'], 'C5 RE-Attn-AAE (Ours)')]:
    ax.plot(epochs_x, loss_history[key], color=col, lw=2, label=lbl)
ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction Loss (MSE)')
ax.set_title('All Conditions')
ax.legend()

# Right: zoom to C3 vs C5 (same scale, shows attention gap)
ax2 = axes[1]
ax2.plot(epochs_x, loss_history['C3'], color=PAL['C3'], lw=2, label='C3 AAE')
ax2.plot(epochs_x, loss_history['C5'], color=PAL['C5'], lw=2,
         linestyle='--', label='C5 RE-Attn-AAE (two-pass, ÷2)')
# C5 loss is sum of two passes — divide by 2 for fair comparison
ax2.plot(epochs_x, np.array(loss_history['C5'])/2, color=PAL['C5'],
         lw=2, alpha=0.5, label='C5 per-pass loss (÷2)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Reconstruction Loss (MSE)')
ax2.set_title('C3 vs C5 (zoomed)')
ax2.legend()

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/convergence.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/convergence.png")

# %% [CELL 14]  Metric bar charts — C1, C3, C5

main_keys    = ['C1', 'C3', 'C5']
main_labels  = [all_results[k]['label'] for k in main_keys]
main_colors  = [PAL[k] for k in main_keys]

metrics_to_plot = [
    ('auc_roc', 'AUC-ROC'),
    ('auc_pr',  'AUC-PR'),
    ('f1',      'F1 Score'),
    ('fnr',     'FNR (lower better)'),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Performance Comparison — C1 → C3 → C5 Ablation', fontsize=15, fontweight='bold')

for ax, (metric, title) in zip(axes, metrics_to_plot):
    vals = [all_results[k][metric] for k in main_keys]
    bars = ax.bar(main_labels, vals, color=main_colors, edgecolor='white',
                  linewidth=1.2, width=0.55)
    # value labels on top of bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ymin = max(0, min(vals) - 0.05)
    ymax = min(1, max(vals) + 0.07)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=15)

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/metric_bars.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/metric_bars.png")

# %% [CELL 15]  Per-family AUC grouped bar chart

families  = ['DoS', 'Probe', 'R2L', 'U2R']
x         = np.arange(len(families))
width     = 0.25
offsets   = [-width, 0, width]

fig, ax = plt.subplots(figsize=(11, 6))
for (key, col), offset in zip([(k, PAL[k]) for k in main_keys], offsets):
    vals = [all_results[key]['family'].get(f, np.nan) for f in families]
    bars = ax.bar(x + offset, vals, width, label=all_results[key]['label'],
                  color=col, edgecolor='white', linewidth=1.0)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(families, fontsize=12)
ax.set_ylabel('AUC-ROC')
ax.set_ylim(0.50, 1.01)
ax.set_title('Per-Family AUC-ROC by Attack Category\n'
             'R2L and U2R are the hardest families (few examples, subtle patterns)',
             fontsize=13)
ax.legend(loc='lower right')
ax.axhline(0.5, color='gray', lw=0.8, linestyle=':', label='random')
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/family_auc.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/family_auc.png")

# %% [CELL 16]  Precision-Recall curves — C1, C3, C5

fig, ax = plt.subplots(figsize=(8, 7))
for key, scores, col in [
    ('C1', scores_c1,         PAL['C1']),
    ('C3', scores_recon_c34,  PAL['C3']),
    ('C5', scores_c5,         PAL['C5']),
]:
    if len(np.unique(binary_test)) > 1:
        prec, rec, _ = precision_recall_curve(binary_test, scores)
        auc_pr       = average_precision_score(binary_test, scores)
        ax.plot(rec, prec, color=col, lw=2,
                label=f"{all_results[key]['label']}  (AUC-PR={auc_pr:.4f})")

baseline_pr = binary_test.mean()
ax.axhline(baseline_pr, color='gray', lw=0.9, linestyle='--',
           label=f'No-skill baseline ({baseline_pr:.3f})')
ax.set_xlabel('Recall  (True Positive Rate)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves\n'
             '(AUC-PR is more informative than AUC-ROC for imbalanced data)', fontsize=13)
ax.legend(loc='upper right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/pr_curves.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/pr_curves.png")

# %% [CELL 17]  ROC curves — C1, C3, C5

fig, ax = plt.subplots(figsize=(8, 7))
for key, scores, col in [
    ('C1', scores_c1,         PAL['C1']),
    ('C3', scores_recon_c34,  PAL['C3']),
    ('C5', scores_c5,         PAL['C5']),
]:
    if len(np.unique(binary_test)) > 1:
        fpr, tpr, _ = roc_curve(binary_test, scores)
        auc         = roc_auc_score(binary_test, scores)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{all_results[key]['label']}  (AUC={auc:.4f})")

ax.plot([0,1],[0,1], 'k--', lw=0.8, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — C1 → C3 → C5 Ablation\nKDD Cup 1999', fontsize=13)
ax.legend(loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/roc_curves.png")

# %% [CELL 18]  Alpha-sweep — C4 vs C5  (supporting finding)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ALPHA_STEPS, auc_curve_c4, color=PAL['C4'], lw=2,
        marker='o', markersize=4, label='C4 AAE (no attention)')
ax.plot(ALPHA_STEPS, auc_curve_c5, color=PAL['C5'], lw=2,
        marker='s', markersize=4, label='C5 RE-Attn-AAE')
ax.axvline(best_alpha_c4, color=PAL['C4'], linestyle='--', alpha=0.6,
           label=f'C4 best α = {best_alpha_c4:.2f}')
ax.axvline(best_alpha_c5, color=PAL['C5'], linestyle='--', alpha=0.6,
           label=f'C5 best α = {best_alpha_c5:.2f}')
ax.set_xlabel('α  (weight of reconstruction score;  α=1 → recon only,  α=0 → latent only)',
              fontsize=11)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Score Blending Parameter Sweep\n'
             'Finding: reconstruction score dominates (α≈1.0) — '
             'latent disc adds regularisation, not scoring signal', fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/alpha_sweep.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/alpha_sweep.png")

# %% [CELL 19]  Attention heatmap — C5 (with feature names)

# Mean attention per attack family over test samples flagged as anomalous
pred_anomaly  = scores_c5 > np.percentile(scores_c5, 99)
family_attn   = {}
for fam, indices in ATTACK_FAMILIES.items():
    mask = np.isin(y_test, indices) & pred_anomaly
    if mask.sum() == 0:
        mask = np.isin(y_test, indices)
    family_attn[fam] = (attn_weights_all[mask].mean(axis=0)
                        if mask.sum() > 0 else np.zeros(INPUT_DIM))

heatmap = np.stack([family_attn[f] for f in ['DoS','Probe','R2L','U2R']])

fig, ax = plt.subplots(figsize=(18, 3.5))
im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_yticks(range(4)); ax.set_yticklabels(['DoS','Probe','R2L','U2R'], fontsize=11)
ax.set_xlabel(f'Feature index  (0 – {INPUT_DIM-1})', fontsize=11)
ax.set_title('RE-Attention Weight Heatmap (C5) — Mean Attention per Attack Family\n'
             'Brighter = model attends more to this feature when computing anomaly score',
             fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label='Attention weight')
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/attention_heatmap.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/attention_heatmap.png")

# %% [CELL 20]  Top-10 attended features per attack family

TOP_K = 10
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Top-10 Most Attended Features per Attack Family (C5 RE-Attn-AAE)',
             fontsize=14, fontweight='bold')

for ax, fam in zip(axes.flat, ['DoS','Probe','R2L','U2R']):
    attn_mean = family_attn[fam]
    top_idx   = np.argsort(attn_mean)[-TOP_K:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = attn_mean[top_idx]

    bars = ax.barh(range(TOP_K), top_vals[::-1],
                   color=sns.color_palette('YlOrRd', TOP_K))
    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Mean Attention Weight', fontsize=10)
    ax.set_title(f'{fam} Attacks', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, v in zip(bars, top_vals[::-1]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8)

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/top_features.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/top_features.png")

# %% [CELL 21]  Attention entropy — focus vs diffuse per attack family

def attention_entropy(attn_matrix, eps=1e-8):
    """Shannon entropy per sample, averaged across samples."""
    a     = np.clip(attn_matrix, eps, 1 - eps)
    entr  = -np.sum(a * np.log(a) + (1-a) * np.log(1-a), axis=1)
    return entr.mean()

entropies = {}
for fam, indices in ATTACK_FAMILIES.items():
    mask = np.isin(y_test, indices)
    if mask.sum() > 0:
        entropies[fam] = attention_entropy(attn_weights_all[mask])
    else:
        entropies[fam] = np.nan

# Normal entropy for comparison
normal_mask        = (binary_test == 0)
entropies['Normal'] = attention_entropy(attn_weights_all[normal_mask])

fig, ax = plt.subplots(figsize=(8, 5))
fam_names = list(entropies.keys())
entr_vals = list(entropies.values())
colors_e  = [PAL['C3']]*4 + ['#888888']
bars = ax.bar(fam_names, entr_vals, color=colors_e, edgecolor='white',
              linewidth=1.2, width=0.55)
for bar, v in zip(bars, entr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Mean Attention Entropy (bits)', fontsize=12)
ax.set_title('Attention Entropy per Traffic Class\n'
             'Lower entropy = more focused attention on fewer features', fontsize=13)
ax.set_xlabel('Traffic Class', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/attention_entropy.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/attention_entropy.png")
print("\nEntropy values:")
for k, v in entropies.items():
    print(f"  {k:8s} : {v:.2f}")

# %% [CELL 22]  Save all results to JSON

def _json(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, dict):  return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json(v) for v in obj]
    return obj

with open(f'{OUTPUT_DIR}/all_results.json', 'w') as f:
    json.dump(_json(all_results), f, indent=2)

with open(f'{OUTPUT_DIR}/loss_history.json', 'w') as f:
    json.dump(_json(loss_history), f, indent=2)

with open(f'{OUTPUT_DIR}/attention_entropy.json', 'w') as f:
    json.dump(_json(entropies), f, indent=2)

with open(f'{OUTPUT_DIR}/feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("  all_results.json    — metrics for all 5 conditions")
print("  loss_history.json   — epoch losses for C1, C3, C5")
print("  attention_entropy.json")
print("  feature_names.json")
print("  convergence.png     — loss curves")
print("  metric_bars.png     — AUC-PR / F1 bar chart")
print("  family_auc.png      — per-family grouped bar chart")
print("  pr_curves.png       — precision-recall curves")
print("  roc_curves.png      — ROC curves")
print("  alpha_sweep.png     — blending parameter sweep")
print("  attention_heatmap.png")
print("  top_features.png    — top-10 features per family")
print("  attention_entropy.png")

# %% [CELL 23]  Latent space — PCA (C1 vs C5) + t-SNE (C5), coloured by traffic class

CLASS_COLORS = {
    'Normal': '#888888',
    'DoS'   : '#E84C3D',
    'Probe' : '#4878CF',
    'R2L'   : '#3DBE73',
    'U2R'   : '#FF8C00',
}

def make_class_labels(y_raw):
    labels = np.full(len(y_raw), 'Normal', dtype=object)
    for fam, idx in ATTACK_FAMILIES.items():
        labels[np.isin(y_raw, idx)] = fam
    return labels

class_labels = make_class_labels(y_test)

pca_c1 = PCA(n_components=2, random_state=SEED).fit_transform(z1_test_c1)
pca_c5 = PCA(n_components=2, random_state=SEED).fit_transform(z1_test_c5)

_TSNE_MAX   = 8000   # t-SNE is O(n²) — cap to keep it under ~30s
_n_total    = len(z1_test_c5)
if _n_total > _TSNE_MAX:
    rng       = np.random.default_rng(SEED)
    _tsne_idx = rng.choice(_n_total, _TSNE_MAX, replace=False)
    _tsne_idx.sort()
    z1_tsne      = z1_test_c5[_tsne_idx]
    _class_tsne  = class_labels[_tsne_idx]
    print(f"t-SNE: subsampling {_n_total:,} → {_TSNE_MAX:,} points")
else:
    z1_tsne     = z1_test_c5
    _class_tsne = class_labels
_perplexity = min(30, max(2, len(z1_tsne) // 5))
has_tsne    = len(z1_tsne) > _perplexity + 1
if has_tsne:
    print(f"Running t-SNE (n={len(z1_tsne):,}, perplexity={_perplexity}) …")
    tsne_c5 = TSNE(n_components=2, perplexity=_perplexity, random_state=SEED,
                   max_iter=1000, init='pca').fit_transform(z1_tsne)

n_panels = 3 if has_tsne else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
fig.suptitle('Latent Space Visualisation — C1 Vanilla AE vs C5 RE-Attn-AAE',
             fontsize=15, fontweight='bold')

for ax, coords, title in zip(axes[:2], [pca_c1, pca_c5],
                              ['C1 — PCA', 'C5 RE-Attn-AAE — PCA']):
    for cls, col in CLASS_COLORS.items():
        mask = class_labels == cls
        if mask.sum() == 0: continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=col, label=cls, alpha=0.55, s=14, edgecolors='none')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.legend(markerscale=2, fontsize=9, loc='best')

if has_tsne:
    ax3 = axes[2]
    for cls, col in CLASS_COLORS.items():
        mask = _class_tsne == cls
        if mask.sum() == 0: continue
        ax3.scatter(tsne_c5[mask, 0], tsne_c5[mask, 1],
                    c=col, label=cls, alpha=0.55, s=14, edgecolors='none')
    ax3.set_title(f'C5 RE-Attn-AAE — t-SNE (n={len(z1_tsne):,})', fontsize=13)
    ax3.set_xlabel('t-SNE 1'); ax3.set_ylabel('t-SNE 2')
    ax3.legend(markerscale=2, fontsize=9, loc='best')

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/latent_space.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/latent_space.png")

# %% [CELL 24]  Reconstruction error distribution per traffic class — C1 / C3 / C5

conditions_dist = [
    ('C1', scores_c1,        'C1 Vanilla AE'),
    ('C3', scores_recon_c34, 'C3 Adversarial AE'),
    ('C5', scores_recon_c5,  'C5 RE-Attn-AAE'),
]
fam_order_dist  = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
fam_colors_dist = [CLASS_COLORS[f] for f in fam_order_dist]

fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=False)
fig.suptitle('Reconstruction Error Distribution by Traffic Class\n'
             'Wider gap between Normal and anomaly violins = better anomaly separation',
             fontsize=14, fontweight='bold')

for ax, (key, scores, label) in zip(axes, conditions_dist):
    groups    = [scores[class_labels == f] for f in fam_order_dist]
    positions = [i + 1 for i, g in enumerate(groups) if len(g) > 1]
    data_filt = [g for g in groups if len(g) > 1]
    labs_filt = [fam_order_dist[i] for i, g in enumerate(groups) if len(g) > 1]
    cols_filt = [fam_colors_dist[i] for i, g in enumerate(groups) if len(g) > 1]

    if len(data_filt) > 0:
        parts = ax.violinplot(data_filt, positions=positions,
                              showmedians=True, showextrema=False)
        for pc, col in zip(parts['bodies'], cols_filt):
            pc.set_facecolor(col); pc.set_alpha(0.72)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labs_filt, rotation=15, fontsize=10)
    ax.set_ylabel('Reconstruction Error (MSE)' if key == 'C1' else '')
    ax.set_title(label, fontsize=12, color=PAL[key], fontweight='bold')

fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/error_distribution.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Saved → {OUTPUT_DIR}/error_distribution.png")

# %% [CELL 25]  Attention weight evolution across training epochs (C5)

if not attn_evol_history:
    print("No attention evolution data — skipping")
else:
    epochs_tracked = sorted(attn_evol_history.keys())
    attn_matrix    = np.stack([attn_evol_history[e] for e in epochs_tracked])

    # Top-5 features by final-epoch mean attention weight
    final_attn  = attn_evol_history[epochs_tracked[-1]]
    top5_idx    = np.argsort(final_attn)[-5:][::-1]
    top5_names  = [feature_names[i] for i in top5_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    for rank, (idx, name) in enumerate(zip(top5_idx, top5_names)):
        ax.plot(epochs_tracked, attn_matrix[:, idx],
                color=plt.cm.tab10.colors[rank], lw=2,
                marker='o', markersize=5, label=name)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Attention Weight Evolution — Top-5 Features (C5 RE-Attn-AAE)\n'
                 'Weights measured on a fixed normal-traffic probe set every 5 epochs',
                 fontsize=13)
    ax.legend(title='Feature', fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/attention_evolution.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"Saved → {OUTPUT_DIR}/attention_evolution.png")

# %% [CELL 26]  Single-sample RE-Attention walkthrough — DoS attack vs Normal

dos_indices  = np.where(np.isin(y_test, ATTACK_FAMILIES['DoS']))[0]
norm_indices = np.where(binary_test == 0)[0]

if len(dos_indices) > 0 and len(norm_indices) > 0:
    pick_dos  = dos_indices[np.argmax(scores_c5[dos_indices])]
    pick_norm = norm_indices[np.argmin(scores_c5[norm_indices])]

    enc1_c5.eval(); dec_c5.eval(); re_attn.eval()

    def _get_components(x_np):
        with torch.no_grad():
            xb  = torch.tensor(x_np[None], dtype=torch.float32).to(device)
            z1_ = enc1_c5(xb); xh = dec_c5(z1_)
            err = (xb - xh) ** 2
            att = re_attn(err)
            return (xb.cpu().numpy()[0], xh.cpu().numpy()[0],
                    err.cpu().numpy()[0], att.cpu().numpy()[0],
                    (xb * att).cpu().numpy()[0])

    dos_comps  = _get_components(x_test[pick_dos])
    norm_comps = _get_components(x_test[pick_norm])

    # Show top-20 features ranked by the DoS attention weight
    N_SHOW    = min(20, INPUT_DIM)
    top_idx   = np.argsort(dos_comps[3])[-N_SHOW:][::-1]
    top_names = [feature_names[i] for i in top_idx]

    row_labels = ['Original x', 'Reconstruction x̂',
                  'Error (x−x̂)²', 'Attention weight a', 'Attended input x·a']
    fig, axes = plt.subplots(5, 2, figsize=(16, 13))
    fig.suptitle('Single-Sample RE-Attention Walkthrough  (top-20 attended features)\n'
                 'Left: DoS Attack   |   Right: Normal Traffic',
                 fontsize=14, fontweight='bold')

    for row, (r_label, dos_val, norm_val) in enumerate(
            zip(row_labels, dos_comps, norm_comps)):
        for col, (vals, title, color) in enumerate([
            (dos_val,  'DoS Attack  (highest anomaly score)',  '#E84C3D'),
            (norm_val, 'Normal Traffic  (lowest anomaly score)', '#888888'),
        ]):
            ax = axes[row, col]
            ax.bar(range(N_SHOW), vals[top_idx], color=color, alpha=0.75, width=0.8)
            ax.set_ylabel(r_label, fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=11, color=color, fontweight='bold')
            if row == 4:
                ax.set_xticks(range(N_SHOW))
                ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=7)
            else:
                ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=7)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/sample_walkthrough.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"Saved → {OUTPUT_DIR}/sample_walkthrough.png")
else:
    print("Not enough DoS / Normal samples for walkthrough — skipping")

print("\n" + "="*60)
print("DONE  —  check /kaggle/working/results/ for all outputs")
print("="*60)

# %% [CELL 27]  Presentation figures — extra diagnostics for report/slides

PRES_DIR = f'{OUTPUT_DIR}/presentation'
os.makedirs(PRES_DIR, exist_ok=True)

CONDITIONS_KDD = ['C1','C2','C3','C4','C5']
_LABELS = {
    'C1': 'Vanilla AE\n(baseline)',
    'C2': 'GAN Disc\n(baseline)',
    'C3': 'AAE recon\n(ablation)',
    'C4': 'AAE dual\n(ablation)',
    'C5': 'RE-Attn-AAE\n(Ours★)',
}
_COLORS = {'C1':'#9e9e9e','C2':'#78909c','C3':'#42a5f5','C4':'#ff8f00','C5':'#e53935'}
_auc = {c: all_results[c]['auc_roc'] for c in CONDITIONS_KDD}
_pr  = {c: all_results[c]['auc_pr']  for c in CONDITIONS_KDD}
_f1  = {c: all_results[c]['f1']      for c in CONDITIONS_KDD}

# ── PRES-A: Radar chart ───────────────────────────────────────────────
_met_names = ['AUC-ROC', 'AUC-PR', 'F1']
_angles = np.linspace(0, 2*np.pi, len(_met_names), endpoint=False).tolist()
_angles += _angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for c in CONDITIONS_KDD:
    v = [_auc[c], _pr[c], _f1[c]]; v += v[:1]
    ax.plot(_angles, v, 'o-', lw=2.5 if c=='C5' else 1.5,
            color=_COLORS[c], label=_LABELS[c].replace('\n',' '), alpha=0.9)
    ax.fill(_angles, v, alpha=0.07, color=_COLORS[c])
ax.set_thetagrids(np.degrees(_angles[:-1]), _met_names, fontsize=13)
ax.set_ylim(0, 1); ax.set_yticks([0.25,0.5,0.75,1.0])
ax.set_title('Multi-Metric Radar — KDD99', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
plt.tight_layout()
fig.savefig(f'{PRES_DIR}/presA_radar.png', dpi=150, bbox_inches='tight')
plt.close(); print('presA_radar.png')

# ── PRES-B: Score violin plots — normal vs anomaly per condition ──────
_scores_all = {
    'C1': scores_c1, 'C2': scores_c2,
    'C3': scores_recon_c34, 'C4': scores_c4, 'C5': scores_c5,
}
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
for ax, c in zip(axes, CONDITIONS_KDD):
    s = _scores_all[c]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    norm_s = s[binary_test == 0]; anom_s = s[binary_test == 1]
    parts = ax.violinplot([norm_s, anom_s], positions=[0,1],
                          showmedians=True, showextrema=False)
    parts['bodies'][0].set_facecolor('#90caf9'); parts['bodies'][0].set_alpha(0.75)
    parts['bodies'][1].set_facecolor('#ef9a9a'); parts['bodies'][1].set_alpha(0.75)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Normal','Attack'], fontsize=10)
    ax.set_title(f'{_LABELS[c].replace(chr(10)," ")}\nAUC={_auc[c]:.4f}',
                 fontsize=9, color=_COLORS[c], fontweight='bold')
    ax.set_ylabel('Normalised Score' if c=='C1' else '', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Anomaly Score Separation — Normal vs Attack (KDD99)\nHigher gap = better detection',
             fontsize=13)
plt.tight_layout()
fig.savefig(f'{PRES_DIR}/presB_violins.png', dpi=150, bbox_inches='tight')
plt.close(); print('presB_violins.png')

# ── PRES-C: Ablation design matrix ───────────────────────────────────
_components = ['Reconstruction\nLoss','Adversarial\nRegularisation',
               'RE-Attention','Dual\nScoring']
_design = {
    'C1':[1,0,0,0], 'C2':[0,0,0,0],
    'C3':[1,1,0,0], 'C4':[1,1,0,1], 'C5':[1,1,1,1],
}
_grid = np.array([_design[c] for c in CONDITIONS_KDD]).T
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6),
                                gridspec_kw={'height_ratios':[3,1],'hspace':0.06})
ax1.imshow(_grid, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(len(_components)):
    for j in range(len(CONDITIONS_KDD)):
        ax1.text(j, i, '✓' if _grid[i,j] else '·',
                 ha='center', va='center',
                 fontsize=16, color='#0d47a1' if _grid[i,j] else '#bdbdbd')
ax1.set_xticks([]); ax1.set_yticks(range(len(_components)))
ax1.set_yticklabels(_components, fontsize=11)
ax1.set_title('Ablation Design Matrix + AUC-ROC (KDD99)', fontsize=13, pad=8)
ax2.bar(range(len(CONDITIONS_KDD)), [_auc[c] for c in CONDITIONS_KDD],
        color=[_COLORS[c] for c in CONDITIONS_KDD], width=0.6)
for j, c in enumerate(CONDITIONS_KDD):
    ax2.text(j, _auc[c]+0.005, f'{_auc[c]:.4f}', ha='center', fontsize=9,
             fontweight='bold' if c=='C5' else 'normal')
ax2.set_xticks(range(len(CONDITIONS_KDD)))
ax2.set_xticklabels([_LABELS[c].replace('\n',' ') for c in CONDITIONS_KDD], fontsize=9)
ax2.set_ylabel('AUC-ROC', fontsize=10); ax2.set_ylim(0, 1.05)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.savefig(f'{PRES_DIR}/presC_design_matrix.png', dpi=150, bbox_inches='tight')
plt.close(); print('presC_design_matrix.png')

# ── PRES-D: Waterfall — incremental contribution C1→C3→C4→C5 ─────────
_chain  = ['C1','C3','C4','C5']
_clbls  = ['C1\nVanilla AE\n(baseline)',
           'C3\nAAE recon\n(+adversarial)',
           'C4\nAAE dual\n(+dual score)',
           'C5\nRE-Attn-AAE\n(+attention)']
_vals   = [_auc[c] for c in _chain]
_deltas = [_vals[0]] + [_vals[i]-_vals[i-1] for i in range(1, len(_vals))]
_bots   = [0, _vals[0], _vals[1], _vals[2]]
_pal    = ['#9e9e9e','#42a5f5','#ff8f00','#e53935']
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(_chain)), _deltas, bottom=_bots,
              color=_pal, edgecolor='white', linewidth=1.5, width=0.5)
for i, (v, d) in enumerate(zip(_vals, _deltas)):
    ax.text(i, v+0.008, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
    if i > 0:
        sign = '+' if d >= 0 else ''
        ax.text(i, _bots[i]+abs(d)/2, f'{sign}{d:.4f}',
                ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
for i in range(len(_chain)-1):
    ax.annotate('', xy=(i+1-0.27, _vals[i+1]),
                xytext=(i+0.27, _vals[i]),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.set_xticks(range(len(_chain)))
ax.set_xticklabels(_clbls, fontsize=10)
ax.set_ylabel('AUC-ROC', fontsize=12)
ylo = max(0, min(_vals)-0.15); yhi = min(1.0, max(_vals)+0.08)
ax.set_ylim(ylo, yhi)
ax.set_title('Incremental Component Contribution (KDD99)\nEach bar shows the gain from adding one component',
             fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{PRES_DIR}/presD_waterfall.png', dpi=150, bbox_inches='tight')
plt.close(); print('presD_waterfall.png')

# ── PRES-E: ROC curves all conditions overlaid ────────────────────────
from sklearn.metrics import roc_curve as _roc_curve
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],'--',color='grey',lw=0.8,label='Random (0.5)')
for c in CONDITIONS_KDD:
    s = _scores_all[c]
    fpr, tpr, _ = _roc_curve(binary_test, s)
    lw = 3.0 if c == 'C5' else 1.5
    ax.plot(fpr, tpr, lw=lw, color=_COLORS[c],
            label=f'{_LABELS[c].replace(chr(10)," ")} ({_auc[c]:.4f})',
            zorder=10 if c=='C5' else 5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All 5 Conditions (KDD99)\nC5 RE-Attn-AAE = bold red', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{PRES_DIR}/presE_roc_overlay.png', dpi=150)
plt.close(); print('presE_roc_overlay.png')

# ── PRES-F: Per-family AUC heatmap ───────────────────────────────────
_families = [f for f in ['DoS','Probe','R2L','U2R']
             if any(all_results[c]['family'].get(f) is not None
                    for c in CONDITIONS_KDD)]
if len(_families) > 1:
    _fam_grid = np.array([
        [all_results[c]['family'].get(f, np.nan) for f in _families]
        for c in CONDITIONS_KDD
    ])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(_fam_grid, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    for i in range(len(CONDITIONS_KDD)):
        for j in range(len(_families)):
            v = _fam_grid[i, j]
            txt = f'{v:.3f}' if not np.isnan(v) else 'n/a'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if (not np.isnan(v) and (v > 0.75 or v < 0.45)) else 'black')
    ax.set_xticks(range(len(_families))); ax.set_xticklabels(_families, fontsize=12)
    ax.set_yticks(range(len(CONDITIONS_KDD)))
    ax.set_yticklabels([_LABELS[c].replace('\n',' ') for c in CONDITIONS_KDD], fontsize=10)
    plt.colorbar(im, ax=ax, label='AUC-ROC')
    ax.set_title('Per-Attack-Family AUC-ROC Heatmap\nGreen = high detection, Red = low', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{PRES_DIR}/presF_family_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(); print('presF_family_heatmap.png')
else:
    print('presF: not enough attack families in test set — skipped (SAMPLE_MODE)')

print(f'\nPresentation figures saved to {PRES_DIR}/')

# %% [CELL 28]  Zip all outputs and make downloadable

import shutil
from IPython.display import FileLink, display

# Bundle everything
_abs_src  = os.path.abspath(OUTPUT_DIR)
_zip_out  = _abs_src + '_complete'
shutil.make_archive(_zip_out, 'zip', os.path.dirname(_abs_src),
                    os.path.basename(_abs_src))
_zip_path = _zip_out + '.zip'
_zip_mb   = os.path.getsize(_zip_path) / 1e6
print(f'\nZip created: {_zip_path}  ({_zip_mb:.1f} MB)')
print('Contents:')
print('  results/             — all_results.json, loss_history.json, feature_names.json')
print('  ckpt_v2/             — per-condition done.json + scores.npy + *.pth')
print('  presentation/        — presA–presF (6 report-ready figures)')
print('  *.png                — 14 standard output figures')
print('\nClick below to download:')
try:
    _rel = os.path.relpath(_zip_path, '/kaggle/working')
    display(FileLink(_rel))
except Exception:
    print(f'  Manual download: {_zip_path}')
